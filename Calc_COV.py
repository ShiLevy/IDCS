#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Shiran Levy)s
"""
import numpy as np
import os.path as path
import scipy.ndimage as ndimage
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import NearestNDInterpolator
from csv import writer
import time
import pygimli as pg

def pairwise_dist(m):
    xx, yy = np.meshgrid(np.arange(0,m.shape[1]), np.arange(0,m.shape[0]))
    pairs = np.concatenate([yy.flatten()[:,np.newaxis],xx.flatten()[:,np.newaxis]],axis=1) # [x,y] coords of points
    
    return pairs

def cov_calc(h,a0,var,azimuth=0,alpha=1):
    if a0.size==1:
        #calculate the covariance matrix
        H = squareform(pdist(h,metric= 'euclidean'))
        Cov = var * np.exp(-np.power(H/a0,alpha)) # covariance function
    else:
        ang=azimuth; cang=np.cos(ang/180*np.pi); sang=np.sin(ang/180*np.pi);
        rot = np.array([[cang,-sang],[sang,cang]]);
        cx = np.where(np.diag(a0)==0, 0, rot/np.diag(a0)) 
        H = squareform(pdist(h@cx,metric= 'euclidean'))
        Cov = var * np.exp(-np.power(H,alpha)) # covariance function  
        
    return Cov

def extract_path(d,ind,ind2):
    #extracts the coordinates to calculate covariance matrix for 
    d_new = d[ind,:].copy()
    if len(d_new.shape)<2:
        d_new = d_new[np.newaxis]
    d_new = d_new[:,ind2]
    if len(d_new.shape)<2:
        d_new = d_new[:,np.newaxis]
        
    return d_new

# def woodbury_identity(A, U, V, C):
#     A_inv_diag = 1./np.diag(A)  # note! A_inv_diag is a vector!
#     B_inv = np.linalg.inv(np.linalg.inv(C) + (V * A_inv_diag) @ U)
#     return np.diag(A_inv_diag) - (A_inv_diag.reshape(-1,1) * U @ B_inv @ V * A_inv_diag)

def update_J(s_m, tt):
    G = np.zeros((len(tt.fop.data["valid"]), s_m.size))
    tt.Velocity = pg.Vector(np.float64(1./s_m.flatten()))
    tt.fop.createJacobian(1./tt.Velocity)
    J = pg.utils.sparseMatrix2Dense(tt.fop.jacobian())
    return J

def calc_likelihood(dist,s,modelObj,cand_index,ti,**kwargs):
    ''' calculates the likelihood for each MPS simulation step
        #input#
        @ dist: simulation grid with informed and uninformed points indicated
        @ s: simulated point location on the a flat grid
        @ sigma_d: standard deviation of the observational noise
        @ sigma_m: standard deviation of the prior
        @ A: linear forward operator
        @ V: covariance factor in the exponent (see function cov_calc)
        #output#
        @ mu_L: mean of likelihood
        @ Sigma_L: covariance of likelihood
    '''
    
    ind_InS = np.flatnonzero(dist)
    ind_I = np.delete(ind_InS,np.argwhere(ind_InS==s))
    ind_tot = np.arange(0,dist.size)

    pairs = pairwise_dist(dist.copy())
    # calculating mean and covariance of theta_2 given theta_1 
    mu1 = np.ones(ind_InS.size,dtype = np.float32)*modelObj.mu_m
    mu2 = np.ones(ind_tot.size,dtype = np.float32)*modelObj.mu_m
    
    reshDist = np.tile(dist,(cand_index.size,1,1))
    reshDist = reshDist.reshape((cand_index.size,-1),order='C')
    if 'chan' in kwargs:
        reshDist[:,s] = 1/(0.06 + (1-ti.flat[cand_index])*0.02)
    else:
        reshDist[:,s] = ti.flat[cand_index]
  
    #If it is the first simulated pixel or if the model is categorical enters here
    if modelObj.update==0:
        # start = time.time()
        if modelObj.LikeProb==1:
            # calculating the covariance function based on real training image parameters
            COV_tot = cov_calc(pairs,modelObj.a0,np.power(modelObj.sigma_m,2))
        if modelObj.LikeProb==2 or modelObj.LikeProb==4:
            # calculating the covariance function using the estimated parameters from the variogram
            COV_tot = cov_calc(pairs,modelObj.est_a0,modelObj.est_var,alpha=modelObj.est_alpha)
        elif (modelObj.LikeProb==3):
            # calculating the liklihood based on covariance function using the estimated parameters from the variogram and prior mean
            COV_tot = cov_calc(pairs,modelObj.est_a0,modelObj.est_var,alpha=modelObj.est_alpha)
    
        # Covariance matrices    
        E_11 = extract_path(COV_tot,ind_InS,ind_InS)       
        E_22 = extract_path(COV_tot,ind_tot,ind_tot)                 
        E_12 = extract_path(COV_tot,ind_InS,ind_tot)        
        E_21 = extract_path(COV_tot,ind_tot,ind_InS) 

        if (modelObj.LikeProb==3):
            mu_2g1 = mu2[:,np.newaxis]
            Sigma_2g1 = E_22
        else:
            mu_2g1 = mu2[:,np.newaxis] + np.matmul(np.matmul(E_21,np.linalg.inv(E_11)),(reshDist[:,ind_InS]-mu1)[...,np.newaxis])
            Sigma_2g1 = E_22 - np.matmul(np.matmul(E_21,np.linalg.inv(E_11)),E_12)
        # Computing the analytical likelihood given correlated unknown variables theta_2
        modelObj.update=1
        # print('Normal cov calculation time'+str(time.time()-start))
    # for continuous models the simulation is based on updates to the mean and covariance from the first step
    else:
        # start=time.time()
        mu_Xs = modelObj.mu_2g1[s]
        Cke = np.array(modelObj.sigma_2g1[s,s],ndmin=2)
        covUS = modelObj.sigma_2g1[s,:][np.newaxis]
        lambda_t = np.matmul(np.linalg.inv(Cke), covUS)

        mu_2g1 = (modelObj.mu_2g1 + np.matmul(lambda_t.transpose(),(reshDist[:,s] - mu_Xs)[...,np.newaxis,np.newaxis]))
        Sigma_2g1 = modelObj.sigma_2g1 - np.matmul(np.matmul(lambda_t.transpose(),Cke),lambda_t)
        # print('update cov calculation time'+str((time.time()-start)))
        
    # start=time.time()
    if modelObj.fw=='linear':
        A=modelObj.A
    elif modelObj.mu_2g1 is None:
        A=update_J(mu2, modelObj.tt)
    else:
        A=update_J(modelObj.mu_2g1, modelObj.tt)
    mu_L = np.matmul(A,mu_2g1)
    Sigma_L = np.eye(A.shape[0], dtype = np.float32)*modelObj.sigma_d**2 + np.matmul(np.matmul(A,Sigma_2g1),A.transpose())

    # Cd = np.eye(A.shape[0], dtype = np.float32)*modelObj.sigma_d
    # Sigma_L_inv = woodbury_identity(Cd, A, A.transpose(), Sigma_2g1)

    if modelObj.index.size>0:
        mu_L = np.delete(mu_L,modelObj.index, axis=1)
        Sigma_L = np.delete(Sigma_L,modelObj.index, axis=0)
        Sigma_L = np.delete(Sigma_L,modelObj.index, axis=1)
    misf = modelObj.d_obs - mu_L
    if modelObj.LikeProb==4:
        Sigma_L += modelObj.C_Tapp
        misf += modelObj.d_Tapp
    # logP = -( modelObj.d_obs.size / 2.0) * np.log(2.0 * np.pi) - 0.5 * np.linalg.det(Sigma_L_inv) - 0.5 * (np.matmul(np.matmul(misf.transpose(0,2,1),Sigma_L_inv),misf))
    logP = -( modelObj.d_obs.size / 2.0) * np.log(2.0 * np.pi) - 0.5 * np.linalg.slogdet(Sigma_L)[1] - 0.5 * (np.matmul(np.matmul(misf.transpose(0,2,1),np.linalg.inv(Sigma_L)),misf))
    # print('likelihood calculation time'+str((time.time()-start)))

        # if ii==0:
        # s_model = np.matmul(A,mu_2g1[0])
        # with open('/home/slevy/Desktop/3rd_project_MPS/PyQS-PyDS/results/n10k50/channels_prob2/mean_models.csv', 'a') as file:
        #     writer_object = writer(file,dialect='excel')
        #     writer_object.writerow(s_model.squeeze())   

    return logP.squeeze(), mu_L, mu_2g1, Sigma_2g1

def calc_covPrior(dist,sigma,a0,alpha=1):
    pairs = pairwise_dist(dist.copy())
    Cov = cov_calc(pairs,a0,np.power(sigma,2),alpha)
    
    return Cov

'''still need to take case for the randomness to be able to reproduce results'''
def sampleLike(logProb):
    meanP = np.mean(logProb)        #the mean is used to normalize the log-prob and prevent precision issues
    prob = np.exp(logProb-meanP)
    normProb = prob/np.sum(prob)
    indices = np.argsort(normProb)
    sorted_arr = np.array(normProb)[indices]
    cum_arr = np.cumsum(sorted_arr)
    rand = np.random.uniform(0,1,1)
    index = np.min(np.where(rand<cum_arr)[0])
    ind = indices[index]
    
    return ind