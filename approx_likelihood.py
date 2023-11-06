import numpy as np
import os.path as path
from scipy.spatial.distance import pdist, squareform
from set_fw import update_J

def pairwise_dist(m):
    ''' Calculate the lag distance between pair of points '''
    xx, yy = np.meshgrid(np.arange(0,m.shape[1]), np.arange(0,m.shape[0]))
    pairs = np.concatenate([yy.flatten()[:,np.newaxis],xx.flatten()[:,np.newaxis]],axis=1) # [x,y] coords of points
    
    return pairs

def cov_calc(h,a0,var,azimuth=0,alpha=1):
    ''' Calculate the covariance function '''
    if a0.size==1:
        #for isotropic model
        H = squareform(pdist(h,metric= 'euclidean'))
        Cov = var * np.exp(-np.power(H/a0,alpha)) # covariance function
    else:
        # for anisotropic model
        ang=azimuth; cang=np.cos(ang/180*np.pi); sang=np.sin(ang/180*np.pi);
        rot = np.array([[cang,-sang],[sang,cang]]);
        cx = np.where(np.diag(a0)==0, 0, rot/np.diag(a0)) 
        H = squareform(pdist(h@cx,metric= 'euclidean'))
        Cov = var * np.exp(-np.power(H,alpha)) # covariance function  
        
    return Cov

def extract_path(d,ind,ind2):
    ''' extracts the coordinates to calculate the covariance metrices for informed or uninformed pixels '''
    d_new = d[ind,:].copy()
    if len(d_new.shape)<2:
        d_new = d_new[np.newaxis]
    d_new = d_new[:,ind2]
    if len(d_new.shape)<2:
        d_new = d_new[:,np.newaxis]
        
    return d_new

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
    # calculating the prior mean of theta and theta_c 
    mu1 = np.ones(ind_InS.size,dtype = np.float32)*modelObj.mu_m
    mu2 = np.ones(ind_tot.size,dtype = np.float32)*modelObj.mu_m
    
    reshDist = np.tile(dist,(cand_index.size,1,1))
    reshDist = reshDist.reshape((cand_index.size,-1),order='C')
    if 'chan' in kwargs:
        reshDist[:,s] = 1/(0.06 + (1-ti.flat[cand_index])*0.02)
    else:
        reshDist[:,s] = ti.flat[cand_index]
  
    #If it is the first simulated pixel
    if modelObj.update==0:
        if modelObj.LikeProb==1:
            # calculating the covariance function based on real training image parameters (available for the MultiGaussian case only)
            COV_tot = cov_calc(pairs,modelObj.a0,np.power(modelObj.sigma_m,2))
        if modelObj.LikeProb==2:
            # calculating the covariance function using the estimated parameters from the variogram
            COV_tot = cov_calc(pairs,modelObj.est_a0,modelObj.est_var,alpha=modelObj.est_alpha)
    
        # Prior covariance matrices    
        E_11 = extract_path(COV_tot,ind_InS,ind_InS)       
        E_22 = extract_path(COV_tot,ind_tot,ind_tot)                 
        E_12 = extract_path(COV_tot,ind_InS,ind_tot)        
        E_21 = extract_path(COV_tot,ind_tot,ind_InS) 

        mu_2g1 = mu2[:,np.newaxis] + np.matmul(np.matmul(E_21,np.linalg.inv(E_11)),(reshDist[:,ind_InS]-mu1)[...,np.newaxis])
        Sigma_2g1 = E_22 - np.matmul(np.matmul(E_21,np.linalg.inv(E_11)),E_12)
        
        modelObj.update=1

    else:
        # Fast update of conditional mean and covariance
        mu_Xs = modelObj.mu_2g1[s]
        Cke = np.array(modelObj.sigma_2g1[s,s],ndmin=2)
        covUS = modelObj.sigma_2g1[s,:][np.newaxis]
        lambda_t = np.matmul(np.linalg.inv(Cke), covUS)

        mu_2g1 = (modelObj.mu_2g1 + np.matmul(lambda_t.transpose(),(reshDist[:,s] - mu_Xs)[...,np.newaxis,np.newaxis]))
        Sigma_2g1 = modelObj.sigma_2g1 - np.matmul(np.matmul(lambda_t.transpose(),Cke),lambda_t)
        
    if modelObj.fw=='linear':
        A=modelObj.A
    elif modelObj.mu_2g1 is None:
        A=update_J(mu2, modelObj.tt) # first update to the Jacobian
    else:
        A=update_J(modelObj.mu_2g1, modelObj.tt) # updating the Jacobian based on the last step mean if solver is non-linear
        
    mu_L = np.matmul(A,mu_2g1)  # Mean Eq. (10)
    Sigma_L = np.eye(A.shape[0], dtype = np.float32)*modelObj.sigma_d**2 + np.matmul(np.matmul(A,Sigma_2g1),A.transpose())  # Covariance Eq. (11)

    if modelObj.index.size>0:
        # if source-receiver angle is limited, deleting shot index outside of range
        mu_L = np.delete(mu_L,modelObj.index, axis=1)
        Sigma_L = np.delete(Sigma_L,modelObj.index, axis=0)
        Sigma_L = np.delete(Sigma_L,modelObj.index, axis=1)
        
    misf = modelObj.d_obs - mu_L    #calculating the data misfit with respect to the observed data 
        
    logP = -( modelObj.d_obs.size / 2.0) * np.log(2.0 * np.pi) - 0.5 * np.linalg.slogdet(Sigma_L)[1] - 0.5 * (np.matmul(np.matmul(misf.transpose(0,2,1),np.linalg.inv(Sigma_L)),misf)) 

    # returnung the (1) log-likelihood, (2) mean likelihood, kriging (3) mean and (4) covariance
    return logP.squeeze(), mu_L, mu_2g1, Sigma_2g1

def sampleLike(logProb):
    ''' sampling one MPS candidate proportionally to the approximate likelihood '''
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