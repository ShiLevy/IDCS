#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Shiran Levy)s
"""

import numpy
import scipy.ndimage as ndimage
import math
import matplotlib.pyplot as plt
import os
import array
from scipy.linalg import fractional_matrix_power
import time

os.environ["MKL_NUM_THREADS"] = "1" # removes fft paralelization

try:
	import mkl_fft as fft
except ImportError:
	try:
		import pyfftw.interfaces.numpy_fft  as fft
	except ImportError:
		import numpy.fft as fft
        
def qsSample(parameter,source,template,data_cond=0):
    (fftim,fftim2, imSize,dist,n,k)=parameter;
    for r in range(1,numpy.max(numpy.array(template.shape))//2):
        if numpy.logical_not(numpy.isnan(source[dist<=r])).sum()>=n:
            break;
    source[dist>r]=numpy.nan;
    extendSource=numpy.pad(source,((0,imSize[0]-source.shape[0]),(0,imSize[1]-source.shape[1])),'constant', constant_values=numpy.nan);
    extendtemplate=numpy.pad(template,((0,imSize[0]-source.shape[0]),(0,imSize[1]-source.shape[1])),'constant', constant_values=0);
    mismatchMap=numpy.real( fft.ifft2( fftim2 * numpy.conj(fft.fft2(extendtemplate* numpy.nan_to_num(extendSource*0+1))) 
                                      - 2 * fftim * numpy.conj(fft.fft2(extendtemplate* numpy.nan_to_num(extendSource)))));
    mismatchMap[-template.shape[0]+1:,:]=numpy.nan;
    mismatchMap[:,-template.shape[1]+1:]=numpy.nan;
    indexes=numpy.argpartition(numpy.roll(mismatchMap,tuple(x//2 for x in template.shape),(0,1)).flat,math.ceil(k));
    return indexes[int(math.floor(numpy.random.uniform(k)))];

def qs(ti,dst,path,template,n,k,true_model=None,modelObj=None,thresh=100):
    dist=numpy.zeros(shape=template.shape);
    dist[math.floor(dist.shape[0]/2),math.floor(dist.shape[1]/2)]=1;
    dist=ndimage.morphology.distance_transform_edt(1-dist);
    return runsim((fft.fft2(ti), fft.fft2(ti**2),ti.shape, dist, n, k),ti, dst,path,template,qsSample,thresh,true_model,modelObj);

def qsSampleCat(parameter,source,template,data_cond=0):
    (fftim, imSize,dist,n,k)=parameter;
    for r in range(1,numpy.max(numpy.array(template.shape))//2):
        if numpy.logical_not(numpy.isnan(source[dist<=r])).sum()>=n:
            break;
    source[dist>r]=numpy.nan;
    extendSource=numpy.pad(source,((0,imSize[0]-source.shape[0]),(0,imSize[1]-source.shape[1])),'constant', constant_values=numpy.nan);
    extendtemplate=numpy.pad(template,((0,imSize[0]-source.shape[0]),(0,imSize[1]-source.shape[1])),'constant', constant_values=0);
    mismatchMap=numpy.real( fft.ifft2( numpy.sum(numpy.stack([ fftim[:,:,x] * numpy.conj(fft.fft2(extendtemplate* numpy.nan_to_num(extendSource==x))) for x in range(fftim.shape[-1])],axis=2),axis=2)));
    mismatchMap[-template.shape[0]+1:,:]=numpy.nan;
    mismatchMap[:,-template.shape[1]+1:]=numpy.nan;
    indexes=numpy.argpartition(numpy.roll(mismatchMap,tuple(x//2 for x in template.shape),(0,1)).flat,math.ceil(k));
# 	return indexes[int(math.floor(numpy.random.uniform(k)))];
    return indexes[int(math.floor(numpy.random.uniform(k)))] if data_cond==0 else indexes[:int(k)];

def qsCat(ti,dst,path,template,n,k,true_model=None,modelObj=None,thresh=100,idr=None):
    dist=numpy.zeros(shape=template.shape);
    dist[math.floor(dist.shape[0]/2),math.floor(dist.shape[1]/2)]=1;
    dist=ndimage.morphology.distance_transform_edt(1-dist);
    unValue=numpy.unique(ti.flat);
    fftim = numpy.stack([fft.fft2(-1*(ti==x)) for x in unValue],axis=2);
    adjustedTi=numpy.ones(ti.shape)*numpy.nan;
    for x in range(unValue.size):
        adjustedTi[ti==unValue[x]]=x;
    dst, WRMSE, LP = runsim((fftim ,ti.shape, dist, n, k), adjustedTi, dst, path, template, qsSampleCat,thresh,true_model,modelObj);
    return unValue[dst.astype(int)], WRMSE, LP;

def runsim(parameter,ti,dst,path,template,sampler,thresh,true_model=None,modelObj=None):
    xL,yL=numpy.unravel_index(path, dst.shape)
    hx=template.shape[0]//2;
    hy=template.shape[1]//2;
    source=template.copy(); 
    
    for x,y,p,idx in zip(xL,yL,path,range(path.size)):
        #print(x,y,p,idx)
        # if(idx%(path.size//100)==0):
            # print(idx*100//path.size," %");
        source*=numpy.nan;
        source[max(0,x-hx)-(x-hx):min(dst.shape[0]-1,x+hx)-x+hx+1,max(0,y-hy)-(y-hy):min(dst.shape[1]-1,y+hy)-y+hy+1]=\
            dst[max(0,x-hx):min(dst.shape[0],x+hx)+1,max(0,y-hy):min(dst.shape[1],y+hy)+1];
        if (idx*100/path.size)>thresh:
            modelObj.data_cond=0
            
        simIndex=sampler(parameter,source,template)
        dst.flat[p]=ti.flat[simIndex];
    
    if sampler.__name__=='qsSampleCat':
        s_model = 1/(0.06 + (1-dst)*0.02)
    else:
        s_model = dst
    
    d_sim = numpy.matmul(modelObj.A,s_model.reshape((-1,1)))
    if modelObj.index.size>0:
        d_sim = numpy.delete(d_sim,modelObj.index, axis=0)
        
    e = modelObj.d_obs - d_sim
    WRMSE = numpy.sqrt(numpy.sum(numpy.power(e,2.0))/e.size) # e is a vector and not a 1 x d array 
    log_p = - ( modelObj.d_obs.size / 2.0) * numpy.log(2.0 * numpy.pi) - modelObj.d_obs.size * numpy.log(modelObj.sigma_d) - 0.5 * numpy.power(modelObj.sigma_d,-2.0) * numpy.sum(numpy.power(e,2.0))

    return dst, WRMSE, log_p;

def GelmanRubin(Sequences):
    """
    code snippet modified from line 121 in
    https://github.com/elaloy/SGANinv/blob/master/example_inversion_pytorch/mcmc_func.py 
    
    See:
    Gelman, A. and D.R. Rubin, 1992. 
    Inference from Iterative Simulation Using Multiple Sequences, 
    Statistical Science, Volume 7, Issue 4, 457-472.
    """

    n = Sequences.shape[1]                          # number of samples
    nrp = Sequences.shape[2]*Sequences.shape[3]     # number of params
    m = Sequences.shape[0]                          # number of chains

    if n < 10:
        R_stat = -2 * numpy.ones(nrp)  # n= nr. model parameters
        
    else:
        Sequences = Sequences.reshape((m,n,-1))
        Xi_bar = numpy.mean(Sequences,axis=1)
        mu_hat = numpy.mean(Sequences,axis=(0,1))
        
        si2 = numpy.var(Sequences,axis=1)
        
        B = n * (1/(m-1)) * numpy.sum((Xi_bar-mu_hat)**2,axis=0)
                
        s2 = numpy.mean(si2,axis=0).flatten()
        
        sigma_hat2 = ((n-1)/n) * s2 + B   
        
        # R-statistics
        R_stat = numpy.sqrt(sigma_hat2/s2)    
        
    
        # meanSeq = numpy.mean(Sequences.reshape((m,n,-1)),axis=1)
    
        # # Variance between the sequence means 
        # B = n * numpy.var(meanSeq,axis=0)
        
        # # Variances of the various sequences
        # varSeq=numpy.zeros((m,nrp))
        # for zz in range(0,m):
        #     varSeq[zz,:] = numpy.var(Sequences.reshape((m,n,-1))[zz,:],axis=0)
        
        # # Average of the within sequence variances
        # W = numpy.mean(varSeq,axis=0)
        
        # # Target variance
        # sigma2 = ((n - 1)/numpy.float(n)) * W + (1.0/n) * B
        
        # # R-statistic
        # R_stat2 = numpy.sqrt((m + 1)/numpy.float(m) * sigma2 / W - (n-1)/numpy.float(m)/numpy.float(n))
    
    return R_stat