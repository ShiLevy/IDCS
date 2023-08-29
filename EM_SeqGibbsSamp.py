#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Shiran Levy)s

This script combines the extended metropolis routine with a sequential Gibbs sampler based on QS
that generates prior realizations from the conditional pdf
"""

import numpy as np
import math
import os
import time
import json
import argparse
import pickle
import scipy.ndimage as ndimage

'''internal imports'''
from MCMC_sim import qs, qsCat, GelmanRubin
from set_fw import set_fw
from Model_obj import Model_obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended Metropolis with sequential Gibbs sampler based on QS simulation"
    )
    parser.add_argument("--numChains",default=8,type=int,help="number of MCMC chains",
    )
    parser.add_argument("--restart",default=1,type=bool,help="Restart existing run (1), new run (0)",
    )    
    parser.add_argument("--Iter",default=20000,type=int,help="number of MCMC iterations per chain",
    )
    parser.add_argument("--thin",default=100,type=int,help="iteration skip to tune delta and calculate statistics",
    )    
    parser.add_argument("--delta_adjust", default=20, type=int, help="number of step before delta is adjusted"
    )  
    parser.add_argument("--Ptarget", default=0.3, type=float, help="target acceptance rate"
    )
    parser.add_argument("--random", default=0, type=bool, help="reproducible results (0), random paths (1)"
    )   
    parser.add_argument("--sigma-d", default=1, type=float, help="standard deviation of the observational noise in [ns]"
    )    
    parser.add_argument("--TIsize", default=500, type=int, help="size of TI to use for MPS simulation"
    )   
    parser.add_argument("--x", default=50, type=int, help="X size"
    )   
    parser.add_argument("--y", default=100, type=int, help="Y size"
    )   
    parser.add_argument("--n", default=30, type=int, help="number of neighbors"
    )  
    parser.add_argument("--k", default=1.2, type=int, help="number of candidates (low value give more importance to MPS)"
    )  
    parser.add_argument("--alpha", default=0, type=int, help="factor of the weighting kernel"
    ) 
    parser.add_argument("--kernel-size", default=51, type=int, help="weighting kernel size"
    ) 
    parser.add_argument("--distributed", default=1, type=bool, help="parallel computing of realizations if more than 1"
    )
    parser.add_argument("--workdir", default='/users/slevy4/working_dir/conditional_MPS/', type=str, help="working directory (where all the models are"
    )
    parser.add_argument("--outdir", default= '/users/slevy4/working_dir/conditional_MPS/results/MCMC/', type=str, help="directory to save results in"
    )
    parser.add_argument("--case", default= 'channels', type=str, help="Test case and results name"
    )
    
    args = parser.parse_args()
    outdir = args.outdir+args.case+'_k'+str(args.k)+'_n'+str(args.n)+'_TIsize'+str(args.TIsize)+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(outdir+'run_commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    homeDir = args.workdir
    numChains = args.numChains
    Iter = args.Iter
    thin = args.thin
    # forward simulation settings
    rand = args.random        # set to random (1) or reproducable (0)
    cutSize = args.TIsize
    x = args.x
    y = args.y
    sigma_d  = args.sigma_d
    Pc = args.Ptarget
    delta_adjust = args.delta_adjust
    
    # QS parameters 
    n = args.n   # number of neighbors to caclulate based oI thought that maybe for the channels (and most probably for the lenses) case we try to implement something like what Gregoire suggested. Doing some "burn-in" n 
    k = args.k    # probability of choosing k candidates (choosing 1 has the risk of 
            # verbatim copy. 2 is 50% chance to choose either the first or the secod)
    kernel=np.ones((args.kernel_size,args.kernel_size));      # map of euclidian distances
    kernel[math.floor(kernel.shape[0]/2),math.floor(kernel.shape[1]/2)]=0;
    kernel = np.exp(-args.alpha*ndimage.morphology.distance_transform_edt(kernel))
    
    field = np.load('./TIs/'+args.case+'.npy')
    #field = np.load('./TIs/GaussianRandomField_Iso10.npy')
    if args.case!='channels' and args.case!='lenses':
        field = field*0.004+0.07  #assign valocity values
        sampler = qs
        fullTI = 1/field   #assign new mean and standard deviation to the field   
        if args.case=='GaussianRandomField': 
            true_model = fullTI[1200:1200+y,1200:1200+x]
            #true_model = 1/(np.load('./TIs/LowConnectedGaussian.npy')[633:633+y,1831:1831+x]*0.004+0.07)
        else:
            #true_model = fullTI[1182:1182+y,1083:1083+x]
            true_model = fullTI[633:633+y,1831:1831+x]

    else:
        a0=None
        sampler = qsCat
        fullTI = 1/field   #assign new mean and standard deviation to the field   
        if args.case=='channels':
            #true_model = fullTI[1020:1020+y,1040:1040+x]
            true_model = fullTI[1700:1700+y,2194:2194+x]    # test case
            #true_model = fullTI[1200:1200+y,1200:1200+x]   #test 1
            # true_model = fullTI[1560:1560+y,1350:1350+x]   #test 2
        else:
            #true_model = fullTI[1127:1127+y,1032:1032+x]
            true_model = fullTI[1738:1738+y,2052:2052+x]    # test case
            #true_model = fullTI[1200:1200+y,1200:1200+x]   #test 1
            # true_model = fullTI[1770:1770+y,1435:1435+x]   #test 2

    ti = fullTI[:cutSize,:cutSize];
    mu_m = np.mean(fullTI)
    sigma_m = np.std(fullTI)
    
    if args.restart:
        print("Restarting MCMC chains")
        with open(outdir+'MPSrun.pkl', 'rb') as f:
            tmp = pickle.load(f)
        model = tmp[0]
        realz = tmp[1]
        WRMSE = tmp[2]
        LP = tmp[3]
        delta_ar = tmp[4]
        AR = tmp[5]
        R_stat = tmp[6]
        Iter = tmp[7]
        thin = tmp[8]
        del tmp
        Mnew = realz[:,-1]
        log_p_old = LP[:,-1]
        wrmse = WRMSE[:,-1]
        delta = int(delta_ar[-1])
        x = model.x
        y = model.y
        numChains = realz.shape[0]
        if delta_ar[-1]!=0:            
            iterations = np.arange((realz.shape[1]-1)*thin,((realz.shape[1]-1)*thin)+args.Iter)
            realz = np.concatenate([realz,np.zeros((numChains,int(iterations.size/thin),y,x)).astype(np.float32)],axis=1)
            WRMSE = np.concatenate([WRMSE,np.zeros((numChains,int(iterations.size/thin))).astype(np.float32)],axis=1)
            LP = np.concatenate([LP,np.zeros((numChains,int(iterations.size/thin))).astype(np.float32)],axis=1)
            AR = np.concatenate([AR, np.zeros(int(iterations.size/delta_adjust)).astype(np.float16)],axis=0)
            R_stat = np.concatenate([R_stat,np.zeros((int(iterations.size/thin),x*y)).astype(np.float16)],axis=0)
            delta_ar = np.concatenate([delta_ar,np.zeros(int(iterations.size/delta_adjust))],axis=0)
        else:
            continueInd =  np.argwhere(LP==0)[0][1]-1
            iterations = np.arange(continueInd*thin,Iter)
            Mnew = realz[:,continueInd]
            log_p_old = LP[:,continueInd]
            wrmse = WRMSE[:,continueInd]
            delta = int(delta_ar[continueInd*10])
        
        Mnew = Mnew.astype(np.float64)
        cp = np.array([np.random.randint(0,y),np.random.randint(0,x)])
        dst = Mnew.copy()
        if sampler.__name__=='qsCat':
            dst = np.where((1/dst).round(2) == 0.06, 1, 0).astype(np.float16)
        dst[:,max(0,cp[0]-delta):min(y,cp[0]+delta)+1,max(0,cp[1]-delta):min(x,cp[1]+delta)+1] = np.nan
    else:
        
        d_obs, A, index = set_fw(y,x,s_model=true_model,loc=0,scale=sigma_d,spacing=1,SnR_spacing=1,limit_angle=0,dir=outdir)
        
        '''save A as sparse matrix, search if it is possible to make caclulation on sparse matrix'''
        model = Model_obj(data_cond=0, LikeProb=2, sampProp=0, x=x, y=y)
        model.sigma_d = sigma_d 
        model.sigma_m = sigma_m 
        model.mu_m = mu_m
        model.A = A
        model.d_obs = d_obs
        model.index = index
        
        realz = np.zeros((numChains,int(Iter/thin)+1,y,x)).astype(np.float32)
        WRMSE = np.zeros((numChains,int(Iter/thin)+1)).astype(np.float32)
        LP = np.zeros((numChains,int(Iter/thin)+1)).astype(np.float32)
        AR = np.zeros(int(Iter/delta_adjust)+1).astype(np.float16)
        R_stat = np.zeros((int(Iter/thin)+1,x*y)).astype(np.float16)
        wrmse = np.zeros(numChains).astype(np.float32)
        delta_ar = np.zeros(int(Iter/delta_adjust)+1).astype(int)
        delta=2#int(max(x,y)*0.5)#np.random.randint(0,int(min(x,y)/2))
        iterations = np.arange(0,Iter)
        
        '''generate initial model'''
        dst = np.zeros(realz[:,0].shape)*np.nan;   # grid to be simulated
        
        with open(outdir+'trueModel.pkl', 'wb') as f:
            pickle.dump(true_model, f)

    Mprop = np.zeros((numChains,y,x))    
    log_p_new = np.zeros(numChains)
    # path = np.zeros((numChains,x*y))
    
#%%     parallel chains avolution
    if args.distributed==1 and numChains>1:
        import dask
        from dask.distributed import Client, LocalCluster
        
        if not rand:
            np.random.seed(0)
            
        accept = 0
        
        with LocalCluster(n_workers = min(numChains,os.cpu_count()), threads_per_worker=1) as cluster, Client(address=cluster) as client:
            # cluster.adapt(maximum=48, minimum_cores=10, maximum_cores=48) 
            print(cluster)      
            
            start = time.time()
            
            for step in iterations:
                futures = []
                path = np.zeros((numChains,np.argwhere(np.isnan(dst[0].flat)).size)).astype(int)
                for r in range(numChains):
                    path[r] = np.random.permutation(np.argwhere(np.isnan(dst[r].flat)).squeeze())
                for r in range(numChains):
                    future = client.submit(sampler,ti,dst[r],path[r],kernel,n,k,modelObj=model)
                    futures.append(future)
                result = client.gather(futures)
                assert future.done(), "No computation was made"

                for r in range(numChains):
                    Mprop[r] = result[r][0]
                    log_p_new[r] = result[r][2]                                   
                
                if step>0: 
                    '''tuning delta during burn in (see  Hansen et al. 2012)'''
                    alfa = np.exp(log_p_new - log_p_old)
                    Z = np.random.rand(numChains)
                    idx = np.where(alfa > Z)[0]
                    accept += idx.size
                    Mnew[idx] = Mprop[idx]
                    log_p_old[idx] = log_p_new[idx]
                    wrmse[idx] = np.array([result[ind][1] for ind in idx])
                    if (step+1)%delta_adjust==0:    
                        delta_ar[int(step/delta_adjust)+1] = delta
                        AR[int(step/delta_adjust)+1] = 100 * accept/(delta_adjust * numChains)
                        if step<2000:
                            delta = max(int(delta * (accept/(delta_adjust * numChains * Pc))),1)
                        # if step<int(Iter*0.2) and (AR[int((step)/delta_adjust)+1]<25): 
                        #     delta-=1
                        # elif step<int(Iter*0.2) and (AR[int((step)/delta_adjust)+1]>40):
                        #     delta+=1
                        accept = 0
                    if (step+1)%thin==0: 
                        realz[:,int(step/thin)+1] = Mnew
                        WRMSE[:,int(step/thin)+1] = wrmse
                        LP[:,int(step/thin)+1] = log_p_old            
                        R_stat[int((step)/thin)+1,:] = GelmanRubin(realz[:,int(step/thin*0.5):int(step/thin)+1,:])
                        with open(outdir+'MPSrun.pkl', 'wb') as f:
                            pickle.dump((model,realz,WRMSE,LP,delta_ar,AR,R_stat,Iter,thin), f)

                else:
                    R_stat[0] = -2*np.ones(x*y)
                    delta_ar[0] = delta
                    AR[0] = 0
                    Mnew = Mprop.copy()
                    realz[:,0] = Mprop
                    wrmse = np.array([result[r][1] for r in range(numChains)])
                    WRMSE[:,0] = wrmse
                    log_p_old = log_p_new.copy()
                    LP[:,0] = log_p_old
                    

                
                '''remove portion of the model given size delta'''
                cp = np.array([np.random.randint(0,y),np.random.randint(0,x)])
                # cp = np.array([np.random.randint(0,y,size=numChains),np.random.randint(0,x,size=numChains)])
                dst = Mnew.copy()
                if sampler.__name__=='qsCat':
                    dst = np.where((1/dst).round(2) == 0.06, 1, 0).astype(np.float16)
                dst[:,max(0,cp[0]-delta):min(y,cp[0]+delta)+1,max(0,cp[1]-delta):min(x,cp[1]+delta)+1] = np.nan
                
                if (step+1)%100==0:
                    print((100*(step+1)/(iterations[-1]+1)).round(1)," %");
        
        end = time.time()                
        
    #%%             Save results
    
    with open(outdir+'run_commandline_args.txt', 'w') as f:
        json.dump((args.__dict__,{"time [hr]":round((end-start),4)/3600}), f, indent=2)
    with open(outdir+'MPSrun.pkl', 'wb') as f:
        pickle.dump((model,realz,WRMSE,LP,delta_ar,AR,R_stat,Iter,thin), f)
    