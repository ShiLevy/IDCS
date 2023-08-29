from dask_mpi import initialize
initialize()

import numpy as np
import math
import os
import time
import json
import argparse
import pickle
from PIL import Image
import scipy.ndimage as ndimage
from io import BytesIO
from scipy.io import loadmat

import itertools

'''internal imports'''
from sim import qs, qsCat
from set_fw import set_fw, pygimli_fw
from fit_vario import fit_vario
from Model_obj import Model_obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional MPS simulation"
    )
    parser.add_argument("--numRealz",default=1,type=int,help="number of simulations to perform",
    )
    parser.add_argument("--random", default=0, type=int, help="reproducible results (0), random paths (1)"
    )
    parser.add_argument("--LikeProb", default=2, type=int, help="which likelihood to calculate (analytical (1), est. kriging cov-mean (2) or est. prior cov-mean (3) based)"
    ) 
    parser.add_argument("--sampProp", default=1, type=int, help="sample proportional to the likelihood"
    )     
    parser.add_argument( "--data-cond", default=1, type=int, help="Condition MPS on data (0) no (1) yes"
    )
    parser.add_argument( "--linear", default=0, type=int, help="linear (1) non-linear (0) forward solver"
    )
    parser.add_argument( "--resim", default=1, type=int, help="re-simulate on the best model for non linear solvers"
    )
    parser.add_argument("--sigma-d", default=1, type=float, help="standard deviation of the observational noise in [ns]"
    )    
    parser.add_argument("--TIsize", default=500, type=int, help="size of TI to use for MPS simulation"
    )   
    parser.add_argument("--x", default=50, type=int, help="X size"
    )   
    parser.add_argument("--y", default=100, type=int, help="Y size"
    )   
    parser.add_argument("--n", default=25, type=int, help="number of neighbors"
    )  
    parser.add_argument("--k", default=100, type=int, help="number of candidates (low value give more importance to MPS)"
    )  
    parser.add_argument("--alpha", default=0, type=float, help="factor of the weighting kernel"
    ) 
    parser.add_argument("--kernel-size", default=51, type=int, help="weighting kernel size"
    ) 
    parser.add_argument("--distributed", default=0, type=int, help="parallel computing of realizations if more than 1"
    )
    parser.add_argument("--workdir", default='./', type=str, help="working directory (where all the models are"
    )
    parser.add_argument("--outdir", default= './results/', type=str, help="directory to save results in"
    )
    parser.add_argument("--case", default= 'LowConnectedGaussian', type=str, help="Test case and results name"
    )
    
    args = parser.parse_args()
    outdir = args.outdir+args.case+'_k'+str(args.k)+'_n'+str(args.n)+'_alpha'+str(args.alpha)+'_debug/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(outdir+'run_commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    homeDir = args.workdir
    # forward simulation settings
    data_cond = args.data_cond
    numRealz = args.numRealz   # number of realizations to simulate
    rand = args.random        # set to random (1) or reproducable (0)
    cutSize = args.TIsize
    x = args.x
    y = args.y
    sigma_d  = args.sigma_d
    stopDataCond = 100
    
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
        #field = np.where(field==0.06,0,1) #if channels are slower (1,0) if matrix is faster (0,1)
        #field = 0.06 + 0.02*(1-field)
        fullTI = 1/field   #assign new mean and standard deviation to the field   
        if args.case=='channels':
            #true_model = fullTI[1020:1020+y,1040:1040+x]
            true_model = fullTI[1700:1700+y,2194:2194+x]    # test case
            #true_model = fullTI[1200:1200+y,1200:1200+x]   #test 1
            #true_model = fullTI[1560:1560+y,1350:1350+x]   #test 2
        else:
            #true_model = fullTI[1127:1127+y,1032:1032+x]
            true_model = fullTI[1738:1738+y,2052:2052+x]    # test case
            #true_model = fullTI[1200:1200+y,1200:1200+x]   #test 1
            #true_model = fullTI[1770:1770+y,1435:1435+x]   #test 2


    ti = fullTI[:cutSize,:cutSize];
    mu_m = np.mean(fullTI)
    sigma_m = np.std(fullTI)
    
    #%%
    if args.linear:
        d_obs, A, index = set_fw(y,x,s_model=true_model,loc=0,scale=sigma_d,spacing=0.1,SnR_spacing=4,limit_angle=1,dir=outdir)
        model = Model_obj(data_cond,args.LikeProb,args.sampProp,x,y,sigma_d,sigma_m,mu_m,A,d_obs)
        model.fw = "linear"
    else:
        d_obs, param, index = pygimli_fw(true_model, bh_spacing=x/10, bh_length=y/10, sensor_spacing=0.4,sigma_d=sigma_d,limit_angle=1,dir=outdir)
        if args.resim:
            from sim import set_J
            from Calc_COV import update_J
            bestModel = np.load(outdir+'bestModel.npy')
            # bestModel = true_model
            temp_tt = set_J(param,bestModel)
            A=update_J(bestModel, temp_tt)
            model = Model_obj(data_cond,args.LikeProb,args.sampProp,x,y,sigma_d,sigma_m,mu_m,A,d_obs)
            del temp_tt
            model.fw = "linear"
        else:
            model = Model_obj(data_cond,args.LikeProb,args.sampProp,x,y,sigma_d,sigma_m,mu_m,None,d_obs)
            model.fw = "pygimli"
            model.param = param
        
        
    model.index = index
    
    if args.case=="GaussianRandomField":
        model.a0 = loadmat('./TIs/GaussianRandomField_mean0unitVar5_10.mat')['range'][0]
    if args.LikeProb>1:
        model.est_a0, model.est_alpha, model.est_var  = fit_vario(ti,seed=0,case=args.case)

        #with open('./TIs/'+args.case+'_fitvario.pkl', 'rb') as f:
        #    tmp = pickle.load(f)
        #model.est_a0 = tmp.len_scale_vec 
        #model.est_alpha = tmp.alpha
        #model.est_var = tmp.var
        #del tmp
    if not rand:
        model.rand=rand
        seed = np.arange(0,numRealz*10,10)   #if rand=0 array of seeds to use 
        model.seed=seed
        
    #%%             Dask distributed
    if args.distributed==1 and numRealz>1:
        import dask
        from dask.distributed import Client, LocalCluster
        from time import sleep
        
        realz = []
        WRMSE = []
        LP = []
        path = []
        dst=np.zeros((numRealz,y,x))*np.nan; 
        for r in range(numRealz):  
            if not rand:
                np.random.seed(seed[r])
            path.append(np.random.permutation(dst[0].size))
            
        futures = []
        # Uncomment the next line for running the simulations using **dask**
        #with LocalCluster(threads_per_worker=1) as cluster, Client(address=cluster) as client:
        # Uncomment the next line for running the simulations using **dask-MPI**
        with Client() as client:
            # cluster.adapt(maximum_jobs=10, interval="10000 ms", wait_count=10)
            #print(cluster)           
            start = time.time()
            for idr,(p,d) in enumerate(zip(path,dst)):    
                future = client.submit(sampler,ti,d,p,kernel,n,k,modelObj=model,thresh=stopDataCond,idr=idr)
                futures.append(future)
            result = client.gather(futures)
            end = time.time()
            # dask.distributed.get_worker().log_event("runtimes", {"start": start, "stop": stop})
            print(round((end-start),4))
            assert future.done(), "No computation was made"
            client.retire_workers()
            sleep(1)
            
        if model.data_cond:
            for r in range(numRealz):
                realz.append(result[r][0])   
                WRMSE.append(result[r][1])
                LP.append(result[r][2])
            realz = np.array(realz)
            WRMSE = np.array(WRMSE)
            LP = np.array(LP)
        else:
            for r in range(numRealz):
                realz.append(result[r][0])             
    
    #%%             Sequential simulation
    else:
        realz = np.zeros((numRealz,y,x))
        WRMSE = np.zeros((numRealz,y*x))
        LP = []
        start = time.time()
        for r in range(numRealz):
            dst=np.zeros((y,x))*np.nan;   # grid to be simulated
            if not rand:
                np.random.seed(seed[r])
            path = np.random.permutation(dst.size)
            # start = time.time()
            model.update = 0
            sim=sampler(ti,dst,path,kernel,n,k,true_model=true_model,modelObj=model,thresh=stopDataCond,idr=r)
            # end = time.time()
            # print(end - start)
            if model.data_cond:
                realz[r] = sim[0]
                WRMSE[r] = sim[1]
                LP.append(sim[2])
        end = time.time()
        print(round((end-start),4))
        LP = np.array(LP)
        
    #%%             Save results
    
    with open(outdir+'run_commandline_args.txt', 'w') as f:
        json.dump((args.__dict__,{"time [s]":round((end-start),4)}), f, indent=2)
    if not args.resim or args.linear==1:
        with open(outdir+'MPSrun.pkl', 'wb') as f:
            pickle.dump((model,realz,WRMSE,LP), f)
    else:
        with open(outdir+'MPSrun_resim.pkl', 'wb') as f:
            pickle.dump((model,realz,WRMSE,LP), f)  
    with open(outdir+'trueModel.pkl', 'wb') as f:
        pickle.dump(true_model, f)
    