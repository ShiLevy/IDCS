import numpy as np
import math
import os
import time
import json
import argparse
import pickle
import scipy.ndimage as ndimage
from scipy.io import loadmat

'''internal imports'''
from sim import qs, qsCat
from set_fw import set_fw, pygimli_fw, update_J, set_J
from fit_vario import fit_vario
from Model_obj import Model_obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional MPS simulation"
    )
    parser.add_argument("--numRealz",default=1,type=int,help="number of simulations to perform",
    )
    parser.add_argument("--random", default=0, type=int, help="reproducible results (0), random paths (1)"
    )
    parser.add_argument("--LikeProb", default=2, type=int, help="which likelihood to calculate (1) analytical (only available for MG case) or (2) kriging-based"
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
    outdir = args.outdir+args.case+'_k'+str(args.k)+'_n'+str(args.n)+'_alpha'+str(args.alpha)+'testing/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(outdir+'run_commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    homeDir = args.workdir
    rand = args.random        
    cutSize = args.TIsize
    
    # forward simulation settings
    data_cond = args.data_cond
    numRealz = args.numRealz   
    x = args.x
    y = args.y
    sigma_d  = args.sigma_d
    stopDataCond = 100  # the percentage of simulated pixels after which to stop data conditioning (100 to have all pixels conditioned on data)
    
    # QS parameters 
    n = args.n   # number of neighbors
    k = args.k   # number of MPS candidates

    kernel=np.ones((args.kernel_size,args.kernel_size));      # map of euclidian distances
    kernel[math.floor(kernel.shape[0]/2),math.floor(kernel.shape[1]/2)]=0;
    kernel = np.exp(-args.alpha*ndimage.morphology.distance_transform_edt(kernel))
    
    field = np.load('./TIs/'+args.case+'.npy')

    if args.case!='channels' and args.case!='lenses':
        field = field*0.004+0.07  # assign new mean and standard deviation to the field  
        sampler = qs
        fullTI = 1/field   # convert to slowness (working with to avoid additional nonlinear transformations)
        
        if args.case=='GaussianRandomField': 
            true_model = fullTI[1200:1200+y,1200:1200+x] # test case multiGaussian
        else:
            true_model = fullTI[633:633+y,1831:1831+x] # test case Zin & Harvey

    else:
        a0=None
        sampler = qsCat
        fullTI = 1/field  # convert to slowness (working with to avoid additional nonlinear transformations)
         
        if args.case=='channels':
            true_model = fullTI[1700:1700+y,2194:2194+x]    # test case binary channels
        else:
            true_model = fullTI[1738:1738+y,2052:2052+x]    # test case binary lenses

    ti = fullTI[:cutSize,:cutSize]
    mu_m = np.mean(fullTI)      # training image mean
    sigma_m = np.std(fullTI)    # training image standard deviation
    
    #%%
    # run the straight-ray tomography solver (linear)
    if args.linear:
        d_obs, A, index = set_fw(y,x,s_model=true_model,loc=0,scale=sigma_d,spacing=0.1,SnR_spacing=4,limit_angle=1,dir=outdir)
        model = Model_obj(data_cond,args.LikeProb,args.sampProp,x,y,sigma_d,sigma_m,mu_m,A,d_obs)
        model.fw = "linear"
        
    # run the travel-time tomography with pygimli solver (non-linear)
    else:
        d_obs, param, index = pygimli_fw(true_model, bh_spacing=x/10, bh_length=y/10, sensor_spacing=0.4,sigma_d=sigma_d,limit_angle=1,dir=outdir)
        if args.resim:
            # if running re-simulation based on best data fitting realisation (described in Section 4.2.)
            bestModel = np.load(outdir+'bestModel.npy')  # reads the saved based model from previous run
            temp_tt = set_J(param,bestModel)
            A=update_J(bestModel, temp_tt)
            model = Model_obj(data_cond,args.LikeProb,args.sampProp,x,y,sigma_d,sigma_m,mu_m,A,d_obs)
            del temp_tt
            model.fw = "linear"
        else:
            # running normal simulation based on Jacobian updates using pygimli
            model = Model_obj(data_cond,args.LikeProb,args.sampProp,x,y,sigma_d,sigma_m,mu_m,None,d_obs)
            model.fw = "pygimli"
            model.param = param
        
        
    model.index = index  # indices of source-receiver pairs that are outside the angle limit range
    
    if args.case=="GaussianRandomField":
        # Only available for the MG case for later comparison with the analytical solution
        model.a0 = loadmat('./TIs/GaussianRandomField_mean0unitVar5_10.mat')['range'][0]
        
    if args.LikeProb>1:
        # fitting a variogram to samples from the training image
        model.est_a0, model.est_alpha, model.est_var  = fit_vario(ti,seed=0,case=args.case)

    if not rand:
        # setting the seed for reproducibility 
        model.rand=rand
        seed = np.arange(0,numRealz*10,10)   #if rand=0 array of seeds to use 
        model.seed=seed
        
    #%%             Running Running IDCS in parallel using Dask distributed
    if args.distributed==1 and numRealz>1:
        import dask
        from dask.distributed import Client, LocalCluster
        from time import sleep
        
        realz = []
        WRMSE = []
        LP = []
        path = []
        dst=np.zeros((numRealz,y,x))*np.nan; 
        if not rand:
            np.random.seed(7)
        for r in range(numRealz):  
            path.append(np.random.permutation(dst[0].size))
        np.save('./path_random.npy',path)
            
        futures = []
        # running the simulations in parallel using dask client
        with LocalCluster(threads_per_worker=1) as cluster, Client(address=cluster) as client:          
            start = time.time()
            for idr,(p,d) in enumerate(zip(path,dst)):    
                future = client.submit(sampler,ti,d,p,kernel,n,k,modelObj=model,thresh=stopDataCond,idr=idr)
                futures.append(future)
            result = client.gather(futures)
            end = time.time()
            print(round((end-start),4))
            assert future.done(), "No computation was made"
            client.retire_workers()

        if model.data_cond:
            # conditional simulations results
            for r in range(numRealz):
                realz.append(result[r][0])   
                WRMSE.append(result[r][1])
                LP.append(result[r][2])
            realz = np.array(realz)
            WRMSE = np.array(WRMSE)
            LP = np.array(LP)
        else:
            # unconditional simulations results 
            for r in range(numRealz):
                realz.append(result[r][0])             
    
    #%%             Running IDCS sequentially
    else:
        realz = np.zeros((numRealz,y,x))
        WRMSE = np.zeros((numRealz,y*x))
        LP = []
        start = time.time()
        for r in range(numRealz):
            dst=np.zeros((y,x))*np.nan;   # grid to be simulated
            if not rand:
                np.random.seed(seed[r])
            if args.preferential:
                path = preferential_path(model.A)
            else: 
                path = np.random.permutation(dst.size)
            path = np.random.permutation(dst.size)
            model.update = 0
            sim=sampler(ti,dst,path,kernel,n,k,true_model=true_model,modelObj=model,thresh=stopDataCond,idr=r)

            if model.data_cond:
                realz[r] = sim[0]
                WRMSE[r] = sim[1]
                LP.append(sim[2])
            else:
                realz[r] = sim[0]
                
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
    