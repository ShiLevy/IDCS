import numpy
import scipy.ndimage as ndimage
import math
import matplotlib.pyplot as plt
import os
from Calc_COV import calc_likelihood, sampleLike
import array
from set_fw import set_J
import time

os.environ["MKL_NUM_THREADS"] = "1" # remove fft paralelization

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
    return indexes[int(math.floor(numpy.random.uniform(k)))] if data_cond==0 else indexes[:int(k)];

# @dask.delayed
def qs(ti,dst,path,template,n,k,true_model=None,modelObj=None,thresh=100,idr=None):
    modelObj.ID = idr
    if not modelObj.rand:
        numpy.random.seed(modelObj.seed[idr])
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
    modelObj.ID = idr
    if not modelObj.rand:
        numpy.random.seed(modelObj.seed[idr])
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

def dsSample(parameter,source,template):
	(ti,dist,allowedPosition,n,th,f,)=parameter;
	for r in range(1,numpy.max(numpy.array(template.shape))//2):
		if numpy.logical_not(numpy.isnan(source[dist<=r])).sum()>=n:
			break;
	source[dist>r]=numpy.nan;
	dataLoc=numpy.where(numpy.logical_not(numpy.isnan(source)).flat);
	data=source.flat[dataLoc];
	dxL,dyL=numpy.unravel_index(dataLoc,template.shape);
	deltas=numpy.ravel_multi_index([dxL, dyL],ti.shape)
	scanPath=numpy.random.permutation(allowedPosition)[:math.ceil(ti.size*f)];

	hx=template.shape[0]//2;
	hy=template.shape[1]//2;

	bestP=numpy.random.randint(ti.size);
	if(numpy.sum(numpy.logical_not(numpy.isnan(source)))<1):
		return bestP
	bestError=numpy.inf;
	sourcelocal=numpy.zeros(source.shape);
	for p in scanPath:
		missmatch=numpy.mean((ti.flat[deltas+p]-data)**2);
		
		if(missmatch<bestError):
			bestP=p;
			bestError=missmatch;
		if(bestError<th):
			break;
	return bestP+numpy.ravel_multi_index([hx, hy],ti.shape);

def ds(ti,dst,path,template,n,th,f):
	dist=numpy.zeros(shape=template.shape);
	dist[math.floor(dist.shape[0]/2),math.floor(dist.shape[1]/2)]=1;
	dist=ndimage.morphology.distance_transform_edt(1-dist);
	allowedPosition=ti.copy();
	allowedPosition.flat[:]=range(allowedPosition.size);
	allowedPosition=allowedPosition[:-template.shape[0],:-template.shape[1]].flatten().astype(int);
	return runsim((ti, dist,allowedPosition, n,th,f),ti, dst,path,template,dsSample);

def dsSampleCat(parameter,source,template):
	(ti,dist,allowedPosition,n,th,f,)=parameter;
	for r in range(1,numpy.max(numpy.array(template.shape))//2):
		if numpy.logical_not(numpy.isnan(source[dist<=r])).sum()>=n:
			break;
	source[dist>r]=numpy.nan;
	dataLoc=numpy.where(numpy.logical_not(numpy.isnan(source)).flat);
	data=source.flat[dataLoc];
	dxL,dyL=numpy.unravel_index(dataLoc,template.shape);
	deltas=numpy.ravel_multi_index([dxL, dyL],ti.shape)
	scanPath=numpy.random.permutation(allowedPosition)[:math.ceil(ti.size*f)];

	hx=template.shape[0]//2;
	hy=template.shape[1]//2;

	bestP=numpy.random.randint(ti.size);
	if(numpy.sum(numpy.logical_not(numpy.isnan(source)))<1):
		return bestP
	bestError=numpy.inf;
	sourcelocal=numpy.zeros(source.shape);
	for p in scanPath:
		missmatch=numpy.mean(ti.flat[deltas+p]!=data);
		if(missmatch<bestError):
			bestP=p;
			bestError=missmatch;
		if(bestError<th):
			break;
	return bestP+numpy.ravel_multi_index([hx, hy],ti.shape);

def dsCat(ti,dst,path,template,n,th,f):
	ti=ti.astype(int)
	dist=numpy.zeros(shape=template.shape);
	dist[math.floor(dist.shape[0]/2),math.floor(dist.shape[1]/2)]=1;
	dist=ndimage.morphology.distance_transform_edt(1-dist);
	allowedPosition=ti.copy();
	allowedPosition.flat[:]=range(allowedPosition.size);
	allowedPosition=allowedPosition[:-template.shape[0],:-template.shape[1]].flatten().astype(int);
	return runsim((ti, dist,allowedPosition, n,th,f),ti, dst,path,template,dsSampleCat);

def fw(ti,cand_index,p,simGrid,s_name,modelObj):
    
    if s_name=='qsSampleCat':
        simGrid.flat[p]=1
        simGrid = 1/(0.06 + (1-simGrid)*0.02)
        simGrid[numpy.isnan(simGrid)]=0 
        LogProb, mu_L, mu_2g1, sigma_2g1 = calc_likelihood(simGrid.copy(),p,modelObj,cand_index,ti,chan=1)
    else:
        simGrid.flat[p]=1
        simGrid[numpy.isnan(simGrid)]=0  
        LogProb, mu_L, mu_2g1, sigma_2g1 = calc_likelihood(simGrid.copy(),p,modelObj,cand_index,ti)
    
    if modelObj.data_cond!=1:
        d_pred = mu_L
        simIndex = cand_index
    else: 
        if modelObj.sampProp:
            ind = sampleLike(LogProb)
        else:
            ind = numpy.argmax(LogProb)
            
        modelObj.mu_2g1 = mu_2g1[ind]
        modelObj.sigma_2g1 = sigma_2g1
        d_pred = mu_L[ind]
    
        simIndex = cand_index[ind]
    e = modelObj.d_obs - d_pred
    wrmse = numpy.sqrt(numpy.sum(numpy.power(e/modelObj.sigma_d,2.0))/e.size)
    
    return simIndex, wrmse, LogProb

def runsim(parameter,ti,dst,path,template,sampler,thresh,true_model=None,modelObj=None):
    xL,yL=numpy.unravel_index(path, dst.shape)
    hx=template.shape[0]//2;
    hy=template.shape[1]//2;
    source=template.copy(); 
    WRMSE = []
    LP = []
    if modelObj.fw=='pygimli':
        modelObj.tt = set_J(modelObj.param,dst)
    # fig = plt.figure(1,figsize=(15,5));
    # ax = fig.add_subplot(1, 3, 1);
    # ax2 = fig.add_subplot(1, 3, 2);
    # ax3 = fig.add_subplot(1, 3, 3);
    # ax.set_title('Data fit for std %.1f ns' %(modelObj.sigma_d))
    # ax.set_xlabel('Simulated pixels')
    # ax.set_ylabel('Data misfit WRMSE')
    # ax2.set_title('Simulation grid')
    # ax2.set_xlabel('x [pixel unit]')
    # ax2.set_ylabel('y [pixel unit]')
    # ax3.set_title('True model')
    # ax3.set_xlabel('x [pixel unit]')
    # im = ax2.imshow(dst)
    # im2 = ax3.imshow(true_model)
    # # ax2.set_xticks(numpy.arange(-.5, 8, 1), major=True)
    # # ax2.set_yticks(numpy.arange(-.5, 16, 1), major=True)
    # # ax2.set_xticklabels([])
    # # ax2.set_yticklabels([])
    # # ax2.grid(visible=1,which='both')
    # cbar = plt.colorbar(im2)
    for x,y,p,idx in zip(xL,yL,path,range(path.size)):
        start_cand = time.time()
        #print(x,y,p,idx)
        # if(idx%(path.size//100)==0):
            # print(idx*100//path.size," %");
        source*=numpy.nan;
        source[max(0,x-hx)-(x-hx):min(dst.shape[0]-1,x+hx)-x+hx+1,max(0,y-hy)-(y-hy):min(dst.shape[1]-1,y+hy)-y+hy+1]=\
            dst[max(0,x-hx):min(dst.shape[0],x+hx)+1,max(0,y-hy):min(dst.shape[1],y+hy)+1];

        if modelObj.data_cond!=0 and ((idx+1)*100/path.size)<=thresh:
            startQS = time.time()
            cand_index = sampler(parameter,source,template,modelObj.data_cond);
            print("It takes "+str((time.time()-startQS))+" s to comp. QS")
            start = time.time()
            simIndex, wrmse, LogProb = fw(ti,cand_index,p,dst.copy(),sampler.__name__,modelObj) 
            print("It takes "+str((time.time()-start))+" s to comp. the chosen candidate")
            WRMSE.append(wrmse)
            LP.append(LogProb)
            
            # # ax.set_xlim(0,path.size)
            # if idx>0:
            #     ax.scatter(idx+1,WRMSE[-1],color='b')
            # else:
            #     ax.axhline(y = modelObj.sigma_d, color = 'r', linestyle = '-')
            # display(fig) 
        else:
            simIndex=sampler(parameter,source,template);
            # someInd, wrmse, LogProb = fw(ti,simIndex[numpy.newaxis],p,dst.copy(),sampler.__name__,modelObj) 
        dst.flat[p]=ti.flat[simIndex];

        # im.set_data(dst)
        # im.set_clim([numpy.min(true_model),numpy.max(true_model)])
        # cbar.draw_all() 
        # fig.canvas.draw()
        # plt.pause(0.01)
        print("It takes "+str((time.time()-start_cand))+" s to comp. one step")
    if modelObj.fw=='pygimli':
        del modelObj.tt


    return dst, WRMSE, LP;