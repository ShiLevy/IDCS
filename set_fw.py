#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Shiran Levy)s
"""
import numpy as np
from tomokernel_straight import tomokernel_straight_2D #tomokernel solver
import pygimli as pg
from pygimli.physics.traveltime import TravelTimeManager
from itertools import product

def set_fw(y,x,s_model,loc,scale,spacing=0.1,SnR_spacing=1,limit_angle=0, dir='./'):
    #%% experiment setup
    ny = y; nx = x
    start = 2             # in pixel units!
    sourcex = 0*spacing
    sourcez = np.arange(start,ny,SnR_spacing)*spacing      #sources positions in meters
    receiverx = x*spacing
    receiverz = np.arange(start,ny,SnR_spacing)*spacing       #receivers positions in meters
    nsource = len(sourcez); nreceiver = len(receiverz)                  
    ndata=nsource*nreceiver
    x = np.arange(0,nx+1,1)*spacing
    z = np.arange(0,ny+1,1)*spacing
        
    # Calculate acquisition geometry (multiple-offset gather)
    data=np.zeros((ndata,4))
    for jj in range(0,nsource):
        for ii in range(0,nreceiver):
            data[ ( jj ) * nreceiver + ii , :] = np.array([sourcex, sourcez[jj]+spacing/2, receiverx, receiverz[ii]+spacing/2])
    # Calculate forward modeling kernel (from Matlab code by Dr. James Irving, UNIL)
    A = tomokernel_straight_2D(data,x,z) # Distance of ray-segment in each cell for each ray
    A=np.array(A.todense())
    del data
    np.random.seed(0)
    noise = np.random.normal(loc=loc, scale=scale, size=A.shape[0])[:,np.newaxis]
    d_obs = A@s_model.flatten(order='C')[:,np.newaxis] + noise

    if limit_angle == True:
            count = 0
            i = 0
            zmin = sourcez[0];  zmax = sourcez[-1]; 
            index = []
            # Calculate acquisition geometry (multiple-offset gather)
            for jj in range(0,nsource):
                for ii in range(0,nreceiver):
                    aOVERb = abs((receiverz[ii]-sourcez[jj]))/(receiverx-sourcex)
                    count+=1
                    if np.arctan(aOVERb) > (50*np.pi/180):            
                        # data = data[:-1,:]
                        index = np.append(index,i) 
                    i+=1
            index = np.asarray(index).astype(int) 
            d_obs = np.delete(d_obs,index, axis=0)
            noise = np.delete(noise,index, axis=0)
    else:
        index = np.zeros(0)
    
    np.save(dir+'noise.npy',noise)
    return d_obs, A, index

def pygimli_fw(x_true, bh_spacing, bh_length, sensor_spacing, sigma_d,limit_angle=0,dir='./'):

    model_true = np.copy(x_true) # shape: [1, 1, ny, nx]
    # x_true = x_true.detach().numpy()
    x = model_true.shape[-1]/10 # x in meters
    y = model_true.shape[-2]/10 # y in meters
    xcells = model_true.shape[-1]+1
    ycells = model_true.shape[-2]+1
    ############################################################
    '''Simulate synthetic traveltime data'''
    ############################################################
    
    depth = -np.arange(0.2, bh_length+0.01, sensor_spacing)
    
    sensors = np.zeros((len(depth) * 2, 2))  # two boreholes
    sensors[:len(depth), 0] = 0.0  # x
    sensors[len(depth):, 0] = bh_spacing  # x
    sensors[:, 1] = np.hstack([depth] * 2)  # y
    
    numbers = np.arange(len(depth))
    rays = list(product(numbers, numbers + len(numbers)))
    
    # Empty container
    scheme = pg.DataContainer()
    
    # Add sensors
    for sen in sensors:
        scheme.createSensor(sen)
    
    # Add measurements
    rays = np.array(rays)
    scheme.resize(len(rays))
    scheme.add("s", rays[:, 0])
    scheme.add("g", rays[:, 1])
    scheme.add("valid", np.ones(len(rays)))
    scheme.registerSensorIndex("s")
    scheme.registerSensorIndex("g")
    
    mygrid = pg.meshtools.createMesh2D(x = np.linspace(0.0, x, xcells), y = np.linspace(0.0, -y, ycells))
    #mygrid.createNeighbourInfos()
    print(mygrid)
    # read channels simulation:
    model_true = np.reshape(x_true,np.size(x_true))
    mslow = model_true
    mvel = 1.0/mslow
    print(np.max(mslow))
    
    # set traveltime forward model
    ttfwd = TravelTimeManager()
    resp = ttfwd.simulate(mesh=mygrid, scheme=scheme, slowness=mslow, secNodes=2)
    ttfwd.applyData(resp)
    print(ttfwd.fop.data)
    sim_true = ttfwd.fop.data.get("t") # ns
    # add synthetic noise
    #nnoise = noise_lvl*np.random.randn(len(sim_true))
    np.random.seed(0)
    noise = np.random.normal(loc=0, scale=sigma_d, size=len(sim_true))
    d_obs = np.array(sim_true +  noise)[:,np.newaxis]# noise_lvl scales the noise.
    
    if limit_angle == True:
            count = 0
            i = 0
            data = np.concatenate(sensors)
            zmin = np.min(-depth);  zmax = np.min(-depth); 
            index = []
            # Calculate acquisition geometry (multiple-offset gather)
            for jj in numbers:
                for ii in numbers:
                    aOVERb = abs((depth[ii]-depth[jj]))/(bh_spacing)
                    count+=1
                    if np.arctan(aOVERb) > (50*np.pi/180):            
                        # data = data[:-1,:]
                        index = np.append(index,i) 
                    i+=1
            index = np.asarray(index).astype(int) 
            d_obs = np.delete(d_obs,index, axis=0)
            noise = np.delete(noise,index, axis=0)
    else:
        index = np.zeros(0)
        
    np.save(dir+'noise.npy',noise)
    
    return d_obs, (bh_spacing, bh_length, sensor_spacing), index

def set_J(parameters,x_true):
    
    ############################################################
    '''Set forward model'''
    ############################################################
    (bh_spacing, bh_length, sensor_spacing) = parameters
    model_true = np.copy(x_true) # shape: [1, 1, ny, nx]
    model_true[:] = 1
    # x_true = x_true.detach().numpy()
    x = model_true.shape[-1]/10 # x in meters
    y = model_true.shape[-2]/10 # y in meters
    xcells = model_true.shape[-1]+1
    ycells = model_true.shape[-2]+1
    ############################################################
    '''Simulate synthetic traveltime data'''
    ############################################################
    
    depth = -np.arange(0.2, bh_length+0.01, sensor_spacing)
    
    sensors = np.zeros((len(depth) * 2, 2))  # two boreholes
    sensors[:len(depth), 0] = 0.0  # x
    sensors[len(depth):, 0] = bh_spacing  # x
    sensors[:, 1] = np.hstack([depth] * 2)  # y
    
    numbers = np.arange(len(depth))
    rays = list(product(numbers, numbers + len(numbers)))
    
    # Empty container
    scheme = pg.DataContainer()
    
    # Add sensors
    for sen in sensors:
        scheme.createSensor(sen)
    
    # Add measurements
    rays = np.array(rays)
    scheme.resize(len(rays))
    scheme.add("s", rays[:, 0])
    scheme.add("g", rays[:, 1])
    scheme.add("valid", np.ones(len(rays)))
    scheme.registerSensorIndex("s")
    scheme.registerSensorIndex("g")
    
    mygrid = pg.meshtools.createMesh2D(x = np.linspace(0.0, x, xcells), y = np.linspace(0.0, -y, ycells))    
    # set traveltime forward model
    tt = TravelTimeManager()
    tt.fop.data = scheme
    tt.applyMesh(mygrid, secNodes=2) 
    
    return tt
