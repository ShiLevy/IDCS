#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Shiran Levy)s
"""
import numpy as np
import gstools as gs
import pickle

def fit_vario(TI,seed,case):
    
    x = np.arange(TI.shape[1])
    y = np.arange(TI.shape[0])
    
    angle = 0
    bins = range(0, 40, 1)
    bin_center, dir_vario = gs.vario_estimate(
        *((x, y), TI, bins), sampling_size = 30000,
        direction=gs.rotated_main_axes(dim=2, angles=angle),
        sampling_seed=seed,
        angles_tol=np.pi/16,
        mesh_type="structured",
        return_counts=False,
    )
    
    model = gs.Stable(dim=2)
    
    print("Original:")
    print(model)
    model.fit_variogram(bin_center, dir_vario)
    print("Fitted:")
    print(model)
    
    with open('./TIs/'+case+'_fitvario.pkl', 'wb') as f:
        pickle.dump((model), f)
    
    ''''can we use the mean and std over the TI??'''
    return model.len_scale_vec, model.alpha, model.var

