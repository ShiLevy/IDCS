#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Shiran Levy)s
"""
import numpy as np
from dataclasses import dataclass, field
import array

@dataclass
class Model_obj: 
    
    data_cond: bool;
    LikeProb: int;
    sampProp: bool;
    x: int;
    y: int;
    sigma_d: np.float32 = field(default=None);
    sigma_m: np.float32 = field(default=None);
    mu_m: np.float32 = field(default=None);
    A: np.ndarray = field(default=None);
    d_obs: np.ndarray = field(default=None);  
    a0: np.float32 = field(default=None);
    est_a0: np.float32 = field(default=None);
    est_alpha: np.float32 = field(default=None);
    est_var: np.float32 = field(default=None);
    mu_2g1: np.float32 = field(default=None);
    sigma_2g1: np.float32 = field(default=None);
    
    index: np.int32 = field(default=None);
    ID: int = None;
    # true_model: np.ndarray = np.empty([],dtype=np.float32);
    seed: np.int32 = field(default=None);
    update: int = 0;
    rand: int = 1;
    fw: str = None;
    param: np.float32 = field(default=None);
    tt: str = None
    d_Tapp: np.float32 = None
    C_Tapp: np.float32 = None

def preferential_path(J):
    #%% Preferntial path according to raypath length in a cell
    sum_path = J.sum(axis=0)
    norm_sum = sum_path/(sum_path.sum())
    non0 = np.nonzero(sum_path)[0]
    partialPath=np.random.choice(non0,non0.size,replace=False,p=norm_sum[non0])
    exist = np.arange(0,J.shape[1])

    index = list()
    rest_ind =  lambda x: x==non0
    for i in exist:    
        if rest_ind(i).any() == False:
            index.append(i)
            
    index = np.array(index)
    restInd = np.random.choice(index,index.size,replace=False,p=None)
    path = np.concatenate((partialPath,restInd))
    
    return path