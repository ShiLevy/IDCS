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

    