import numpy as np
import gstools as gs

def fit_vario(TI,seed,case):
    ''' sampling from the training image and fitting a variogram'''
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
    model.fit_variogram(bin_center, dir_vario)

    return model.len_scale_vec, model.alpha, model.var

