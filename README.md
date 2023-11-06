# IDCS

This is a python implementation of the IDCS method presented in "Conditioning of multiple-point statistics simulations to indirect geophysical data" (submitted to Computers & Geosciences). 
The folder includes the scripts to run the IDCS with a linear straight-ray and non-linear (pygimli based) solvers and training image examples from the paper. 

## Scripts and folders:

**TIs** folder --> contains all trainin image examples used in the paper and an additional training image of lenses.

***qs.py*** --> main code to run the IDCS simulations

***sim.py*** --> the script in which the simulation is performed (QS and likelihood functions)

***approx_likelihood.py*** --> contains all functions associated with the likelihood approimation and MPS candidate sampling

***fit_vario*** --> contains the fuction that fits a variogram

***set_fw.py*** --> all functions associated with the forward response or Jacobian update

***Model_obj*** --> saves all parameters into an object

***tomokernel_straight.py*** --> script to run the linear response (straight-ray traveltime tomography)

## Requirement:

gstools==1.4.1

pygimli==1.4.0

dask==2021.10.0

## Citation :

To be added...

## License:

GNU GENERAL PUBLIC LICENSE

See LICENSE file

## Contact:

Shiran Levy (shiran.levy@unil.ch)
