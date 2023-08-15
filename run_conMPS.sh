#!/bin/bash -l

#SBATCH --mail-type ALL 
#SBATCH --mail-user shiran.levy@unil.ch
#SBATCH --chdir /work/FAC/FGSE/ISTE/nlinde/default/slevy/conditional_MPS/
#SBATCH --job-name con_MPS

# if running with LocalCLuster() run with --nodes 1 (LocalCluster can only work on one node!)
##SBATCH --nodes 1

# if running with dask-MPI set ntasks to (nr. CPUs + 2), if with dask distributed set to (nr. CPUs) 
#SBATCH --ntasks 7

#SBATCH --partition cpu
#SBATCH --cpus-per-task 1

##SBATCH --gres=gpu:1
##SBATCH --gres-flags enforce-binding
##SBATCH --mem 100G

#SBATCH --time 1:00:00

#SBATCH --output=con_MPS_%A.out
#SBATCH --error=con_MPS_%A.error

#SBATCH --export=NONE

module purge
#uncomment for running with dask-MPI
source /work/FAC/FGSE/ISTE/nlinde/default/slevy/spack/share/spack/setup-env.sh
module load gcc/10.4.0 python mvapich2 openblas py-dask-mpi-2022.4.0-gcc-10.4.0-7pwbcfx 

source /users/slevy4/working_dir/pygimli_penv/bin/activate # an environment for running with dask-MPI
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=1

#uncomment for running with dask distributed only
#module load gcc python
#conda activate Neutra   #an environment with pygimli

#python qs.py --numRealz=10 --random=0 --LikeProb=2 --sampProp=1 --data-cond=1 --linear=0 --resim=0 --sigma-d=1 --TIsize=500 --x=8 --y=16 --n=10 --k=100 --alpha=0 --distributed=1 --outdir='results/' --case='GaussianRandomField' --workdir='/users/slevy4/conditional_MPS/'
#python qs.py --numRealz=5 --random=0 --LikeProb=2 --sampProp=1 --data-cond=1 --linear=1 --resim=0 --sigma-d=1 --TIsize=500 --x=8 --y=16 --n=25 --k=100 --alpha=0 --distributed=1 --outdir='results/' --case='channels' --workdir='/users/slevy4/conditional_MPS/'

#srun python qs.py --numRealz=10 --random=0 --LikeProb=2 --sampProp=1 --data-cond=1 --linear=1 --resim=0 --sigma-d=1 --TIsize=500 --x=50 --y=100 --n=10 --k=100 --alpha=0 --distributed=1 --outdir='results/' --case='GaussianRandomField' --workdir='/users/slevy4/conditional_MPS/'
srun python qs.py --numRealz=5 --random=0 --LikeProb=2 --sampProp=1 --data-cond=1 --linear=1 --resim=0 --sigma-d=1 --TIsize=500 --x=8 --y=16 --n=25 --k=100 --alpha=0 --distributed=1 --outdir='results/' --case='channels' --workdir='/users/slevy4/conditional_MPS/'










