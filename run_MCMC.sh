#!/bin/bash -l

#SBATCH --mail-type ALL 
#SBATCH --mail-user shiran.levy@unil.ch
#SBATCH --chdir /work/FAC/FGSE/ISTE/nlinde/default/slevy/conditional_MPS/
#SBATCH --job-name Gibbs_samp

## if running with LocalCLuster() run with --nodes 1 (LocalCluster can only work on one node!)
#SBATCH --nodes 1
#SBATCH --ntasks 8

#SBATCH --partition cpu
#SBATCH --cpus-per-task 1

##SBATCH --gres=gpu:1
##SBATCH --gres-flags enforce-binding
##SBATCH --mem 100G
#SBATCH --time 72:00:00

#SBATCH --output=Gibbs_samp_%A.out
#SBATCH --error=Gibbs_samp_%A.error

module purge
module load gcc miniconda3
module load gcc python
#module load cuda

#conda activate conMPS
conda activate Neutra

python EM_SeqGibbsSamp.py --restart=1 --numChains=8 --Iter=20000 --thin=100 --delta_adjust=20 --x=50 --y=100 --case='LowConnectedGaussian'
