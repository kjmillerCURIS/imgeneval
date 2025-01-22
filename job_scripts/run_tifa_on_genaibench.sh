#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=2:59:59
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -j y
#$ -m ea
#$ -N run_tifa_on_genaibench_7

module load miniconda
conda activate imgeneval
cd ~/data/imgeneval
python run_tifa_on_genaibench.py 7

