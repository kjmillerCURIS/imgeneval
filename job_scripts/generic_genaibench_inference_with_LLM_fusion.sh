#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -l h_rt=2:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate imgeneval-meow
cd ~/data/imgeneval
python genaibench_inference_with_LLM_fusion.py ${OFFSET}

