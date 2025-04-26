#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=2:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate imgeneval-meow
cd ~/data/imgeneval
python generate_questions_from_semantic_graph.py ${OFFSET}

