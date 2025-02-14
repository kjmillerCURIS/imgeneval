import os
import sys
from collections import deque
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
from influence_on_human_ratings import load_human_ratings_data, adjust_impath, GENERATORS
from influence_on_tifa_residuals import load_tifa_results_dict
from ontology_questions import get_log_filename


SKIP_FILENAMES = ['SDXL_2_1_00149_Three_pink_peonies_and_four_white_daisies_in_a_garden.txt']


def get_ontology_score(log_filename):
    #print(log_filename)
    with open(log_filename, 'r') as f:
        lines = list(deque(f, 10))

    lines = [line for line in lines if 'best_score = ' in line]
    assert(len(lines) == 1)
    return float(lines[0].split('best_score = ')[1])


def ontology_eval():
    d_human = load_human_ratings_data()
    d_tifa = load_tifa_results_dict()
    human_scores = []
    tifa_scores = []
    ontology_scores = []
    for k in tqdm(sorted(d_human.keys())):
        for generator in GENERATORS:
            prompt = d_human[k]['prompt']
            image_filename = adjust_impath(d_human[k]['images'][generator]['filename'])
            human_score = d_human[k]['images'][generator]['rating']
            if image_filename not in d_tifa:
                print('!')
                continue

            tifa_score = d_tifa[image_filename]['tifa_score']
            log_filename = get_log_filename(prompt, image_filename)
            if not os.path.exists(log_filename):
                continue

            if os.path.basename(log_filename) in SKIP_FILENAMES:
                continue

            ontology_score = get_ontology_score(log_filename)
            human_scores.append(human_score)
            tifa_scores.append(tifa_score)
            ontology_scores.append(ontology_score)

    print('Found %d GenAI-Bench examples'%(len(human_scores)))
    print('TIFA: spearman_rho = %f'%(spearmanr(human_scores, tifa_scores).statistic))
    print('Ontology: spearman_rho = %f'%(spearmanr(human_scores, ontology_scores).statistic))


if __name__ == '__main__':
    ontology_eval()
