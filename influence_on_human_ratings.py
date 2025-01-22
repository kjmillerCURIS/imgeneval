import os
import sys
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from make_examples_html import generate_html


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/imgeneval-data')
HUMAN_RATINGS_FILENAME = os.path.join(BASE_DIR, '2024-12-17T03-22_export.csv')
IMAGE_DIR = os.path.join(BASE_DIR, 'GenAIBenchImages')
GENERATORS = ['DALLE_3', 'DeepFloyd_I_XL_v1', 'Midjourney_6', 'SDXL_2_1', 'SDXL_Base', 'SDXL_Turbo']
ALL_CONDITIONS = sorted(['Action Relation', 'Attribute', 'Comparison', 'Counting', 'Differentiation', 'Negation', 'Part Relation', 'Scene', 'Spatial Relation', 'Universal'])
CONDITIONS_TO_SHOW = ['Differentiation', 'Comparison', 'Negation', 'Spatial Relation', 'Scene']
NUM_EXAMPLES = 5
RANDOM_SEED = 0


#def generate_html(image_paths, below, captions, output_file):


def specialsplit(z):
    if isinstance(z, float):
        assert(np.isnan(z))
        return []
    else:
        return z.split(',')


#return d[k] = {'prompt' : prompt, 'conditions' : conditions, 'images' : {generator : {'filename' : filename, 'rating' : rating}}}
def load_human_ratings_data(human_ratings_filename=HUMAN_RATINGS_FILENAME):
    d = {}
    df = pd.read_csv(human_ratings_filename)
    rows = df.to_dict(orient='records')
    for row in rows:
        k = row['id']
        v = {'prompt' : row['prompt'], 'conditions' : specialsplit(row['basic']) + specialsplit(row['advanced']), 'images' : {}}
        for generator in GENERATORS:
            v['images'][generator] = {'filename' : row[generator], 'rating' : min(row[generator + '_Human'], 5.0)}

        d[k] = v

    return d


#return X, Y where X is binary with columns same order as ALL_CONDITIONS and Y has columns same order as GENERATORS
#d should come from load_human_ratings_data()
def preprocess_data(d):
    X, Y = [], []
    for k in tqdm(sorted(d.keys())):
        assert(all([c in ALL_CONDITIONS for c in d[k]['conditions']]))
        X_row = [int(c in d[k]['conditions']) for c in ALL_CONDITIONS]
        Y_row = [d[k]['images'][generator]['rating'] for generator in GENERATORS]
        X.append(X_row)
        Y.append(Y_row)

    return np.array(X), np.array(Y)


#return dict mapping each condition to its influence weight
#X should come from preprocess_data()
#y should be a vector, same height as X
def compute_influences(X, y):
    X_pad = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.squeeze(np.linalg.pinv(X_pad.T @ X_pad) @ X_pad.T @ y[:,np.newaxis])
    assert(w.shape == (len(ALL_CONDITIONS) + 1,))
    influences = {c : v for c, v in zip(ALL_CONDITIONS, w[:-1])}
    return influences


def beautifulprint(influences):
    print({c : '%.3f'%(influences[c]) for c in ALL_CONDITIONS})


def adjust_impath(impath, full_path=True):
    if full_path:
        return os.path.join(IMAGE_DIR, os.path.basename(os.path.dirname(impath)), os.path.basename(impath))
    else:
        return os.path.join(os.path.basename(os.path.dirname(impath)), os.path.basename(impath))


def stitch_examples_one_condition(d, c):
    my_keys = [k for k in sorted(d.keys()) if c in d[k]['conditions']]
    my_keys = random.sample(my_keys, NUM_EXAMPLES)
    image_paths, below, captions = [], [], []
    output_file = os.path.join(BASE_DIR, 'GenAIBench_examples', c.replace(' ', '_') + '.html')
    for k in my_keys:
        image_paths.append([adjust_impath(d[k]['images'][generator]['filename']) for generator in GENERATORS])
        below.append([str(d[k]['images'][generator]['rating']) for generator in GENERATORS])
        captions.append(d[k]['prompt'])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    generate_html(image_paths, below, captions, output_file)


def stitch_examples(d):
    random.seed(RANDOM_SEED)
    for c in tqdm(CONDITIONS_TO_SHOW):
        stitch_examples_one_condition(d, c)


def influence_on_human_ratings():
    d = load_human_ratings_data(HUMAN_RATINGS_FILENAME)
    X, Y = preprocess_data(d)
    influences_on_mean = compute_influences(X, np.mean(Y, axis=1))
    influences_on_min = compute_influences(X, np.amin(Y, axis=1))
    influences_on_max = compute_influences(X, np.amax(Y, axis=1))
    print('influences on mean:')
    beautifulprint(influences_on_mean)
    print('')
    print('influences on min:')
    beautifulprint(influences_on_min)
    print('')
    print('influences on max:')
    beautifulprint(influences_on_max)
    stitch_examples(d)


if __name__ == '__main__':
    influence_on_human_ratings()
