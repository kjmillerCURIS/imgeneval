import os
import sys
import numpy as np
import pickle
import random
from benchmark_negclip_on_our_synthetic_dataset import OUR_OUTPUT_FILENAME


NUM_TO_SHOW = 5


def is_in_range(v, min_thresh, max_thresh):
    return v >= min_thresh and v <= max_thresh


def explore_our_synthetic_data_negclip_outputs(min_perc, max_perc):
    min_perc = float(min_perc)
    max_perc = float(max_perc)
    min_perc, max_perc = min(min_perc, max_perc), max(min_perc, max_perc)
    assert(min_perc <= max_perc)
    assert(min_perc >= 0)
    assert(max_perc <= 100)

    with open(OUR_OUTPUT_FILENAME, 'rb') as f:
        outputs = pickle.load(f)

    vals = [output['negative_cossim'] - output['positive_cossim'] for output in outputs]
    min_thresh = np.percentile(vals, min_perc)
    max_thresh = np.percentile(vals, max_perc)
    print((min_thresh, max_thresh))
    selected_outputs = [output for output in outputs if is_in_range(output['negative_cossim'] - output['positive_cossim'], min_thresh, max_thresh)]
    selected_outputs = random.sample(selected_outputs, NUM_TO_SHOW)
    for output in selected_outputs:
        print('ANCHOR: ' + output['caption'])
        print('T+: ' + output['positive_example'])
        print('T-: ' + output['negative_example'])
        print('image_url: ' + output['image_url'])
        print('cossim_diff: %s'%(str(output['negative_cossim'] - output['positive_cossim'])))
        print('')


def usage():
    print('Usage: python explore_our_synthetic_data_negclip_outputs.py <min_perc> <max_perc>')


if __name__ == '__main__':
    explore_our_synthetic_data_negclip_outputs(*(sys.argv[1:]))
