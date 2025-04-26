import os
import sys
import random
from plot_cossim_stats import load_data


def pick_orig_examples():
    orig_data, _, _ = load_data()
    while True:
        result_one = random.choice(orig_data['raw'])
        i = random.choice([0,1])
        print('')
        print('CAPTION: ' + result_one[f'caption{i}'])
        print(f'OTHER CAPTION (cossim={result_one["caption0_caption1_cossim"]}): ' + result_one[f'caption{1-i}'])
        k = f'caption{i}_image{i}_cossim'
        print(f'CORRECT SUMMARY (cossim={result_one[k]}): ' + result_one['LLM_generated_description_dict'][(i, i)])
        k = f'caption{i}_image{1-i}_cossim'
        print(f'INCORRECT SUMMARY (cossim={result_one[k]}): ' + result_one['LLM_generated_description_dict'][(i, 1-i)])
        input()


if __name__ == '__main__':
    pick_orig_examples()
