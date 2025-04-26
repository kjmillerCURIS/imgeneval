import os
import sys
import random
from plot_cossim_stats import load_data


def pick_reshuffle_rep_examples():
    orig_data, reshuffle_data, rep_data = load_data()
    while True:
        t = random.choice(range(len(orig_data['raw'])))
        orig_one, reshuffle_one, rep_one = orig_data['raw'][t], reshuffle_data['raw'][t], rep_data['raw'][t]
        i = random.choice([0,1])
        j = random.choice([0,1])
        print('')
        print(len(orig_one['checklist_trues_dict'][(i,j)]))
        print(len(reshuffle_one['checklist_trues_dict'][(i,j)]))
        print(len(rep_one['checklist_trues_dict'][(i,j)]))
        print('checklist_trues: ' + str(orig_one['checklist_trues_dict'][(i,j)]))
        print('checklist_false: ' + str(orig_one['checklist_false_dict'][(i,j)]))
        print('ORIG SUMMARY: ' + orig_one['LLM_generated_description_dict'][(i,j)])
        print('SUMMARY AFTER RESHUFFLING: ' + reshuffle_one['LLM_generated_description_dict'][(i,j)])
        print('SUMMARY AFTER REP: ' + rep_one['LLM_generated_description_dict'][(i,j)])
        input()


if __name__ == '__main__':
    pick_reshuffle_rep_examples()
