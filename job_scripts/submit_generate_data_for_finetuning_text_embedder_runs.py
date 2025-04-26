import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_generate_data_for_finetuning_text_embedder.sh'
OFFSETS = [0,1,2,3,4,5]


def submit_generate_data_for_finetuning_text_embedder_runs():
    for offset in OFFSETS:
        job_name = 'genfinembdata_%d'%(offset)
        my_cmd = 'qsub -N %s -v OFFSET=%d %s'%(job_name, offset, SCRIPT_NAME)
        print('submitting training run: "%s"'%(my_cmd))
        os.system(my_cmd)
        if DEBUG:
            print('DEBUG MODE: let\'s see how that first run goes...')
            return


if __name__ == '__main__':
    submit_generate_data_for_finetuning_text_embedder_runs()
