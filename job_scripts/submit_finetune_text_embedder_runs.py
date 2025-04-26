import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_finetune_text_embedder.sh'
LOSS_TYPES = ['CoSENT', 'AnglE']
LRS = [32e-5, 64e-5] #[8e-5, 16e-5] #[1e-5, 2e-5, 4e-5]


def submit_finetune_text_embedder_runs():
    for loss_type in LOSS_TYPES:
        for lr in LRS:
            job_name = 'ftemb_%s_%s'%(loss_type, str(lr))
            my_cmd = 'qsub -N %s -v LOSS_TYPE=%s,LR=%s %s'%(job_name, loss_type, str(lr), SCRIPT_NAME)
            print('submitting training run: "%s"'%(my_cmd))
            os.system(my_cmd)
            if DEBUG:
                print('DEBUG MODE: let\'s see how that first run goes...')
                return


if __name__ == '__main__':
    submit_finetune_text_embedder_runs()
