import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_genaibench_questions_and_tools.sh'
OFFSETS = [0,1,2,3,4,5,6,7,8,9,10,11]
MODE = 'not_OFA'


def submit_genaibench_questions_and_tools_runs():
    for offset in OFFSETS:
        job_name = 'qat_%d_%s'%(offset, MODE)
        my_cmd = 'qsub -N %s -v OFFSET=%d,MODE=%s %s'%(job_name, offset, MODE, SCRIPT_NAME)
        print('submitting training run: "%s"'%(my_cmd))
        os.system(my_cmd)
        if DEBUG:
            print('DEBUG MODE: let\'s see how that first run goes...')
            return


if __name__ == '__main__':
    submit_genaibench_questions_and_tools_runs()
