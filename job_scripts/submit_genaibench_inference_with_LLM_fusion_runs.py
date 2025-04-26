import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_genaibench_inference_with_LLM_fusion.sh'
OFFSETS = [0,1,2,3,4,5,6,7,8,9,10,11]


def submit_genaibench_inference_with_LLM_fusion_runs():
    for offset in OFFSETS:
        job_name = 'infLLMfus_%d'%(offset)
        my_cmd = 'qsub -N %s -v OFFSET=%d %s'%(job_name, offset, SCRIPT_NAME)
        print('submitting training run: "%s"'%(my_cmd))
        os.system(my_cmd)
        if DEBUG:
            print('DEBUG MODE: let\'s see how that first run goes...')
            return


if __name__ == '__main__':
    submit_genaibench_inference_with_LLM_fusion_runs()
