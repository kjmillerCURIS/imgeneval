import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_ontology_questions.sh'
GENERATORS = ['DALLE_3', 'DeepFloyd_I_XL_v1', 'Midjourney_6', 'SDXL_2_1', 'SDXL_Base', 'SDXL_Turbo']
OFFSETS = [0,1]


def submit_ontology_questions_runs():
    for generator in ['SDXL_Base']: #GENERATORS:
        for offset in [1]: #OFFSETS:
            job_name = 'ontology_questions_%s_%d'%(generator, offset)
            my_cmd = 'qsub -N %s -v GENERATOR=%s,OFFSET=%d %s'%(job_name, generator, offset, SCRIPT_NAME)
            print('submitting training run: "%s"'%(my_cmd))
            os.system(my_cmd)
            if DEBUG:
                print('DEBUG MODE: let\'s see how that first run goes...')
                return


if __name__ == '__main__':
    submit_ontology_questions_runs()
