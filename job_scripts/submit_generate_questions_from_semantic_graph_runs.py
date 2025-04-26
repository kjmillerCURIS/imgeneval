import os
import sys


DEBUG = False
SCRIPT_NAME = 'generic_generate_questions_from_semantic_graph.sh'
OFFSETS = range(24)


def submit_generate_questions_from_semantic_graph_runs():
    for offset in OFFSETS:
        job_name = 'genquestfromgraph_%d'%(offset)
        my_cmd = 'qsub -N %s -v OFFSET=%d %s'%(job_name, offset, SCRIPT_NAME)
        print('submitting training run: "%s"'%(my_cmd))
        os.system(my_cmd)
        if DEBUG:
            print('DEBUG MODE: let\'s see how that first run goes...')
            return


if __name__ == '__main__':
    submit_generate_questions_from_semantic_graph_runs()
