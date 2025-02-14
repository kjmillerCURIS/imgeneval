import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
sys.path.append('VPEval/src')
sys.path.append('VPEval/src/dino')
from dino.vpeval.model.modeling import Model as DinoModel


if __name__ == '__main__':
    print('meow')
    my_dino_model = DinoModel()
    my_dino_model = my_dino_model.to('cuda')
    my_dino_model.eval()
    print('mrow')
    image_pil = Image.open('person_in_front_of_train.jpg').convert('RGB')
    datum = {'image_pil' : image_pil, 'gt_labels' : ['person', 'train']}
    final_labels, final_boxes, images = my_dino_model([datum])
    import pdb
    pdb.set_trace()
