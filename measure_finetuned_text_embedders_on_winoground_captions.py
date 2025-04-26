import os
import sys
import numpy as np
import pickle
import random
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from winoground_inference_with_LLM_fusion import load_winoground_data, BASE_DIR
from finetune_text_embedder import EMBEDDER_TYPE


LOSS_TYPES = ['CoSENT', 'AnglE']
LRS = [1e-5, 2e-5, 4e-5, 8e-5, 16e-5, 32e-5, 64e-5]
STEPS_LIST = [i*250 for i in range(1,44)]
PLOT_PREFIX = os.path.join(BASE_DIR, 'winoground_caption_cossims_on_finetuned_embedders')


def load_model(loss_type, lr, steps):
    if steps is None:
        return SentenceTransformer(EMBEDDER_TYPE, device='cuda')

    model_dir = os.path.join(BASE_DIR, 'embedder_finetuning_%s_%s_lr%f'%(EMBEDDER_TYPE, loss_type, lr), 'checkpoint-%d'%(steps))
    return SentenceTransformer(model_dir, device='cuda')


#return pos_mean, neg_mean
def measure_one(loss_type, lr, steps, captionsA, captionsB):
    model = load_model(loss_type, lr, steps)
    embeddingsA = model.encode(captionsA)
    embeddingsB = model.encode(captionsB)
    embeddingsA = embeddingsA / np.linalg.norm(embeddingsA, axis=-1, keepdims=True)
    embeddingsB = embeddingsB / np.linalg.norm(embeddingsB, axis=-1, keepdims=True)
    cossims = embeddingsA @ embeddingsB.T
    pos_cossims, neg_cossims = [], []
    for i in range(cossims.shape[0]):
        for j in range(cossims.shape[1]):
            if i == j:
                pos_cossims.append(cossims[i,j])
            else:
                neg_cossims.append(cossims[i,j])

    pos_mean = np.mean(pos_cossims)
    neg_mean = np.mean(neg_cossims)
    return pos_mean, neg_mean


def measure_finetuned_text_embedders_on_winoground_captions():
    data = load_winoground_data()
    captionsA = [example[0][0]['prompt'] for example in data]
    captionsB = [example[1][0]['prompt'] for example in data]
    xs = [0]
    xs.extend(STEPS_LIST)
    pos_mean_zero, neg_mean_zero = measure_one(None, None, None, captionsA, captionsB)
    ys_pos, ys_neg = {}, {}
    for loss_type in LOSS_TYPES:
        for lr in LRS:
            k = (loss_type, lr)
            ys_pos[k] = [pos_mean_zero]
            ys_neg[k] = [neg_mean_zero]
            for steps in tqdm(STEPS_LIST):
                pos_mean, neg_mean = measure_one(loss_type, lr, steps, captionsA, captionsB)
                ys_pos[k].append(pos_mean)
                ys_neg[k].append(neg_mean)

    for ys, my_title, plot_filename in zip([ys_pos, ys_neg], ['Winoground caption cossims (paired)', 'Winoground caption cossims (unpaired)'], [PLOT_PREFIX + '_paired.png', PLOT_PREFIX + '_unpaired.png']):
        plt.clf()
        plt.title(my_title)
        plt.xlabel('training steps')
        plt.ylabel('mean cossim')
        for loss_type, linestyle in zip(LOSS_TYPES, ['solid', 'dashed']):
            for lr, color in zip(LRS, ['b', 'k', 'r', 'orange', 'grey', 'green', 'cyan']):
                k = (loss_type, lr)
                plt.plot(xs, ys[k], linestyle=linestyle, color=color, label='loss=%s, lr=%s'%(loss_type, str(lr)))

        plt.legend()
        plt.savefig(plot_filename)
        plt.clf()


def usage():
    print('Usage: python measure_finetuned_text_embedders_on_winoground_captions.py')


if __name__ == '__main__':
    measure_finetuned_text_embedders_on_winoground_captions()
