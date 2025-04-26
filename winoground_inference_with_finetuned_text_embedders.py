import os
import sys
import copy
import numpy as np
import pickle
import random
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from winoground_inference_with_LLM_fusion import cosine_similarity, BASE_DIR
from finetune_text_embedder import EMBEDDER_TYPE


LOSS_TYPES = ['CoSENT', 'AnglE']
LRS = [1e-5, 2e-5, 4e-5, 8e-5, 16e-5, 32e-5, 64e-5]
STEPS_LIST = [i*250 for i in range(1,44)]
PLOT_FILENAME = os.path.join(BASE_DIR, 'winoground_results_with_finetuned_embedders.png')


def load_model(loss_type, lr, steps):
    if steps is None:
        return SentenceTransformer(EMBEDDER_TYPE, device='cuda')

    model_dir = os.path.join(BASE_DIR, 'embedder_finetuning_%s_%s_lr%f'%(EMBEDDER_TYPE, loss_type, lr), 'checkpoint-%d'%(steps))
    return SentenceTransformer(model_dir, device='cuda')


def get_new_results_filename(loss_type, lr, steps):
    model_dir = os.path.join(BASE_DIR, 'embedder_finetuning_%s_%s_lr%f'%(EMBEDDER_TYPE, loss_type, lr), 'checkpoint-%d'%(steps))
    return os.path.join(model_dir, 'winoground_results.pth')


#return num_correct
def winoground_one(loss_type, lr, steps, winoground_results):
    new_results = {'aggregate' : {'correct_answers' : 0, 'incorrect_answers' : 0}, 'raw' : []}
    model = load_model(loss_type, lr, steps)
    caption0_embedding_list = model.encode([result_one['caption0'] for result_one in winoground_results['raw']])
    caption1_embedding_list = model.encode([result_one['caption1'] for result_one in winoground_results['raw']])
    desc_embedding_lists = {}
    for caption_idx in [0,1]:
        for image_idx in [0,1]:
            desc_embedding_lists[(caption_idx, image_idx)] = model.encode([result_one['LLM_generated_description_dict'][(caption_idx, image_idx)] for result_one in winoground_results['raw']])

    for t, result_one in tqdm(enumerate(winoground_results['raw'])):
        new_result_one = copy.deepcopy(result_one)
        caption0_embedding = caption0_embedding_list[t]
        caption1_embedding = caption1_embedding_list[t]
        new_result_one['caption0_caption1_cossim'] = cosine_similarity(caption0_embedding, caption1_embedding)
        for caption_idx in [0,1]:
            for image_idx in [0,1]:
                LLM_generated_description_embedding = desc_embedding_lists[(caption_idx, image_idx)][t]
                cossim = cosine_similarity(LLM_generated_description_embedding, [caption0_embedding, caption1_embedding][caption_idx])
                new_result_one['caption%d_image%d_cossim'%(caption_idx, image_idx)] = cossim

        if new_result_one['caption0_image0_cossim'] > new_result_one['caption0_image1_cossim'] and new_result_one['caption1_image0_cossim'] < new_result_one['caption1_image1_cossim']:
            new_results['aggregate']['correct_answers'] += 1
        else:
            new_results['aggregate']['incorrect_answers'] += 1

        new_results['raw'].append(new_result_one)

    if steps is not None:
        new_results_filename = get_new_results_filename(loss_type, lr, steps)
        torch.save(new_results, new_results_filename)

    return new_results['aggregate']['correct_answers']


def load_winoground_results():
    results_filename = os.path.join(BASE_DIR, 'trained_q_network_vanilla_supervision_hidden_dims_128_64/cossimexploration_repindex0_reshuffle0_winoground_results_LLM_fusion_vanilla_supervision_hidden_dims_128_64_epoch_70.pth')
    results = torch.load(results_filename, map_location='cpu', weights_only=False)
    return results


def winoground_inference_with_finetuned_text_embedders():
    winoground_results = load_winoground_results()
    xs = [0]
    xs.extend(STEPS_LIST)
    correct_answers_zero = winoground_one(None, None, None, winoground_results)
    print('ZERO: correct_answers=%s'%(str(correct_answers_zero)))
    ys = {}
    for loss_type in LOSS_TYPES:
        for lr in LRS:
            k = (loss_type, lr)
            ys[k] = [correct_answers_zero]
            for steps in tqdm(STEPS_LIST):
                correct_answers = winoground_one(loss_type, lr, steps, winoground_results)
                print('loss_type=%s, lr=%s, steps=%d: correct_answers=%s'%(loss_type, str(lr), steps, str(correct_answers)))
                ys[k].append(correct_answers)

    plt.clf()
    plt.figure(figsize=(6.4,2*4.8))
    plt.title('Winoground performance with finetuned text embedders')
    plt.xlabel('training steps')
    plt.ylabel('num correct answers')
    for loss_type, linestyle in zip(LOSS_TYPES, ['solid', 'dashed']):
        for lr, color in zip(LRS, ['b', 'k', 'r', 'orange', 'grey', 'green', 'cyan']):
            k = (loss_type, lr)
            plt.plot(xs, ys[k], linestyle=linestyle, color=color, label='loss=%s, lr=%s'%(loss_type, str(lr)))

    plt.legend()
    plt.savefig(PLOT_FILENAME)
    plt.clf()


def usage():
    print('Usage: python winoground_inference_with_finetuned_text_embedders.py')


if __name__ == '__main__':
    winoground_inference_with_finetuned_text_embedders()
