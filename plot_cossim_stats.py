import os
import sys
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/imgeneval-data')
ORIG_DATA_FILENAME = os.path.join(BASE_DIR, 'trained_q_network_vanilla_supervision_hidden_dims_128_64/cossimexploration_repindex0_reshuffle0_winoground_results_LLM_fusion_vanilla_supervision_hidden_dims_128_64_epoch_70.pth')
RESHUFFLE_DATA_FILENAME = os.path.join(BASE_DIR, 'trained_q_network_vanilla_supervision_hidden_dims_128_64/cossimexploration_repindex0_reshuffle1_winoground_results_LLM_fusion_vanilla_supervision_hidden_dims_128_64_epoch_70.pth')
REP_DATA_FILENAME = os.path.join(BASE_DIR, 'trained_q_network_vanilla_supervision_hidden_dims_128_64/cossimexploration_repindex1_reshuffle0_winoground_results_LLM_fusion_vanilla_supervision_hidden_dims_128_64_epoch_70.pth')
PLOT_PREFIX = os.path.join(BASE_DIR, 'trained_q_network_vanilla_supervision_hidden_dims_128_64/cossimexploration_stats')


def load_data():
    orig_data = torch.load(ORIG_DATA_FILENAME, map_location='cpu', weights_only=False)
    reshuffle_data = torch.load(RESHUFFLE_DATA_FILENAME, map_location='cpu', weights_only=False)
    rep_data = torch.load(REP_DATA_FILENAME, map_location='cpu', weights_only=False)
    print(f'ORIG: correct_answers = {orig_data["aggregate"]["correct_answers"]}')
    print(f'RESHUFFLE: correct_answers = {reshuffle_data["aggregate"]["correct_answers"]}')
    print(f'REP: correct_answers = {rep_data["aggregate"]["correct_answers"]}')
    return orig_data, reshuffle_data, rep_data


def compute_kde(vals, start, end, steps=1000):
    xs = np.linspace(start, end, steps)
    my_kde = gaussian_kde(vals)
    densities = my_kde(xs)
    mean = np.mean(vals)
    sd = np.std(vals, ddof=1)
    return xs, densities, mean, sd


#plot caption-caption cossim, correct-match cossim, and incorrect-match cossim
def plot_cossim_stats_orig_only(orig_data):
    caption_caption_cossims, correct_cossims, incorrect_cossims = [], [], []
    for result_one in orig_data['raw']:
        caption_caption_cossims.append(result_one['caption0_caption1_cossim'])
        for i in range(2):
            correct_cossims.append(result_one[f'caption{i}_image{i}_cossim'])
            incorrect_cossims.append(result_one[f'caption{i}_image{1-i}_cossim'])

    caption_caption_xs, caption_caption_densities, caption_caption_mean, caption_caption_sd = compute_kde(caption_caption_cossims, 0, 1)
    correct_xs, correct_densities, correct_mean, correct_sd = compute_kde(correct_cossims, 0, 1)
    incorrect_xs, incorrect_densities, incorrect_mean, incorrect_sd = compute_kde(incorrect_cossims, 0, 1)

    plt.clf()
    plt.title('Winoground cossims - is the embedder capturing semantics?')
    plt.xlabel('cossim')
    plt.ylabel('density')
    plt.plot(caption_caption_xs, caption_caption_densities, color='k', label='caption-caption cossim')
    plt.plot(correct_xs, correct_densities, color='b', label='correct-match cossim')
    plt.plot(incorrect_xs, incorrect_densities, color='r', label='incorrect-match cossim')
    plt.legend()
    _, ymax = plt.ylim()
    plt.vlines(caption_caption_mean, 0, ymax, color='k')
    plt.vlines([caption_caption_mean - caption_caption_sd, caption_caption_mean + caption_caption_sd], 0, ymax, color='k', linestyle='dashed')
    plt.vlines(correct_mean, 0, ymax, color='b')
    plt.vlines([correct_mean - correct_sd, correct_mean + correct_sd], 0, ymax, color='b', linestyle='dashed')
    plt.vlines(incorrect_mean, 0, ymax, color='r')
    plt.vlines([incorrect_mean - incorrect_sd, incorrect_mean + incorrect_sd], 0, ymax, color='r', linestyle='dashed')
    plt.xlim((0,1))
    plt.ylim((0, ymax))
    plt.savefig(PLOT_PREFIX + '_orig_only.png')
    plt.clf()


def plot_cossim_stats_reshuffle_and_rep_helper(orig_data, reshuffle_data, rep_data, correct_or_incorrect):
    assert(correct_or_incorrect in ['correct', 'incorrect'])
    correct_incorrect_diffs, reshuffle_orig_diffs, rep_orig_diffs = [], [], []
    for orig_one, reshuffle_one, rep_one in zip(orig_data['raw'], reshuffle_data['raw'], rep_data['raw']):
        for i in range(2):
            assert(orig_one[f'caption{i}'] == reshuffle_one[f'caption{i}'])
            assert(orig_one[f'caption{i}'] == rep_one[f'caption{i}'])
            correct_incorrect_diffs.append(orig_one[f'caption{i}_image{i}_cossim'] - orig_one[f'caption{i}_image{1-i}_cossim'])
            if correct_or_incorrect == 'correct':
                reshuffle_orig_diffs.append(reshuffle_one[f'caption{i}_image{i}_cossim'] - orig_one[f'caption{i}_image{i}_cossim'])
                rep_orig_diffs.append(rep_one[f'caption{i}_image{i}_cossim'] - orig_one[f'caption{i}_image{i}_cossim'])
            else:
                reshuffle_orig_diffs.append(reshuffle_one[f'caption{i}_image{1-i}_cossim'] - orig_one[f'caption{i}_image{1-i}_cossim'])
                rep_orig_diffs.append(rep_one[f'caption{i}_image{1-i}_cossim'] - orig_one[f'caption{i}_image{1-i}_cossim'])

    cid_xs, cid_densities, cid_mean, cid_sd = compute_kde(correct_incorrect_diffs, -0.25, 0.25)
    resod_xs, resod_densities, resod_mean, resod_sd = compute_kde(reshuffle_orig_diffs, -0.25, 0.25)
    repod_xs, repod_densities, repod_mean, repod_sd = compute_kde(rep_orig_diffs, -0.25, 0.25)

    plt.clf()
    plt.title('Winoground cossim diffs - effect of reshuffling + LLM nondeterminism')
    plt.xlabel('cossim diff')
    plt.ylabel('density')
    plt.plot(cid_xs, cid_densities, color='k', label='correct - incorrect cossim')
    plt.plot(resod_xs, resod_densities, color='b', label=f'reshuffle - orig cossim ({correct_or_incorrect} match)')
    plt.plot(repod_xs, repod_densities, color='r', label=f'rep - orig cossim ({correct_or_incorrect} match)')
    plt.legend()
    _, ymax = plt.ylim()
    plt.vlines(cid_mean, 0, ymax, color='k')
    plt.vlines([cid_mean - cid_sd, cid_mean + cid_sd], 0, ymax, color='k', linestyle='dashed')
    plt.vlines(resod_mean, 0, ymax, color='b')
    plt.vlines([resod_mean - resod_sd, resod_mean + resod_sd], 0, ymax, color='b', linestyle='dashed')
    plt.vlines(repod_mean, 0, ymax, color='r')
    plt.vlines([repod_mean - repod_sd, repod_mean + repod_sd], 0, ymax, color='r', linestyle='dashed')
    plt.xlim((-0.25,0.25))
    plt.ylim((0, ymax))
    plt.savefig(PLOT_PREFIX + f'_reshuffle_and_rep_{correct_or_incorrect}.png')
    plt.clf()


#plot correct-incorrect diff, correct_orig-correct_reshuffle diff, correct_orig-correct_rep diff
#and also plot correct-incorrect diff, incorrect_orig-incorrect_reshuffle diff, incorrect_orig-incorrect_rep diff
def plot_cossim_stats_reshuffle_and_rep(orig_data, reshuffle_data, rep_data):
    plot_cossim_stats_reshuffle_and_rep_helper(orig_data, reshuffle_data, rep_data, 'correct')
    plot_cossim_stats_reshuffle_and_rep_helper(orig_data, reshuffle_data, rep_data, 'incorrect')


def plot_cossim_stats():
    orig_data, reshuffle_data, rep_data = load_data()
    plot_cossim_stats_orig_only(orig_data)
    plot_cossim_stats_reshuffle_and_rep(orig_data, reshuffle_data, rep_data)


if __name__ == '__main__':
    plot_cossim_stats()
