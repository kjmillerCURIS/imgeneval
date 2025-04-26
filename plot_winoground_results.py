import os
import sys
import torch
from matplotlib import pyplot as plt
from winoground_inference_with_LLM_fusion import EPOCH_LIST, BASE_DIR


def get_result(learning_type, hidden_dims, epoch):
    hidden_dims_str = '_'.join([str(h) for h in hidden_dims])
    model_prefix = f"trained_q_network_{learning_type}_hidden_dims_{hidden_dims_str}"
    model_dir = os.path.join(BASE_DIR, model_prefix)
    results_prefix = f"winoground_results_LLM_fusion_{learning_type}_hidden_dims_{hidden_dims_str}"
    results_suffix = f"_epoch_{epoch}.pth"
    results_path = os.path.join(model_dir, results_prefix + results_suffix)
    if not os.path.exists(results_path):
        return None

    results = torch.load(results_path, map_location='cpu', weights_only=False)
    return results['aggregate']['correct_answers']


def load_winoground_results_one(learning_type, hidden_dims):
    xs, ys = [], []
    for epoch in EPOCH_LIST:
        y = get_result(learning_type, hidden_dims, epoch)
        if y is not None:
            ys.append(y)
            xs.append(epoch)

    return xs, ys


def plot_winoground_results():
    plt.clf()
    plt.title('Winoground, LLM-fusion, learning exploration')
    plt.xlabel('epoch')
    plt.ylabel('# correct')
    for learning_type in ['RL', 'vanilla_supervision']:
        for hidden_dims in [[128], [128,64]]:
            label = f'learning_type={learning_type}, hidden_dims={hidden_dims}'
            color = {1 : 'blue', 2 : 'red'}[len(hidden_dims)]
            linestyle = {'RL' : 'dashed', 'vanilla_supervision' : 'solid'}[learning_type]
            xs, ys = load_winoground_results_one(learning_type, hidden_dims)
            plt.plot(xs, ys, marker='o', color=color, linestyle=linestyle, label=label)

    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, 'winoground_LLM_fusion_learning_exploration.png'))
    plt.clf()


if __name__ == '__main__':
    plot_winoground_results()
