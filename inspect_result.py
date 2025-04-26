import os
import sys
import torch


if __name__ == '__main__':
    path = sys.argv[1]
    results = torch.load(path, map_location='cpu', weights_only=False)
    print(results['aggregate'])
