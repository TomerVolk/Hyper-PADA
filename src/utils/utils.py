import torch
import random
import numpy as np
import os


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_last_version(folder):
    files = list(os.listdir(folder))
    files = [(int(file.split("_")[1]), file) for file in files]
    files = sorted(files, reverse=True)[0][1]
    cp_file = list(os.listdir(f"{folder}{files}/checkpoints"))
    if len(cp_file) != 1:
        raise NotImplementedError
    cp_file = cp_file[0]
    cp_file = f"{folder}{files}/checkpoints/{cp_file}"
    return cp_file


def get_source(task, target, data_dir):
    all_pairs = sorted(list({dom for dom in list(os.listdir(data_dir)) if "." not in dom}))
    if task == 'sentiment':
        lang, dom = target.split("-")
        source = sorted([x for x in all_pairs if x.split("-")[0] != lang and x.split("-")[1] != dom])
    else:
        source = sorted([x for x in all_pairs if x != target])
    return source
