import os
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, cohen_kappa_score, f1_score,
                             balanced_accuracy_score, roc_curve, auc, silhouette_score)

from utils.sampler import BalancedBatchSampler

def save_dict(dict: dict,
              path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

def read_dict(path: str) -> pickle:
    with open(path, 'rb') as f:
        return pickle.load(f)

def print_color(statment: str,
                color: str ='RED',
                end: str ='\n') -> None:
    assert color in ['RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN']

    if color == 'RED':
        print(Fore.RED + statment + Style.RESET_ALL, end=end)
    elif color == 'GREEN':
        print(Fore.GREEN + statment + Style.RESET_ALL, end=end)
    elif color == 'YELLOW':
        print(Fore.YELLOW + statment + Style.RESET_ALL, end=end)
    elif color == 'BLUE':
        print(Fore.BLUE + statment + Style.RESET_ALL, end=end)
    elif color == 'MAGENTA':
        print(Fore.MAGENTA + statment + Style.RESET_ALL, end=end)
    elif color == 'CYAN':
        print(Fore.CYAN + statment + Style.RESET_ALL, end=end)

def print_title(statment: str) -> None:
    len_title = len(statment)
    print_color('\n' + len_title*'-' + '\n' + len_title*'-', 'YELLOW')
    print_color(statment, 'MAGENTA')
    print_color(len_title*'-' + '\n' + len_title*'-' + '\n', 'YELLOW')

def print_title_(statment: str) -> None:
    len_title = len(statment)
    print('\n' + len_title*'-' + '\n' + len_title*'-')
    print(statment)
    print(len_title*'-' + '\n' + len_title*'-' + '\n')

def print_mix(statment: str,
              color: str) -> None:
    stat = statment.split('__')
    for i, words in enumerate(stat):
        if i % 2 == 0:
            print(words, end='')
        else:
            print_color(words, color=color, end='')
    print()

def my_collate(batch):
    """
    The dimension of data in each batch should be same to apply torch.stack
    since each slide has its own number of patches, we get an error.
    Solution: write a collate_fn function that in each batch padd zero to the data
    at the end of it, and return the size of padding, so in model.py we could remove
    that part.
    """
    max_nb = max([item[0].shape[0] for item in batch])
    data      = []
    pad_patch = []
    coords    = []
    for item in batch:
        data_  = item[0]
        coord_ = item[3]
        pad = max_nb - data_.shape[0]
        if pad != 0:
            feature = data_.shape[1]
            data_   = torch.cat((data_, torch.zeros(pad,feature)), dim=0)
            coord_ += pad * [(-100, -100)] # add (-100,-100) coord for padded ones
        data.append(data_)
        pad_patch.append(torch.tensor(pad))
        coords.append(coord_)
    target = [item[1] for item in batch] # ints should be returned as torch.tensor
    id     = [item[2] for item in batch] # string should be returned as tuple
    return torch.stack(data), torch.stack(target), tuple(id), torch.stack(pad_patch), coords

def metrics_(gt_labels, pred_labels, pred_probs, num_classes):
    pred_probs = np.asarray(pred_probs)
    if num_classes > 2:
        overall_auc = roc_auc_score(gt_labels, pred_probs, multi_class='ovr', average='macro')
    else:
        overall_auc = roc_auc_score(gt_labels, pred_probs[:, 1], average='macro')
    overall_acc     = accuracy_score(gt_labels, pred_labels)
    balanced_acc    = balanced_accuracy_score(gt_labels, pred_labels)
    overall_kappa   = cohen_kappa_score(gt_labels, pred_labels)
    overall_f1      = f1_score(gt_labels, pred_labels, average='macro')
    conf_mat        = confusion_matrix(gt_labels, pred_labels).T
    acc_per_subtype = conf_mat.diagonal() / conf_mat.sum(axis=0) * 100
    acc_per_subtype[np.isinf(acc_per_subtype)] = 0.00

    # roc curve for classes
    ovr_roc_curve = {'fpr': {}, 'tpr': {}, 'thresh': {}}
    for num_ in range(num_classes):
        ovr_roc_curve['fpr'][num_], ovr_roc_curve['tpr'][num_], ovr_roc_curve['thresh'][num_] = roc_curve(gt_labels,
                                                pred_probs[:, num_], pos_label=num_)

    return {'overall_auc': overall_auc, 'overall_acc': overall_acc, 'overall_kappa': overall_kappa,
            'overall_f1': overall_f1, 'conf_mat': conf_mat, 'acc_per_subtype': acc_per_subtype,
            'balanced_acc': balanced_acc, 'roc_curve': ovr_roc_curve}

def metrics(info, num_classes):
    """Calculate all metrics for given dataloader
    """
    metrics = metrics_(info['gt_label'], info['prediction'],
                       info['probability'], num_classes)
    return metrics

def cutoff_youdens_j(fpr, tpr, thresh):
    opt_idx    = np.argmax(tpr - fpr)
    opt_thresh = thresh[opt_idx]
    return opt_thresh

def plot_roc_curve(ovr_roc_curve, num_classes, path, dataloader):
    plt.figure()
    if num_classes > 2:
        for i in range(num_classes):
            plt.plot(ovr_roc_curve['fpr'][i], ovr_roc_curve['tpr'][i],
            label=f"class {dataloader.dataset.classes[i]} vs rest (area = "
                  f"{auc(ovr_roc_curve['fpr'][i], ovr_roc_curve['tpr'][i]):0.2f})")
    else:
        i = 1
        plt.plot(ovr_roc_curve['fpr'][i], ovr_roc_curve['tpr'][i],
        label=f"{dataloader.dataset.classes[0]} vs {dataloader.dataset.classes[1]} (area = "
              f"{auc(ovr_roc_curve['fpr'][i], ovr_roc_curve['tpr'][i]):0.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(path)


def handle_log_folder(cfg):
    cfg["log_dir"]     = os.path.join(cfg["log_dir"], f"{cfg['experiment_name']}")
    cfg["representation_dir"]  = os.path.join(cfg["log_dir"], 'representation')

    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["representation_dir"], exist_ok=True)

    if cfg['model_method'] == 'train-attention':
        cfg["roc_dir"]     = os.path.join(cfg["log_dir"], f"roc_curves/{cfg['model']}")
        cfg["dict_dir"]    = os.path.join(cfg["log_dir"], f"information/{cfg['model']}")
        cfg["checkpoints"] = os.path.join(cfg["log_dir"], f"checkpoints/{cfg['model']}")
        cfg["tensorboard_dir"] = os.path.join(cfg["log_dir"], f"tensorboard/{cfg['model']}")
        os.makedirs(cfg["roc_dir"], exist_ok=True)
        os.makedirs(cfg["dict_dir"], exist_ok=True)
        os.makedirs(cfg["checkpoints"], exist_ok=True)
        os.makedirs(cfg["tensorboard_dir"], exist_ok=True)
