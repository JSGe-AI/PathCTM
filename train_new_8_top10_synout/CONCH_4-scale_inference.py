import argparse
import os
import random
from datetime import datetime

import time

import h5py
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
import torch.nn as nn
from tqdm.auto import tqdm

from data.custom_datasets import ImageNet
from torchvision import datasets
from torchvision import transforms
from tasks.image_classification.imagenet_classes import IMAGENET2012_CLASSES
from models.pathctm_inference_conch_top10_synout import ContinuousThoughtMachine
from models.lstm import LSTMBaseline
from models.ff import FFBaseline
from tasks.image_classification.plotting import plot_neural_dynamics, make_classification_gif
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import image_classification_loss
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup

from autoclip.torch import QuantileClip

import gc
import torchvision
torchvision.disable_beta_transforms_warning()


import warnings
warnings.filterwarnings("ignore", message="using precomputed metric; inverse_transform will be unavailable")
warnings.filterwarnings('ignore', message='divide by zero encountered in power', category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    "Corrupt EXIF data",
    UserWarning,
    r"^PIL\.TiffImagePlugin$"
)
warnings.filterwarnings(
    "ignore",
    "UserWarning: Metadata Warning",
    UserWarning,
    r"^PIL\.TiffImagePlugin$"
)
warnings.filterwarnings(
    "ignore",
    "UserWarning: Truncated File Read",
    UserWarning,
    r"^PIL\.TiffImagePlugin$"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='ctm', choices=['ctm', 'lstm', 'ff'])

    parser.add_argument('--d_model', type=int, default=4096)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--backbone_type', type=str, default='resnet18-4')
    parser.add_argument('--d_input', type=int, default=1024)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=80)
    parser.add_argument('--positional_embedding_type', type=str, default='none', choices=['none','learnable-fourier','multi-learnable-fourier','custom-rotational'])
    parser.add_argument('--synapse_depth', type=int, default=12)
    parser.add_argument('--n_synch_out', type=int, default=150)
    parser.add_argument('--n_synch_action', type=int, default=150)
    parser.add_argument('--neuron_select_type', type=str, default='random')
    parser.add_argument('--n_random_pairing_self', type=int, default=0)
    parser.add_argument('--memory_length', type=int, default=30)
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--memory_hidden_dims', type=int, default=64)
    parser.add_argument('--dropout_nlm', type=float, default=None)
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batch_size_test', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'])
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--weight_decay_exclusion_list', type=str, nargs='+', default=[])
    parser.add_argument('--gradient_clipping', type=float, default=-1)
    parser.add_argument('--do_compile', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--num_workers_train', type=int, default=1)

    parser.add_argument('--dataset', type=str, default='BRACS')
    parser.add_argument('--data_root', type=str, default='data/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--strict_reload', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--n_test_batches', type=int, default=20)
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument('--num_class', type=int, default=7)
    parser.add_argument('--threshold', type=float, default=1)
    parser.add_argument('--fold', type=int, default=4)

    args, _ = parser.parse_known_args()

    parser.set_defaults(
        log_dir=f"",
        test_h5_dir=f"",
        checkpoint_path=f""
    )

    args = parser.parse_args()
    return args


class CustomDataset(Dataset):
    def __init__(self, txt_file, device='cuda'):
        self.device = torch.device(device)

        self.samples = []
        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                label_str, path = line.split(";", 1)
                label = int(label_str)
                self.samples.append((label, path))

        example_path = self.samples[0][1]
        root_part = example_path.split('0_1024', 1)[0]

        self.scale_dirs = {
            '0_1024': os.path.join(root_part, '0_1024'),
            '1_1024': os.path.join(root_part, '1_1024'),
            '1_512': os.path.join(root_part, '1_512'),
            '0_8192': os.path.join(root_part, '0_8192')
        }

        self.scale_relation_dirs = {
            '2048_1024': os.path.join(root_part, 'relation_2048-1024'),
            '4096_2048': os.path.join(root_part, 'relation_4096-2048'),
            '8192_4096': os.path.join(root_part, 'relation_8192-4096')
        }
    

    def load_npy_data(self, full_path):
        data = np.load(full_path, allow_pickle=True).item()
        feature = data['feature']

        raw_index_list = data['index']
        index_tensor = []
        for idx_str in raw_index_list:
            parts = idx_str.replace('.png', '').split('_')
            xy = [int(parts[0]), int(parts[1])]
            index_tensor.append(xy)
        index_tensor = torch.tensor(index_tensor, dtype=torch.long)

        return feature, index_tensor

    def load_relation(self, full_path):
        relation = np.load(full_path, allow_pickle=True).item()
        relation_tensor = {}
        for large_idx, small_indices in relation.items():
            small_coords_and_classes = torch.tensor(small_indices, dtype=torch.long)
            relation_tensor[large_idx] = small_coords_and_classes
        return relation_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, file_path = self.samples[idx]
        fname = os.path.basename(file_path)

        feature_0_1024, index_0_1024 = self.load_npy_data(os.path.join(self.scale_dirs['0_1024'], fname))
        feature_1_1024, index_1_1024 = self.load_npy_data(os.path.join(self.scale_dirs['1_1024'], fname))
        feature_1_512, index_1_512 = self.load_npy_data(os.path.join(self.scale_dirs['1_512'], fname))
        feature_0_8192, index_0_8192 = self.load_npy_data(os.path.join(self.scale_dirs['0_8192'], fname))

        features = {
            '1024': feature_0_1024,
            '2048': feature_1_512,
            '4096': feature_1_1024,
            '8192': feature_0_8192
        }

        index_list = {
            '1024': index_0_1024,
            '2048': index_1_512,
            '4096': index_1_1024,
            '8192': index_0_8192
        }

        relation_2048_1024 = self.load_relation(os.path.join(self.scale_relation_dirs['2048_1024'], fname))
        relation_4096_2048 = self.load_relation(os.path.join(self.scale_relation_dirs['4096_2048'], fname))
        relation_8192_4096 = self.load_relation(os.path.join(self.scale_relation_dirs['8192_4096'], fname))

        relations = {
            '2048_1024': relation_2048_1024,
            '4096_2048': relation_4096_2048,
            '8192_4096': relation_8192_4096
        }

        return features, label, index_list, relations


if __name__=='__main__':
    best_test_acc = 0
    current_test_accuracies_most_certain = 0
    set_seed(args.seed, False)
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

    assert args.dataset in ['cifar10', 'cifar100', 'imagenet']
    
    if args.device[0] != -1:
        device = f'cuda:{args.device[0]}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Running model {args.model} on {device}')
    
    
    
    num_class = args.num_class
    
    test_h5_dir = args.test_h5_dir

    test_dataset = CustomDataset(test_h5_dir, device=device)

    testloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    

    threshold = args.threshold
    
    
    num_workers_test = 1
    
    prediction_reshaper = [-1]
    args.out_dims = num_class


    model = None
    if args.model == 'ctm':
        model = ContinuousThoughtMachine(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=args.do_normalisation,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
            dropout_nlm=args.dropout_nlm,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
        ).to(device)
    
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    
    decay_params = []
    no_decay_params = []
    no_decay_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(exclusion_str in name for exclusion_str in args.weight_decay_exclusion_list):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
    if len(no_decay_names):
        print(f'WARNING, excluding: {no_decay_names}')

    if len(no_decay_names) and args.weight_decay!=0:
        optimizer = torch.optim.AdamW([{'params': decay_params, 'weight_decay':args.weight_decay},
                                       {'params': no_decay_params, 'weight_decay':0}],
                                     lr=args.lr,
                                     eps=1e-8 if not args.use_amp else 1e-6)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                     lr=args.lr,
                                     eps=1e-8 if not args.use_amp else 1e-6,
                                     weight_decay=args.weight_decay)
    


    start_iter = 0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    iters = []
    train_accuracies_most_certain = [] if args.model in ['ctm', 'lstm'] else None
    test_accuracies_most_certain = [] if args.model in ['ctm', 'lstm'] else None

    scaler = torch.amp.GradScaler("cuda" if "cuda" in device else "cpu", enabled=args.use_amp)


    if args.do_compile:
        print('Compiling...')
        if hasattr(model, 'backbone'):
            model.backbone = torch.compile(model.backbone, mode='reduce-overhead', fullgraph=True)

        if args.model == 'ctm':
            model.synapses = torch.compile(model.synapses, mode='reduce-overhead', fullgraph=True)
            
    checkpoint_path = args.checkpoint_path
    if os.path.isfile(checkpoint_path):
        print(f'Reloading from: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        iters = checkpoint['iters']
        if not args.strict_reload: print('WARNING: not using strict reload for model weights!')
        load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=args.strict_reload)
        print(f" Loaded state_dict. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")




    # current_train_losses = []
    current_test_losses = []
    # current_train_accuracies = []
    current_test_accuracies = []
    # current_train_accuracies_most_certain = []
    current_test_accuracies_most_certain = []
    slide_num = 0
    patch_num_all = 0
    patch_num_avg = 0
    patch_num_Standard_scale = 0
    patch_num_Standard_scale_avg = 0


    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.reset_running_stats()

    

    
    total_inference_time_ms = 0.0
    if torch.cuda.is_available():
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    with torch.inference_mode():
        loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        all_targets_list = []
        all_predictions_list = []
        all_predictions_most_certain_list = []
        all_losses = []
    
        acc_list = []
  

        with open(log_txt_path, 'w') as f:
            f.write("certainty\tstepi\n")
            f.write("=========================\n")

        with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
            for inferi, (inputs, targets, indexs, relations) in enumerate(loader):
                slide_num += 1
                targets = targets.to(device)
                all_targets_list.append(targets.detach().cpu().numpy())

                if args.model == 'ctm':
                    if torch.cuda.is_available():
                        starter.record()
                    
                    current_prediction, current_certainty, stepi, patch_num_current, patch_num_Standard_scale_current = model(
                        inputs, indexs, relations, device, threshold
                    )
                    
                    if torch.cuda.is_available():
                        ender.record()
                        torch.cuda.synchronize()
                        curr_time_ms = starter.elapsed_time(ender)
                        total_inference_time_ms += curr_time_ms
                    
                    
                    patch_num_all = patch_num_all + patch_num_current
                    patch_num_Standard_scale = patch_num_Standard_scale + patch_num_Standard_scale_current
                    probabilities = torch.softmax(current_prediction, dim=1) 
                    all_predictions_list.append(probabilities.detach().cpu().numpy())
                                            
                    try:
                        with open(log_txt_path, 'a') as f:
                            f.write(f"{current_certainty[:, 1].item():.4f}\t{stepi}\n")
                    except Exception as e:
                        print(f" {e}")

                    pred_label = current_prediction.argmax(dim=1)
                    acc = (pred_label == targets).float().mean().item()
                    acc_list.append(acc)

                pbar_inner.set_description(f'Computing metrics for test (Batch {inferi+1})')
                pbar_inner.update(1)
            
            average_inference_time_ms = total_inference_time_ms / len(loader)


    patch_num_avg = patch_num_all/slide_num
    patch_num_Standard_scale_avg = patch_num_Standard_scale/slide_num
    mean_acc = sum(acc_list) / len(acc_list)
    print(" fold: ", args.fold)

    print("patch_num_all: ", patch_num_all)
    print("patch_num_avg: ", patch_num_avg)
    print("slide_num: ", slide_num)
    print("patch_num_Standard-scale_avg: ", patch_num_Standard_scale_avg)
    
    
    from sklearn.metrics import roc_auc_score
    if all_targets_list and all_predictions_list:
        all_targets_np = np.concatenate(all_targets_list).flatten()
        all_predictions_np = np.concatenate(all_predictions_list)

        num_classes = all_predictions_np.shape[1]
        
        if num_classes == 2:
            positive_class_scores = all_predictions_np[:, 1]
            auc_score = roc_auc_score(all_targets_np, positive_class_scores)

        elif num_classes > 2:
            try:
                auc_score = roc_auc_score(all_targets_np, all_predictions_np, multi_class='ovr', average='micro')

            except ValueError as e:
                print(f"")
    else:
        print("")
    print("="*50 + "\n")