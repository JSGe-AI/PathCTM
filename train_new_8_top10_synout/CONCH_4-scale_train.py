import argparse
import os
import random
from datetime import datetime

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
from torchvision import transforms
from tasks.image_classification.imagenet_classes import IMAGENET2012_CLASSES
from models.pathctm_top10_synout import ContinuousThoughtMachine
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

    # Model Selection
    parser.add_argument('--model', type=str, default='ctm', choices=['ctm', 'lstm', 'ff'])

    # Model Architecture
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

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batch_size_test', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--training_iterations', type=int, default=43600)
    parser.add_argument('--warmup_steps', type=int, default=4360)
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'])
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--weight_decay_exclusion_list', type=str, nargs='+', default=[])
    parser.add_argument('--gradient_clipping', type=float, default=-1)
    parser.add_argument('--do_compile', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--num_workers_train', type=int, default=1)

    # Housekeeping
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--data_root', type=str, default='data/')
    parser.add_argument('--save_every', type=int, default=436)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--strict_reload', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--track_every', type=int, default=436)
    parser.add_argument('--n_test_batches', type=int, default=20)
    
    parser.add_argument('--device', type=int, nargs='+', default=[7])
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False)
    
    parser.add_argument('--fold', type=int, default=4)
    parser.add_argument('--num_class', type=int, default=3)

    args, _ = parser.parse_known_args()

    parser.set_defaults(
        log_dir=f"",
        train_h5_dir=f"",
        val_h5_dir=f""
    )

    args = parser.parse_args()
    return args



import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
            '0_1024': os.path.join(root_part, ''),
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
        index_tensor = torch.tensor(index_tensor, dtype=torch.long)  # [N, 2]

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
        feature_1_512,  index_1_512  = self.load_npy_data(os.path.join(self.scale_dirs['1_512'], fname))
        feature_0_8192,  index_0_8192  = self.load_npy_data(os.path.join(self.scale_dirs['0_8192'], fname))

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
    best_test_auc = 0
    best_test_loss = 100
    current_test_accuracies_most_certain = 0
    current_test_auc_most_certain = 0
    # Hosuekeeping
    args = parse_args()

    set_seed(args.seed, False)
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

    assert args.dataset in ['cifar10', 'cifar100', 'imagenet']
    
    # Configure device string (support MPS on macOS)
    if args.device[0] != -1:
        device = f'cuda:{args.device[0]}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Running model {args.model} on {device}')
    
    
    
    num_class = args.num_class
    
    train_h5_dir = args.train_h5_dir
    val_h5_dir = args.val_h5_dir

    
    
  
    train_dataset = CustomDataset(train_h5_dir, device=device)
    test_dataset = CustomDataset(val_h5_dir, device=device)
    
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    

    
    num_workers_test = 1 # Defaulting to 1, change if needed
    
    prediction_reshaper = [-1]  # Problem specific
    args.out_dims = num_class # len(class_labels)

    # For total reproducibility
    zip_python_code(f'{args.log_dir}/repo_state.zip')
    with open(f'{args.log_dir}/args.txt', 'w') as f:
        print(args, file=f)

    

    # Build model conditionally
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
    elif args.model == 'lstm':
         model = LSTMBaseline(
            num_layers=args.num_layers,
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
        ).to(device)
    elif args.model == 'ff':
        model = FFBaseline(
            d_model=args.d_model,
            backbone_type=args.backbone_type,
            out_dims=args.out_dims,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")




    model.train()

    

    decay_params = []
    no_decay_params = []
    no_decay_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue # Skip parameters that don't require gradients
        if any(exclusion_str in name for exclusion_str in args.weight_decay_exclusion_list):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
    if len(no_decay_names):
        print(f'WARNING, excluding: {no_decay_names}')

    # Optimizer and scheduler (Common setup)
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
    

    warmup_schedule = warmup(args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_schedule.step)
    if args.use_scheduler:
        if args.scheduler_type == 'multistep':
            scheduler = WarmupMultiStepLR(optimizer, warmup_steps=args.warmup_steps, milestones=args.milestones, gamma=args.gamma)
        elif args.scheduler_type == 'cosine':
            scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_steps, args.training_iterations, warmup_start_lr=1e-20, eta_min=1e-7)
        else:
            raise NotImplementedError


    # Metrics tracking
    start_iter = 0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    test_auc_list = []# 
    test_auc_most_certain = [] 

    iters = []
    # Conditional metrics for CTM/LSTM
    train_accuracies_most_certain = [] if args.model in ['ctm', 'lstm'] else None
    test_accuracies_most_certain = [] if args.model in ['ctm', 'lstm'] else None

    scaler = torch.amp.GradScaler("cuda" if "cuda" in device else "cpu", enabled=args.use_amp)

    # Reloading logic
    if args.reload:
        checkpoint_path = f'{args.log_dir}/checkpoint.pt'
        if os.path.isfile(checkpoint_path):
            print(f'Reloading from: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if not args.strict_reload: print('WARNING: not using strict reload for model weights!')
            load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=args.strict_reload)
            print(f" Loaded state_dict. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")

            if not args.reload_model_only:
                print('Reloading optimizer etc.')
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_iter = checkpoint['iteration']
                # Load common metrics
                train_losses = checkpoint['train_losses']
                test_losses = checkpoint['test_losses']
                train_accuracies = checkpoint['train_accuracies'] 
                test_accuracies = checkpoint['test_accuracies'] 
                iters = checkpoint['iters']

                # Load conditional metrics if they exist in checkpoint and are expected for current model
                if args.model in ['ctm', 'lstm']:
                    train_accuracies_most_certain = checkpoint['train_accuracies_most_certain']
                    test_accuracies_most_certain = checkpoint['test_accuracies_most_certain']

            else:
                print('Only reloading model!')

            if 'torch_rng_state' in checkpoint:
                # Reset seeds
                torch.set_rng_state(checkpoint['torch_rng_state'].cpu().byte())
                np.random.set_state(checkpoint['numpy_rng_state'])
                random.setstate(checkpoint['random_rng_state'])

            del checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Conditional Compilation
    if args.do_compile:
        print('Compiling...')
        if hasattr(model, 'backbone'):
            model.backbone = torch.compile(model.backbone, mode='reduce-overhead', fullgraph=True)

        # Compile synapses only for CTM
        if args.model == 'ctm':
            model.synapses = torch.compile(model.synapses, mode='reduce-overhead', fullgraph=True)

    # Training
    iterator = iter(trainloader)


    with tqdm(total=args.training_iterations, initial=start_iter, leave=False, position=0, dynamic_ncols=True) as pbar:
        for bi in range(start_iter, args.training_iterations):

            current_lr = optimizer.param_groups[-1]['lr']

            try:
                inputs, targets, indexs, relations = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, targets, indexs, relations = next(iterator)
            # inputs = inputs.to(device)
            targets = targets.to(device)

            loss = None
            accuracy = None
            # Model-specific forward and loss calculation
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=args.use_amp):
                if args.do_compile: # CUDAGraph marking for clean compile
                     torch.compiler.cudagraph_mark_step_begin()

                if args.model == 'ctm':
                    predictions, certainties, synchronisation = model(inputs, indexs, relations, device)
                    loss, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
            
                    accuracy = (predictions.argmax(1)[torch.arange(predictions.size(0), device=predictions.device),where_most_certain] == targets).float().mean().item()
                    pbar_desc = f'CTM Loss={loss.item():0.3f}. Acc={accuracy:0.3f}. LR={current_lr:0.6f}. Where_certain={where_most_certain.float().mean().item():0.2f}+-{where_most_certain.float().std().item():0.2f} ({where_most_certain.min().item():d}<->{where_most_certain.max().item():d})'

                elif args.model == 'lstm':
                    predictions, certainties, synchronisation = model(inputs)
                    loss, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
                    # LSTM where_most_certain will just be -1 because use_most_certain is False owing to stability issues with LSTM training
                    accuracy = (predictions.argmax(1)[torch.arange(predictions.size(0), device=predictions.device),where_most_certain] == targets).float().mean().item()
                    pbar_desc = f'LSTM Loss={loss.item():0.3f}. Acc={accuracy:0.3f}. LR={current_lr:0.6f}. Where_certain={where_most_certain.float().mean().item():0.2f}+-{where_most_certain.float().std().item():0.2f} ({where_most_certain.min().item():d}<->{where_most_certain.max().item():d})'

                elif args.model == 'ff':
                    predictions = model(inputs)
                    loss = nn.CrossEntropyLoss()(predictions, targets)
                    accuracy = (predictions.argmax(1) == targets).float().mean().item()
                    pbar_desc = f'FF Loss={loss.item():0.3f}. Acc={accuracy:0.3f}. LR={current_lr:0.6f}'

            scaler.scale(loss).backward()

            if args.gradient_clipping!=-1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            pbar.set_description(f'Dataset={args.dataset}. Model={args.model}. {pbar_desc}')


            ######################### Metrics tracking and plotting (conditional logic needed)  ########################
            if (bi % args.track_every == 0 or bi == args.warmup_steps) and (bi != 0 or args.reload_model_only):

                iters.append(bi)
                current_train_losses = []
                current_test_losses = []
                current_train_accuracies = [] 
                current_test_accuracies = [] 
                current_train_accuracies_most_certain = [] # Only for CTM/LSTM
                current_test_accuracies_most_certain = [] # Only for CTM/LSTM

                
                    

                

                ######################### Switch to eval mode for test metrics (fixed BN stats)  ########################
                model.eval()
                pbar.set_description('Tracking: Computing TEST metrics')
                all_where_most_certain_list = [] 
                all_probs_list = []
                with torch.inference_mode(): # Use inference_mode for test eval
                    
                    loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
                    all_targets_list = []
                    all_predictions_list = []
                    all_predictions_most_certain_list = [] # Only for CTM/LSTM
                    all_losses = []
                    current_test_loss = 0

                    with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
                       for inferi, (inputs, targets, indexs, relations) in enumerate(loader):
                            # inputs = inputs.to(device)
                            targets = targets.to(device)
                            all_targets_list.append(targets.detach().cpu().numpy())

                            # Model-specific forward and loss for evaluation
                            if args.model == 'ctm':
                                these_predictions, certainties, _ = model(inputs, indexs, relations, device)
                                loss, where_most_certain = image_classification_loss(these_predictions, certainties, targets, use_most_certain=True)
                                all_predictions_list.append(these_predictions.argmax(1).detach().cpu().numpy())
                                all_predictions_most_certain_list.append(these_predictions.argmax(1)[torch.arange(these_predictions.size(0), device=these_predictions.device), where_most_certain].detach().cpu().numpy())
                                all_probs_list.append(these_predictions.softmax(dim=1).detach().cpu().numpy())
                                all_where_most_certain_list.append(where_most_certain.detach().cpu().numpy()) 
                                
                            elif args.model == 'lstm':
                                these_predictions, certainties, _ = model(inputs)
                                loss, where_most_certain = image_classification_loss(these_predictions, certainties, targets, use_most_certain=True)
                                all_predictions_list.append(these_predictions.argmax(1).detach().cpu().numpy())
                                all_predictions_most_certain_list.append(these_predictions.argmax(1)[torch.arange(these_predictions.size(0), device=these_predictions.device), where_most_certain].detach().cpu().numpy())

                            elif args.model == 'ff':
                                these_predictions = model(inputs)
                                loss = nn.CrossEntropyLoss()(these_predictions, targets)
                                all_predictions_list.append(these_predictions.argmax(1).detach().cpu().numpy())

                            all_losses.append(loss.item())

                            
                            pbar_inner.set_description(f'Computing metrics for test (Batch {inferi+1})')
                            pbar_inner.update(1)

                    all_targets = np.concatenate(all_targets_list)
                    all_predictions = np.concatenate(all_predictions_list)
                    current_test_loss = np.mean(all_losses)
                    test_losses.append(np.mean(all_losses))

                    if args.model in ['ctm', 'lstm']:
                        current_test_accuracies = np.mean(all_predictions == all_targets[...,np.newaxis], axis=0)
                        test_accuracies.append(current_test_accuracies)
                        all_predictions_most_certain = np.concatenate(all_predictions_most_certain_list)
                        current_test_accuracies_most_certain = (all_targets == all_predictions_most_certain).mean()
                        test_accuracies_most_certain.append(current_test_accuracies_most_certain)
                       
                        all_where_most_certain = np.concatenate(all_where_most_certain_list).flatten()
                        
                        -
                        from sklearn.metrics import roc_auc_score
                        from sklearn.preprocessing import label_binarize

                        all_probs = np.concatenate(all_probs_list)     
                        all_targets_flat = all_targets.reshape(-1) # [N]
                        num_samples = all_probs.shape[0]
                        num_classes = all_probs.shape[1]
                        num_steps = all_probs.shape[2]


                        
                        auc_per_step = []
                        for t in range(num_steps):
                            probs_t = all_probs[:, :, t]  # [N, C]

                            if num_classes == 2:
                                auc_t = roc_auc_score(all_targets_flat, probs_t[:, 1])
                            else:
                                all_targets_bin = label_binarize(all_targets_flat, classes=np.arange(num_classes))
                                auc_t = roc_auc_score(all_targets_bin, probs_t, average='macro', multi_class='ovr')

                            auc_per_step.append(auc_t)
                        test_auc_list.append(auc_per_step)
                        
                        if args.model in ['ctm', 'lstm']:
                            
                            probs_most_certain = all_probs[np.arange(num_samples), :, all_where_most_certain] #  [N, C]

                            
                            if num_classes == 2:
                                auc_mc = roc_auc_score(all_targets_flat, probs_most_certain[:, 1])
                                current_test_auc_most_certain= auc_mc
                            else:
                                
                                auc_mc = roc_auc_score(all_targets_bin, probs_most_certain, average='macro', multi_class='ovr')
                                current_test_auc_most_certain= auc_mc
                            
                            test_auc_most_certain.append(auc_mc)
                            
                    else: 
                         current_test_accuracies = (all_targets == all_predictions).mean()
                         test_accuracies.append(current_test_accuracies)
                         
                         

          
                figacc = plt.figure(figsize=(10, 10))
                axacc_train = figacc.add_subplot(211)
                axacc_test = figacc.add_subplot(212)
                cm = sns.color_palette("viridis", as_cmap=True)

                if args.model in ['ctm', 'lstm']:
                    
                    test_auc_arr = np.array(test_auc_list)
                    test_acc_arr = np.array(test_accuracies) # Shape (N_iters, T)
                    num_ticks = test_auc_arr.shape[1]
                    for ti in range(num_ticks):
                         axacc_train.plot(iters, test_auc_arr[:, ti], color=cm(ti / num_ticks), alpha=0.3)
                         axacc_test.plot(iters, test_acc_arr[:, ti], color=cm(ti / num_ticks), alpha=0.3)
                    # Plot most certain accuracy
                    axacc_train.plot(iters, test_auc_most_certain, 'k--', alpha=0.7, label='Most certain')
                    axacc_test.plot(iters, test_accuracies_most_certain, 'k--', alpha=0.7, label='Most certain')
                else: # FF
                    axacc_train.plot(iters, auc_per_step, 'k-', alpha=0.7, label='AUC') # Simple line
                    axacc_test.plot(iters, test_accuracies, 'k-', alpha=0.7, label='Accuracy')

                axacc_train.set_title('Test AUC')
                axacc_test.set_title('Test Accuracy')
                axacc_train.legend(loc='lower right')
                axacc_test.legend(loc='lower right')
                axacc_train.set_xlim([0, args.training_iterations])
                axacc_test.set_xlim([0, args.training_iterations])
                if args.dataset=='cifar10':
                    axacc_train.set_ylim([0.75, 1])
                    axacc_test.set_ylim([0.75, 1])



                figacc.tight_layout()
                figacc.savefig(f'{args.log_dir}/AUC.png', dpi=150)
                plt.close(figacc)

                figloss = plt.figure(figsize=(10, 5))
                axloss = figloss.add_subplot(111)
                # axloss.plot(iters, train_losses, 'b-', linewidth=1, alpha=0.8, label=f'Train: {train_losses[-1]:.4f}')
                axloss.plot(iters, test_losses, 'r-', linewidth=1, alpha=0.8, label=f'Test: {test_losses[-1]:.4f}')
                axloss.legend(loc='upper right')
                axloss.set_xlim([0, args.training_iterations])
                axloss.set_ylim(bottom=0)

                figloss.tight_layout()
                figloss.savefig(f'{args.log_dir}/losses.png', dpi=150)
                plt.close(figloss)

                


            pbar.update(1)
