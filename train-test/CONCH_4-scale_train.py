import argparse
import gc
import os
import random
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pathctm import PatchCTM
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import image_classification_loss
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup


sns.set_style("darkgrid")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
torchvision.disable_beta_transforms_warning()

warnings.filterwarnings(
    "ignore",
    message="using precomputed metric; inverse_transform will be unavailable",
)
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in power",
    category=RuntimeWarning,
)
warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning, r"^PIL\.TiffImagePlugin$")
warnings.filterwarnings("ignore", "UserWarning: Metadata Warning", UserWarning, r"^PIL\.TiffImagePlugin$")
warnings.filterwarnings("ignore", "UserWarning: Truncated File Read", UserWarning, r"^PIL\.TiffImagePlugin$")

FEATURE_SCALE_DIR_CANDIDATES = {
    "2048": ["0_2048"],
    "4096": ["0_4096"],
    "8192": ["0_8192"],
}

RELATION_DIR_CANDIDATES = {
    "2048_1024": [
        "relation_2048-1024",
        "relation_2048-1024_index",
    ],
    "4096_2048": [
        "relation_4096-2048",
        "relation_4096-2048_index",
    ],
    "8192_4096": [
        "relation_8192-4096",
        "relation_8192-4096_index",
    ],
}


def first_existing_path(candidates):
    for candidate in candidates:
        if Path(candidate).exists():
            return str(candidate)
    return str(candidates[0])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--d_model", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--backbone_type", type=str, default="resnet18-4")
    parser.add_argument("--d_input", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument(
        "--positional_embedding_type",
        type=str,
        default="none",
        choices=["none", "learnable-fourier", "multi-learnable-fourier", "custom-rotational"],
    )
    parser.add_argument("--synapse_depth", type=int, default=12)
    parser.add_argument("--n_synch_out", type=int, default=150)
    parser.add_argument("--n_synch_action", type=int, default=150)
    parser.add_argument("--neuron_select_type", type=str, default="random")
    parser.add_argument("--n_random_pairing_self", type=int, default=0)
    parser.add_argument("--memory_length", type=int, default=30)
    parser.add_argument("--deep_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory_hidden_dims", type=int, default=64)
    parser.add_argument("--dropout_nlm", type=float, default=None)
    parser.add_argument("--do_normalisation", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--batch_size_test", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--training_iterations", type=int, default=43600)
    parser.add_argument("--warmup_steps", type=int, default=4360)
    parser.add_argument("--use_scheduler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["multistep", "cosine"])
    parser.add_argument("--milestones", type=int, default=[8000, 15000, 20000], nargs="+")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--weight_decay_exclusion_list", type=str, nargs="+", default=[])
    parser.add_argument("--gradient_clipping", type=float, default=-1)
    parser.add_argument("--do_compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_workers_train", type=int, default=1)

    parser.add_argument("--dataset", type=str, default="BRACS")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--save_every", type=int, default=436)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reload_model_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict_reload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--track_every", type=int, default=436)
    parser.add_argument("--n_test_batches", type=int, default=20)
    parser.add_argument("--device", type=int, nargs="+", default=[7])
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--num_class", type=int, default=3)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--train_h5_dir", type=str)
    parser.add_argument("--val_h5_dir", type=str)

    args, _ = parser.parse_known_args()

    parser.set_defaults(
        log_dir=str(PROJECT_ROOT / "train_test" / f"fold_{args.fold}"),
        train_h5_dir=f"path/{args.num_class}cls_fold/fold_{args.fold}_train.txt",
        val_h5_dir=f"path/{args.num_class}cls_fold/fold_{args.fold}_val.txt",
    )
    return parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, txt_file):
        self.samples = []
        with open(txt_file, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                label_str, path = line.split(";", 1)
                self.samples.append((int(label_str), path))

        example_path = Path(self.samples[0][1]).resolve()
        self.data_root = self.find_data_root(example_path)
        self.scale_dirs = {
            scale: self.resolve_existing_dir(self.data_root, directory_candidates)
            for scale, directory_candidates in FEATURE_SCALE_DIR_CANDIDATES.items()
        }
        self.scale_relation_dirs = {
            scale: self.resolve_existing_dir(self.data_root, directory_candidates)
            for scale, directory_candidates in RELATION_DIR_CANDIDATES.items()
        }

    @staticmethod
    def load_npy_data(full_path):
        data = np.load(full_path, allow_pickle=True).item()
        feature = data["feature"]

        index_tensor = []
        for idx_str in data["index"]:
            x_coord, y_coord = idx_str.replace(".png", "").split("_")[:2]
            index_tensor.append([int(x_coord), int(y_coord)])
        return feature, torch.tensor(index_tensor, dtype=torch.long)

    @staticmethod
    def load_relation(full_path):
        relation = np.load(full_path, allow_pickle=True).item()
        return {
            large_idx: torch.tensor(small_indices, dtype=torch.long)
            for large_idx, small_indices in relation.items()
        }

    @staticmethod
    def find_data_root(example_path):
        for parent in (example_path.parent, *example_path.parents):
            if parent.name.startswith("0_1024"):
                return parent.parent
        raise ValueError(f"Could not infer data root from sample path: {example_path}")

    @staticmethod
    def resolve_existing_dir(data_root, candidates):
        for candidate in candidates:
            candidate_path = data_root / candidate
            if candidate_path.is_dir():
                return candidate_path
        raise FileNotFoundError(
            f"Could not find any of {candidates} under data root {data_root}"
        )

    @staticmethod
    def resolve_existing_file(candidates):
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "Could not find file in any candidate path: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    def resolve_feature_path(self, scale, file_path, filename):
        sample_path = Path(file_path)
        if scale == "1024":
            return sample_path

        label_dir = sample_path.parent.name
        scale_dir = self.scale_dirs[scale]
        return self.resolve_existing_file(
            [
                scale_dir / filename,
                scale_dir / label_dir / filename,
                scale_dir / "all_class" / filename,
            ]
        )

    def resolve_relation_path(self, relation_key, file_path, filename):
        label_dir = Path(file_path).parent.name
        relation_dir = self.scale_relation_dirs[relation_key]
        return self.resolve_existing_file(
            [
                relation_dir / filename,
                relation_dir / label_dir / filename,
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, file_path = self.samples[idx]
        filename = Path(file_path).name

        features = {}
        index_list = {}
        for scale in ["1024", "2048", "4096", "8192"]:
            feature_path = self.resolve_feature_path(scale, file_path, filename)
            feature, index_tensor = self.load_npy_data(feature_path)
            features[scale] = feature
            index_list[scale] = index_tensor

        relations = {
            relation_key: self.load_relation(self.resolve_relation_path(relation_key, file_path, filename))
            for relation_key in self.scale_relation_dirs
        }

        return features, label, index_list, relations


def require_args(args, names):
    missing = [f"--{name}" for name in names if not getattr(args, name)]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")


def get_device(device_ids):
    if device_ids[0] != -1:
        return f"cuda:{device_ids[0]}"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_autocast_device_type(device):
    return "cuda" if device.startswith("cuda") else "cpu"


def build_model(args, device):
    return PatchCTM(
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
        out_dims=args.num_class,
        prediction_reshaper=[-1],
        dropout=args.dropout,
        dropout_nlm=args.dropout_nlm,
        neuron_select_type=args.neuron_select_type,
        n_random_pairing_self=args.n_random_pairing_self,
    ).to(device)


def build_optimizer(model, args):
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

    if no_decay_names:
        print(f"WARNING, excluding: {no_decay_names}")

    if no_decay_names and args.weight_decay != 0:
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0},
            ],
            lr=args.lr,
            eps=1e-8 if not args.use_amp else 1e-6,
        )

    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        eps=1e-8 if not args.use_amp else 1e-6,
        weight_decay=args.weight_decay,
    )


def build_scheduler(optimizer, args):
    warmup_schedule = warmup(args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_schedule.step)

    if not args.use_scheduler:
        return scheduler

    if args.scheduler_type == "multistep":
        return WarmupMultiStepLR(
            optimizer,
            warmup_steps=args.warmup_steps,
            milestones=args.milestones,
            gamma=args.gamma,
        )

    if args.scheduler_type == "cosine":
        return WarmupCosineAnnealingLR(
            optimizer,
            args.warmup_steps,
            args.training_iterations,
            warmup_start_lr=1e-20,
            eta_min=1e-7,
        )

    raise NotImplementedError


def compute_auc(targets, probabilities, average="macro"):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    num_classes = probabilities.shape[1]
    try:
        if num_classes == 2:
            return float(roc_auc_score(targets, probabilities[:, 1]))

        targets_bin = label_binarize(targets, classes=np.arange(num_classes))
        return float(
            roc_auc_score(
                targets_bin,
                probabilities,
                average=average,
                multi_class="ovr",
            )
        )
    except ValueError:
        return float("nan")


def evaluate_model(model, dataloader, args, device):
    model.eval()

    all_targets_list = []
    all_predictions_list = []
    all_predictions_most_certain_list = []
    all_where_most_certain_list = []
    all_probabilities_list = []
    all_losses = []

    with torch.inference_mode():
        with tqdm(total=len(dataloader), leave=False, position=1, dynamic_ncols=True) as pbar_inner:
            for batch_index, (inputs, targets, indexs, relations) in enumerate(dataloader):
                targets = targets.to(device)
                all_targets_list.append(targets.detach().cpu().numpy())

                predictions, certainties, _ = model(inputs, indexs, relations, device)
                loss, where_most_certain = image_classification_loss(
                    predictions,
                    certainties,
                    targets,
                    use_most_certain=True,
                )
                all_predictions_list.append(predictions.argmax(1).detach().cpu().numpy())
                all_predictions_most_certain_list.append(
                    predictions.argmax(1)[
                        torch.arange(predictions.size(0), device=predictions.device),
                        where_most_certain,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )
                all_where_most_certain_list.append(where_most_certain.detach().cpu().numpy())
                all_probabilities_list.append(predictions.softmax(dim=1).detach().cpu().numpy())

                all_losses.append(loss.item())
                pbar_inner.set_description(f"Computing metrics for test (Batch {batch_index + 1})")
                pbar_inner.update(1)

    all_targets = np.concatenate(all_targets_list)
    all_predictions = np.concatenate(all_predictions_list)
    metrics = {"loss": float(np.mean(all_losses))}

    all_probabilities = np.concatenate(all_probabilities_list)
    all_targets_flat = all_targets.reshape(-1)
    num_samples = all_probabilities.shape[0]
    num_steps = all_probabilities.shape[2]

    auc_per_step = []
    for step_index in range(num_steps):
        auc_per_step.append(
            compute_auc(all_targets_flat, all_probabilities[:, :, step_index], average="macro")
        )

    all_predictions_most_certain = np.concatenate(all_predictions_most_certain_list)
    all_where_most_certain = np.concatenate(all_where_most_certain_list).flatten()
    most_certain_probabilities = all_probabilities[
        np.arange(num_samples),
        :,
        all_where_most_certain,
    ]

    metrics["accuracy_per_step"] = np.mean(
        all_predictions == all_targets[..., np.newaxis],
        axis=0,
    )
    metrics["accuracy_most_certain"] = float(
        (all_targets == all_predictions_most_certain).mean()
    )
    metrics["auc_per_step"] = auc_per_step
    metrics["auc_most_certain"] = compute_auc(
        all_targets_flat,
        most_certain_probabilities,
        average="macro",
    )

    return metrics


def save_plots(args, iters, test_losses, test_accuracies, test_auc_list, test_accuracies_most_certain, test_auc_most_certain):
    figacc = plt.figure(figsize=(10, 10))
    axacc_auc = figacc.add_subplot(211)
    axacc_acc = figacc.add_subplot(212)
    cmap = sns.color_palette("viridis", as_cmap=True)

    test_auc_arr = np.array(test_auc_list, dtype=float)
    test_acc_arr = np.array(test_accuracies, dtype=float)
    num_steps = test_auc_arr.shape[1]

    for step_index in range(num_steps):
        color_value = cmap(step_index / max(num_steps, 1))
        axacc_auc.plot(iters, test_auc_arr[:, step_index], color=color_value, alpha=0.3)
        axacc_acc.plot(iters, test_acc_arr[:, step_index], color=color_value, alpha=0.3)

    axacc_auc.plot(iters, test_auc_most_certain, "k--", alpha=0.7, label="Most certain")
    axacc_acc.plot(iters, test_accuracies_most_certain, "k--", alpha=0.7, label="Most certain")

    axacc_auc.set_title("Test AUC")
    axacc_acc.set_title("Test Accuracy")
    axacc_auc.legend(loc="lower right")
    axacc_acc.legend(loc="lower right")
    axacc_auc.set_xlim([0, args.training_iterations])
    axacc_acc.set_xlim([0, args.training_iterations])

    if args.dataset == "cifar10":
        axacc_auc.set_ylim([0.75, 1])
        axacc_acc.set_ylim([0.75, 1])

    figacc.tight_layout()
    figacc.savefig(f"{args.log_dir}/AUC.png", dpi=150)
    plt.close(figacc)

    figloss = plt.figure(figsize=(10, 5))
    axloss = figloss.add_subplot(111)
    axloss.plot(iters, test_losses, "r-", linewidth=1, alpha=0.8, label=f"Test: {test_losses[-1]:.4f}")
    axloss.legend(loc="upper right")
    axloss.set_xlim([0, args.training_iterations])
    axloss.set_ylim(bottom=0)

    figloss.tight_layout()
    figloss.savefig(f"{args.log_dir}/losses.png", dpi=150)
    plt.close(figloss)


if __name__ == "__main__":
    args = parse_args()
    require_args(args, ["log_dir", "train_h5_dir", "val_h5_dir"])

    set_seed(args.seed, False)
    os.makedirs(args.log_dir, exist_ok=True)

    device = get_device(args.device)
    print(f"Running model ctm on {device}")

    train_dataset = CustomDataset(args.train_h5_dir)
    test_dataset = CustomDataset(args.val_h5_dir)
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers_train)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    zip_python_code(f"{args.log_dir}/repo_state.zip", root_dir=PROJECT_ROOT)
    with open(f"{args.log_dir}/args.txt", "w") as handle:
        print(args, file=handle)

    model = build_model(args, device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler(get_autocast_device_type(device), enabled=args.use_amp)

    start_iter = 0
    test_losses = []
    test_accuracies = []
    test_auc_list = []
    test_auc_most_certain = []
    iters = []
    test_accuracies_most_certain = []

    if args.reload:
        checkpoint_path = f"{args.log_dir}/checkpoint.pt"
        if os.path.isfile(checkpoint_path):
            print(f"Reloading from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if not args.strict_reload:
                print("WARNING: not using strict reload for model weights!")

            load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=args.strict_reload)
            print(
                f" Loaded state_dict. Missing: {load_result.missing_keys}, "
                f"Unexpected: {load_result.unexpected_keys}"
            )

            if not args.reload_model_only:
                print("Reloading optimizer etc.")
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                start_iter = checkpoint["iteration"]
                test_losses = checkpoint.get("test_losses", test_losses)
                test_accuracies = checkpoint.get("test_accuracies", test_accuracies)
                test_auc_list = checkpoint.get("test_auc_list", test_auc_list)
                test_auc_most_certain = checkpoint.get("test_auc_most_certain", test_auc_most_certain)
                iters = checkpoint.get("iters", iters)
                test_accuracies_most_certain = checkpoint.get(
                    "test_accuracies_most_certain",
                    test_accuracies_most_certain,
                )
            else:
                print("Only reloading model!")

            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu().byte())
                np.random.set_state(checkpoint["numpy_rng_state"])
                random.setstate(checkpoint["random_rng_state"])

            del checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.do_compile:
        print("Compiling...")
        if hasattr(model, "backbone"):
            model.backbone = torch.compile(model.backbone, mode="reduce-overhead", fullgraph=True)
        model.synapses = torch.compile(model.synapses, mode="reduce-overhead", fullgraph=True)

    model.train()
    iterator = iter(trainloader)
    autocast_device_type = get_autocast_device_type(device)

    with tqdm(total=args.training_iterations, initial=start_iter, leave=False, position=0, dynamic_ncols=True) as pbar:
        for bi in range(start_iter, args.training_iterations):
            current_lr = optimizer.param_groups[-1]["lr"]

            try:
                inputs, targets, indexs, relations = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, targets, indexs, relations = next(iterator)

            targets = targets.to(device)

            with torch.autocast(device_type=autocast_device_type, dtype=torch.float16, enabled=args.use_amp):
                if args.do_compile and device.startswith("cuda"):
                    torch.compiler.cudagraph_mark_step_begin()

                predictions, certainties, _ = model(inputs, indexs, relations, device)
                loss, where_most_certain = image_classification_loss(
                    predictions,
                    certainties,
                    targets,
                    use_most_certain=True,
                )
                accuracy = (
                    predictions.argmax(1)[
                        torch.arange(predictions.size(0), device=predictions.device),
                        where_most_certain,
                    ]
                    == targets
                ).float().mean().item()
                pbar_desc = (
                    f"CTM Loss={loss.item():0.3f}. Acc={accuracy:0.3f}. "
                    f"LR={current_lr:0.6f}. "
                    f"Where_certain={where_most_certain.float().mean().item():0.2f}+-"
                    f"{where_most_certain.float().std().item():0.2f} "
                    f"({where_most_certain.min().item():d}<->{where_most_certain.max().item():d})"
                )

            scaler.scale(loss).backward()

            if args.gradient_clipping != -1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            pbar.set_description(f"Dataset={args.dataset}. Model=ctm. {pbar_desc}")

            should_track = (bi % args.track_every == 0 or bi == args.warmup_steps) and (
                bi != 0 or args.reload_model_only
            )
            if should_track:
                iters.append(bi)
                pbar.set_description("Tracking: Computing TEST metrics")
                metrics = evaluate_model(model, testloader, args, device)

                test_losses.append(metrics["loss"])
                test_accuracies.append(metrics["accuracy_per_step"])
                test_accuracies_most_certain.append(metrics["accuracy_most_certain"])
                test_auc_list.append(metrics["auc_per_step"])
                test_auc_most_certain.append(metrics["auc_most_certain"])

                save_plots(
                    args,
                    iters,
                    test_losses,
                    test_accuracies,
                    test_auc_list,
                    test_accuracies_most_certain,
                    test_auc_most_certain,
                )
                model.train()

            pbar.update(1)
