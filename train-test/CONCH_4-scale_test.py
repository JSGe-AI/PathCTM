import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pathctm_inference import PatchCTM
from utils.housekeeping import set_seed


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

    parser.add_argument("--batch_size_test", type=int, default=1)
    parser.add_argument("--do_compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataset", type=str, default="BRACS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict_reload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=int, nargs="+", default=[0])
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_class", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--test_h5_dir", type=str)
    parser.add_argument("--checkpoint_path", type=str)

    args, _ = parser.parse_known_args()

    checkpoint_candidates = [
        f"path/CONCH_BRACS_{args.num_class}cls/fold_{args.fold}/best_checkpoint.pt",
        f"path/CONCH_BRACS_{args.num_class}cls/fold_{args.fold}/best_checkpoint.pt",
    ]
    parser.set_defaults(
        log_dir=str(PROJECT_ROOT / "inference_test" / f"fold_{args.fold}"),
        test_h5_dir=f"path/{args.num_class}cls_fold/fold_{args.fold}_test.txt",
        checkpoint_path=first_existing_path(checkpoint_candidates),
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


def compute_auc(targets, probabilities, average="micro"):
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


if __name__ == "__main__":
    args = parse_args()
    require_args(args, ["log_dir", "test_h5_dir", "checkpoint_path"])

    set_seed(args.seed, False)
    os.makedirs(args.log_dir, exist_ok=True)

    device = get_device(args.device)
    print(f"Running model ctm on {device}")

    test_dataset = CustomDataset(args.test_h5_dir)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = build_model(args, device)

    if args.do_compile:
        print("Compiling...")
        if hasattr(model, "backbone"):
            model.backbone = torch.compile(model.backbone, mode="reduce-overhead", fullgraph=True)
        model.synapses = torch.compile(model.synapses, mode="reduce-overhead", fullgraph=True)

    checkpoint_path = args.checkpoint_path
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Reloading from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not args.strict_reload:
        print("WARNING: not using strict reload for model weights!")
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=args.strict_reload)
    print(
        f" Loaded state_dict. Missing: {load_result.missing_keys}, "
        f"Unexpected: {load_result.unexpected_keys}"
    )
    model.eval()

    log_txt_path = os.path.join(args.log_dir, "certainty_stepi.txt")
    slide_num = 0
    patch_num_all = 0
    patch_num_standard_scale = 0
    total_inference_time_ms = 0.0
    all_targets_list = []
    all_predictions_list = []
    acc_list = []

    autocast_device_type = get_autocast_device_type(device)
    with open(log_txt_path, "w") as handle:
        handle.write("certainty\tstepi\n")
        handle.write("=========================\n")

    with torch.inference_mode():
        with tqdm(total=len(testloader), leave=False, position=1, dynamic_ncols=True) as pbar_inner:
            for batch_index, (inputs, targets, indexs, relations) in enumerate(testloader):
                slide_num += 1
                targets = targets.to(device)
                all_targets_list.append(targets.detach().cpu().numpy())

                start_time = time.perf_counter()
                with torch.autocast(device_type=autocast_device_type, dtype=torch.float16, enabled=args.use_amp):
                    current_prediction, current_certainty, stepi, patch_num_current, patch_num_standard_scale_current = model(
                        inputs,
                        indexs,
                        relations,
                        device,
                        args.threshold,
                    )
                total_inference_time_ms += (time.perf_counter() - start_time) * 1000

                patch_num_all += patch_num_current
                patch_num_standard_scale += patch_num_standard_scale_current
                probabilities = torch.softmax(current_prediction, dim=1)
                all_predictions_list.append(probabilities.detach().cpu().numpy())

                with open(log_txt_path, "a") as handle:
                    handle.write(f"{current_certainty[:, 1].item():.4f}\t{stepi}\n")

                pred_label = current_prediction.argmax(dim=1)
                acc_list.append((pred_label == targets).float().mean().item())

                pbar_inner.set_description(f"Computing metrics for test (Batch {batch_index + 1})")
                pbar_inner.update(1)

    average_inference_time_ms = total_inference_time_ms / len(testloader) if len(testloader) else 0.0
    patch_num_avg = patch_num_all / slide_num if slide_num else 0.0
    patch_num_standard_scale_avg = patch_num_standard_scale / slide_num if slide_num else 0.0
    mean_acc = float(np.mean(acc_list)) if acc_list else float("nan")

    auc_score = float("nan")
    if all_targets_list and all_predictions_list:
        all_targets_np = np.concatenate(all_targets_list).flatten()
        all_predictions_np = np.concatenate(all_predictions_list)
        auc_score = compute_auc(all_targets_np, all_predictions_np, average="micro")

    print("fold:", args.fold)
    print("patch_num_all:", patch_num_all)
    print("patch_num_avg:", patch_num_avg)
    print("slide_num:", slide_num)
    print("patch_num_standard_scale_avg:", patch_num_standard_scale_avg)
    print("mean_acc:", mean_acc)
    print("auc:", auc_score)
    print("average_inference_time_ms:", average_inference_time_ms)
    print("=" * 50 + "\n")
