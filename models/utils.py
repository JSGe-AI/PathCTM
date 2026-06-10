import torch
import torch.nn.functional as F
import re
import os

def compute_decay(T, params, clamp_lims=(0, 15)):
    assert len(clamp_lims), 'Clamp lims should be length 2'
    assert type(clamp_lims) == tuple, 'Clamp lims should be tuple'
    
    indices = torch.arange(T-1, -1, -1, device=params.device).reshape(T, 1).expand(T, params.shape[0])
    out = torch.exp(-indices * torch.clamp(params, clamp_lims[0], clamp_lims[1]).unsqueeze(0))
    return out

def add_coord_dim(x, scaled=True):
    B, H, W = x.shape
    x_coords = torch.arange(W, device=x.device, dtype=x.dtype).repeat(H, 1)
    y_coords = torch.arange(H, device=x.device, dtype=x.dtype).unsqueeze(-1).repeat(1, W)
    if scaled:
        x_coords /= (W-1)
        y_coords /= (H-1)
    coords = torch.stack((x_coords, y_coords), dim=-1)
    coords = coords.unsqueeze(0)
    coords = coords.repeat(B, 1, 1, 1)
    return coords

def compute_normalized_entropy(logits, reduction='mean'):
    preds = F.softmax(logits, dim=-1)

    log_preds = torch.log_softmax(logits, dim=-1)

    entropy = -torch.sum(preds * log_preds, dim=-1)

    num_classes = preds.shape[-1]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))

    normalized_entropy = entropy / max_entropy
    if len(logits.shape)>2 and reduction == 'mean':
        normalized_entropy = normalized_entropy.flatten(1).mean(-1)

    return normalized_entropy
    
def reshape_predictions(predictions, prediction_reshaper):
    B, T = predictions.size(0), predictions.size(-1)
    new_shape = [B] + prediction_reshaper + [T]
    rehaped_predictions = predictions.reshape(new_shape)
    return rehaped_predictions

def get_all_log_dirs(root_dir):
    folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.endswith(".pt") for f in filenames):
            folders.append(dirpath)
    return folders

def get_latest_checkpoint(log_dir):
    files = [f for f in os.listdir(log_dir) if re.match(r'checkpoint_\d+\.pt', f)]
    return os.path.join(log_dir, max(files, key=lambda f: int(re.search(r'\d+', f).group()))) if files else None

def get_latest_checkpoint_file(filepath, limit=300000):
    checkpoint_files = get_checkpoint_files(filepath)
    checkpoint_files = [
        f for f in checkpoint_files if int(re.search(r'checkpoint_(\d+)\.pt', f).group(1)) <= limit
    ]
    if not checkpoint_files:
        return None
    return checkpoint_files[-1]

def get_checkpoint_files(filepath):
    regex = r'checkpoint_(\d+)\.pt'
    files = [f for f in os.listdir(filepath) if re.match(regex, f)]
    files = sorted(files, key=lambda f: int(re.search(regex, f).group(1)))
    return [os.path.join(filepath, f) for f in files]

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint

def get_model_args_from_checkpoint(checkpoint):
    if "args" in checkpoint:
        return(checkpoint["args"])
    else:
        raise ValueError("Checkpoint does not contain saved args.")

def get_accuracy_and_loss_from_checkpoint(checkpoint, device="cpu"):
    training_iteration = checkpoint.get('training_iteration', 0)
    train_losses = checkpoint.get('train_losses', [])
    test_losses = checkpoint.get('test_losses', [])
    train_accuracies = checkpoint.get('train_accuracies_most_certain', [])
    test_accuracies = checkpoint.get('test_accuracies_most_certain', [])
    return training_iteration, train_losses, test_losses, train_accuracies, test_accuracies