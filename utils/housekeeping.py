import numpy as np
import random
import torch

import os
import zipfile

def zip_python_code(output_filename, root_dir=None):
    root_dir = os.path.abspath(root_dir or os.getcwd())
    with zipfile.ZipFile(output_filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for current_root, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                full_path = os.path.join(current_root, filename)
                arcname = os.path.relpath(full_path, root_dir)
                zipf.write(full_path, arcname)

def set_seed(seed=42, deterministic=True):
 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
