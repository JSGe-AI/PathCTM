import numpy as np
import random
import torch


import os
import zipfile
import glob

def zip_python_code(output_filename):


    with zipfile.ZipFile(output_filename, 'w') as zipf:
        files = glob.glob('/', recursive=True) + glob.glob('/', recursive=True) + glob.glob('/', recursive=True) + glob.glob('*.py', recursive=True)
        for file in files:
            root = '/'.join(file.split('/')[:-1])
            nm = file.split('/')[-1]
            zipf.write(os.path.join(root, nm))

def set_seed(seed=42, deterministic=True):
 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
