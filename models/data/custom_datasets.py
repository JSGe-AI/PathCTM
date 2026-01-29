import os
import torch
import random
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from PIL import Image
from datasets import load_dataset


class SortDataset(Dataset):
    def __init__(self, N):
        self.N = N

    def __len__(self):
        return 10000000

    def __getitem__(self, idx):
        data = torch.zeros(self.N).normal_()
        ordering = torch.argsort(data)
        return data, ordering


class QAMNISTDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        num_images,
        num_images_delta,
        num_repeats_per_input,
        num_operations,
        num_operations_delta,
    ):
        self.base_dataset = base_dataset

        self.num_images = num_images
        self.num_images_delta = num_images_delta
        self.num_images_range = self._calc_num_images_range()

        self.operators = ["+", "-"]

        self.num_operations = num_operations
        self.num_operations_delta = num_operations_delta
        self.num_operations_range = self._calc_num_operations_range()

        self.num_repeats_per_input = num_repeats_per_input

        self.current_num_digits = num_images
        self.current_num_operations = num_operations

        self.modulo_base = 10

    def _calc_num_images_range(self):
        min_val = self.num_images - self.num_images_delta
        max_val = self.num_images + self.num_images_delta
        assert min_val >= 1
        return [min_val, max_val]

    def _calc_num_operations_range(self):
        min_val = self.num_operations - self.num_operations_delta
        max_val = self.num_operations + self.num_operations_delta
        assert min_val >= 1
        return [min_val, max_val]

    def set_num_digits(self, num_digits):
        self.current_num_digits = num_digits

    def set_num_operations(self, num_operations):
        self.current_num_operations = num_operations

    def _build_question(self, targets):

        question = []
        equations = []

        num_digits = self.current_num_digits
        num_ops = self.current_num_operations

        idx = np.random.randint(num_digits)
        cur_val = targets[idx] % self.modulo_base

        question.extend([idx] * self.num_repeats_per_input)

        for _ in range(num_ops):

            op_idx = np.random.randint(2)
            op = self.operators[op_idx]
            question.extend([-(op_idx + 1)] * self.num_repeats_per_input)

            idx = np.random.randint(num_digits)
            digit = targets[idx]

            question.extend([idx] * self.num_repeats_per_input)

            if op == "+":
                new_val = (cur_val + digit) % self.modulo_base
            else:
                new_val = (cur_val - digit) % self.modulo_base

            equations.append(
                f"({cur_val} {op} {digit}) mod {self.modulo_base} = {new_val}"
            )

            cur_val = new_val

        return cur_val, question, "\n".join(equations)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):

        images = []
        targets = []

        for _ in range(self.current_num_digits):
            img, tgt = self.base_dataset[np.random.randint(len(self.base_dataset))]
            images.append(img)
            targets.append(tgt)

        obs = torch.repeat_interleave(
            torch.stack(images, 0),
            repeats=self.num_repeats_per_input,
            dim=0,
        )

        target, question, readable = self._build_question(targets)

        return obs, question, readable, target


class ImageNet(Dataset):

    def __init__(self, which_split, transform, data_dir=None):

        if data_dir is None:
            data_dir = os.environ.get("DATASET_DIR", None)

        if data_dir is None:
            raise ValueError("Dataset directory not specified.")

        dataset = load_dataset(
            path="imagefolder",
            data_dir=data_dir,
            split=which_split,
        )

        self.base_dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):

        item = self.base_dataset[idx]

        image = self.transform(item["image"].convert("RGB"))
        target = item["label"]

        return image, target


class MazeImageFolder(ImageFolder):

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=Image.open,
        is_valid_file=None,
        which_set="train",
        augment_p=0.5,
        maze_route_length=10,
        trunc=False,
        expand_range=True,
    ):

        super().__init__(
            root,
            transform,
            target_transform,
            loader,
            is_valid_file,
        )

        self.which_set = which_set
        self.augment_p = augment_p
        self.maze_route_length = maze_route_length
        self.trunc = trunc
        self.expand_range = expand_range

        self.all_paths = {}

        self._preload()

        for i in range(len(self.preloaded_samples)):
            self.all_paths[i] = self._solve(self.preloaded_samples[i])

    def _preload(self):

        samples = []

        with tqdm(total=len(self.samples)) as pbar:

            for i, (path, _) in enumerate(self.samples):

                img = self.loader(path)
                img = np.array(img).astype(np.float32) / 255.0

                samples.append(img)

                pbar.update(1)

                if self.trunc and i >= 999:
                    break

        self.preloaded_samples = samples

    def __len__(self):

        if hasattr(self, "preloaded_samples"):
            return len(self.preloaded_samples)

        return super().__len__()

    def _solve(self, x):

        x = np.copy(x)

        start = np.argwhere((x == [1, 0, 0]).all(axis=2))
        end = np.argwhere((x == [0, 1, 0]).all(axis=2))

        if len(start) == 0 or len(end) == 0:
            return None

        y, x0 = start[0]
        ty, tx = end[0]

        path = [4] * self.maze_route_length

        i = 0

        while (y, x0) != (ty, tx) and i < len(path):

            ny, nx, d = -1, -1, -1

            if y > 0 and ((x[y - 1, x0] == [0, 0, 1]).all() or (x[y - 1, x0] == [0, 1, 0]).all()):
                ny, nx, d = y - 1, x0, 0

            elif y < x.shape[0] - 1 and ((x[y + 1, x0] == [0, 0, 1]).all() or (x[y + 1, x0] == [0, 1, 0]).all()):
                ny, nx, d = y + 1, x0, 1

            elif x0 > 0 and ((x[y, x0 - 1] == [0, 0, 1]).all() or (x[y, x0 - 1] == [0, 1, 0]).all()):
                ny, nx, d = y, x0 - 1, 2

            elif x0 < x.shape[1] - 1 and ((x[y, x0 + 1] == [0, 0, 1]).all() or (x[y, x0 + 1] == [0, 1, 0]).all()):
                ny, nx, d = y, x0 + 1, 3

            path[i] = d
            i += 1

            x[y, x0] = [255, 255, 255]

            y, x0 = ny, nx

        return np.array(path)

    def __getitem__(self, index):

        sample = np.copy(self.preloaded_samples[index])
        path = np.copy(self.all_paths[index])

        if self.which_set == "train":

            if random.random() < self.augment_p:

                k = random.choice([-1, 1])
                sample = np.rot90(sample, k, axes=(0, 1))

                for i in range(len(path)):

                    if path[i] == 0:
                        path[i] = 3 if k == -1 else 2
                    elif path[i] == 1:
                        path[i] = 2 if k == -1 else 3
                    elif path[i] == 2:
                        path[i] = 0 if k == -1 else 1
                    elif path[i] == 3:
                        path[i] = 1 if k == -1 else 0

            if random.random() < self.augment_p:

                sample = np.fliplr(sample)

                for i in range(len(path)):
                    if path[i] == 2:
                        path[i] = 3
                    elif path[i] == 3:
                        path[i] = 2

            if random.random() < self.augment_p:

                sample = np.flipud(sample)

                for i in range(len(path)):
                    if path[i] == 0:
                        path[i] = 1
                    elif path[i] == 1:
                        path[i] = 0

        sample = torch.from_numpy(sample.copy()).permute(2, 0, 1)

        mask = (
            (sample[0] == 0)
            & (sample[1] == 0)
            & (sample[2] == 1)
        )

        sample[:, mask] = 1

        if not self.expand_range:
            return sample, path

        return (sample * 2) - 1, path


class ParityDataset(Dataset):

    def __init__(self, sequence_length=64, length=100000):

        self.sequence_length = sequence_length
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        vec = 2 * torch.randint(0, 2, (self.sequence_length,)) - 1
        vec = vec.float()

        neg = (vec == -1).long()

        cs = torch.cumsum(neg, dim=0)

        tgt = (cs % 2 != 0).long()

        return vec, tgt
