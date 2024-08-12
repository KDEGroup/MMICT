"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import gzip
import logging
import os
import random as rnd
import tarfile
import zipfile

import decord
import webdataset as wds
import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset, ChainDataset
from torch.utils.data.distributed import DistributedSampler
from decord import VideoReader
from lavis.common.registry import registry
from lavis.datasets.datasets.base_dataset import ConcatDataset
from lavis.common.dist_utils import get_world_size, get_rank
from tqdm import tqdm

decord.bridge.set_bridge("torch")
MAX_INT = registry.get("MAX_INT")


def load_video(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform"):
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int)
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2)) # 从输入中随机选取 k 个不重复的元素列表
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    frms = vr.get_batch(indices).permute(3, 0, 1, 2).float()  # (C, T, H, W)

    return frms


def load_video_clip(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", start=0, end=0):
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = int(start), int(end)
    fps = vr.get_avg_fps()
    n_frms = min(n_frms, end - start)

    if sampling == "uniform":
        indices = np.arange(start * fps, end * fps, fps).astype(int)
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2)) 
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    # frms = vr.get_batch(indices).permute(3, 0, 1, 2).float()  # (C, T, H, W)
    frms = vr.get_batch(indices).permute(0, 3, 1, 2).float()  # (T, C, H, W)

    return frms


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples


def reorg_datasets_by_split(datasets):
    """
    Organizes datasets by split.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by name.

    Returns:
        Dict of datasets by split {split_name: List[Datasets]}.
    """
    # if len(datasets) == 1:
    #     return datasets[list(datasets.keys())[0]]
    # else:
    reorg_datasets = dict()

    # reorganize by split
    for _, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():
            if split_name not in reorg_datasets:
                reorg_datasets[split_name] = [dataset_split]
            else:
                reorg_datasets[split_name].append(dataset_split)

    return reorg_datasets


def concat_datasets(datasets):
    """
    Concatenates multiple datasets into a single dataset.

    It supports may-style datasets and DataPipeline from WebDataset. Currently, does not support
    generic IterableDataset because it requires creating separate samplers.

    Now only supports conctenating training datasets and assuming validation and testing
    have only a single dataset. This is because metrics should not be computed on the concatenated
    datasets.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by split.

    Returns:
        Dict of concatenated datasets by split, "train" is the concatenation of multiple datasets,
        "val" and "test" remain the same.

        If the input training datasets contain both map-style and DataPipeline datasets, returns
        a tuple, where the first element is a concatenated map-style dataset and the second
        element is a chained DataPipeline dataset.

    """
    # concatenate datasets in the same split
    for split_name in datasets:
        if split_name != "train":
            assert (
                len(datasets[split_name]) == 1
            ), "Do not support multiple {} datasets.".format(split_name)
            datasets[split_name] = datasets[split_name][0]
        else:
            iterable_datasets, map_datasets = [], []
            for dataset in datasets[split_name]:
                if isinstance(dataset, wds.DataPipeline):
                    logging.info(
                        "Dataset {} is IterableDataset, can't be concatenated.".format(
                            dataset
                        )
                    )
                    iterable_datasets.append(dataset)
                elif isinstance(dataset, IterableDataset):
                    raise NotImplementedError(
                        "Do not support concatenation of generic IterableDataset."
                    )
                else:
                    map_datasets.append(dataset)

            # if len(iterable_datasets) > 0:
            # concatenate map-style datasets and iterable-style datasets separately
            chained_datasets = (
                ChainDataset(iterable_datasets) if len(iterable_datasets) > 0 else None
            )
            concat_datasets = (
                ConcatDataset(map_datasets) if len(map_datasets) > 0 else None
            )

            train_datasets = concat_datasets, chained_datasets
            train_datasets = tuple([x for x in train_datasets if x is not None])
            train_datasets = (
                train_datasets[0] if len(train_datasets) == 1 else train_datasets
            )

            datasets[split_name] = train_datasets

    return datasets


def extract_archive(from_path, to_path=None, overwrite=False):
    """Extract archive.

    Args:
        from_path: the path of the archive.
        to_path: the root path of the extracted files (directory of from_path)
        overwrite: overwrite existing files (False)

    Returns:
        List of paths to extracted files even if not overwritten.

    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> from_path = './validation.tar.gz'
        >>> to_path = './'
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']

    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    if from_path.endswith((".tar.gz", ".tgz")):
        logging.info("Opening tar file {} to {}.".format(from_path, to_path))
        with tarfile.open(from_path, "r") as tar:
            files = []
            for file_ in tqdm(tar):
                file_path = os.path.join(to_path, file_.name)
                if file_.isfile():
                    files.append(file_path)
                    if os.path.exists(file_path):
                        logging.info("{} already extracted.".format(file_path))
                        if not overwrite:
                            continue
                tar.extract(file_, to_path)
            logging.info("Finished extracting tar file {}.".format(from_path))
            return files

    elif from_path.endswith(".zip"):
        assert zipfile.is_zipfile(from_path), from_path
        logging.info("Opening zip file {} to {}.".format(from_path, to_path))
        with zipfile.ZipFile(from_path, "r") as zfile:
            files = []
            for file_ in tqdm(zfile.namelist()):
                file_path = os.path.join(to_path, file_)
                files.append(file_path)
                if os.path.exists(file_path):
                    logging.info("{} already extracted.".format(file_path))
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        files = [f for f in files if os.path.isfile(f)]
        logging.info("Finished extracting zip file {}.".format(from_path))
        return files

    elif from_path.endswith(".gz"):
        logging.info("Opening gz file {} to {}.".format(from_path, to_path))
        default_block_size = 65536
        filename = from_path[:-3]
        files = [filename]
        with gzip.open(from_path, "rb") as gzfile, open(filename, "wb") as d_file:
            while True:
                block = gzfile.read(default_block_size)
                if not block:
                    break
                else:
                    d_file.write(block)
            d_file.write(block)
        logging.info("Finished extracting gz file {}.".format(from_path))
        return files

    else:
        raise NotImplementedError(
            "We currently only support tar.gz, .tgz, .gz and zip achives."
        )


def save_frames_grid(img_array, out_path):
    import torch
    from PIL import Image
    from torchvision.utils import make_grid

    if len(img_array.shape) == 3:
        img_array = img_array.unsqueeze(0)
    elif len(img_array.shape) == 5:
        b, t, c, h, w = img_array.shape
        img_array = img_array.view(-1, c, h, w)
    elif len(img_array.shape) == 4:
        pass
    else:
        raise NotImplementedError(
            "Supports only (b,t,c,h,w)-shaped inputs. First two dimensions can be ignored."
        )

    assert img_array.shape[1] == 3, "Exepcting input shape of (H, W, 3), i.e. RGB-only."

    grid = make_grid(img_array)
    ndarr = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    img = Image.fromarray(ndarr)

    img.save(out_path)

class CustomDistributedSampler(DistributedSampler):
    """
    support setting indics sequence and custom operations
    """
    def __init__(self, dataset, batch_size=None, epoch=0, seed: int = 0):
        self.start_iter = 0
        self.global_rank = get_rank()
        self.world_size = get_world_size()
        self.batch_size = batch_size
        self.num_samples = len(dataset)
        self.epoch = epoch
        self.seed = seed
        self.dialogs_index = dataset.datasets[0].dialogs_to_ITpair #! TODO, revise
        # self.dialogs_index = dataset.get_dialogs_to_ITpair()
        self.dialog2type = dataset.datasets[0].dialog2type
        self.type2dialog = dataset.datasets[0].type2dialog
        self.length = len(dataset)
        self.pre_iter = -100
        logging.info(f"sample_num: {len(dataset)}")

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed + self.global_rank)

        # indices = torch.randperm(self.num_samples, generator=g).tolist()
        dialog_length = len(self.dialogs_index)
        dialog_indices = torch.randperm(dialog_length, generator=g).tolist()
        indices = []
        dialog_flag = []

        indices_batch = []
        dialog_flag_batch = []
        flag = 0

        for dialog_id in self.dialogs_index:
            dialog_i = self.dialogs_index[dialog_id]
            dialog_i_length = len(dialog_i)
            # drop last
            if len(indices_batch) + dialog_i_length <= self.batch_size:
                indices_batch += dialog_i
                dialog_flag_batch += [flag] * dialog_i_length
                flag = (flag+1) % 2
            else:
                if len(indices_batch) < self.batch_size:
                    padding_indices = [-1] * (self.batch_size - len(indices_batch))
                    padding_flags = [-1] * (self.batch_size - len(dialog_flag_batch))
                    indices_batch += padding_indices
                    dialog_flag_batch += padding_flags
                indices += indices_batch
                dialog_flag += dialog_flag_batch
                indices_batch = []
                dialog_flag_batch = []
                flag = 0
                indices_batch += dialog_i
                dialog_flag_batch += [flag] * dialog_i_length
                flag = (flag+1) % 2

        start_index = self.start_iter * self.batch_size
        indices = indices[start_index:]

        return iter(indices)
    
    def set_epoch(self, epoch=0):
        self.epoch += 1

    def set_start_iter(self, start_iter):
        """
        Set the iteration number from which the sampling should start. This is
        used to find the marker in the data permutation order from where the
        sampler should start sampling.
        """
        self.start_iter = start_iter
        

class er(DistributedSampler):
    """
    More fine-grained state DataSampler that uses training iteration and epoch
    both for shuffling data. PyTorch DistributedSampler only uses epoch
    for the shuffling and starts sampling data from the start. In case of training
    on very large data, we train for one epoch only and when we resume training,
    we want to resume the data sampler from the training iteration.
    """

    def __init__(self, dataset, batch_size=None, args=None, seed: int = 0):
        """
        Initializes the instance of StatefulDistributedSampler. Random seed is set
        for the epoch set and data is shuffled. For starting the sampling, use
        the start_iter (set to 0 or set by checkpointing resuming) to
        sample data from the remaining images.
        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle
            batch_size (int): batch size we want the sampler to sample
            seed (int): Seed for the torch generator.
        """
        super().__init__(dataset, shuffle=False, seed=seed)

        self.start_iter = 0
        self.global_rank = args.global_rank
        self.batch_size = batch_size
        self.total_size = len(dataset) - (len(dataset) % self.num_replicas)
        # self.num_samples = self.total_size // self.num_replicas
        self.num_samples = self.total_size
        # logging.info(f"rank: {self.rank}: Sampler created...")
        logging.info(f"sample_num: {len(dataset) * self.num_replicas}")
        # print(f"rank: {self.rank}: Sampler created...")

    def __iter__(self):
        # partition data into num_replicas and optionally shuffle within a rank
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed + self.global_rank)
        shuffling = torch.randperm(self.num_samples, generator=g).tolist()
        # indices = np.array(list(range((self.rank * self.num_samples), (self.rank + 1) * self.num_samples)))[shuffling].tolist()
        # indices = np.array(list(range(self.global_rank, self.num_samples*self.num_replicas, self.num_replicas)))[shuffling].tolist()
        indices = np.array(list(range(self.num_samples)))[shuffling].tolist()

        # make sure we have correct number of samples per replica
        assert len(indices) == self.num_samples
        
        assert self.batch_size > 0, "batch_size not set for the sampler"

        # resume the sampler
        start_index = self.start_iter * self.batch_size
        indices = indices[start_index:]
        return iter(indices)

    def set_start_iter(self, start_iter):
        """
        Set the iteration number from which the sampling should start. This is
        used to find the marker in the data permutation order from where the
        sampler should start sampling.
        """
        self.start_iter = start_iter

