import numpy as np
import torch
import random
import torchnet as tnt
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

def group_images(dataset, labels):
    # Group images based on the label in LABELS (using labels not reordered)
    idxs = {lab: [] for lab in labels}

    labels_cum = labels + [0, 255]
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if all(x in labels_cum for x in cls):
            for x in cls:
                if x in labels:
                    idxs[x].append(i)
    return idxs


def filter_images(dataset, labels, labels_old=None, overlap=True):
    # Filter images without any label in LABELS (using labels not reordered)
    idxs = []

    if 0 in labels:
        labels.remove(0)

    if labels_old is None:
        labels_old = []
    labels_cum = labels + labels_old + [0, 255]

    if overlap:
        # overlap指的是可包括未来类，只要存在当前类的即可
        fil = lambda c: any(x in labels for x in cls)
    else:
        fil = lambda c: any(x in labels for x in cls) and all(x in labels_cum for x in c)
        # 存在类是当前训练的新类，且所有类都得是学过的类或者是0和255

    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if fil(cls):
            idxs.append(i)
    return idxs


class Subset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (callable): way to transform the images and the targets
        target_transform(callable): way to transform the target labels
    """

    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, idx):
        image_path, sample, target = self.dataset[self.indices[idx]]

        if self.transform is not None:
            sample, target = self.transform(sample, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        while (set(np.unique(np.array(target))) == {0}) \
                or (set(np.unique(np.array(target))) == {0, 255}) \
                or (set(np.unique(np.array(target))) == {255}):
            # 加了个全零或全0和255的target删除条件，跳过后随机再选一张
            random_idx = random.choice(self.indices)
            image_path, sample, target = self.dataset[random_idx]
            if self.transform is not None:
                sample, target = self.transform(sample, target)

            if self.target_transform is not None:
                target = self.target_transform(target)

        cls = list(np.unique(np.array(target)))
        if 0 in cls:
            cls.remove(0)
        if 255 in cls:
            cls.remove(255)


        return (image_path, sample, target, cls)


    def __len__(self):
        return len(self.indices)



class MaskLabels:
    """
    Use this class to mask labels that you don't want in your dataset.
    Arguments:
    labels_to_keep (list): The list of labels to keep in the target images
    mask_value (int): The value to replace ignored values (def: 0)
    """

    def __init__(self, labels_to_keep, mask_value=0):
        self.labels = labels_to_keep
        self.value = torch.tensor(mask_value, dtype=torch.uint8)

    def __call__(self, sample):
        # sample must be a tensor
        assert isinstance(sample, torch.Tensor), "Sample must be a tensor"

        sample.apply_(lambda t: t.apply_(lambda x: x if x in self.labels else self.value))

        return sample



class FewShotDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 nKnovel=5,  # number of novel categories.
                 nKbase=15,  # number of base categories.
                 nShotNovel=5,  # number of test examples for all the novel categories.
                 epoch_size=2000,  # episode size
                 cls_info_path=None,
                 meta_test=False
                 ):

        self.dataset = dataset
        self.nKnovel = nKnovel
        self.nKbase = nKbase

        self.nShotNovel = nShotNovel
        self.epoch_size = epoch_size
        self.meta_test = meta_test

        # cls_info.json
        if os.path.isfile(cls_info_path):
            f = open(cls_info_path, 'r')
            a = f.read()
            self.label2ind = eval(a)
            f.close()
        else:
            raise FileNotFoundError

    def sample_episode(
            self, Knovel, nShotNovel):
        Tnovel = []
        for Knovel_idx in range(len(Knovel)):
            count = 0
            while count < nShotNovel:
                imds_tnovel = random.choice(self.label2ind[Knovel[Knovel_idx]])
                true_idx = self.dataset.idxs.index(imds_tnovel)
                if imds_tnovel in self.dataset.idxs:
                    count += 1
                    Tnovel += [(true_idx, Knovel[Knovel_idx])]

        return Tnovel

    def createExamplesTensorData(self, examples):
        nNovel = len(examples) // self.nShotNovel
        images_list = []
        labels_list = []
        for cls in range(nNovel):
            images_ = torch.stack(
                [self.dataset[examples[cls*self.nShotNovel+n][0]][0] for n in range(self.nShotNovel)], dim=0)
            labels_ = torch.stack(
                [self.dataset[examples[cls*self.nShotNovel+n][0]][1] for n in range(self.nShotNovel)], dim=0)
            images_list.append(images_)
            labels_list.append(labels_)
        images = torch.stack(images_list, dim=0)
        labels = torch.stack(labels_list, dim=0)
        return images, labels



    def __getitem__(self, idx):
        rand_seed = idx
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        if not self.meta_test:
            labels_base = random.sample(self.__strip_zero(self.dataset.labels), self.nKbase)
            labels_novel = [label for label in self.__strip_zero(self.dataset.labels) if label not in labels_base]
            random.shuffle(labels_base)
            random.shuffle(labels_novel)
        else:
            labels_base = self.__strip_zero(self.dataset.labels_old)
            labels_novel = self.__strip_zero(self.dataset.labels)
            random.shuffle(labels_base)
            random.shuffle(labels_novel)

        Tnovel = self.sample_episode(
            labels_novel, self.nShotNovel)
        Xt, Yt = self.createExamplesTensorData(Tnovel)
        return Xt, Yt, labels_base, labels_novel

    def __len__(self):
        return self.epoch_size

    def __strip_zero(self, labels):
        while 0 in labels:
            labels.remove(0)
        return labels