import os
import random
import copy

import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from tqdm import tqdm

from .utils import Subset, filter_images

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, image_set='train', is_aug=True, transform=None):

        self.root = root
        self.transform = transform

        self.image_set = image_set
        voc_root = self.root
        splits_dir = os.path.join(voc_root, 'splits', 'list')

        if not os.path.isdir(voc_root):
            raise RuntimeError(
                'Dataset not found or corrupted.' + ' You can use download=True to download it'
                f'at location = {voc_root}'
            )

        if is_aug and image_set == 'train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            split_f = os.path.join(splits_dir, 'train_aug.txt')
        else:
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')


        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        self.images = [
            (
                os.path.join(voc_root, "PascalVOC12",
                             x[0][1:]), os.path.join(voc_root, 'SegmentationClassAug', x[1][1:])
            ) for x in file_names
        ]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        image_path = self.images[index][0]
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return image_path, img, target

    def __len__(self):
        return len(self.images)


class VOCSegmentationIncremental(data.Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        labels=None,
        labels_old=None,
        idxs_path=None,
        overlap=True,
        data_masking="current",
        nshot=-1,
        ishot=0,
        opts=None
    ):

        full_voc = VOCSegmentation(root, 'train' if train else 'val', is_aug=True, transform=None)

        # cls_info.json
        temp = 'train' if train else 'val'
        cls_info_path = root + '/cls_info_' + f'{temp}.json'
        if not os.path.isfile(cls_info_path):
            self.label2ind = self.build_class_Label_Index(full_voc, cls_info_path)
        else:
            f = open(cls_info_path, 'r')
            a = f.read()
            self.label2ind = eval(a)
            f.close()



        self.nshot = nshot
        self.ishot = ishot

        self.labels = []
        self.labels_old = []

        if labels is not None:
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(
                l in labels_old for l in labels
            ), "labels and labels_old must be disjoint sets"

            self.labels = [0] + labels
            self.labels_old = [0] + labels_old

            self.order = [0] + labels_old + labels

            # take index of images with at least one class in labels and all classes in labels+labels_old+[0,255]
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = filter_images(full_voc, labels, labels_old, overlap=overlap)
                if idxs_path is not None:
                    np.save(idxs_path, np.array(idxs, dtype=int))

            test_on_val = False
            if test_on_val:
                rnd = np.random.RandomState(1)
                rnd.shuffle(idxs)
                train_len = int(0.8 * len(idxs))
                if train:
                    idxs = idxs[:train_len]
                else:
                    idxs = idxs[train_len:]

            #if train:
            #    masking_value = 0
            #else:
            #    masking_value = 255

            #self.inverted_order = {label: self.order.index(label) for label in self.order}
            #self.inverted_order[255] = masking_value

            masking_value = 0  # Future classes will be considered as background.
            self.inverted_order = {label: self.order.index(label) for label in self.order}
            self.inverted_order[255] = 255

            reorder_transform = tv.transforms.Lambda(
                lambda t: t.apply_(
                    lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value
                )
            )

            if data_masking == "current":
                tmp_labels = self.labels + [255]
            elif data_masking == "current+old":
                tmp_labels = labels_old + self.labels + [255]
            elif data_masking == "all":
                raise NotImplementedError(
                    f"data_masking={data_masking} not yet implemented sorry not sorry."
                )
            elif data_masking == "new":
                tmp_labels = self.labels
                masking_value = 255

            target_transform = tv.transforms.Lambda(
                lambda t: t.
                apply_(lambda x: self.inverted_order[x] if x in tmp_labels else masking_value)
            )

            # Few shot setting here
            if nshot != -1:
                print(f'get {nshot} shots each class')
                idxs_final = []

                for cls in self.labels:
                    if cls == 0:
                        continue
                    count = 0
                    while count < nshot:
                        idx_cls = random.choice(self.label2ind[cls])
                        if idx_cls in idxs:
                            count += 1
                            idxs_final.append(idx_cls)
                idxs = idxs_final

            self.idxs = idxs
            # make the subset of the dataset
            self.dataset = Subset(full_voc, idxs, transform, target_transform)

        else:
            self.dataset = full_voc

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]


    def __len__(self):
        return len(self.dataset)

    def build_class_Label_Index(self, dataset, save_path):
        label2inds = {}
        for idx, data in enumerate(tqdm(dataset)):
            # data -> label
            _, target = data
            classes = list(np.unique(np.array(target)))
            if 0 in classes:
                classes.remove(0)
            if 255 in classes:
                classes.remove(255)

            for label in classes:
                # save label sorted by class
                if label not in label2inds:
                    label2inds[label] = []
                label2inds[label].append(idx)

        f = open(save_path, 'w')
        f.write(str(label2inds))
        f.close()

        return label2inds
    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)
