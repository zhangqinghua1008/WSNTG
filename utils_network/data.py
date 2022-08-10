"""
Data loading utilities. 数据加载工具
"""

import csv
from functools import partial
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from skimage.io import imread
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries
from skimage.transform import resize
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

empty_tensor = torch.tensor(0)

resize_mask = partial(
    resize, order=0, preserve_range=True, anti_aliasing=False)


def resize_img(img, target_size):
    img = resize(img, target_size, order=1, anti_aliasing=False)
    return (img * 255).astype('uint8')


class SegmentationDataset(Dataset):
    """Dataset for segmentation task.  用于分割任务的数据集！！！

    This dataset returns following data when indexing: 此数据集在索引时返回以下数据:
        - img: tensor of size (3, H, W) with type float32
        - mask: tensor of size (C, H, W) with type long or an empty tensor
        - cont (optional): tensor of size (C, H, W) with type long, only when `contour` is `True`
    """

    def __init__(self, root_dir, mode=None, contour=False, target_size=None, rescale_factor=None,
                 multiscale_range=None, train=True, proportion=1, n_classes=2, seed=0):
        """Initialize a new SegmentationDataset.  初始化一个新的SegmentationDataset
    
        Args:
            root_dir: path to dataset root  数据集根的路径
            mode: one of `mask`, `area` or `point`
            contour: whether to include contours 是否包括轮廓线
            target_size: desired output spatial size
            rescale_factor: multiplier for spatial size
            multiscale_range: a tuple containing the limits of random rescaling  包含随机缩放限制的元组(0.3,0.4)
            train: whether in training mode
            proportion: proportion of data to be used (between 0 and 1)   使用的数据比例(0到1之间)
            n_classes: number of target classes
            seed: random seed
        """

        self.root_dir = Path(root_dir).expanduser()  # root_dir: G:\py_code\pycharm_Code\WESUP-comparison-models\data_glas\train

        # path to original images
        self.img_paths = sorted((self.root_dir / 'images').iterdir()) # 一个list,存放imgpath

        # path to mask annotations (optional)
        self.mask_paths = None
        if (self.root_dir / 'masks').exists():
            self.mask_paths = sorted((self.root_dir / 'masks').iterdir())

        # 如果mask不存在mode为None，存在的话为mode or 'mask' (就是mode存在为mode，不存在为‘mask’,因为会返回第一个逻辑判断为True字符串)
        self.mode = mode or 'mask' if self.mask_paths is not None else None

        if self.mode != 'mask' and contour: # 需要轮廓 但 没有mask
            raise ValueError('mask is required for providing contours/ 需要mask来提供轮廓')

        self.contour = contour
        self.target_size = target_size
        self.rescale_factor = rescale_factor

        self.train = train
        self.proportion = proportion
        self.n_classes = n_classes
        self.multiscale_range = multiscale_range

        # indexes to pick image/mask from  从中选择图像/掩码的索引
        self.picked = np.arange(len(self.img_paths))  # 【0，img的个数），都是整数，后续被挑选的数据
        if self.proportion < 1:
            np.random.seed(seed)
            np.random.shuffle(self.picked)
            self.picked = self.picked[:len(self)]
            self.picked.sort()

    def __len__(self):
        return int(self.proportion * len(self.img_paths))

    def _resize_image_and_mask(self, img, mask=None):
        height, width = img.shape[:2]
        if self.target_size is not None:
            target_height, target_width = self.target_size
        elif self.multiscale_range is not None:   # 一般执行这个条件
            self.rescale_factor = np.random.uniform(*self.multiscale_range)  # rescale_factor:缩放随机数。
            target_height = int(np.ceil(self.rescale_factor * height))
            target_width = int(np.ceil(self.rescale_factor * width))
        elif self.rescale_factor is not None:
            target_height = int(np.ceil(self.rescale_factor * height))
            target_width = int(np.ceil(self.rescale_factor * width))
        else:
            target_height, target_width = height, width

        img = resize_img(img, (target_height, target_width))

        # pixel-level annotation mask 进行像素级注释
        if mask is not None:
            mask = resize_mask(mask, (target_height, target_width))
            # mask = resize_mask(mask, (270, 270))

        return img, mask

    # augment： 增强
    def _augment(self, *data):
        img, mask = data

        transformer = A.Compose([
            A.RandomRotate90(),
            A.transforms.Flip(),
        ])

        # print("img.shape", img.shape,  "mask: ", mask.shape)
        augmented = transformer(image=img, mask=mask)

        return augmented['image'], augmented.get('mask', None)

    # 转变image and mask为tensor
    def _convert_image_and_mask_to_tensor(self, img, mask):
        img = TF.to_tensor(img)
        if mask is not None:
            if self.contour:
                cont = dilation(find_boundaries(mask))
            mask = np.concatenate([np.expand_dims(mask == i, 0)
                                   for i in range(self.n_classes)])
            mask = torch.as_tensor(mask.astype('int64'), dtype=torch.long)
        else:
            mask = empty_tensor

        if self.contour:
            cont = np.concatenate([np.expand_dims(cont == i, 0)
                                   for i in range(self.n_classes)])
            cont = torch.as_tensor(cont.astype('int64'), dtype=torch.long)
            return img, mask, cont

        return img, mask

    def __getitem__(self, idx):
        '''    1. 按照 index，读取文件中对应的数据  （读取一个数据！！！！我们常读取的数据是图片，一般我们送入模型的数据成批的，但在这里只是读取一张图片，成批后面会说到）
               2. 对读取到的数据进行数据增强
               3. 返回数据对 （一般我们要返回 图片，对应的标签）'''
        idx = self.picked[idx]
        img = imread(str(self.img_paths[idx]))[:,:,:3]  # 防止png四通道的情况
        mask = None
        if self.mask_paths is not None:
            mask = imread(str(self.mask_paths[idx]))
            # mask[mask <= 127] = 0  # 将掩码的像素从[0,255]转换为[0,1]。
            # mask[mask > 127] = 1

        # 如果大小不一样就resize
        if img.shape[:2] != self.target_size or mask.shape[:2] != self.target_size:
            img, mask = self._resize_image_and_mask(img, mask)

        if self.train:
            img, mask = self._augment(img, mask)  # 进行数据增强

        return self._convert_image_and_mask_to_tensor(img, mask)

    def summary(self, logger=None):
        """Print summary information.  打印摘要信息."""

        lines = [
            f"Segmentation dataset ({'training' if self.train else 'inference'}) ",
            f"initialized with {len(self)} images from {self.root_dir}.",
        ]

        if self.mode is not None:
            lines.append(f"Supervision mode: {self.mode}")
        else:
            lines.append("No supervision provided.")

        lines = '\n'.join(lines)

        # 记录信息到logger
        if logger is not None:
            logger.info(lines)
        else:
            print(lines)


class AreaConstraintDataset(SegmentationDataset):
    """Segmentation dataset with area information.

    This dataset returns following data when indexing:
        - img: tensor of size (3, H, W) with type float32
        - mask: tensor of size (C, H, W) with type long or an empty tensor
        - area: a 2-element (lower and upper bound) vector tensor with type float32
    """

    def __init__(self, root_dir, target_size=None, rescale_factor=None, area_type='decimal',
                 constraint='equality', margin=0.1, train=True, proportion=1.0):
        """Construct a new AreaConstraintDataset instance.

        Args:
            root_dir: path to dataset root
            target_size: desired output spatial size
            rescale_factor: multiplier for spatial size
            area_type: either 'decimal' (relative size) or 'integer' (total number of positive pixels)
            constraint: either 'equality' (equality area constraint), 'common' (inequality
                common bound constraint) or 'individual' (inequality individual bound constraint)
            margin: soft margin of inequality constraint, only relevant when `constraint` is
                set to 'individual'
            train: whether in training mode
            proportion: proportion of data to be used (between 0 and 1)

        Returns:
            dataset: a new AreaConstraintDataset instance
        """
        super().__init__(root_dir, mode='area', target_size=target_size,
                         rescale_factor=rescale_factor, train=train, proportion=proportion)

        # area information (# foreground pixels divided by total pixels, between 0 and 1)
        self.area_info = pd.read_csv(
            self.root_dir / 'area.csv', usecols=['img', 'area'])

        self.area_type = area_type
        self.constraint = constraint
        self.margin = margin

    def _augment(self, *data):
        img, mask = data

        transformer = A.Compose([
            A.HueSaturationValue(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                       contrast_limit=0.3, p=1),
            A.CLAHE(p=0.5),
            A.ElasticTransform(p=0.5),
            A.Blur(blur_limit=3, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])
        augmented = transformer(image=img, mask=mask)

        return augmented['image'], augmented.get('mask', None)

    def __getitem__(self, idx):
        idx = self.picked[idx]
        img = imread(str(self.img_paths[idx]))[:, :, :3]

        mask = None
        if self.mask_paths is not None:
            mask = imread(str(self.mask_paths[idx]))
            mask[mask <= 127] = 0  # 将掩码的像素从[0,255]转换为[0,1]。
            mask[mask > 127] = 1
        img, mask = self._resize_image_and_mask(img, mask)

        if self.train:
            img, mask = self._augment(img, mask)

        img, mask = self._convert_image_and_mask_to_tensor(img, mask)

        if self.area_type == 'decimal':
            area = self.area_info.loc[idx]['area']
        else:  # integer
            area = mask[1].sum().float()

        if self.constraint == 'equality':
            area = torch.tensor([area, area])
        elif self.constraint == 'individual':
            area = torch.tensor(
                [area * (1 - self.margin), area * (1 + self.margin)]).long()
        else:  # common
            lower = self.area_info.area.min()
            upper = self.area_info.area.max()
            if self.area_type == 'integer':
                lower = int(lower * np.prod(self.target_size))
                upper = int(upper * np.prod(self.target_size))
            area = torch.tensor([lower, upper])

        return img, mask, area


class PointSupervisionDataset(SegmentationDataset):
    """One-shot segmentation dataset.

    This dataset returns following data when indexing:
        - img: tensor of size (3, H, W) with type float32
        - pixel_mask: pixel-level annotation of size (C, H, W) with type long or an empty tensor
        - point_mask: point-level annotation of size (C, H, W) with type long or an empty tensor
    """

    def __init__(self, root_dir, target_size=None, rescale_factor=None,
                 multiscale_range=None, radius=0, train=True, proportion=1):
        super().__init__(root_dir, mode='point', target_size=target_size,
                         rescale_factor=rescale_factor, train=train,
                         proportion=proportion, multiscale_range=multiscale_range)

        # path to point supervision directory  到点监督目录的路径
        self.point_root = self.root_dir / 'points'

        # path to point annotation files  Path.glob() : 列出匹配的文件或目录
        self.point_paths = sorted(self.point_root.glob('*.csv'))

        self.radius = radius

    def _augment(self, *data):
        img, mask, points = data

        # transforms applied to images and masks  应用于图像和mask的变换
        appearance_transformer = A.Compose([
            A.HueSaturationValue(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                       contrast_limit=0.3, p=1),
            A.CLAHE(p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ])

        # transforms applied to images, masks and points 转换应用于图像，masks和点
        position_transformer = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=1),
        ], keypoint_params={'format': 'xy'})

        augmented = appearance_transformer(image=img, mask=mask)
        temp_img, temp_mask = augmented['image'], augmented['mask']

        augmented = position_transformer(
            image=temp_img, mask=temp_mask, keypoints=points)

        return augmented['image'], augmented.get('mask', None), augmented.get('keypoints', None)

    def __getitem__(self, idx):
        # matplotlib.use('TkAgg')
        idx = self.picked[idx]
        # img = imread(str(self.img_paths[idx]))
        img = imread(str(self.img_paths[idx]))[:, :, :3]  # 防止png四通道的情况

        # pixel_mask : 像素级mask(分割掩膜)
        pixel_mask = None
        if self.mask_paths is not None:
            pixel_mask = imread(str(self.mask_paths[idx]))
            pixel_mask[pixel_mask <= 127] = 0  # 将掩码的像素从[0,255]转换为[0,1]。
            pixel_mask[pixel_mask > 127] = 1

        orig_height, orig_width = img.shape[:2]
        img, pixel_mask = self._resize_image_and_mask(img, pixel_mask)

        # how much we would like to rescale coordinates of each point
        # (the last dimension is target class, which should be kept the same)
        # 我们需要对每个点的坐标进行多少缩放   (最后一个维度是目标类，应该保持不变)
        if self.rescale_factor is None:
            rescaler = np.array([
                [self.target_size[1] / orig_width,
                    self.target_size[0] / orig_height, 1]
            ])
        else:  # 缩放因子存在走这个条件(一般走这个条件)
            rescaler = np.array(
                [[self.rescale_factor, self.rescale_factor, 1]])

        # read points from csv file  从CSV文件读取点,并相应的缩放
        with open(str(self.point_paths[idx])) as fp:
            points = np.array([[int(d) for d in point]
                               for point in csv.reader(fp)])
            points = np.floor(points * rescaler).astype('int')

        if self.train:  #训练数据进行数据增强
            img, pixel_mask, points = self._augment(img, pixel_mask, points)

        point_mask = np.zeros((self.n_classes, *img.shape[:2]), dtype='uint8') # shape: (class,缩放后宽,缩放后高)
        for x, y, class_ in points:
            cv2.circle(point_mask[class_], (x, y), self.radius, 1, -1)  # 画圆，根据点坐标生成对应的mask,将点位置半径内的值改为1。 半径为0时候，当前点为1

        if point_mask is not None:
            point_mask = torch.as_tensor( point_mask.astype('int64'), dtype=torch.long)
        else:
            point_mask = empty_tensor

        img, pixel_mask = self._convert_image_and_mask_to_tensor(img, pixel_mask)

        return img, pixel_mask, point_mask


class WESUPV2Dataset(SegmentationDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.root_dir / 'spl-masks').exists():
            self.mask_paths = sorted((self.root_dir / 'spl-masks').iterdir())

    def _generate_coords(self, shape):
        x = np.linspace(0, 1, shape[0])
        y = np.linspace(0, 1, shape[1])
        coords = torch.as_tensor([np.tile(x, len(y)), np.repeat(y, len(x))],
                                 dtype=torch.float32)

        return coords.view(2, shape[0], shape[1])

    def __getitem__(self, idx):
        idx = self.picked[idx]
        img = imread(str(self.img_paths[idx]))
        mask = None
        if self.mask_paths is not None:
            mask = np.load(self.mask_paths[idx])
        img, mask = self._resize_image_and_mask(img, mask)

        if self.train:
            img, mask = self._augment(img, mask)

        coords = self._generate_coords(img.shape)
        img = TF.to_tensor(img)
        mask = torch.as_tensor(mask.transpose(2, 0, 1), dtype=torch.long)

        return img, mask, coords


class Digest2019PointDataset(SegmentationDataset):
    """One-shot segmentation dataset.

    This dataset returns following data when indexing:
        - img: tensor of size (3, H, W) with type float32
        - pixel_mask: pixel-level annotation of size (C, H, W) with type long or an empty tensor
        - point_mask: point-level annotation of size (C, H, W) with type long or an empty tensor
    """

    def __init__(self, root_dir, target_size=None, rescale_factor=None,
                 multiscale_range=None, radius=0, train=True, proportion=1):
        super().__init__(root_dir, mode='point', target_size=target_size,
                         rescale_factor=rescale_factor, train=train,
                         proportion=proportion, multiscale_range=multiscale_range)

        # path to point supervision directory  路径到点监督目录
        self.point_root = self.root_dir / 'points'

        # path to point annotation files  指向注释文件的路径
        self.point_paths = sorted(self.point_root.glob('*.csv'))

        self.radius = radius

    def _augment(self, *data):
        img, mask, points = data

        # transforms applied to images and masks  应用于图像和masks的变换
        appearance_transformer = A.Compose([
            A.HueSaturationValue(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                       contrast_limit=0.3, p=1),
            A.CLAHE(p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ])

        # transforms applied to images, masks and points
        position_transformer = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=1),
        ], keypoint_params={'format': 'xy'})

        augmented = appearance_transformer(image=img, mask=mask)
        temp_img, temp_mask = augmented['image'], augmented['mask']

        augmented = position_transformer(
            image=temp_img, mask=temp_mask, keypoints=points)

        return augmented['image'], augmented.get('mask', None), augmented.get('keypoints', None)

    def __getitem__(self, idx):
        idx = self.picked[idx]
        img_path = self.img_paths[idx]
        img = imread(str(img_path))
        is_negative = img_path.name.startswith('negative')

        pixel_mask = None
        if self.mask_paths is not None:
            if not is_negative:
                pixel_mask = imread(str(self.mask_paths[idx]))
            else:
                pixel_mask = np.full_like(img.shape,0)

        orig_height, orig_width = img.shape[:2]
        img, pixel_mask = self._resize_image_and_mask(img, pixel_mask)

        # how much we would like to rescale coordinates of each point 我们希望将每个点的坐标重新缩放多少
        # (the last dimension is target class, which should be kept the same) (最后一个维度是目标类，应该保持不变)
        if self.rescale_factor is None:
            rescaler = np.array([
                [self.target_size[1] / orig_width,
                    self.target_size[0] / orig_height, 1]
            ])
        else:
            rescaler = np.array(
                [[self.rescale_factor, self.rescale_factor, 1]])

        if is_negative:
            points = np.array([[0, 0, 0]])
        else:
            # read points from csv file
            with open(str(self.point_paths[idx])) as fp:
                points = np.array([[int(d) for d in point]
                                   for point in csv.reader(fp)])
                points = np.floor(points * rescaler).astype('int')

        if self.train:
            img, pixel_mask, points = self._augment(img, pixel_mask, points)

        img, pixel_mask = self._convert_image_and_mask_to_tensor(
            img, pixel_mask)

        if img_path.name.startswith('negative'):
            point_mask = pixel_mask
        else:
            point_mask = np.zeros(
                (self.n_classes, *img.shape[-2:]), dtype='uint8')
            for x, y, class_ in points:
                if class_>1:
                    class_ = 1
                cv2.circle(point_mask[class_], (x, y), self.radius, 1, -1)

            if point_mask is not None:
                point_mask = torch.as_tensor(
                    point_mask.astype('int64'), dtype=torch.long)
            else:
                point_mask = empty_tensor

        return img, pixel_mask, point_mask


class CompoundDataset(Dataset):

    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return tuple(dataset[idx] for dataset in self.datasets)

    def summary(self, logger=None):
        for dataset in self.datasets:
            dataset.summary(logger=logger)


class PointSupervisionDataset_Edit(SegmentationDataset):
    """One-shot segmentation dataset.

    This dataset returns following data when indexing:
        - img: tensor of size (3, H, W) with type float32
        - pixel_mask: pixel-level annotation of size (C, H, W) with type long or an empty tensor
        - point_mask: point-level annotation of size (C, H, W) with type long or an empty tensor
    """

    def __init__(self, root_dir, target_size=None, rescale_factor=None,
                 multiscale_range=None, radius=0, train=True, proportion=1):
        super().__init__(root_dir, mode='point', target_size=target_size,
                         rescale_factor=rescale_factor, train=train,
                         proportion=proportion, multiscale_range=multiscale_range)

        # path to point supervision directory  到点监督目录的路径
        self.point_root = self.root_dir / 'points'

        # path to point annotation files  Path.glob() : 列出匹配的文件或目录
        self.point_paths = sorted(self.point_root.glob('*.csv'))

        self.radius = radius

    def _augment(self, *data):
        img, mask, points = data

        # transforms applied to images and masks  应用于图像和mask的变换
        appearance_transformer = A.Compose([
            A.HueSaturationValue(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                       contrast_limit=0.3, p=1),
            A.CLAHE(p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ])

        # transforms applied to images, masks and points 转换应用于图像，masks和点
        position_transformer = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=1),
        ], keypoint_params={'format': 'xy'})

        augmented = appearance_transformer(image=img, mask=mask)
        temp_img, temp_mask = augmented['image'], augmented['mask']

        augmented = position_transformer(
            image=temp_img, mask=temp_mask, keypoints=points)

        return augmented['image'], augmented.get('mask', None), augmented.get('keypoints', None)

    def __getitem__(self, idx):
        idx = self.picked[idx]
        img = imread(str(self.img_paths[idx]))

        # 像素级mask
        pixel_mask = None
        if self.mask_paths is not None:
            pixel_mask = imread(str(self.mask_paths[idx]))

        orig_height, orig_width = img.shape[:2]
        img, pixel_mask = self._resize_image_and_mask(img, pixel_mask)

        # how much we would like to rescale coordinates of each point
        # (the last dimension is target class, which should be kept the same)
        # 我们需要对每个点的坐标进行多少缩放   (最后一个维度是目标类，应该保持不变)
        if self.rescale_factor is None:
            rescaler = np.array([
                [self.target_size[1] / orig_width,
                    self.target_size[0] / orig_height, 1]
            ])
        else:
            rescaler = np.array(
                [[self.rescale_factor, self.rescale_factor, 1]])

        # read points from csv file  从CSV文件读取点
        with open(str(self.point_paths[idx])) as fp:
            points = np.array([[int(d) for d in point]
                               for point in csv.reader(fp)])
            points = np.floor(points * rescaler).astype('int')

        if self.train:  #训练数据进行数据增强
            img, pixel_mask, points = self._augment(img, pixel_mask, points)

        point_mask = np.zeros((self.n_classes, *img.shape[:2]), dtype='uint8')
        for x, y, class_ in points:
            cv2.circle(point_mask[class_], (x, y), self.radius, 1, -1)

        if point_mask is not None:
            point_mask = torch.as_tensor( point_mask.astype('int64'), dtype=torch.long)
        else:
            point_mask = empty_tensor

        img, pixel_mask = self._convert_image_and_mask_to_tensor(img, pixel_mask)

        return img, pixel_mask, point_mask
