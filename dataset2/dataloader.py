from argparse import Namespace

import cv2
import torch
import numpy as np
import yaml
from torch.utils import data
import random
from dataset2.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize
from dataset2.RGBXDataset import RGBXDataset

def random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, gt, modal_x

def random_scale(rgb, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, gt, modal_x, scale

def random_mosaic(image, thermal, mosaic_size, num_mosaics):

    height, width, _ = image.shape
    # 记录已选择的区域
    selected_regions = []
    n = 0
    for i in range(num_mosaics):
        # 随机选择mosaic块的位置，确保不重叠
        while True:
            top = np.random.randint(0, height - mosaic_size)
            left = np.random.randint(0, width - mosaic_size)
            region = (top, left, top + mosaic_size, left + mosaic_size)

            # 检查新选择的区域是否与已有的区域重叠
            overlap = any(
                [(
                        x1 < region[2] and x2 > region[0] and
                        y1 < region[3] and y2 > region[1]
                ) for x1, y1, x2, y2 in selected_regions]
            )

            if not overlap:
                selected_regions.append(region)
                break
            n += 1
            if n > 2000:
                break
        if n > 2000:
            break
        # 创建mosaic块
        if num_mosaics % 2 == 0:
            mosaic_block1 = thermal[top:top + mosaic_size, left:left + mosaic_size, :]
            thermal[top:top + mosaic_size, left:left + mosaic_size, :] = mosaic_block1.mean()
        else:
            mosaic_block2 = image[top:top + mosaic_size, left:left + mosaic_size, :]
            image[top:top + mosaic_size, left:left + mosaic_size, :] = mosaic_block2.mean()
    return image, thermal

class TrainPre(object):
    def __init__(self, norm_mean, norm_std, config):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.config = config

    def __call__(self, rgb, gt, modal_x):
        rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        if self.config.random_mosaic:
            rgb, modal_x = random_mosaic(rgb, modal_x, self.config.mosaic_size, self.config.num_mosaics)
        if self.config.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, self.config.train_scale_array)

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 0)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)

        return p_rgb, p_gt, p_modal_x


class ValPre(object):
    def __init__(self, norm_mean, norm_std, config):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.config = config
    def __call__(self, rgb, gt, modal_x):
        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        modal_x = normalize(modal_x, self.norm_mean, self.norm_std)
        rgb = rgb.transpose(2, 0, 1)
        modal_x = modal_x.transpose(2, 0, 1)
        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
        modal_x = torch.from_numpy(np.ascontiguousarray(modal_x)).float()
        return rgb, gt, modal_x

def get_dataloader(config):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'test_source': config.test_source}
    train_preprocess = TrainPre(config.norm_mean, config.norm_std, config)

    #train_dataset = RGBXDataset(data_setting, "train", train_preprocess)
    train_dataset = RGBXDataset(data_setting, "train", ValPre(config.norm_mean, config.norm_std, config))

    is_shuffle = False
    batch_size = config.batch_size


    train_loader = data.DataLoader(train_dataset,
                                   batch_size=config.batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True)

    val_dataset = RGBXDataset(data_setting, "val", ValPre(config.norm_mean, config.norm_std, config))

    val_loader = data.DataLoader(val_dataset,
                                 batch_size=1,
                                 num_workers=0,
                                 drop_last=False,
                                 shuffle=False,
                                 pin_memory=True)

    test_dataset = RGBXDataset(data_setting, "test", ValPre(config.norm_mean, config.norm_std, config))

    test_loader = data.DataLoader(test_dataset,
                                 batch_size=1,
                                 num_workers=0,
                                 drop_last=False,
                                 shuffle=False,
                                 pin_memory=True)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    with open("configs/train_MF_aug.yaml", 'r') as f:
        args = yaml.safe_load(f)
    train_config = Namespace(**args)
    dataloader = get_dataloader(train_config)
    train_loader, val_loader, val_dataset = dataloader
    for _, (image, label, index) in enumerate(train_loader):
        print(image.shape)
        print(label.shape)
        print(index)


