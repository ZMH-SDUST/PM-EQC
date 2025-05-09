"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""
import copy
import logging
import os
import time
import torch
import pickle
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from openpose import util
from openpose.body import Body
import torchvision.ops.boxes as box_ops

try:
    import wandb
except ImportError:
    pass

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet

import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

from ops import recover_boxes
from detr.datasets import transforms as T


hico_unseen_index = {
    "default": [],
    # start from 0
    'uc0': [
        0, 1, 10, 29, 30, 41, 48, 50, 56, 57, 69, 72, 80, 81, 92, 93, 96, 109,
        110, 114, 127, 134, 139, 161, 170, 177, 183, 189, 191, 197, 198, 201,
        208, 209, 221, 227, 229, 232, 233, 235, 239, 242, 245, 247, 250, 252,
        260, 263, 270, 271, 280, 286, 288, 290, 299, 301, 308, 316, 325, 334,
        336, 343, 344, 352, 355, 356, 357, 363, 375, 376, 380, 384, 387, 389,
        395, 396, 397, 404, 408, 413, 414, 417, 422, 425, 433, 434, 436, 444,
        448, 452, 454, 455, 463, 480, 484, 488, 498, 503, 505, 507, 513, 516,
        527, 530, 532, 536, 537, 540, 546, 547, 550, 555, 561, 562, 566, 567,
        572, 581, 587, 598
    ],
    'uc1': [
        0, 3, 22, 29, 32, 52, 58, 63, 72, 73, 78, 89, 91, 92, 105, 106, 107,
        113, 137, 148, 163, 165, 172, 178, 179, 194, 196, 207, 209, 210, 214,
        215, 229, 231, 233, 234, 236, 240, 241, 243, 245, 247, 252, 254, 260,
        262, 269, 272, 282, 286, 289, 292, 296, 302, 310, 315, 322, 326, 333,
        335, 338, 340, 343, 347, 350, 351, 353, 354, 358, 362, 367, 368, 376,
        380, 388, 389, 393, 395, 397, 399, 410, 412, 416, 417, 419, 420, 429,
        434, 439, 441, 445, 449, 454, 467, 476, 483, 495, 503, 507, 511, 519,
        528, 529, 535, 537, 539, 547, 548, 556, 557, 561, 563, 565, 569, 579,
        587, 589, 591, 595, 597
    ],
    'uc2': [
        9, 25, 30, 49, 51, 61, 71, 74, 77, 82, 94, 108, 110, 116, 126, 131,
        143, 164, 168, 177, 185, 200, 201, 208, 212, 229, 232, 234, 239, 241,
        243, 244, 248, 255, 256, 258, 259, 266, 272, 279, 281, 287, 288, 290,
        294, 295, 301, 305, 308, 319, 322, 325, 328, 330, 332, 337, 344, 347,
        349, 350, 356, 359, 366, 367, 370, 375, 378, 380, 386, 387, 390, 391,
        400, 406, 409, 411, 416, 419, 428, 429, 431, 436, 439, 443, 445, 447,
        449, 451, 454, 457, 466, 468, 477, 479, 485, 486, 491, 497, 504, 508,
        510, 516, 527, 529, 531, 533, 536, 544, 545, 546, 549, 550, 552, 558,
        561, 568, 589, 594, 596, 599
    ],
    'uc3': [
        4, 14, 26, 27, 41, 45, 51, 53, 62, 69, 74, 80, 88, 91, 92, 93, 100,
        107, 110, 125, 127, 130, 136, 152, 153, 163, 167, 170, 177, 183, 186,
        188, 196, 200, 207, 210, 217, 220, 225, 232, 237, 242, 243, 246, 248,
        252, 253, 263, 267, 270, 280, 285, 289, 291, 292, 302, 312, 316, 325,
        335, 341, 343, 348, 355, 356, 362, 363, 368, 378, 382, 384, 385, 390,
        394, 396, 404, 406, 407, 415, 416, 426, 428, 429, 431, 435, 441, 443,
        448, 450, 452, 454, 460, 467, 469, 479, 480, 483, 498, 503, 505, 509,
        518, 524, 532, 533, 541, 549, 551, 560, 561, 566, 572, 573, 579, 580,
        585, 587, 594, 595, 599
    ],
    'uc4': [
        0, 4, 28, 29, 42, 43, 49, 53, 55, 56, 66, 72, 80, 81, 87, 90, 92, 94,
        100, 103, 109, 110, 129, 137, 149, 159, 166, 167, 170, 171, 179, 182,
        189, 193, 194, 195, 201, 206, 236, 237, 244, 245, 248, 249, 254, 255,
        257, 258, 266, 270, 290, 292, 300, 303, 316, 317, 326, 327, 331, 333,
        339, 340, 345, 347, 349, 350, 352, 353, 357, 362, 365, 366, 375, 380,
        381, 383, 385, 395, 396, 425, 426, 446, 448, 450, 451, 458, 466, 470,
        474, 476, 485, 487, 494, 495, 504, 505, 509, 515, 516, 525, 528, 529,
        536, 537, 539, 541, 546, 548, 556, 557, 568, 572, 578, 582, 585, 586,
        590, 593, 595, 597
    ],
    "rare_first": [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418,
                   70, 416,
                   389, 90, 395, 76, 397, 84, 135, 262, 401, 592, 560, 586, 548, 593, 526, 181, 257, 539, 535, 260, 596,
                   345, 189,
                   205, 206, 429, 179, 350, 405, 522, 449, 261, 255, 546, 547, 44, 22, 334, 599, 239, 315, 317, 229,
                   158, 195,
                   238, 364, 222, 281, 149, 399, 83, 127, 254, 398, 403, 555, 552, 520, 531, 440, 436, 482, 274, 8, 188,
                   216, 597,
                   77, 407, 556, 469, 474, 107, 390, 410, 27, 381, 463, 99, 184, 100, 292, 517, 80, 333, 62, 354, 104,
                   55, 50,
                   198, 168, 391, 192, 595, 136, 581],  # 120
    "non_rare_first": [38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75,
                       212, 472, 61,
                       457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479,
                       230, 385, 73,
                       159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338,
                       29, 594, 346,
                       456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191,
                       266, 304, 6, 572,
                       529, 312, 9, 308, 417, 197, 193, 163, 455, 25, 54, 575, 446, 387, 483, 534, 340, 508, 110, 329,
                       246, 173, 506,
                       383, 93, 516, 64],  # 120
    "unseen_object": [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                      126, 127, 128, 224, 225, 226, 227, 228, 229, 230, 231, 290, 291, 292, 293,
                      294, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 336, 337,
                      338, 339, 340, 341, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
                      429, 430, 431, 432, 433, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
                      463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 533, 534, 535, 536,
                      537, 558, 559, 560, 561, 595, 596, 597, 598, 599],  # 100
    "unseen_verb": [4, 6, 12, 15, 18, 25, 34, 38, 40, 49, 58, 60, 68, 69, 72, 73, 77, 82, 96, 97,
                    104, 113, 116, 118, 122, 129, 139, 147, 150, 153, 165, 166, 172, 175, 176, 181, 190, 202, 210, 212,
                    219, 227, 228, 233, 235, 243, 298, 313, 315, 320, 326, 336, 342, 345, 354, 372, 401, 404, 409, 431,
                    436, 459, 466, 470, 472, 479, 481, 488, 491, 494, 498, 504, 519, 523, 535, 536, 541, 544, 562, 565,
                    569, 572, 591, 595]}

def custom_collate(batch):
    images = []
    targets = []
    poses = []
    for im, pose, tar in batch:
        images.append(im)
        targets.append(tar)
        poses.append(pose)
    return images, poses, targets


class DataFactory(Dataset):
    def __init__(self, name, partition, data_root, zs=False, zs_type=""):
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)
        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            if partition == "train2015":
                pose_file = os.path.join(data_root, "HICODET_train_pose.json")
            else:
                pose_file = os.path.join(data_root, "HICODET_test_pose.json")
            self.dataset = HICODet(
                root=os.path.join(data_root, "hico_20160224_det/images", partition),
                anno_file=os.path.join(data_root, f"instances_{partition}.json"),
                pose_file=pose_file,
                target_transform=pocket.ops.ToTensor(input_format='dict'),
                pose_transform=pocket.ops.ToTensor(input_format='dict')
            )
            self.index_to_file = self.dataset.index_to_file()
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            if partition == "trainval":
                pose_file = os.path.join(data_root, "VCOCO_train_pose.json")
            else:
                pose_file = os.path.join(data_root, "VCOCO_test_pose.json")
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, f"instances_vcoco_{partition}.json"),
                pose_file=pose_file,
                target_transform=pocket.ops.ToTensor(input_format='dict'),
                pose_transform=pocket.ops.ToTensor(input_format='dict')
            )

        # Prepare dataset transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
                normalize,
            ])
        else:
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

        self.name = name

        self.keep = [i for i in range(len(self.dataset))]
        self.zs = zs
        self.zs_type = zs_type
        if self.zs:
            self.zs_keep = []
            self.filtered_hoi_idx = hico_unseen_index[self.zs_type]
            self.remain_hoi_idx = [i for i in np.arange(600) if i not in self.filtered_hoi_idx]
            for i in self.keep:
                [image, pose, target] = self.dataset[i]
                # 图片中的HOI类型与过滤后的HOI有重叠，那么就保留这个图像样本
                mutual_hoi = set(self.remain_hoi_idx) & set([_h.item() for _h in target['hoi']])
                if len(mutual_hoi) != 0:
                    self.zs_keep.append(i)
            self.keep = self.zs_keep



    def __len__(self):
        return len(self.keep)

    def __getitem__(self, i):
        [image, pose, target] = self.dataset[self.keep[i]]
        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')
            # 原pose已经被修改了
        image_, pose_, target_ = self.transforms(image, pose, target)
        return image_, pose_, target_


def draw(image, target):
    # image h, w, c
    import cv2
    color = (0, 255, 0)
    thickness = 2
    [h, w, c] = image.shape
    image = np.array(image.cpu())

    # target = box_cxcywh_to_xyxy(target)  # 坐标值类型转换
    # 定义矩形框的左上角和右下角坐标
    for i, item in enumerate(target):
        # item[0] = item[0] * w # 相对值转换为绝对值坐标
        # item[2] = item[2] * w
        # item[1] = item[1] * h
        # item[3] = item[3] * h
        # cv2.rectangle(image, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), color, thickness)
        roi = image[int(item[1]):int(item[3]), int(item[0]):int(item[2])]
        cv2.imwrite(str(i) + '.jpg', roi)

    # 显示图像
    # cv2.imshow('Image with Rectangle', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v

    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]


class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, train_dataloader, test_dataloader, config):
        super().__init__(
            net, None, train_dataloader,
            print_interval=config.print_interval,
            cache_dir=config.output_dir,
            find_unused_parameters=True
        )
        self.config = config
        self.max_norm = config.clip_max_norm
        self.test_dataloader = test_dataloader

    def _on_start(self):
        wandb.init(config=self.config)
        # if self._train_loader.dataset.name == "hicodet":
        #     ap = self.test_hico()
        #     if self._rank == 0:
        #         # Fetch indices for rare and non-rare classes
        #         rare = self.test_dataloader.dataset.dataset.rare
        #         non_rare = self.test_dataloader.dataset.dataset.non_rare
        #         perf = [ap.mean().item(), ap[rare].mean().item(), ap[non_rare].mean().item()]
        #         print(
        #             f"Epoch {self._state.epoch} =>\t"
        #             f"mAP: {perf[0]:.4f}, rare: {perf[1]:.4f}, none-rare: {perf[2]:.4f}."
        #         )
        #         logging.info(
        #             f"Epoch {self._state.epoch} =>\t"
        #             f"mAP: {perf[0]:.4f}, rare: {perf[1]:.4f}, none-rare: {perf[2]:.4f}."
        #         )
        #         self.best_perf = perf[0]
        #         wandb.init(config=self.config)
        #         wandb.watch(self._state.net.module)
        #         wandb.define_metric("epochs")
        #         wandb.define_metric("mAP full", step_metric="epochs", summary="max")
        #         wandb.define_metric("mAP rare", step_metric="epochs", summary="max")
        #         wandb.define_metric("mAP non_rare", step_metric="epochs", summary="max")
        #
        #         wandb.define_metric("training_steps")
        #         wandb.define_metric("elapsed_time", step_metric="training_steps", summary="max")
        #         wandb.define_metric("loss", step_metric="training_steps", summary="min")
        #
        #         wandb.log({
        #             "epochs": self._state.epoch, "mAP full": perf[0],
        #             "mAP rare": perf[1], "mAP non_rare": perf[2]
        #         })
        # else:
        #     ap = self.test_vcoco()
        #     if self._rank == 0:
        #         perf = [ap.mean().item(), ]
        #         print(
        #             f"Epoch {self._state.epoch} =>\t"
        #             f"mAP: {perf[0]:.4f}."
        #         )
        #         logging.info(
        #             f"Epoch {self._state.epoch} =>\t"
        #             f"mAP: {perf[0]:.4f}."
        #         )
        #         self.best_perf = perf[0]
        #         """
        #         NOTE wandb was not setup for V-COCO as the dataset was only used for evaluation
        #         """
        #         wandb.init(config=self.config)
    def _on_end(self):
        if self._rank == 0:
            wandb.finish()

    def _on_each_iteration(self):
        loss_dict = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        if loss_dict['cls_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

    def _print_statistics(self):
        running_loss = self._state.running_loss.mean()
        t_data = self._state.t_data.sum() / self._world_size
        t_iter = self._state.t_iteration.sum() / self._world_size

        # Print stats in the master process
        if self._rank == 0:
            num_iter = len(self._train_loader)
            n_d = len(str(num_iter))
            print(
                "Epoch [{}/{}], Iter. [{}/{}], "
                "Loss: {:.4f}, "
                "Time[Data/Iter.]: [{:.2f}s/{:.2f}s]".format(
                    self._state.epoch, self.epochs,
                    str(self._state.iteration - num_iter * (self._state.epoch - 1)).zfill(n_d),
                    num_iter, running_loss, t_data, t_iter
                ))
            logging.info(
                "Epoch [{}/{}], Iter. [{}/{}], "
                "Loss: {:.4f}, "
                "Time[Data/Iter.]: [{:.2f}s/{:.2f}s]".format(
                    self._state.epoch, self.epochs,
                    str(self._state.iteration - num_iter * (self._state.epoch - 1)).zfill(n_d),
                    num_iter, running_loss, t_data, t_iter
                ))
            wandb.log({
                "elapsed_time": (time.time() - self._dawn) / 3600,
                "training_steps": self._state.iteration,
                "loss": running_loss
            })
        self._state.t_iteration.reset()
        self._state.t_data.reset()
        self._state.running_loss.reset()

    def _on_end_epoch(self):
        if self._train_loader.dataset.name == "hicodet":
            ap = self.test_hico()
            if self._rank == 0:
                # Fetch indices for rare and non-rare classes
                rare = self.test_dataloader.dataset.dataset.rare
                non_rare = self.test_dataloader.dataset.dataset.non_rare
                perf = [ap.mean().item(), ap[rare].mean().item(), ap[non_rare].mean().item()]
                print(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}, rare: {perf[1]:.4f}, none-rare: {perf[2]:.4f}."
                )
                if self.config.zs:
                    zs_hoi_idx = hico_unseen_index[self.config.zs_type]
                    print(f'>>> zero-shot setting({self.config.zs_type}!!)')
                    ap_unseen = []
                    ap_seen = []
                    for i, value in enumerate(ap):
                        if i in zs_hoi_idx:
                            ap_unseen.append(value)
                        else:
                            ap_seen.append(value)
                    ap_unseen = torch.as_tensor(ap_unseen).mean()
                    ap_seen = torch.as_tensor(ap_seen).mean()
                    print(
                        f"full mAP: {ap.mean() * 100:.2f}",
                        f"unseen: {ap_unseen * 100:.2f}",
                        f"seen: {ap_seen * 100:.2f}",)

                logging.info(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}, rare: {perf[1]:.4f}, none-rare: {perf[2]:.4f}."
                )
                wandb.log({
                    "epochs": self._state.epoch, "mAP full": perf[0],
                    "mAP rare": perf[1], "mAP non_rare": perf[2]
                })
        else:
            ap = self.test_vcoco()
            if self._rank == 0:
                perf = [ap.mean().item(), ]
                print(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}."
                )
                logging.info(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}."
                )
                """
                NOTE wandb was not setup for V-COCO as the dataset was only used for evaluation
                """

        if self._rank == 0:
            # Save checkpoints
            checkpoint = {
                'iteration': self._state.iteration,
                'epoch': self._state.epoch,
                'performance': perf,
                'model_state_dict': self._state.net.module.state_dict(),
                'optim_state_dict': self._state.optimizer.state_dict(),
                'scaler_state_dict': self._state.scaler.state_dict()
            }
            if self._state.lr_scheduler is not None:
                checkpoint['scheduler_state_dict'] = self._state.lr_scheduler.state_dict()
            torch.save(checkpoint, os.path.join(self._cache_dir, "latest.pth"))
            if perf[0] > self.best_perf:
                self.best_perf = perf[0]
                torch.save(checkpoint, os.path.join(self._cache_dir, "best.pth"))
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()

    @torch.no_grad()
    def test_hico(self):
        dataloader = self.test_dataloader
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))

        if self._rank == 0:
            meter = DetectionAPMeter(
                600, nproc=1, algorithm='11P',
                num_gt=dataset.anno_interaction,
            )
        for batch in tqdm(dataloader, disable=(self._world_size != 1)):
            inputs = pocket.ops.relocate_to_cuda(batch[:-1])
            outputs = net(*inputs)
            outputs = pocket.ops.relocate_to_cpu(outputs, ignore=True)
            targets = batch[-1]

            scores_clt = []
            preds_clt = []
            labels_clt = []
            # 遍历batch中的每一个图片
            for output, target in zip(outputs, targets):
                # Format detection
                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
                scores = output['scores']
                verbs = output['labels']
                objects = output['objects']

                # # 在Known object的设置下，此处应该提前将 outputs根据target进行筛选；
                # target_objects = target['object']
                # objects = output['objects']
                # valid_inds = [i for i, obj in enumerate(objects) if obj in target_objects]
                # objects = [output['objects'][i] for i in valid_inds]
                # boxes = output['boxes']
                # boxes_h, boxes_o = boxes[[output['pairing'][i] for i in valid_inds]].unbind(1)
                # scores = [output['scores'][i] for i in valid_inds]
                # verbs = [output['labels'][i] for i in valid_inds]

                interactions = conversion[objects, verbs]  # object + verbs to HOI class
                # Recover target box scale
                # 左上角坐标xy，右下角坐标xy
                gt_bx_h = recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = recover_boxes(target['boxes_o'], target['size'])

                # Associate detected pairs with ground truth pairs
                labels = torch.zeros_like(scores)  # 用于标记正负样本，正样本为1,负样本为0，正样本数量不大于gt数量
                unique_hoi = interactions.unique()  # 图片中预测的所有的HOI类别
                for hoi_idx in unique_hoi:
                    # 对于预测结果中的每个类别
                    gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)  # target中该类别是否出现，返回出现的索引
                    det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)  # 预测结果中，该类别的pair的索引
                    if len(gt_idx):  # 将预测结果中属于某类HOI的pairs与真实值中该类别的HOI信息关联，包括box以及score
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                             gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                             boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )
                    #
                scores_clt.append(scores)  # 每张图片的预测结果的置信度得分
                preds_clt.append(interactions)  # 每张图片的预测结果的交互类型
                labels_clt.append(labels)  # 每张图片的预测结果的正负样本标记
            # Collate results into one tensor
            scores_clt = torch.cat(scores_clt)
            preds_clt = torch.cat(preds_clt)
            labels_clt = torch.cat(labels_clt)
            # Gather data from all processes
            scores_ddp = pocket.utils.all_gather(scores_clt)
            preds_ddp = pocket.utils.all_gather(preds_clt)
            labels_ddp = pocket.utils.all_gather(labels_clt)

            if self._rank == 0:
                meter.append(torch.cat(scores_ddp), torch.cat(preds_ddp), torch.cat(labels_ddp))

        if self._rank == 0:
            ap = meter.eval()
            return ap
        else:
            return -1

    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, (image, pose, target) in enumerate(tqdm(dataloader.dataset)):
            inputs = pocket.ops.relocate_to_cuda([image, pose])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num

        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def test_vcoco(self):
        dataloader = self.test_dataloader
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)

        if self._rank == 0:
            meter = DetectionAPMeter(
                24, nproc=1, algorithm='11P',
                num_gt=dataset.num_instances,
            )
        for batch in tqdm(dataloader, disable=(self._world_size != 1)):
            inputs = pocket.ops.relocate_to_cuda(batch[:-1])
            outputs = net(*inputs)
            outputs = pocket.ops.relocate_to_cpu(outputs, ignore=True)
            targets = batch[-1]

            scores_clt = []
            preds_clt = []
            labels_clt = []
            for output, target in zip(outputs, targets):
                # Format detections
                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
                scores = output['scores']
                actions = output['labels']
                gt_bx_h = recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = recover_boxes(target['boxes_o'], target['size'])

                # Associate detected pairs with ground truth pairs
                labels = torch.zeros_like(scores)
                unique_actions = actions.unique()
                for act_idx in unique_actions:
                    gt_idx = torch.nonzero(target['actions'] == act_idx).squeeze(1)
                    det_idx = torch.nonzero(actions == act_idx).squeeze(1)
                    if len(gt_idx):
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                             gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                             boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )

                scores_clt.append(scores)
                preds_clt.append(actions)
                labels_clt.append(labels)
            # Collate results into one tensor
            scores_clt = torch.cat(scores_clt)
            preds_clt = torch.cat(preds_clt)
            labels_clt = torch.cat(labels_clt)
            # Gather data from all processes
            scores_ddp = pocket.utils.all_gather(scores_clt)
            preds_ddp = pocket.utils.all_gather(preds_clt)
            labels_ddp = pocket.utils.all_gather(labels_clt)

            if self._rank == 0:
                meter.append(torch.cat(scores_ddp), torch.cat(preds_ddp), torch.cat(labels_ddp))

        if self._rank == 0:
            ap = meter.eval()
            return ap
        else:
            return -1

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, (image, pose, target) in enumerate(tqdm(dataloader.dataset)):
            inputs_image = pocket.ops.relocate_to_cuda([image, ])
            inputs_pose = pocket.ops.relocate_to_cuda([pose, ])
            output = net(inputs_image, inputs_pose)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)
