# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/3 11:01
@Auther ： Zzou
@File ：pose_extractor.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""
import pocket

from hicodet.hicodet import HICODet
from vcoco.vcoco import VCOCO

"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""
import json
import os
import sys
import torch
import random
import warnings
import argparse
import numpy as np
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from pvic import build_detector
from utils import custom_collate, CustomisedDLE, DataFactory
from configs import base_detector_args, advanced_detector_args

warnings.filterwarnings("ignore")
os.environ["DETR"] = "base"

logging.basicConfig(filename="hicodet_log.txt",
                    filemode='a',
                    format='%(message)s',
                    level=logging.DEBUG)


def main(rank, args):
    dist.init_process_group(
        # backend="nccl",
        backend="gloo",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)

    # # HICO-DET
    # trainset = HICODet(
    #     root=os.path.join(args.data_root, "hico_20160224_det/images", args.partitions[0]),
    #     anno_file=os.path.join(args.data_root, f"instances_{args.partitions[0]}.json"),
    #     target_transform=pocket.ops.ToTensor(input_format='dict'))
    #
    # testset = HICODet(
    #     root=os.path.join(args.data_root, "hico_20160224_det/images", args.partitions[1]),
    #     anno_file=os.path.join(args.data_root, f"instances_{args.partitions[1]}.json"),
    #     target_transform=pocket.ops.ToTensor(input_format='dict'))
    #
    # full_dict_train = dict()
    # for _, [filenames, candidate, subset] in trainset:
    #     temp = dict()
    #     temp["candidate"] = candidate.tolist()
    #     temp["subset"] = subset.tolist()
    #     print(filenames)
    #     full_dict_train[filenames] = temp
    # with open('HICODET_train_pose.json', 'w') as f:
    #     json.dump(full_dict_train, f)
    #
    # full_dict_test = dict()
    # for _, [filenames, candidate, subset] in testset:
    #     temp = dict()
    #     temp["candidate"] = candidate.tolist()
    #     temp["subset"] = subset.tolist()
    #     print(filenames)
    #     full_dict_test[filenames] = temp
    # with open('HICODET_test_pose.json', 'w') as f:
    #     json.dump(full_dict_test, f)

    # V-COCO

    image_dir = dict(
        train='mscoco2014/train2014',
        val='mscoco2014/train2014',
        trainval='mscoco2014/train2014',
        test='mscoco2014/val2014'
    )
    trainset = VCOCO(
        root=os.path.join(args.data_root, image_dir[args.partitions[0]]),
        anno_file=os.path.join(args.data_root, f"instances_vcoco_{args.partitions[0]}.json"),
        target_transform=pocket.ops.ToTensor(input_format='dict'))
    testset = VCOCO(
        root=os.path.join(args.data_root, image_dir[args.partitions[1]]),
        anno_file=os.path.join(args.data_root, f"instances_vcoco_{args.partitions[1]}.json"),
        target_transform=pocket.ops.ToTensor(input_format='dict'))

    # full_dict_train = dict()
    # for _, [filenames, candidate, subset] in trainset:
    #     temp = dict()
    #     temp["candidate"] = candidate.tolist()
    #     temp["subset"] = subset.tolist()
    #     print(filenames)
    #     full_dict_train[filenames] = temp
    # with open('VCOCO_train_pose.json', 'w') as f:
    #     json.dump(full_dict_train, f)

    full_dict_test = dict()
    for _, [filenames, candidate, subset] in testset:
        temp = dict()
        temp["candidate"] = candidate.tolist()
        temp["subset"] = subset.tolist()
        print(filenames)
        full_dict_test[filenames] = temp
    with open('VCOCO_test_pose.json', 'w') as f:
        json.dump(full_dict_test, f)

if __name__ == '__main__':

    if "DETR" not in os.environ:
        raise KeyError(f"Specify the detector type with env. variable \"DETR\".")
    elif os.environ["DETR"] == "base":
        parser = argparse.ArgumentParser(parents=[base_detector_args(), ])
        parser.add_argument('--detector', default='base', type=str)
        parser.add_argument('--raw-lambda', default=2.8, type=float)
    elif os.environ["DETR"] == "advanced":
        parser = argparse.ArgumentParser(parents=[advanced_detector_args(), ])
        parser.add_argument('--detector', default='advanced', type=str)
        parser.add_argument('--raw-lambda', default=1.7, type=float)

    parser.add_argument('--kv-src', default='C5', type=str, choices=['C5', 'C4', 'C3'])
    parser.add_argument('--repr-dim', default=384, type=int)
    parser.add_argument('--triplet-enc-layers', default=1, type=int)
    parser.add_argument('--triplet-dec-layers', default=2, type=int)

    parser.add_argument('--alpha', default=.5, type=float)
    parser.add_argument('--gamma', default=.1, type=float)
    parser.add_argument('--box-score-thresh', default=.05, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--use-wandb', default=False, action='store_true')

    parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--seed', default=140, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')

    args = parser.parse_args()
    print(args)
    logging.info(args)

    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    mp.spawn(main, nprocs=args.world_size, args=(args,))
