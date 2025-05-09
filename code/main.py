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
from torchinfo import summary
from pvic import build_detector
from utils import custom_collate, CustomisedDLE, DataFactory, hico_unseen_index
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

    trainset = DataFactory(
        name=args.dataset, partition=args.partitions[0],
        data_root=args.data_root, zs=args.zs, zs_type=args.zs_type
    )
    testset = DataFactory(
        name=args.dataset, partition=args.partitions[1],
        data_root=args.data_root
    )

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size // args.world_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            trainset, num_replicas=args.world_size,
            rank=rank, drop_last=True)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=args.batch_size // args.world_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            testset, num_replicas=args.world_size,
            rank=rank, drop_last=True)
    )

    if args.dataset == 'hicodet':
        object_to_target = train_loader.dataset.dataset.object_to_verb
        args.num_verbs = 117
    elif args.dataset == 'vcoco':
        object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
        args.num_verbs = 24

    model = build_detector(args, object_to_target)

    if os.path.exists(args.resume):
        print(f"=> Rank {rank}: PViC loaded from saved checkpoint {args.resume}.")
        logging.info(f"=> Rank {rank}: PViC loaded from saved checkpoint {args.resume}.")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Rank {rank}: PViC randomly initialised.")
        logging.info(f"=> Rank {rank}: PViC randomly initialised.")

    # summary(model, input_size=(1, 3, 224, 224))

    engine = CustomisedDLE(model, train_loader, test_loader, args)

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return

    if args.eval:
        if args.dataset == 'vcoco':
            """
            NOTE This evaluation results on V-COCO do not necessarily follow the 
            protocol as the official evaluation code, and so are only used for
            diagnostic purposes.
            """
            ap = engine.test_vcoco()
            if rank == 0:
                print(f"The mAP is {ap.mean():.4f}.")
                logging.info(f"The mAP is {ap.mean():.4f}.")
            return
        else:
            ap = engine.test_hico()
            if rank == 0:
                # Fetch indices for rare and non-rare classes
                rare = trainset.dataset.rare
                non_rare = trainset.dataset.non_rare
                print(
                    f"The mAP is {ap.mean():.4f},"
                    f" rare: {ap[rare].mean():.4f},"
                    f" none-rare: {ap[non_rare].mean():.4f}"
                )
                if args.zs:
                    zs_hoi_idx = hico_unseen_index[args.zs_type]
                    print(f'>>> zero-shot setting({args.zs_type}!!)')
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
                    f"The mAP is {ap.mean():.4f},"
                    f" rare: {ap[rare].mean():.4f},"
                    f" none-rare: {ap[non_rare].mean():.4f}"
                )
            return

    model.freeze_detector()
    param_dicts = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    optim = torch.optim.AdamW(param_dicts, lr=args.lr_head, weight_decay=args.weight_decay)
    # 每20轮，学习率降为0.2
    # 一共四十轮，每17轮降低0.2
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop, gamma=args.lr_drop_factor)
    # Override optimiser and learning rate scheduler
    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
    engine(args.epochs)


@torch.no_grad()
def sanity_check(args):
    dataset = DataFactory(name='hicodet', partition=args.partitions[0], data_root=args.data_root)
    args.num_verbs = 117
    args.num_triplets = 600
    object_to_target = dataset.dataset.object_to_verb
    model = build_detector(args, object_to_target)
    if args.eval:
        model.eval()
    if os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        print(f"Loading checkpoints from {args.resume}.")
        logging.info(f"Loading checkpoints from {args.resume}.")
        model.load_state_dict(ckpt['model_state_dict'])

    image, target = dataset[998]
    outputs = model([image], targets=[target])


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

    parser.add_argument('--zs', default=True)
    parser.add_argument('--zs_type', default="rare_first")

    # print(len([4, 6, 12, 15, 18, 25, 34, 38, 40, 49, 58, 60, 68, 69, 72, 73, 77, 82, 96, 97,
    #                 104, 113, 116, 118, 122, 129, 139, 147, 150, 153, 165, 166, 172, 175, 176, 181, 190, 202, 210, 212,
    #                 219, 227, 228, 233, 235, 243, 298, 313, 315, 320, 326, 336, 342, 345, 354, 372, 401, 404, 409, 431,
    #                 436, 459, 466, 470, 472, 479, 481, 488, 491, 494, 498, 504, 519, 523, 535, 536, 541, 544, 562, 565,
    #                 569, 572, 591, 595]))
    args = parser.parse_args()
    print(args)
    logging.info(args)

    if args.sanity:
        sanity_check(args)
        sys.exit()
    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    mp.spawn(main, nprocs=args.world_size, args=(args,))
