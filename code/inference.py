"""
Visualise detected human-object interactions and
the cross-attention weights.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import math
import torch
import pocket
import pocket.advis
import warnings
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff
import numpy as np

from detr.util import box_ops
from utils import DataFactory
from pvic import build_detector
from configs import base_detector_args, advanced_detector_args

warnings.filterwarnings("ignore")


def draw_boxes(ax, boxes):
    xy = boxes[:, :2].unbind(0)
    h, w = (boxes[:, 2:] - boxes[:, :2]).unbind(1)
    for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
        patch = patches.Rectangle(a.tolist(), b, c, facecolor='none', edgecolor='w')
        ax.add_patch(patch)
        txt = plt.text(*a.tolist(), str(i + 1), fontsize=20, fontweight='semibold', color='w')
        txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
        plt.draw()


def visualise_entire_image(image, output, file_name, action=None, thresh=0.2):
    """Visualise bounding box pairs in the whole image by classes"""
    # Rescale the boxes to original image size
    ow, oh = image.size
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = output['boxes'] * scale_fct

    image_copy = image.copy()
    scores = output['scores']
    pred = output['labels']
    # Visualise detected human-object pairs with attached scores
    if action is not None:
        keep = torch.nonzero(torch.logical_and(scores >= thresh, pred == action)).squeeze(1)
        bx_h, bx_o = boxes[output['pairing']].unbind(1)
        pocket.utils.draw_box_pairs(image, bx_h[keep], bx_o[keep], width=5)
        plt.imshow(image)
        plt.axis('off')

        for i in range(len(keep)):
            txt = plt.text(*bx_h[keep[i], :2], f"{scores[keep[i]]:.2f}", fontsize=15, fontweight='semibold', color='w')
            txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
            plt.draw()

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig("E:/HOI/Inference/" + file_name, bbox_inches="tight", pad_inches=0)

        # for i in keep:
        #     ho_pair_idx = output["x"][i]
        #     attn_map = attn[0, :, ho_pair_idx].reshape(8, math.ceil(h / 32), math.ceil(w / 32))
        #     attn_image = image_copy.copy()
        #     pocket.utils.draw_boxes(attn_image, torch.stack([bx_h[i], bx_o[i]]), width=4)
        #     if args.avg_attn:
        #         pocket.advis.heatmap(attn_image, attn_map.mean(0, keepdim=True), save_path=f"pair_{i}_avg_attn.png")
        #         plt.close()
        #     else:
        #         for j in range(8):
        #             pocket.advis.heatmap(attn_image, attn_map[j: j + 1], save_path=f"pair_{i}_attn_head_{j + 1}.png")
        #             plt.close()
        plt.close()


def iou(box1, box2):
    """
    计算两个box的IOU
    box1,box2分别为[x1,y1,x2,y2]
    """
    # 计算交集区域
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    # 计算交集和并集
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou


def match(gt_boxes, pred_boxes):
    """
    计算ground truth boxes 和预测boxes之间最优匹配
    返回匹配后的ground truth box索引和IoU
    """

    n = gt_boxes.shape[0]
    m = pred_boxes.shape[0]

    iou_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = iou(gt_boxes[i], pred_boxes[j])

    matched_indices = np.argmax(iou_matrix, axis=1)
    matched_ious = np.max(iou_matrix, axis=1)

    return matched_indices, matched_ious


def iou_count(data):
    counts = np.zeros(10)
    for x in data:
        idx = int(10 * x)
        counts[idx] += 1
    print(counts)


@torch.no_grad()
def main(args):
    dataset = DataFactory(name=args.dataset, partition=args.partition, data_root=args.data_root)
    conversion = dataset.dataset.object_to_verb if args.dataset == 'hicodet' \
        else list(dataset.dataset.object_to_action.values())
    args.num_verbs = 117 if args.dataset == 'hicodet' else 24

    model = build_detector(args, conversion)
    model.eval()

    if os.path.exists(args.resume):
        print(f"=> Continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Start from a randomly initialised model")
    results = []
    # file_names = ["HICO_test2015_00000314.jpg", "HICO_test2015_00000415.jpg", "HICO_test2015_00000414.jpg",
    #               "HICO_test2015_00000387.jpg", "HICO_test2015_00004390.jpg", "HICO_test2015_00000453.jpg",
    #               "HICO_test2015_00000596.jpg", "HICO_test2015_00000733.jpg", "HICO_test2015_00000775.jpg",
    #               "HICO_test2015_00001556.jpg", "HICO_test2015_00000348.jpg", "HICO_test2015_00004301.jpg",
    #               "HICO_test2015_00004296.jpg",
    #               "HICO_test2015_00004803.jpg", "HICO_test2015_00002969.jpg", "HICO_test2015_00004580.jpg",
    #               "HICO_test2015_00001802.jpg", "HICO_test2015_00001629.jpg", "HICO_test2015_00000195.jpg",
    #               "HICO_test2015_00004136.jpg", "HICO_test2015_00000272.jpg", "HICO_test2015_00001467.jpg"]
    file_names = ["HICO_test2015_00002295.jpg", "HICO_test2015_00002662.jpg", "HICO_test2015_00002656.jpg", "HICO_test2015_00003934.jpg"]
    for index in range(0, len(dataset.index_to_file)):
        image, pose, _ = dataset[index]
        file_name = dataset.index_to_file[index]["file_name"]
        if file_name in file_names:
            print(file_name)
            print(dataset.index_to_file[index]["anno"])
        # annotation = dataset.index_to_file[index]["anno"]
        # verb = annotation["verb"][0]
        # output = model([image], [pose])
        # image = dataset.dataset.load_image(
        #     os.path.join(dataset.dataset._root,
        #                  file_name
        #                  ))
        # gt_boxes = np.array(annotation["boxes_h"] + annotation["boxes_o"])
        # pred_boxes = np.array(output[0]["boxes"])
        # matched_indices, matched_ious = match(gt_boxes, pred_boxes)
        # results.extend(matched_ious)
        # # if index % 1000 == 0:
        # #     iou_count(results)
        # visualise_entire_image(
        #     image, output[0], file_name,
        #     verb, args.action_score_thresh
        # )

    # attn_weights = []
    # hook = model.decoder.layers[-1].qk_attn.register_forward_hook(
    #     lambda self, input, output: attn_weights.append(output[1])
    # )
    #
    # if os.path.exists(args.resume):
    #     print(f"=> Continue from saved checkpoint {args.resume}")
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     model.load_state_dict(checkpoint['model_state_dict'])
    # else:
    #     print(f"=> Start from a randomly initialised model")
    #
    # if args.image_path is None:
    #     image, pose, _ = dataset[args.index]
    #     print(dataset.index_to_file[args.index])
    #     output = model([image], [pose])
    #     image = dataset.dataset.load_image(
    #         os.path.join(dataset.dataset._root,
    #                      dataset.dataset.filename(args.index)
    #                      ))
    # else:
    #     image = dataset.dataset.load_image(args.image_path)
    #     image_tensor, _ = dataset.transforms(image, None)
    #     output = model([image_tensor])
    #
    # hook.remove()
    #
    # visualise_entire_image(
    #     image, output[0], attn_weights[0],
    #     args.action, args.action_score_thresh
    # )


if __name__ == "__main__":
    os.environ["DETR"] = "base"
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

    parser.add_argument('--partition', type=str, default="test2015")

    parser.add_argument('--kv-src', default='C5', type=str, choices=['C5', 'C4', 'C3'])
    parser.add_argument('--repr-dim', default=384, type=int)
    parser.add_argument('--triplet-enc-layers', default=1, type=int)
    parser.add_argument('--triplet-dec-layers', default=2, type=int)

    parser.add_argument('--alpha', default=.5, type=float)
    parser.add_argument('--gamma', default=.1, type=float)
    parser.add_argument('--box-score-thresh', default=.05, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--avg-attn', action='store_true', default=False)

    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--index', default=0, type=int)  # 指定图片
    parser.add_argument('--action', default=None, type=int,  # 指定verb的类别
                        help="Index of the action class to visualise.")
    parser.add_argument('--action-score-thresh', default=0.2, type=float,
                        help="Threshold on action classes.")
    parser.add_argument('--image-path', default=None, type=str,
                        help="Path to an image file.")

    args = parser.parse_args()

    main(args)
