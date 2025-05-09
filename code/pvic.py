"""
Two-stage HOI detector with enhanced visual context

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""
import logging
import os
import torch
import math
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.ops.boxes as box_ops
import numpy as np
from torch import nn, Tensor
from collections import OrderedDict
from typing import Optional, Tuple, List
from torchvision.ops import FeaturePyramidNetwork

from transformers import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    SwinTransformer,
)

from ops import (
    binary_focal_loss_with_logits,
    compute_spatial_encodings,
    prepare_region_proposals,
    associate_with_ground_truth,
    compute_prior_scores,
    compute_sinusoidal_pe
)

from detr.models import build_model as build_base_detr
from h_detr.models import build_model as build_advanced_detr
from detr.models.position_encoding import PositionEmbeddingSine
from detr.util.misc import NestedTensor, nested_tensor_from_tensor_list


class MultiModalFusion(nn.Module):
    def __init__(self, fst_mod_size, scd_mod_size, repr_size):
        super().__init__()
        self.fc1 = nn.Linear(fst_mod_size, repr_size)
        self.fc2 = nn.Linear(scd_mod_size, repr_size)
        self.ln1 = nn.LayerNorm(repr_size)
        self.ln2 = nn.LayerNorm(repr_size)

        mlp = []
        repr_size = [2 * repr_size, int(repr_size * 1.5), repr_size]
        for d_in, d_out in zip(repr_size[:-1], repr_size[1:]):
            mlp.append(nn.Linear(d_in, d_out))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: Tensor, y: Tensor, global_pose: Tensor, local_pose: Tensor, pose_valid: Tensor) -> Tensor:
        x = self.ln1(self.fc1(x))
        y = self.ln2(self.fc2(y))
        z = F.relu(torch.cat([x, y], dim=-1))
        z = self.mlp(z)
        return z


'''
class MultiModalFusion(nn.Module):
    # 512, 384, 384
    def __init__(self, fst_mod_size, scd_mod_size, repr_size):
        super().__init__()

        # Visual features
        self.query_v = nn.Linear(int(fst_mod_size / 2), int(fst_mod_size / 2))
        self.key_v = nn.Linear(int(fst_mod_size / 2), int(fst_mod_size / 2))
        self.value_v = nn.Linear(int(fst_mod_size / 2), int(fst_mod_size / 2))
        self.fc_v = nn.Linear(fst_mod_size, repr_size)
        self.ln_v = nn.LayerNorm(repr_size)

        # Spatial features
        self.fc_s = nn.Linear(128, int(repr_size / 2))
        self.ln_s = nn.LayerNorm(int(repr_size / 2))

        # pose features
        self.fc_g = nn.Linear(36, 128)
        self.att_gl = nn.Linear(128, 13)
        self.fc_l = nn.Linear(26, 128)
        self.fc_p = nn.Linear(128 + 128, int(repr_size / 2))
        self.ln_p = nn.LayerNorm(int(repr_size / 2))

        # concat
        mlp = []
        repr_size = [2 * repr_size, int(repr_size * 1.5), repr_size]
        for d_in, d_out in zip(repr_size[:-1], repr_size[1:]):
            mlp.append(nn.Linear(d_in, d_out))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)
        self.dp = nn.Dropout(p=0.2)

    def forward(self, visual: Tensor, spatial: Tensor, global_pose: Tensor, local_pose: Tensor,
                pose_valid: Tensor) -> Tensor:
        # # visual branch
        # human_visual = visual[:, :256]
        # object_visual = visual[:, 256:]
        # # 计算注意力权重
        # query = self.query_v(human_visual)
        # key = self.key_v(object_visual)
        # attn_weights = torch.softmax(query @ key.T, dim=-1)
        # # 应用注意力机制
        # value = self.value_v(object_visual)
        # context = torch.matmul(attn_weights, value)
        # # 将人物特征和上下文连接
        # temp_visual = torch.cat((human_visual, context), dim=1)
        # visual_feature = F.relu(self.ln_v(self.fc_v(temp_visual)))
        #
        # # spatial branch
        # spatial_feature = F.relu(self.ln_s(self.fc_s(spatial)))
        #
        # # pose branch
        # batch_size = global_pose.shape[0]
        # global_pose_feature = self.fc_g(global_pose.view(batch_size, -1))
        # global_to_pose_att = F.softmax(self.att_gl(global_pose_feature), dim=-1)
        # atted_local_pose = local_pose * global_to_pose_att.unsqueeze(-1)
        # local_pose_feature = self.fc_l(atted_local_pose.view(batch_size, -1))
        # pose_feature = F.relu(self.ln_p(self.fc_p(torch.cat([global_pose_feature, local_pose_feature], dim=-1))))
        #
        # # concat
        # z = torch.cat([visual_feature, spatial_feature, pose_feature], dim=-1)
        # z = self.mlp(z)
        # z = self.dp(z)

        visual_feature = self.ln_v(self.fc_v(visual))

        # spatial branch
        spatial_feature = self.ln_s(self.fc_s(spatial))

        # pose branch
        batch_size = global_pose.shape[0]
        global_pose_feature = self.fc_g(global_pose.view(batch_size, -1))
        global_to_pose_att = F.softmax(self.att_gl(global_pose_feature), dim=-1)
        atted_local_pose = local_pose * global_to_pose_att.unsqueeze(-1)
        local_pose_feature = self.fc_l(atted_local_pose.view(batch_size, -1))
        pose_feature = self.ln_p(self.fc_p(torch.cat([global_pose_feature, local_pose_feature], dim=-1)))

        # concat
        z = F.relu(torch.cat([visual_feature, spatial_feature, pose_feature], dim=-1))
        z = self.mlp(z)
        # z = self.dp(z)

        return z
'''


def boundary(points):  # N*[x,y]
    x_min = torch.min(points[:, 0])
    x_max = torch.max(points[:, 0])
    y_min = torch.min(points[:, 1])
    y_max = torch.max(points[:, 1])
    return torch.Tensor([x_min, y_min, x_max, y_max])


# N,18,3
# 为图片中，每个human pose计算关键点之间的相对位置特征
def pose_feature(pose_point, image_sizes):
    relations = [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [1, 11], [8, 9], [9, 10], [11, 12],
                 [12, 13]]
    num_pose = pose_point.shape[0]
    h, w = image_sizes
    relative_xy = torch.zeros([num_pose, 13])
    angle = torch.zeros([num_pose, 13])
    for i, pose in enumerate(pose_point):
        for j, rel in enumerate(relations):
            relative_x = pose[rel[1]][0] - pose[rel[0]][0]
            relative_y = pose[rel[1]][1] - pose[rel[0]][1]
            ang = math.atan2(relative_y, relative_x)
            ang = (ang + math.pi) / (2 * math.pi)
            relative_xy[i, j] = torch.tensor(math.sqrt((abs(relative_x / w)) ** 2 + (abs(relative_y / h)) ** 2))
            angle[i, j] = ang
    return torch.cat((relative_xy.unsqueeze(-1), angle.unsqueeze(-1)), dim=-1)


def global_pose_feature(pose_point, image_sizes):
    num_pose = pose_point.shape[0]
    re = torch.zeros(num_pose, 18, 2)
    h, w = image_sizes
    for i, pose in enumerate(pose_point):
        xys = pose[:, :2]
        for j, point in enumerate(pose):
            re[i, j, 0] = torch.tensor(xys[j, 0] / w)
            re[i, j, 1] = torch.tensor(xys[j, 1] / h)
    return re


# [N,4], dict
def human_box_pose_matcher(human_boxes, human_poses):
    candidate = human_poses["candidate"]
    subset = human_poses["subset"]
    candidate_dict = dict()
    for item in candidate:
        candidate_dict[int(item[3])] = item[:3]

    pose_points = []  # 存储每个human的pose信息，包括x,y,score
    pose_xyxy = []  # 存储每个human pose的边界信息
    match_ind, match_iou = torch.zeros(human_boxes.shape[0], dtype=torch.long), torch.zeros(human_boxes.shape[0])
    # 如果图片中没有检测到pose，返回全0结果

    if subset.shape[0] != 0:
        mask = subset[:, -1] != 0  # 创建布尔掩码，表示每个子张量最后一个元素是否不等于 0
        subset = subset[mask]  # 使用掩码来筛选保留符合条件的子张量

    if subset.shape[0] == 0:
        return match_iou, match_ind, torch.zeros([1, 18, 3])
    # # 如果图片中没有检测到human box，返回空
    # if len(human_boxes) == 0:
    #     return [], [], []
    # 如果图片中检测到了pose
    for h in subset:
        part_ids = h[:18]
        pose = []
        for part in part_ids:
            part = int(part.item())
            if part != -1:
                info = candidate_dict[part]
                pose.append(info)
            else:
                # 未检测到的人体关键点，坐标和score全部置为0
                pose.append(torch.Tensor([0, 0, 0]).to(human_boxes.device))

        pose_xy = [p[:2] for p in pose if p[2] > 0]  # 只取有效坐标的姿势点
        pose_xyxy.append(boundary(torch.stack(pose_xy)))  # 计算pose的边界
        pose_points.append(torch.stack(pose))
    pose_xyxy = torch.stack(pose_xyxy)
    human_pose_iou = box_ops.box_iou(human_boxes, pose_xyxy.to(human_boxes.device))

    match_iou, match_ind = torch.max(human_pose_iou, dim=1)
    # match_iou = 0, 意味着没有这个box没有与之相匹配的pose
    # pose_points 中 score=0，意味该关键点未检测到
    return match_iou, match_ind, torch.stack(pose_points)


class HumanObjectMatcher(nn.Module):
    def __init__(self, repr_size, num_verbs, obj_to_verb, dropout=.1, human_idx=0):
        super().__init__()
        self.repr_size = repr_size
        self.num_verbs = num_verbs
        self.human_idx = human_idx
        self.obj_to_verb = obj_to_verb

        self.ref_anchor_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, repr_size), nn.ReLU(),
        )
        # 19,256  19,512
        self.pos_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
        )

        self.encoder = TransformerEncoder(num_layers=2, dropout=dropout)
        self.mmf = MultiModalFusion(512, repr_size, repr_size)

    def check_human_instances(self, labels):
        is_human = labels == self.human_idx
        n_h = torch.sum(is_human)
        if not torch.all(labels[:n_h] == self.human_idx):
            raise AssertionError("Human instances are not permuted to the top!")
        return n_h

    def compute_box_pe(self, boxes, embeds, image_size):
        bx_norm = boxes / image_size[[1, 0, 1, 0]]  # 绝对值转换为相对值
        bx_c = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2  # 坐标转换为 中心坐标以及宽度高度 # N,2
        b_wh = bx_norm[:, 2:] - bx_norm[:, :2]

        # 计算添加位置编码之后的结果
        c_pe = compute_sinusoidal_pe(bx_c[:, None], 20).squeeze(1)
        wh_pe = compute_sinusoidal_pe(b_wh[:, None], 20).squeeze(1)

        box_pe = torch.cat([c_pe, wh_pe], dim=-1)  # 每个box，由xy，wh添加位置编码之后的结果拼接

        # Modulate the positional embeddings with box widths and heights by
        # applying different temperatures to x and y
        # box特征变换为低维度
        ref_hw_cond = self.ref_anchor_head(embeds).sigmoid()  # n_query, 2
        # Note that the positional embeddings are stacked as [pe(y), pe(x)]
        c_pe[..., :128] *= (ref_hw_cond[:, 1] / b_wh[:, 1]).unsqueeze(-1)
        c_pe[..., 128:] *= (ref_hw_cond[:, 0] / b_wh[:, 0]).unsqueeze(-1)

        return box_pe, c_pe  # box的位置编码，box中心坐标的位置编码

    def forward(self, region_props, poses, image_sizes, device=None):
        if device is None:
            device = region_props[0]["hidden_states"].device

        ho_queries = []
        paired_indices = []
        prior_scores = []
        object_types = []
        positional_embeds = []
        # 遍历每个图片
        for i, rp in enumerate(region_props):
            boxes, scores, labels, embeds = rp.values()
            nh = self.check_human_instances(labels)
            n = len(boxes)
            # step 1 枚举所有的human-object pair
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            # 此处human 将和所有的human和object 进行匹配
            # 如果设置 human和所有object匹配呢？ 即 y>nh
            # x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < nh)).unbind(1)
            x_keep, y_keep = torch.nonzero(torch.logical_and(torch.logical_and(x != y, x < nh), y > (nh - 1))).unbind(1)
            # ------如果没有human-object pairs：添加运算结果的默认值，全0
            if len(x_keep) == 0:
                ho_queries.append(torch.zeros(0, self.repr_size, device=device))
                paired_indices.append(torch.zeros(0, 2, device=device, dtype=torch.int64))
                prior_scores.append(torch.zeros(0, 2, self.num_verbs, device=device))
                object_types.append(torch.zeros(0, device=device, dtype=torch.int64))
                positional_embeds.append({})
                continue
            # ------ 否则，为每个pair计算各类特征

            # step 2 计算所有human-object pair的空间位置特征
            x = x.flatten()
            y = y.flatten()
            pairwise_spatial = compute_spatial_encodings(
                [boxes[x], ], [boxes[y], ], [image_sizes[i], ]
            )  # 100,36
            # 空间位置特征进一步编码
            pairwise_spatial = self.spatial_head(pairwise_spatial)  # 空间位置编码：通过spatial head从36转换为128维度
            pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)

            # step 3 计算所有human-object pair视觉特征
            # ------ box的位置编码，box中心坐标的位置编码
            box_pe, c_pe = self.compute_box_pe(boxes, embeds, image_sizes[i])
            # ------ 将视觉特征与位置编码输入到transformer编码器中，进一步提炼视觉特征

            # 19,256  19,512
            embeds, _ = self.encoder(embeds.unsqueeze(1), box_pe.unsqueeze(1))  # visual branch 的 注意力机制
            embeds = embeds.squeeze(1)
            # embeds = embeds + self.pos_head(box_pe)

            # step 4 计算human的pose特征
            match_ious, match_inds, pose_points = human_box_pose_matcher(boxes[:nh], poses[i])
            # 其中：
            # match_ious 记录了每个human框与pose集合的最佳匹配的IOU，数值为0，表示没有匹配的pose
            # match_inds 记录了每个human框与pose集合中最匹配的pose的索引
            # pose_points 记录了pose集合中，每个pose的18个关键点的坐标（x,y）以及得分，score为-0，表示未检测到该关键点

            # ------ human 和 pose的匹配
            human_to_pose_index = match_inds[x_keep]
            human_to_pose_iou = match_ious[x_keep]
            # for item in pose_iou, if ==0:, mask = true else false
            # if false, human_to_pose_index标记为-1
            human_to_pose_valid = torch.Tensor([True if element != 0 else False for element in human_to_pose_iou]).to(
                embeds.device)

            # ------ 相对pose信息
            hp_features = pose_feature(pose_points, image_sizes[i])
            local_human_pose_feature = hp_features[human_to_pose_index].to(embeds.device)  # N,18,3
            # ------ 绝对pose信息
            global_human_pose_feature = global_pose_feature(pose_points, image_sizes[i])[human_to_pose_index].to(
                embeds.device)

            # step 5 多分支特征融合，以构建query embedding
            ho_q = self.mmf(
                torch.cat([embeds[x_keep], embeds[y_keep]], dim=1),  # N,512
                pairwise_spatial_reshaped[x_keep, y_keep], global_human_pose_feature, local_human_pose_feature,
                human_to_pose_valid  # N,384
            )
            # step 6, 信息整合
            ho_queries.append(ho_q)
            paired_indices.append(torch.stack([x_keep, y_keep], dim=1))  # 存储配对的human-object pair的box索引对
            prior_scores.append(compute_prior_scores(  # pairs的action得分，具体到每个类比的得分
                x_keep, y_keep, scores, labels, self.num_verbs, self.training,
                self.obj_to_verb
            ))  # N,2(human,object),117(verbs)
            object_types.append(labels[y_keep])  # 存储每个pair中object的种类
            positional_embeds.append({
                "centre": torch.cat([c_pe[x_keep], c_pe[y_keep]], dim=-1).unsqueeze(1),  # pair中human和object中心坐标编码
                "box": torch.cat([box_pe[x_keep], box_pe[y_keep]], dim=-1).unsqueeze(1)  # pair中human和object box位置编码
            })

        return ho_queries, paired_indices, prior_scores, object_types, positional_embeds
        # human-object query embed [N,384], pair的box index对 [N,2], pair中human和object的不同动词得分 [N,2,117]
        # pair中object的类别[N], pair的位置编码信息,cat human和object的结果,centre:[N,1,512],box:[N,1,1024]


class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)


class FeatureHead(nn.Module):
    def __init__(self, dim, dim_backbone, return_layer, num_layers):
        super().__init__()
        self.dim = dim
        self.dim_backbone = dim_backbone
        self.return_layer = return_layer

        in_channel_list = [
            int(dim_backbone * 2 ** i)
            for i in range(return_layer + 1, 1)
        ]
        self.fpn = FeaturePyramidNetwork(in_channel_list, dim)
        self.layers = nn.Sequential(
            Permute([0, 2, 3, 1]),
            SwinTransformer(dim, num_layers)  # 换成普通Transformer Encoder即可
        )
        self.pe = PositionEmbeddingSine(256, normalize=True)
        self.te = TransformerEncoder(hidden_size=256, num_heads=8, num_layers=2, dropout=.1)

    def forward(self, x):
        pyramid = OrderedDict(
            (f"{i}", x[i].tensors)
            for i in range(self.return_layer, 0)  # 取x的最后一个尺度的特征：[4, 2048, 26, 23]
        )  # 多尺度特征采用最后一个尺度作为输入 [4,2048,25,39]
        mask = x[self.return_layer].mask  # mask: [4, 26, 23]
        x = self.fpn(pyramid)[f"{self.return_layer}"]  # FPN的最后一层：[4,256,26,23]
        x = self.layers(x)  # [4,26,23 256]

        # pos = self.pe(NestedTensor(x, mask))  # [4, 26, 23, 256]
        # flatten NxCxHxW to HWxNxC
        # bs, c, h, w = x.shape
        # x = x.flatten(2).permute(2, 0, 1)  # [26*23,4,256]
        # pos = pos.flatten(2).permute(2, 0, 1)
        # x, _ = self.te(x, pos)  # [26*23,4,256]
        # x = x.permute(1, 2, 0).view(bs, c, h, w)
        # x = x.permute(0, 2, 3, 1)
        return x, mask


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class PViC(nn.Module):
    """Two-stage HOI detector with enhanced visual context"""

    def __init__(self,
                 detector: Tuple[nn.Module, str], postprocessor: nn.Module,
                 feature_head: nn.Module, ho_matcher: nn.Module,
                 triplet_decoder: nn.Module, num_verbs: int,
                 repr_size: int = 384, human_idx: int = 0,
                 # Focal loss hyper-parameters
                 alpha: float = 0.5, gamma: float = .1,
                 # Sampling hyper-parameters
                 box_score_thresh: float = .05,
                 min_instances: int = 3,
                 max_instances: int = 15,
                 raw_lambda: float = 2.8,
                 ) -> None:
        super().__init__()

        self.detector = detector[0]
        self.od_forward = {
            "base": self.base_forward,
            "advanced": self.advanced_forward,
        }[detector[1]]
        self.postprocessor = postprocessor

        self.fh = nn.Linear(2048, 256)

        self.ho_matcher = ho_matcher
        self.feature_head = feature_head
        self.kv_pe = PositionEmbeddingSine(128, 20, normalize=True)
        self.decoder = triplet_decoder
        self.binary_classifier = nn.Linear(repr_size, num_verbs)

        self.repr_size = repr_size
        self.human_idx = human_idx
        self.num_verbs = num_verbs
        self.alpha = alpha
        self.gamma = gamma
        self.box_score_thresh = box_score_thresh
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.raw_lambda = raw_lambda

    def freeze_detector(self):
        for p in self.detector.parameters():
            p.requires_grad = False

    # batch级别下HOI pair的类别概率；每个HOI pair的human和object先验得分；每个HOI预测结果与target配对之后的类别标签，1为正确配对
    def compute_classification_loss(self, logits, prior, labels):
        # [num_decoder, batch_query_num, verbs], [batch, query_num, 2, verbs], [batch_query_num, verbs]
        prior = torch.cat(prior, dim=0).prod(1)  # [N,117]
        x, y = torch.nonzero(prior).unbind(1)  # 3909, 3909 # 第x个pair属于类别y的先验得分不为0的索引

        logits = logits[:, x, y]  # [2, 3909] 2表示decoder的数量
        prior = prior[x, y]  # 3909
        labels = labels[None, x, y].repeat(len(logits), 1)  # [2, 3909]

        n_p = labels.sum()  # 正确标签的总数
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
        )

        return loss / n_p

    def postprocessing(self,
                       boxes, paired_inds, object_types,
                       logits, prior, image_sizes
                       ):
        n = [len(p_inds) for p_inds in paired_inds]  # 每个图片中，预测的样本数量，即query num
        logits = logits.split(n)  # 每个图片中，预测的样本所对应的类别得分

        detections = []
        # boxes -> human and object boxes: B,N,4
        # paired_inds -> human 和 object配对的索引集合：B,num_query,2
        # object_types -> 配对结果中，object的类别信息
        # logits[-1] -> 最后一层模型预测的动词结果: B*num_query, 117
        # prior_scores -> human 和object的动词类别先验得分：B, num_query, 2, 117
        # image_sizes -> 图像的大小：【宽，高】，B，2
        for bx, p_inds, objs, lg, pr, size in zip(
                boxes, paired_inds, object_types,
                logits, prior, image_sizes
        ):
            pr = pr.prod(1)  # human和object的动词先验得分相乘，[query_num, classes]
            x, y = torch.nonzero(pr).unbind(1)  # 先验得分非0的query-class样本
            # 根据得分过滤query
            scores = lg[x, y].sigmoid() * pr[x, y].pow(self.raw_lambda)  # 样本的预测得分和先验得分相乘作为最终得分
            # 只根据先验得分是否为0来筛选，样本是不是太多了
            detections.append(dict(
                boxes=bx, pairing=p_inds[x], scores=scores,
                labels=y, objects=objs[x], size=size, x=x
            ))  # pair的boxes，score，label，object

        return detections

    @staticmethod
    def base_forward(ctx, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = ctx.backbone(samples)

        src, mask = features[-1].decompose()  # [4,2048,26,23]
        assert mask is not None
        hs, encoder_mem = ctx.transformer(ctx.input_proj(src), mask, ctx.query_embed.weight, pos[-1])

        outputs_class = ctx.class_embed(hs)
        outputs_coord = ctx.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out, hs, features, encoder_mem

    @staticmethod
    def advanced_forward(ctx, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = ctx.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(ctx.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if ctx.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, ctx.num_feature_levels):
                if l == _len_srcs:
                    src = ctx.input_proj[l](features[-1].tensors)
                else:
                    src = ctx.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = ctx.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not ctx.two_stage or ctx.mixed_selection:
            query_embeds = ctx.query_embed.weight[0: ctx.num_queries, :]

        self_attn_mask = (
            torch.zeros([ctx.num_queries, ctx.num_queries, ]).bool().to(src.device)
        )
        self_attn_mask[ctx.num_queries_one2one:, 0: ctx.num_queries_one2one, ] = True
        self_attn_mask[0: ctx.num_queries_one2one, ctx.num_queries_one2one:, ] = True

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = ctx.transformer(srcs, masks, pos, query_embeds, self_attn_mask)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = ctx.class_embed[lvl](hs[lvl])
            tmp = ctx.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes_one2one.append(outputs_class[:, 0: ctx.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, ctx.num_queries_one2one:])
            outputs_coords_one2one.append(outputs_coord[:, 0: ctx.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, ctx.num_queries_one2one:])
        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
        }

        if ctx.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return out, hs, features

    def forward(self,
                images: List[Tensor],
                poses: [List[dict]],
                targets: Optional[List[dict]] = None
                ) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (M, 2) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `size`: torch.Tensor
                (2,) Image height and width
            `x`: torch.Tensor
                (M,) Index tensor corresponding to the duplications of human-objet pairs. Each
                pair was duplicated once for each valid action.
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        image_sizes = torch.as_tensor([im.size()[-2:] for im in images], device=images[0].device)

        with torch.no_grad():
            # step 1 图像输入到目标检测器中
            # 输出Encoder和Decoder的目标检测结果，Decoder特征，backbone不同尺寸的特征
            results, hs, features, encoder_mem = self.od_forward(self.detector, images)
            # step 2 将坐标信息从相对值转换为绝对值（x1,y1,x2,y2）
            results = self.postprocessor(results, image_sizes)
        # 生成经过筛选后的样本（边框、得分、类别、隐状态），并约束样本数量
        region_props = prepare_region_proposals(
            results, hs[-1], image_sizes,
            box_score_thresh=self.box_score_thresh,
            human_idx=self.human_idx,
            min_instances=self.min_instances,
            max_instances=self.max_instances
        )
        boxes = [r['boxes'] for r in region_props]
        # Produce human-object pairs.
        # human-object query embed [N,384], pair的box index对 [N,2], pair中human和object的不同动词得分 [N,2,117]
        # pair中object的类别[N], pair的位置编码信息,cat human和object的结果,centre:[N,1,512],box:[N,1,1024]
        (
            ho_queries,
            paired_inds, prior_scores,
            object_types, positional_embeds
        ) = self.ho_matcher(region_props, poses, image_sizes)

        # C_pe：box的中心点xy的位置编码（sinusoidal）
        # Wh_pe：box的长宽的位置编码（sinusoidal）
        #
        # Box_pe：concat(C_pe, Wh_pe)
        # C_pe: C_pe * box embedd
        #
        # Positional_embeds: （q_pos）
        # Centre: human和object的C_pe拼接
        # Box：human和object的box_pe的拼接

        # Compute keys/values for triplet decoder.
        # [4, 26, 23, 256]   # 4, 2048, 26, 23
        # memory = encoder_mem.permute(0, 2, 3, 1)
        # memory = features[-1].tensors
        # B, H, W, C[4, 26, 23]
        # mask = features[-1].mask
        # memory = self.fh(memory.permute(0, 2, 3, 1))
        memory, mask = self.feature_head(features)  # feature: backbone中不同尺度的特征图
        b, h, w, c = memory.shape  # backbone中的特征通过feature head 进一步提炼
        memory = memory.reshape(b, h * w, c)
        kv_p_m = mask.reshape(-1, 1, h * w)
        k_pos = self.kv_pe(NestedTensor(memory, mask)).permute(0, 2, 3, 1).reshape(b, h * w, 1, c)  # 特征的位置编码
        # Enhance visual context with triplet decoder.
        query_embeds = []
        for i, (ho_q, mem) in enumerate(zip(ho_queries, memory)):
            query_embeds.append(self.decoder(
                ho_q.unsqueeze(1),  # (n, 1, q_dim)
                mem.unsqueeze(1),  # (hw, 1, kv_dim)
                kv_padding_mask=kv_p_m[i],  # (1, hw)
                q_pos=positional_embeds[i],  # centre: (n, 1, 2*kv_dim), box: (n, 1, 4*kv_dim)
                k_pos=k_pos[i]  # (hw, 1, kv_dim)
            ).squeeze(dim=2))
        # Concatenate queries from all images in the same batch.
        query_embeds = torch.cat(query_embeds, dim=1)  # (num_docoder 2 , num_query 726 , 384)
        logits = self.binary_classifier(query_embeds)

        if self.training:
            labels = associate_with_ground_truth(
                boxes, paired_inds, targets, self.num_verbs
            )  # [query_num, verbs]，数值为1表示，有效query对应的HOI类别
            # batch内计算损失
            # [num_decoder, batch_query_num, verbs], [batch, query_num, 2, verbs], [batch_query_num, verbs]
            # batch级别下HOI pair的类别概率；每个HOI pair的human和object先验得分；每个HOI预测结果与target配对之后的类别标签，1为正确配对
            cls_loss = self.compute_classification_loss(logits, prior_scores, labels)
            loss_dict = dict(cls_loss=cls_loss)
            return loss_dict

        detections = self.postprocessing(
            boxes, paired_inds, object_types,
            logits[-1], prior_scores, image_sizes
        )
        # boxes -> human and object boxes: B,N,4
        # paired_inds -> human 和 object配对的索引集合：B,num_query,2
        # object_types -> 配对结果中，object的类别信息
        # logits[-1] -> 最后一层模型预测的动词结果: B*num_query, 117
        # prior_scores -> human 和object的动词类别先验得分：B, num_query, 2, 117
        # image_sizes -> 图像的大小：【宽，高】，B，2
        return detections


def build_detector(args, obj_to_verb):
    if args.detector == "base":
        detr, _, postprocessors = build_base_detr(args)
    elif args.detector == "advanced":
        detr, _, postprocessors = build_advanced_detr(args)

    # 加载预训练的目标检测器权重
    if os.path.exists(args.pretrained):
        if dist.is_initialized():
            print(f"Rank {dist.get_rank()}: Load weights for the object detector from {args.pretrained}")
            logging.info(f"Rank {dist.get_rank()}: Load weights for the object detector from {args.pretrained}")
        else:
            print(f"Load weights for the object detector from {args.pretrained}")
            logging.info(f"Load weights for the object detector from {args.pretrained}")
        # resnet50
        # detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])
        # resnet101
        detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])

    ho_matcher = HumanObjectMatcher(
        repr_size=args.repr_dim,
        num_verbs=args.num_verbs,
        obj_to_verb=obj_to_verb,
        dropout=args.dropout
    )
    decoder_layer = TransformerDecoderLayer(
        q_dim=args.repr_dim, kv_dim=args.hidden_dim,
        ffn_interm_dim=args.repr_dim * 4,
        num_heads=args.nheads, dropout=args.dropout
    )
    triplet_decoder = TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=args.triplet_dec_layers
    )
    return_layer = {"C5": -1, "C4": -2, "C3": -3}[args.kv_src]
    if isinstance(detr.backbone.num_channels, list):
        num_channels = detr.backbone.num_channels[-1]
    else:
        num_channels = detr.backbone.num_channels
    feature_head = FeatureHead(
        args.hidden_dim, num_channels,
        return_layer, args.triplet_enc_layers
    )
    model = PViC(
        (detr, args.detector), postprocessors['bbox'],
        feature_head=feature_head,
        ho_matcher=ho_matcher,
        triplet_decoder=triplet_decoder,
        num_verbs=args.num_verbs,
        repr_size=args.repr_dim,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        raw_lambda=args.raw_lambda,
    )
    return model
