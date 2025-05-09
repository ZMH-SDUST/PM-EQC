# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/18 10:11
@Auther ： Zzou
@File ：pose_hoi_gt_match.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：统计每张图片中HOI的ground truth是否存在pose与之相匹配
"""
import json
import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import math
import matplotlib.patches as patches
import shutil

def HOI_gt_reader(HOI_file_path):
    with open(HOI_file_path, "r") as f:
        data = json.load(f)
    annotation = data["annotation"]  # 38188
    file_names = data["filenames"]
    file_num = len(file_names)
    result_dict = dict()
    for num in range(file_num):
        file_name = file_names[num]
        anno = annotation[num]
        box_h = anno["boxes_h"]
        result_dict[file_name] = box_h
    return result_dict


# list of [x,y]
def point_to_box(points):
    points = np.array(points)
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    return [x_min, y_min, x_max, y_max]


def pose_reader(pose_file_path):
    with open(pose_file_path, "r") as f:
        data = json.load(f)
    result_dict = dict()
    for key in data.keys():  # 37633
        candidate = data[key]["candidate"]
        subset = data[key]["subset"]
        boxes = list()
        if len(subset) == 0:
            result_dict[key] = []
            continue
        for pose in subset:
            indices = [int(num) for i, num in enumerate(pose[:18]) if num != -1]
            pose_points = [candidate[ind][:2] for ind in indices]
            box = point_to_box(pose_points)
            boxes.append(box)
        result_dict[key] = boxes
    return result_dict


def IOU(box1, box2):
    # 获取矩形框1的坐标
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    # 获取矩形框2的坐标
    x1_box2, y1_box2, x2_box2, y2_box2 = box2
    # 计算矩形框1的面积
    area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    # 计算矩形框2的面积
    area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
    # 计算交集部分的坐标
    x1_intersection = max(x1_box1, x1_box2)
    y1_intersection = max(y1_box1, y1_box2)
    x2_intersection = min(x2_box1, x2_box2)
    y2_intersection = min(y2_box1, y2_box2)
    # 计算交集部分的面积
    width_intersection = max(0, x2_intersection - x1_intersection)
    height_intersection = max(0, y2_intersection - y1_intersection)
    area_intersection = width_intersection * height_intersection
    # 计算并集面积
    area_union = area_box1 + area_box2 - area_intersection
    # 计算IoU
    iou = area_intersection / area_union
    return iou


# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas


def human_pose_visual(img_path, human_boxes, pose_info):
    print("")


if __name__ == "__main__":
    # 统计所有图像中的所有human，匹配或者未匹配pose的数量
    HOI_gt_file_path = "../hicodet/instances_train2015.json"
    pose_file_path = "../hicodet/HICODET_train_pose.json"
    raw_img_dir = "E:/HOI/pvic-main/hicodet/hico_20160224_det/images/train2015"
    target_img_dir = "E:/HOI/Missed_pose"
    HOI_data = HOI_gt_reader(HOI_gt_file_path)
    pose_data = pose_reader(pose_file_path)
    file_names = pose_data.keys()
    thres = 0.3
    num_matched = 0
    num_missed = 0
    num_matched_file = 0
    num_missed_file = 0
    for file_name in file_names:
        file_flag = False
        human_box = HOI_data[file_name]
        pose_box = pose_data[file_name]
        for hb in human_box:
            flag = False
            for pb in pose_box:
                iou = IOU(hb, pb)
                if iou >= thres:
                    flag = True
                    file_flag = True
                    break
            if flag:
                num_matched += 1
            else:
                num_missed += 1
        if file_flag:
            num_matched_file += 1
        else:
            num_missed_file += 1
            # source_file_path = os.path.join(raw_img_dir, file_name)
            # target_file_path = os.path.join(target_img_dir, file_name)
            # shutil.copy(source_file_path, target_file_path)
            # print(file_name)
    print(num_matched)  # 0.3 - 73189
    print(num_missed)  # 0.3 - 44682
    print(num_matched_file)
    print(num_missed_file)

    # # 图片中可视化human box 以及 pose info
    # pose_file_path = "../hicodet/HICODET_train_pose.json"
    # HOI_gt_file_path = "../hicodet/instances_train2015.json"
    # save_img_dir = "E:/HOI/Vis"
    # raw_img_dir = "E:/HOI/pvic-main/hicodet/hico_20160224_det/images/train2015"
    # with open(pose_file_path, "r") as f:
    #     data = json.load(f)
    # HOI_data = HOI_gt_reader(HOI_gt_file_path)
    # for image_name in data.keys():
    #     human_boxes = HOI_data[image_name]
    #     anno = data[image_name]
    #     candidate = np.array(anno["candidate"])  # n,4
    #     subset = np.array(anno["subset"])  # m,20
    #     img_path = os.path.join(raw_img_dir, image_name)
    #     oriImg = cv2.imread(img_path)  # B,G,R order
    #     canvas = copy.deepcopy(oriImg)
    #     canvas = draw_bodypose(canvas, candidate, subset)
    #     canvas = canvas[:, :, [2, 1, 0]]  # h,w,3
    #     fig, ax = plt.subplots()
    #     ax.imshow(canvas)
    #     for human_box in human_boxes:
    #         x1, y1, x2, y2 = human_box
    #         x = x1
    #         y = y1
    #         width = x2 - x1
    #         height = y2 - y1
    #         rectangle = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='red', facecolor='none')
    #         ax.add_patch(rectangle)
    #     pose_boxes = []
    #     if len(subset) == 0:
    #         continue
    #     else:
    #         for pose in subset:
    #             indices = [int(num) for i, num in enumerate(pose[:18]) if num != -1]
    #             pose_points = [candidate[ind][:2] for ind in indices]
    #             box = point_to_box(pose_points)
    #             pose_boxes.append(box)
    #     pose_boxes_valid = [False] * len(pose_boxes)
    #     for human_box in human_boxes:
    #         IOU_values = [IOU(human_box, pose_box) for pose_box in pose_boxes]
    #         max_IOU_value = max(IOU_values)
    #         max_IOU_index = IOU_values.index(max_IOU_value)
    #         if max_IOU_value != 0:
    #             pose_boxes_valid[max_IOU_index] = True
    #     for pi, pose_box in enumerate(pose_boxes):
    #         if pose_boxes_valid[pi]:
    #             x1, y1, x2, y2 = pose_box
    #             x = x1
    #             y = y1
    #             width = x2 - x1
    #             height = y2 - y1
    #             rectangle = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='green', facecolor='none')
    #             ax.add_patch(rectangle)
    #     plt.axis('off')
    #     plt.savefig(os.path.join(save_img_dir, image_name))
    #     plt.close()
