# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/11 20:46
@Auther ： Zzou
@File ：json_para.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""
import json
import os.path
import re
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np

train_json_file = "../hicodet/instances_train2015.json"
list_hoi_file = "../hicodet/hico_20160224_det/hico_list_hoi.txt"

# # 显示每个verb，对应的HOI列表
# with open(list_hoi_file, "r") as f:
#     data = f.readlines()
# hoi_info = list()
# for item in data[2:]:
#     split_data = re.split(r"\s+", item.strip())
#     hoi_info.append(split_data)
# hoi_info = np.array(hoi_info)
# verbs = set(hoi_info[:, 2])
# verb_HOI_dict = dict()
# for verb in verbs:
#     verb_HOI_dict[verb] = list()
#     print(verb)
#     for item in hoi_info:
#         if item[2] == verb:
#             print(item)
#             verb_HOI_dict[verb].append(item)
# #
# # 显示某个动词类别下，所有的图片
# key = "throw"
# HOI_info = verb_HOI_dict[key]
# selected_HOI = [int(item[0]) for item in HOI_info]
#
# with open(train_json_file, "r") as f1:
#     anno = json.load(f1)
# annotation = anno["annotation"]
# filenames = anno["filenames"]
# for i, item in enumerate(annotation):
#     item_hoi = item["hoi"]
#     for ih in item_hoi:
#         if ih in selected_HOI:
#             print(ih, filenames[i])

# # 每个关键点的缺失数量统计
# mis_results = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# total_num = 0
# pose_file = "../hicodet/HICODET_train_pose.json"
# with open(pose_file, "r") as f:
#     data = json.load(f)
# for key in data.keys():
#     anno = data[key]
#     subsets = anno["subset"]
#     for subset in subsets:
#         total_num += 1
#         new = [1 if value == -1.0 else 0 for value in subset]
#         mis_results = [x + y for x, y in zip(mis_results, new)]
# print(mis_results)
# print(total_num)
# pose_file = "../hicodet/HICODET_test_pose.json"
# with open(pose_file, "r") as f:
#     data = json.load(f)
# for key in data.keys():
#     anno = data[key]
#     subsets = anno["subset"]
#     for subset in subsets:
#         total_num += 1
#         new = [1 if value == -1.0 else 0 for value in subset]
#         mis_results = [x + y for x, y in zip(mis_results, new)]
# print(mis_results)
# print(total_num)

# 图片HOI的可视化
img_folder = "E:/HOI/datasets/hico_20160224_det/images/train2015"
img_name = "HICO_train2015_00003783.jpg"
img_path = os.path.join(img_folder, img_name)
image = mpimg.imread(img_path)
with open(train_json_file, "r") as f1:
    anno = json.load(f1)
annotation = anno["annotation"]
filenames = anno["filenames"]
index = filenames.index(img_name)
annn_info = annotation[index]
print(annn_info)
human_box = [[(item[0], item[1]), (item[2], item[3])] for item in annn_info["boxes_h"]]
object_box = [[(item[0], item[1]), (item[2], item[3])] for item in annn_info["boxes_o"]]
fig, ax = plt.subplots()
ax.imshow(image)
for i in range(len(human_box)):
    box_1 = human_box[i]
    box_2 = object_box[i]
    color = "#" + "%06x" % random.randint(0, 0xFFFFFF)

    left_top_1 = box_1[0]
    width_1 = box_1[1][0] - box_1[0][0]
    height_1 = box_1[1][1] - box_1[0][1]
    rect_1 = patches.Rectangle(left_top_1, width_1, height_1, linewidth=3, edgecolor=color, facecolor='none')
    ax.add_patch(rect_1)

    left_top_2 = box_2[0]
    width_2 = box_2[1][0] - box_2[0][0]
    height_2 = box_2[1][1] - box_2[0][1]
    rect_2 = patches.Rectangle(left_top_2, width_2, height_2, linewidth=3, edgecolor=color, facecolor='none')
    ax.add_patch(rect_2)
plt.show()
