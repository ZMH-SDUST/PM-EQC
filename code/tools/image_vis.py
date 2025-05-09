# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/21 10:36
@Auther ： Zzou
@File ：image_vis.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""
import json

import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


# 水平翻转
def flip(image):
    flipped_image = F.hflip(image)
    return flipped_image


# 显示函数
def show_images(original, flipped):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(original.numpy().transpose(1, 2, 0))
    axes[1].imshow(flipped.numpy().transpose(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    # # 读取图片
    # image = Image.open("C:/Users/Zzou/Desktop/HICO_train2015_00003783.jpg")
    # # 转换为Tensor
    # transform = transforms.ToTensor()
    # image = transform(image)
    # # 调用
    # flipped_image = flip(image)
    # show_images(image, flipped_image)

    json_file = "../Pose/HICODET_train_pose.json"
    filtered_files = []
    with open(json_file, 'r') as f:
        pose_info = json.load(f)
    for key in pose_info.keys():
        subsets = pose_info[key]['subset']
        l = len(subsets)
        for i in range(0, l):
            if subsets[i][-1] == 18.0:
                filtered_files.append(key)
                break
    import os
    import shutil

    # 源文件夹路径
    source_folder = 'E:/HOI/datasets/hico_20160224_det/images/train2015'
    # 目标文件夹路径
    destination_folder = 'E:/HOI/full_pose'


    # 遍历文件列表
    for file_name in filtered_files:
        # 检查文件是否为图片文件（可根据需要修改文件类型判断条件）
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            # 构建源文件路径
            source_file = os.path.join(source_folder, file_name)
            # 构建目标文件路径
            destination_file = os.path.join(destination_folder, file_name)
            # 拷贝文件到目标文件夹
            shutil.copyfile(source_file, destination_file)