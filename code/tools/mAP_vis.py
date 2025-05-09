# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/15 11:16
@Auther ： Zzou
@File ：mAP_vis.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""
import os.path
import matplotlib.pyplot as plt
import numpy as np

def HICO_results_extractor(file_path: str):
    with open(file_path, "r") as file:
        lines = file.readlines()
    filtered_lines = [line.strip() for line in lines if "mAP" in line]
    Epoch, Full, Rare, N_rare = [], [], [], []
    for line in filtered_lines:
        items = line.split(": ")
        r0 = items[0].split(" ")[1]
        r1 = float(items[1].split(",")[0])
        r2 = float(items[2].split(",")[0])
        r3 = float(items[3][:-1])
        Epoch.append(r0)
        Full.append(r1)
        Rare.append(r2)
        N_rare.append(r3)
    return Epoch, Full, Rare, N_rare


def draw_mAP(epoch, full, rare, n_rare, save_file_path=None):
    fig, ax = plt.subplots()
    ax.plot(epoch, full, label='Full', marker='o')
    ax.plot(epoch, rare, label='Rare', marker='s')
    ax.plot(epoch, n_rare, label='Non-Rare', marker='^')

    ax.legend()
    ax.set_title('mAP of HICO-DET dataset')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')

    ax.set_xticks(np.arange(0, 31, 1.5))  # 刻度间隔为 1
    # plt.xticks(rotation=45)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5)

    if save_file_path:
        plt.savefig(save_file_path, bbox_inches='tight')
    plt.show()


def get_top_three(lst):
    # 使用enumerate()函数获取索引和值的元组列表
    indexed_lst = list(enumerate(lst))

    # 使用sorted()方法按值进行排序，倒序排列
    sorted_lst = sorted(indexed_lst, key=lambda x: x[1], reverse=True)

    # 获取前三个最大值及其索引
    top_three_values = [x[1] for x in sorted_lst[:3]]
    top_three_indices = [x[0] for x in sorted_lst[:3]]

    return top_three_values, top_three_indices


if __name__ == "__main__":
    # hico_result_file_path = "C:/Users/Zzou/Desktop/博士工作/目标-动作识别/My work/方法&实验/实验结果/pose消融/global-local-2.txt"
    # save_path = "C:/Users/Zzou/Desktop/博士工作/目标-动作识别/My work/方法&实验/实验结果/pose消融/train_results-2.pdf"
    # Epoch, Full, Rare, N_rare = HICO_results_extractor(hico_result_file_path)
    # draw_mAP(Epoch, Full, Rare, N_rare, save_path)

    result_file_dir = "C:/Users/Zzou/Desktop/博士工作/目标-动作识别/My work/方法&实验/实验结果/解码器"
    result_file_name = "G2.txt"
    Epoch, Full, Rare, N_rare = HICO_results_extractor(os.path.join(result_file_dir, result_file_name))
    max_values, max_indices = get_top_three(Full)
    selected_Full = [Full[i] for i in max_indices]
    selected_Rare = [Rare[i] for i in max_indices]
    selected_N_rare = [N_rare[i] for i in max_indices]
    print("max_Full is : %.4f, mean_Full is : %.4f" % (max(selected_Full), sum(selected_Full) / len(selected_Full)))
    print("max_Rare is : %.4f, mean_Rare is : %.4f" % (max(selected_Rare), sum(selected_Rare) / len(selected_Rare)))
    print("max_Non-rare is : %.4f, mean_Non-Rare is : %.4f" % (
        max(selected_N_rare), sum(selected_N_rare) / len(selected_N_rare)))
