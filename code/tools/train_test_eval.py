# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/30 15:08
@Auther ： Zzou
@File ：train_test_eval.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""
import torch

n = 10
nh = 3

x, y = torch.meshgrid(
    torch.arange(n, ),
    torch.arange(n, )
)
x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < nh)).unbind(1)
print(x_keep)
print(y_keep)