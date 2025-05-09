# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/20 16:43
@Auther ： Zzou
@File ：reader.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

import torch

# 加载.pth文件
model_path = 'checkpoints/pvic-detr-r50-hicodet.pth'
model = torch.load(model_path)

# 获取参数表
model_state_dict = model.state_dict()

# 打印参数表
for param_name, param_tensor in model_state_dict.items():
    print(param_name)
    print(param_tensor)
    print('----------------------')
