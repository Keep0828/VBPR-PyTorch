# -*- coding:utf-8 -*-
# @file  :main.py.py
# @time  :2024/02/01
# @author:ylZhang
import os
import torch
from torch import optim
# from vbpr import VBPR, Trainer
from vbpr import BPR, Trainer

from vbpr.datasets import AmazonDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dataset, _ = AmazonDataset.from_files(
    "data/amazon-book/user_items_all.json",
    "data/amazon-book/item_text_feat.npy",
)
print("用户数：", dataset.n_users)
print("物品数：", dataset.n_items)

model = BPR(
    dataset.n_users,
    dataset.n_items,
    dim_gamma=64
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

optimizer = optim.Adam(model.parameters(), lr=5e-04)

trainer = Trainer(model, optimizer)
trainer.fit(dataset, n_epochs=500)


