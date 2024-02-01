# -*- coding:utf-8 -*-
# @file  :main.py.py
# @time  :2024/02/01
# @author:ylZhang

import torch
from torch import optim
from vbpr import VBPR, Trainer
# from vbpr.datasets import TradesyDataset
from vbpr.datasets import AmazonDataset


dataset, features = AmazonDataset.from_files(
    "data/amazon-book/user_items_all.json",
    "data/amazon-book/item_text_feat.npy",
)
print("用户数：", dataset.n_users)
print("物品数：", dataset.n_items)

model = VBPR(
    dataset.n_users,
    dataset.n_items,
    torch.tensor(features, dtype=torch.float32),
    dim_gamma=20,
    dim_theta=20,
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

optimizer = optim.Adam(model.parameters(), lr=5e-04)

trainer = Trainer(model, optimizer)
trainer.fit(dataset, n_epochs=10)



