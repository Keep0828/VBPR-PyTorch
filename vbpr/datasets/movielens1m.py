"""This module contains the Tradesy dataset as a PyTorch Dataset class."""

from __future__ import annotations

import gzip
import json
import os.path
import struct
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

import csv


MovielensSample = Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]


class MovielensDataset(Dataset[MovielensSample]):
    """This class represents the Movielens1m dataset as a PyTorch Dataset.

    The dataset handles the interactions of the Tradesy dataset. It returns
    interactions as a tuple of user U, a positive item I_p (consumed by U),
    and a negative item I_n (not consumed by U).

    The class contains methods to load both interactions and embeddings as
    published at https://cseweb.ucsd.edu/~jmcauley/datasets.html#bartering_data."""

    def __init__(self, interactions: pd.DataFrame, random_seed: Optional[int] = None):
        self.__rng_seed = random_seed
        self.__rng = np.random.default_rng(seed=self.__rng_seed)
        interactions = interactions.sort_values(["uid", "iid"])
        self.uid: npt.NDArray[np.int_] = interactions["uid"].to_numpy(copy=True)
        self.iid: npt.NDArray[np.int_] = interactions["iid"].to_numpy(copy=True)
        self.interactions = interactions.set_index(["uid", "iid"])

    @cached_property
    def n_users(self) -> int:
        return self.interactions.index.get_level_values(0).nunique()

    @cached_property
    def n_items(self) -> int:
        return self.interactions.index.get_level_values(1).nunique()

    def split(
        self,
    ) -> Tuple[Subset[MovielensSample], Subset[MovielensSample], Subset[MovielensSample]]:
        train_indices: List[int] = []
        valid_indices: List[int] = []
        eval_indices: List[int] = []
        for user, df in self.interactions.reset_index().groupby("uid"):
            df = df.sample(frac=1, random_state=self.__rng_seed)
            train_indices += df.index[:-2].tolist()
            valid_indices += df.index[-2:-1].tolist()
            eval_indices += df.index[-1:].tolist()
        return (
            Subset(self, train_indices),
            Subset(self, valid_indices),
            Subset(self, eval_indices),
        )

    def __getitem__(
        self, idx: Union[npt.NDArray[np.int_], torch.Tensor]
    ) -> MovielensSample:
        if isinstance(idx, int):
            uid = np.array([self.uid[idx]])
            iid = np.array([self.iid[idx]])
        else:
            idx_seq: npt.NDArray[np.int_]
            if isinstance(idx, torch.Tensor):
                idx_seq = idx.numpy()
            else:
                idx_seq = idx
            uid = self.uid[idx_seq]
            iid = self.iid[idx_seq]
        jid = np.empty_like(iid)
        for i, u in enumerate(uid):
            negative_item = self.__rng.integers(self.n_items, size=1)[0]
            while (u, negative_item) in self.interactions.index:
                negative_item = self.__rng.integers(self.n_items, size=1)[0]
            jid[i] = negative_item
        return uid, iid, jid

    def get_item_users(self, item: int) -> npt.NDArray[np.int_]:
        item_selector = cast(npt.NDArray[np.bool_], self.iid == item)
        return self.uid[item_selector]

    def get_user_items(self, user: int) -> npt.NDArray[np.int_]:
        user_selector = cast(npt.NDArray[np.bool_], self.uid == user)
        return self.iid[user_selector]

    def __len__(self) -> int:
        return len(self.interactions)

    @classmethod
    def from_files(
        cls: Type[MovielensDataset],
        interactions_path: Path,
        features_path: Path,
        random_seed: Optional[int] = None,
    ) -> Tuple[MovielensDataset, npt.NDArray[np.float64]]:
        """Creates a MovielensDataset instance and loads the images features from files"""
        interactions = []

        def read_user_items_json_only(ui_json):
            with open(ui_json, 'r') as f:
                data = json.load(f)
                return data

        ui_dict = read_user_items_json_only(interactions_path)
        for user, items in ui_dict.items():
            u_id = int(user)
            line_data = {
                "uid": u_id,
                "iid": items
            }
            # if len(line_data['iid']) >=5:
            interactions.append(line_data)

        df_interactions = pd.DataFrame(interactions)
        df_interactions = df_interactions.explode("iid", ignore_index=True).astype(int)

        users_id_to_idx = {
            uid: uidx
            for uidx, uid in enumerate(sorted(df_interactions["uid"].unique()))
        }
        unique_map = {
            iid: iidx
            for iidx, iid in enumerate(sorted(df_interactions["iid"].unique()))
        }  # ui加载出来的物品数3685个。
        # print("ui加载出来的map", unique_map)
        # print("ui加载出来的物品数", len(unique_map))

        with open(os.path.join(os.path.dirname(interactions_path), 'ml-1m-textID-map.json'), 'r') as f:
            raw_id_to_feature_idx_map = {
                int(iid): iidx
                for iid, iidx in json.load(f).items()
            }

        # print("自己加载的map：", raw_id_to_feature_idx_map)
        # print("自己加载的map物品数：", len(raw_id_to_feature_idx_map))

        items_id_to_idx = raw_id_to_feature_idx_map  # 还是用作者代码生成的unique_map作为items idx的map，但是加载feature的时候按照feature_idx列表加载进来

        df_interactions["uid"] = df_interactions["uid"].map(users_id_to_idx)
        df_interactions["iid"] = df_interactions["iid"].map(items_id_to_idx)

        item_features = np.load(features_path)  # 如果本来就是按照我的feature的map，那直接load就好了。

        return cls(df_interactions, random_seed=random_seed), item_features
