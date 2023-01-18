"""This module contains the Tradesy dataset as a PyTorch Dataset class."""

from __future__ import annotations

import gzip
import json
import struct
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

TradesySample = Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]


class TradesyDataset(Dataset[TradesySample]):
    """This class represents the Tradesy dataset as a PyTorch Dataset.

    The dataset handles the interactions of the Tradesy dataset. It returns
    interactions as a tuple of user U, a positive item I_p (consumed by U),
    and a negative item I_n (not consumed by U).

    The class contains methods to load both interactions and embeddings as
    published at https://cseweb.ucsd.edu/~jmcauley/datasets.html#bartering_data."""

    def __init__(self, interactions: pd.DataFrame, random_seed: Optional[int] = None):
        self.__rng_seed = random_seed
        self.__rng = np.random.default_rng(seed=self.__rng_seed)
        self.interactions = interactions[["uid", "iid"]]

    @cached_property
    def n_users(self) -> int:
        return self.interactions["uid"].nunique()

    @cached_property
    def n_items(self) -> int:
        return self.interactions["iid"].nunique()

    def split(
        self,
    ) -> Tuple[Subset[TradesySample], Subset[TradesySample], Subset[TradesySample]]:
        train_indices: List[int] = []
        valid_indices: List[int] = []
        eval_indices: List[int] = []
        for user, df in self.interactions.groupby("uid"):
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
    ) -> TradesySample:
        idx_seq: npt.NDArray[np.int_]
        if isinstance(idx, torch.Tensor):
            idx_seq = idx.numpy()
        elif isinstance(idx, int):
            idx_seq = np.array([idx])
        else:
            idx_seq = idx
        uid = self.interactions.iloc[idx_seq, 0].to_numpy()
        iid = self.interactions.iloc[idx_seq, 1].to_numpy()
        jid = np.empty_like(iid)
        for i, u in enumerate(uid):
            consumed_items = set(
                self.interactions.loc[self.interactions["uid"] == u, "iid"].to_numpy()
            )
            negative_item = self.__rng.integers(self.n_items, size=1)[0]
            while negative_item in consumed_items:
                negative_item = self.__rng.integers(self.n_items, size=1)[0]
            jid[i] = negative_item
        return uid, iid, jid

    def get_user_items(
        self, user: int, indices: Optional[List[int]] = None
    ) -> npt.NDArray[np.int_]:
        if indices is not None:
            interactions = self.interactions.iloc[indices]
        else:
            interactions = self.interactions
        return interactions.loc[interactions["uid"] == user, "iid"].to_numpy()

    def __len__(self) -> int:
        return len(self.interactions)

    @classmethod
    def from_files(
        cls: Type[TradesyDataset],
        interactions_path: Path,
        features_path: Path,
        random_seed: Optional[int] = None,
    ) -> Tuple[TradesyDataset, npt.NDArray[np.float64]]:
        """Creates a TradesyDataset instance and loads the images features from files"""
        interactions = []
        with gzip.open(interactions_path, "rt") as file:
            line = file.readline()
            while line:
                line = line.replace("'", '"')
                line_data = json.loads(line)
                # VBPR: "purchase histories and ‘thumbs-up’,
                # which we use together as positive feedback"
                # NOTE: Authors did not remove duplicated interactions
                line_data = {
                    "uid": line_data["uid"],
                    "iid": list(
                        set(line_data["lists"]["bought"] + line_data["lists"]["want"])
                    ),
                }
                if len(line_data["iid"]) >= 5:
                    interactions.append(line_data)
                line = file.readline()
        df_interactions = pd.DataFrame(interactions)
        df_interactions = df_interactions.explode("iid", ignore_index=True).astype(int)
        users_id_to_idx = {
            uid: uidx
            for uidx, uid in enumerate(sorted(df_interactions["uid"].unique()))
        }
        items_id_to_idx = {
            iid: iidx
            for iidx, iid in enumerate(sorted(df_interactions["iid"].unique()))
        }
        df_interactions["uid"] = df_interactions["uid"].map(users_id_to_idx)
        df_interactions["iid"] = df_interactions["iid"].map(items_id_to_idx)

        item_features = np.empty((len(items_id_to_idx), 4096))
        with open(features_path, "rb") as file:
            while True:
                item_id_bytes = file.read(10).strip()
                if not item_id_bytes:
                    break
                item_id = int(item_id_bytes)
                features = struct.unpack("4096f", file.read(4 * 4096))
                if item_id not in items_id_to_idx:
                    continue
                item_idx = items_id_to_idx[item_id]
                item_features[item_idx] = np.array(features)

        return cls(df_interactions, random_seed=random_seed), item_features
