from typing import Dict

import numpy as np
import torch

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.utils import GaussianNormalizer, dict_apply


class OGBenchDataset(BaseDataset):

    def __init__(
            self,
            dataset: Dict[str, np.ndarray],
            horizon: int = 1,
            only_xy: bool = False,
            with_ball: bool = False, 
            max_path_length: int = 4001,
    ):
        super().__init__()

        observations, actions, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["terminals"])
        if only_xy:
            if with_ball:
                indices_to_keep = [0, 1, 15, 16] 
                observations = observations[:, indices_to_keep]
            else:
                observations = observations[:, :2]

        dones = terminals
        self.normalizers = {
            "state": GaussianNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)

        self.horizon = horizon
        self.only_xy = only_xy
        self.with_ball = with_ball
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

        self.indices = []
        self.seq_obs, self.seq_act = [], []

        self.path_lengths, ptr = [], 0
        path_idx = 0
        for i in range(terminals.shape[0]):

            if i != 0 and ((dones[i - 1] and not dones[i])):

                path_length = i - ptr
                self.path_lengths.append(path_length)

                # 1. agent walks out of the goal
                if path_length < max_path_length:

                    _seq_obs = np.zeros((max_path_length, self.o_dim), dtype=np.float32)
                    _seq_act = np.zeros((max_path_length, self.a_dim), dtype=np.float32)

                    _seq_obs[:i - ptr] = normed_observations[ptr:i]
                    _seq_act[:i - ptr] = actions[ptr:i]

                    # repeat padding
                    _seq_obs[i - ptr:] = normed_observations[i]  # repeat last state
                    _seq_act[i - ptr:] = 0  # repeat zero action

                    self.seq_obs.append(_seq_obs)
                    self.seq_act.append(_seq_act)

                # 2. agent never reaches the goal during the episode
                elif path_length == max_path_length:

                    self.seq_obs.append(normed_observations[ptr:i])
                    self.seq_act.append(actions[ptr:i])

                else:
                    raise ValueError(f"path_length: {path_length} > max_path_length: {max_path_length}")

                max_start = min(self.path_lengths[-1] - 1 - horizon, max_path_length - horizon)
                self.indices += [(path_idx, start, start + horizon) for start in range(max_start + 1)]

                ptr = i
                path_idx += 1

        self.seq_obs = np.array(self.seq_obs)
        self.seq_act = np.array(self.seq_act)

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]

        data = {
            'obs': {
                'state': self.seq_obs[path_idx, start:end]},
            'act': self.seq_act[path_idx, start:end],
        }

        torch_data = dict_apply(data, torch.tensor)

        return torch_data
    

class MultiHorizonOGBenchDataset(BaseDataset):

    def __init__(
            self,
            dataset,
            horizons=(10, 20),
            only_xy: bool = False,
            with_ball: bool = False, 
            max_path_length=4001,
    ):
        super().__init__()

        observations, actions, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["terminals"])
        if only_xy:
            if with_ball:
                indices_to_keep = [0, 1, 15, 16] 
                observations = observations[:, indices_to_keep]
            else:
                observations = observations[:, :2]

        dones = terminals #np.logical_or(timeouts, terminals)
        self.normalizers = {
            "state": GaussianNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)

        self.horizons = horizons
        self.only_xy = only_xy
        self.with_ball = with_ball
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

        self.indices = [[] for _ in range(len(horizons))]
        self.seq_obs, self.seq_act = [], []

        self.path_lengths, ptr = [], 0
        path_idx = 0
        for i in range(terminals.shape[0]):

            if i != 0 and ((dones[i - 1] and not dones[i])):

                path_length = i - ptr
                self.path_lengths.append(path_length)

                # 1. agent walks out of the goal
                if path_length < max_path_length:

                    _seq_obs = np.zeros((max_path_length, self.o_dim), dtype=np.float32)
                    _seq_act = np.zeros((max_path_length, self.a_dim), dtype=np.float32)

                    _seq_obs[:i - ptr] = normed_observations[ptr:i]
                    _seq_act[:i - ptr] = actions[ptr:i]

                    # repeat padding
                    _seq_obs[i - ptr:] = normed_observations[i]  # repeat last state
                    _seq_act[i - ptr:] = 0  # repeat zero action

                    self.seq_obs.append(_seq_obs)
                    self.seq_act.append(_seq_act)

                # 2. agent never reaches the goal during the episode
                elif path_length == max_path_length:

                    self.seq_obs.append(normed_observations[ptr:i])
                    self.seq_act.append(actions[ptr:i])

                else:
                    raise ValueError(f"path_length: {path_length} > max_path_length: {max_path_length}")

                max_starts = [min(self.path_lengths[-1] - 1 - horizon, max_path_length - horizon) for horizon in horizons]
                for k in range(len(horizons)):
                    self.indices[k] += [(path_idx, start, start + horizons[k]) for start in range(max_starts[k] + 1)]

                ptr = i
                path_idx += 1

        self.seq_obs = np.array(self.seq_obs)
        self.seq_act = np.array(self.seq_act)

        self.len_each_horizon = [len(indices) for indices in self.indices]

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return max(self.len_each_horizon)

    def __getitem__(self, idx: int):

        indices = [
            int(self.len_each_horizon[i] * (idx / self.len_each_horizon[-1])) for i in range(len(self.horizons))]

        torch_datas = []

        for i, horizon in enumerate(self.horizons):

            path_idx, start, end = self.indices[i][indices[i]]

            data = {
                'obs': {
                    'state': self.seq_obs[path_idx, start:end]},
                'act': self.seq_act[path_idx, start:end],
            }

            torch_data = dict_apply(data, torch.tensor)

            torch_datas.append({
                "horizon": horizon,
                "data": torch_data,
            })

        return torch_datas
