import pandas as pd
import torch
import numpy as np
import abc
import time
import datetime
import os
import csv
import pandas
from typing import Tuple
from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

from algorithms.sequence_models.config import TrainConfig
from algorithms.sequence_models.utils import Logger
from settings import LOG_DIR
from data.trajectory import TrajectoryDataset


class Trainer:
    def __init__(self, name, model, optimizer, loss_fn, dataset: Dataset,
                 device, config: TrainConfig, scheduler=None):
        # LimitedContextWrapper(traj_dataset, context_len=config.context_len)
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.callbacks = []

        self.traj_data_loader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)
        self.logger = Logger(name=name)

    def register_callback(self, callback):
        self.callbacks.append(callback)

    def train(self) -> pd.DataFrame:
        self.logger.start(iterations=self.config.max_train_iters, update_per_iter=self.config.num_updates_per_iter)

        df = []
        for i_train_iter in range(self.config.max_train_iters):
            df_iter = self.train_iteration()
            df_iter['train/iteration'] = i_train_iter
            df.append(df_iter)
        df = pd.concat(df)

        self.logger.finish()

        return df

    def train_iteration(self) -> pd.DataFrame:
        data_iter = iter(self.traj_data_loader)

        self.model.train()

        df = []
        for update_idx in range(self.config.num_updates_per_iter):
            try:
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(self.traj_data_loader)
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
            res = self.train_step(timesteps, states, actions, returns_to_go, traj_mask)
            res['train/update_idx'] = update_idx
            df.append(res)

        df = pd.DataFrame(df)
        for callback in self.callbacks:
            callback(df=df)
        return df

    def train_step(self, timesteps, states, actions, returns_to_go, traj_mask):
        timesteps = timesteps.to(self.device)  # B x T
        states = states.to(self.device)  # B x T x state_dim
        actions = actions.to(self.device)  # B x T x act_dim
        returns_to_go = returns_to_go.to(self.device).unsqueeze(dim=-1)  # B x T x 1
        traj_mask = traj_mask.to(self.device)  # B x T

        action_target = torch.clone(actions).detach().to(self.device)

        state_preds, action_preds, return_preds = self.model.forward(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go
        )
        # only consider non padded elements
        act_dim = action_preds.shape[-1]
        action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1, ) > 0]
        action_target = action_target.view(-1, act_dim)[traj_mask.view(-1, ) > 0]

        action_loss = self.loss_fn(action_preds, action_target)
        self.optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {"train/loss": action_loss.detach().cpu().item()}
