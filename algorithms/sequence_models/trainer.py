from typing import List

import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset

from algorithms.sequence_models.config import TrainConfig
from algorithms.sequence_models.utils import Logger
from abc import abstractmethod


class TrainCallback:
    @abstractmethod
    def start_training(self, model, iterations):
        pass

    @abstractmethod
    def epoch(self, model, report):
        pass

    @abstractmethod
    def finish_training(self, model, report):
        pass


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
        self.callbacks: List[TrainCallback] = []

        self.traj_data_loader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)
        self.logger = Logger(name=name)

    def register_callback(self, callback: TrainCallback):
        self.callbacks.append(callback)

    def train(self, epochs=None) -> pd.DataFrame:
        if epochs is None:
            epochs = self.config.max_train_iters

        # self.logger.start(iterations=self.config.max_train_iters, update_per_iter=self.config.num_updates_per_iter)
        for callback in self.callbacks:
            callback.start_training(model=self.model, iterations=epochs)

        df = []
        for i_train_iter in range(epochs):
            df_iter = self.train_iteration()
            # for callback in self.callbacks:
            #     callback(df=df_iter, model=self.model, iterations)
            for callback in self.callbacks:
                callback.epoch(model=self.model, report=df_iter)
            df_iter['train/iteration'] = i_train_iter
            df.append(df_iter)
        df = pd.concat(df)

        # self.logger.finish()
        for callback in self.callbacks:
            callback.finish_training(model=self.model, report=df)

        return df

    @abstractmethod
    def train_iteration(self) -> pd.DataFrame:
        pass
