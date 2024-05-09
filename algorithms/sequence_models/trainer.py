import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset

from algorithms.sequence_models.config import TrainConfig
from algorithms.sequence_models.utils import Logger
from abc import abstractmethod


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

    @abstractmethod
    def train_iteration(self) -> pd.DataFrame:
        pass
