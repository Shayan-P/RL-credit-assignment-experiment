import dataclasses
import json
import os
from abc import abstractmethod
from typing import List, Tuple

import numpy as np
import datetime
import gymnasium as gym
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

from algorithms.random_policy import BasePolicy
from algorithms.sequence_models.config import TrainConfig
from algorithms.sequence_models.decision_transformer.decision_transformer import DecisionTransformer
from algorithms.sequence_models.decision_transformer.trainer import TrainerDT
from algorithms.sequence_models.trainer import TrainCallback
from data.trajectory import TrajectoryDataset, LimitedContextWrapper, LimitedContextWrapperSortedLength
from final_experiments.callbacks import LogSaveModelEvaluateCallback
from settings import LOG_DIR, cd_mkdir, FIGURES_DIR


class Experiment:
    """
    trains a model on an env
    the reason we use a class and not a function is that we
    don't want to keep having access to the attributes of the class
    """

    def __init__(self,
                 model_name,
                 model,
                 env_name,
                 env,
                 experiment_name,
                 traj_dataset: TrajectoryDataset,
                 dataset_name,
                 config: TrainConfig,
                 device,
                 eval_policies_and_names: List[Tuple[BasePolicy, str]],
                 final_eval_policies: List[BasePolicy] = []
                 ):
        """
            should we save dataset anytime we are running the experiment?
        """

        self.model_name = model_name
        self.env_name = env_name
        self.experiment_name = experiment_name

        self.log_dir = LOG_DIR
        self.log_dir = cd_mkdir(self.log_dir, env_name)
        self.log_dir = cd_mkdir(self.log_dir, model_name)
        self.log_dir = cd_mkdir(self.log_dir, experiment_name)
        self.experiment_name = f'{experiment_name}_{dataset_name}'
        self.full_experiment_name = f'{env_name}_{model_name}_{self.experiment_name}'
        self.fig_dir = cd_mkdir(self.log_dir, f'{self.experiment_name}_figures')

        self.config_file_path = os.path.join(self.log_dir, f'{self.experiment_name}_config.json')
        self.dataset_path = os.path.join(self.log_dir, f'{self.experiment_name}_dataset.pkl')
        self.model_save_path_prefix = os.path.join(self.log_dir,
                                                   f'{self.experiment_name}_model')  # this is passed to ModelLogger
        self.log_save_path_prefix = os.path.join(self.log_dir,
                                                 f'{self.experiment_name}_log')  # this is passed to ModelLogger
        self.concat_report = pd.DataFrame()

        # save config
        with open(self.config_file_path, 'w') as f:
            json.dump(dataclasses.asdict(config), f, indent=4)
        # save dataset
        traj_dataset.save_to_path(self.dataset_path)
        # todo later write the code to load experiment

        ##############################################################
        # this is all what we need to construct the rest of the stuff
        self.traj_dataset = traj_dataset
        self.model = model
        self.env = env
        self.config = config
        self.device = device
        ##############################################################
        # keep the objects we need for training
        # todo rtg_range_check and rtg_main are used for plotting and monitoring
        # todo is the rtg_range_check correct?

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wt_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps + 1) / config.warmup_steps, 1)
        )
        loss_fn = nn.MSELoss(reduction='mean')

        # todo later customize the loss function
        print(self.model_name, ':', "number of parameters", sum(np.prod(param.shape) for param in model.parameters()))
        ###############################################################
        # todo later change LimitedContextWrapper to something more general for S4 and DT
        # todo later we can change this trainer
        self.dataset = LimitedContextWrapper(traj_dataset, context_len=config.context_len)
        # self.dataset = LimitedContextWrapperSortedLength(traj_dataset, context_len=config.context_len, batch_size=config.batch_size)
        self.trainer = TrainerDT(name=experiment_name, model=self.model,
                                 optimizer=optimizer, loss_fn=loss_fn,
                                 dataset=self.dataset,
                                 device=device, config=config, scheduler=scheduler)

        self.custom_callback = LogSaveModelEvaluateCallback(
            env=self.env,
            policies_and_names=eval_policies_and_names,
            trainer=self.trainer,
            config=config,
            model_save_path_prefix=self.model_save_path_prefix,
            log_save_path_prefix=self.log_save_path_prefix,
            verbose=True
        )
        self.trainer.register_callback(self.custom_callback)
        self.eval_policies_and_names = eval_policies_and_names
        self.final_eval_policies = final_eval_policies

    def plot_loss(self, report):
        return report.sort_values(by=['train/iteration', 'train/update_idx']).reset_index()['train/loss'].plot()

    def save_fig(self, name):
        plt.savefig(os.path.join(FIGURES_DIR, f'{self.full_experiment_name}_{name}.png'))
        plt.savefig(os.path.join(self.fig_dir, f'{self.full_experiment_name}_{name}.png'))

    def train_for(self, epochs=None):
        report = self.trainer.train(epochs=epochs)
        self.concat_report = pd.concat([self.concat_report, report])
        return report
