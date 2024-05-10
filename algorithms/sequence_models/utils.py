import torch
import numpy as np
import abc
import time
import datetime
import os
import csv
from typing import Tuple
from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from settings import LOG_DIR
from data.trajectory import TrajectoryDataset


class Logger:
    def __init__(self, name, check_point_interval=50):
        self.name = name
        self.start_time = datetime.datetime.now().replace(microsecond=0)
        self.best_score = -np.inf
        self.iters = 0
        self.num_updates_per_iter = 0
        self.previous_csv_extra_keys = None

        self.check_point_interval = check_point_interval

        self.log_dir = os.path.join(LOG_DIR, "dt_runs")

        self.save_model_path = ""
        self.save_best_model_path = ""

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.csv_writer = None
        self.pbar = None
        self.is_started = False

    def start(self, iterations, update_per_iter):
        self.start_time = datetime.datetime.now().replace(microsecond=0)
        self.best_score = -np.inf
        self.iters = 0
        self.num_updates_per_iter = update_per_iter
        self.previous_csv_extra_keys = None

        prefix = "dt_"
        start_time_str = self.start_time.strftime("%y-%m-%d-%H-%M-%S")
        save_model_name = prefix + self.name + "_model_" + start_time_str + ".pt"
        self.save_model_path = os.path.join(self.log_dir, save_model_name)
        self.save_best_model_path = self.save_model_path[:-3] + "_best.pt"
        log_csv_name = prefix + "_log_" + start_time_str + ".csv"
        log_csv_path = os.path.join(self.log_dir, log_csv_name)
        self.csv_writer = csv.writer(open(log_csv_path, 'a', 1))

        self.pbar = None
        self.is_started = True
        self.pbar = tqdm(total=iterations)

    def finish(self):
        print("=" * 60)
        print("finished training!")
        print("=" * 60)
        end_time = datetime.datetime.now().replace(microsecond=0)
        time_elapsed = str(end_time - self.start_time)
        end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")

        start_time_str = self.start_time.strftime("%y-%m-%d-%H-%M-%S")

        print("started training at: " + start_time_str)
        print("finished training at: " + end_time_str)
        print("total training time: " + time_elapsed)
        print("best score: " + format(self.best_score, ".5f"))
        print("saved max d4rl score model at: " + self.save_best_model_path)
        print("saved last updated model at: " + self.save_model_path)
        print("=" * 60)

        self.is_started = False
        # I don't know why but closing this leads to runtime error
        # self.csv_writer.close()

    # todo later make it generic so that we can log whatever
    def log(self, model, loss, eval_avg_reward, important=set(), **kwargs):
        if not self.is_started:
            raise Exception("call .start() first")

        if self.previous_csv_extra_keys is None:
            self.previous_csv_extra_keys = list(kwargs.keys())
            csv_header = (["duration", "num_updates", "action_loss", "eval_avg_reward", "best_score", *kwargs.keys()])
            self.csv_writer.writerow(csv_header)
        elif set(self.previous_csv_extra_keys) != set(kwargs.keys()):
            raise Exception(
                f"expected {set(self.previous_csv_extra_keys)} keys but passed {set(kwargs.keys())}. Maybe call finish?")

        self.iters += 1
        time_elapsed = str(datetime.datetime.now().replace(microsecond=0) - self.start_time)
        total_updates = self.iters * self.num_updates_per_iter

        log_str = '\n'.join([
            "=" * 60,
            "time elapsed: " + time_elapsed,
            "num of updates: " + str(total_updates),
            "loss: " + format(loss, ".5f"),
            "eval avg reward: " + format(eval_avg_reward, ".5f"),
            "best score: " + format(self.best_score, ".5f"),
            *[key + " " + format(value, ".5f") for key, value in kwargs.items()]
        ])

        log_data = [time_elapsed, total_updates, loss,
                    eval_avg_reward, self.best_score] + [kwargs[key] for key in self.previous_csv_extra_keys]
        self.csv_writer.writerow(log_data)

        if self.iters % self.check_point_interval == 0:
            print("saving current model at: " + self.save_model_path)
            torch.save(model.state_dict(), self.save_model_path)

        if eval_avg_reward >= self.best_score:
            print('achieved average reward: ', eval_avg_reward)
            print("saving max score model at: " + self.save_best_model_path)

            torch.save(model.state_dict(), self.save_best_model_path)
            self.best_score = eval_avg_reward

        self.pbar.set_description(' '.join([
            f'Loss={loss}',
            f'Best_Score={self.best_score}',
            *[f'{key}={value:.5f}' for key, value in kwargs.items() if key in important]
        ]))

        self.pbar.update(1)

        print(log_str)

