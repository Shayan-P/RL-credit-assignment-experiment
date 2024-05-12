import csv
import datetime

import matplotlib.pyplot as plt
import torch
import pandas as pd
from tqdm.notebook import tqdm

from algorithms.evaluate_policy import evaluate_policy
from algorithms.sequence_models.trainer import TrainCallback
from algorithms.sequence_models.config import TrainConfig


class LogSaveModelEvaluateCallback(TrainCallback):
    def __init__(self, env, policies_and_names, trainer, config, model_save_path_prefix, log_save_path_prefix, verbose=True):
        self.env = env
        self.policies_and_names = policies_and_names
        self.trainer = trainer
        self.config: TrainConfig = config

        self.verbose = verbose

        self.start_time = datetime.datetime.now().replace(microsecond=0)
        self.iters = 0
        self.previous_csv_extra_keys = None

        self.csv_writer = None
        self.pbar = None

        start_time_str = self.start_time.strftime("%y-%m-%d-%H-%M-%S")
        self.save_model_path = model_save_path_prefix + "_model_" + start_time_str + ".pt"
        self.save_model_checkpoint_path = model_save_path_prefix + start_time_str + "checkpoint_"
        self.save_best_model_path = model_save_path_prefix + "_model_" + start_time_str + "_best.pt"
        self.save_csv_path = log_save_path_prefix + "_log_" + start_time_str + ".csv"
        self.save_eval_csv_path = log_save_path_prefix + "_eval_log_" + start_time_str + ".csv"

        self.loss_best_model_saved = float("inf")

        self.eval_results_data = []
        self.losses = []

    def start_training(self, model, iterations):
        self.start_time = datetime.datetime.now().replace(microsecond=0)
        print('training started')

        self.previous_csv_extra_keys = None

        self.csv_writer = csv.writer(open(self.save_csv_path, 'a', 1))
        self.pbar = tqdm(total=iterations)

    def log(self, loss, **kwargs):
        if self.previous_csv_extra_keys is None:
            self.previous_csv_extra_keys = list(kwargs.keys())
            csv_header = (["duration", "num_updates", "action_loss", *kwargs.keys()])
            self.csv_writer.writerow(csv_header)
        elif set(self.previous_csv_extra_keys) != set(kwargs.keys()):
            raise Exception(
                f"expected {set(self.previous_csv_extra_keys)} keys but passed {set(kwargs.keys())}. Maybe call finish?")

        self.iters += 1
        time_elapsed = str(datetime.datetime.now().replace(microsecond=0) - self.start_time)
        total_updates = self.iters * self.config.num_updates_per_iter

        log_str = '\n'.join([
            "=" * 60,
            "time elapsed: " + time_elapsed,
            "num of updates: " + str(total_updates),
            "loss: " + format(loss, ".5f"),
            *[key + " " + format(value, ".5f") for key, value in kwargs.items()]
        ])

        log_data = [time_elapsed, total_updates, loss] + [kwargs[key] for key in self.previous_csv_extra_keys]
        self.csv_writer.writerow(log_data)

        if self.verbose:
            print(log_str)

    def log_model(self, loss, model):
        if self.iters % self.config.model_checkpoint_interval == 0:
            print("saving current model at: " + self.save_model_path)
            torch.save(model.state_dict(), self.save_model_path)
            cpath = f'{self.save_model_checkpoint_path}_{self.iters}.pt'
            print("saving checkpoint model at: " + cpath)
            torch.save(model.state_dict(), cpath)

            if self.loss_best_model_saved > loss:
                print("saving best model at: " + self.save_best_model_path)
                self.loss_best_model_saved = loss
                torch.save(model.state_dict(), self.save_best_model_path)

    def log_eval(self, model):
        if self.iters % self.config.eval_model_interval == 0:
            print("evaluating the model: ")
            for policy, name in self.policies_and_names:
                report = evaluate_policy(policy, self.env, num_eval_ep=self.config.num_eval_ep, max_test_ep_len=self.config.max_eval_ep_len)
                report['train_iter'] = self.iters
                report['policy'] = name
                self.eval_results_data.append(report)

            df = pd.DataFrame(self.eval_results_data)
            df.to_csv(self.save_eval_csv_path)
            print("evaluation saved at:", self.save_eval_csv_path)

        # self.logger.log(
        # 		   model=self.model,
        # 		   loss=df['train/loss'].sum(),
        # 		   eval_avg_reward=results['eval/avg_reward'],
        # 		   eval_avg_ep_len=results['eval/avg_ep_len'],
        # 		   grad_norm=max(torch.norm(param.grad) for param in self.model.parameters() if param.grad is not None),
        # 		   lr=self.optimizer.param_groups[0]['lr'],
        # 		   important={"grad_norm", "lr"})

    # todo kwargs can't be passed. hmmmm
    def epoch(self, model, report: pd.DataFrame):
        loss = report['train/loss'].mean()
        grad_norm = max(torch.norm(param.grad).item() for param in model.parameters() if param.grad is not None)
        lr = self.trainer.optimizer.param_groups[0]['lr']

        self.losses.append(loss)
        self.log(loss=loss, lr=lr, grad_norm=grad_norm)
        self.log_model(loss=loss, model=model)
        self.log_eval(model)

        self.pbar.set_description(' '.join([
            f'Loss={loss}',
            f'Best_Model_Loss={self.loss_best_model_saved}',
            f'Grad_norm {grad_norm}',
            f'lr={lr}',
        ]))
        self.pbar.update(1)

    def finish_training(self, model, report):
        print("saving current model at: " + self.save_model_path)
        torch.save(model.state_dict(), self.save_model_path)
        cpath = f'{self.save_model_checkpoint_path}_{self.iters}.pt'
        print("saving checkpoint model at: " + cpath)
        cpath = f'{self.save_model_checkpoint_path}_{self.iters}_final.pt'
        print("saving final checkpoint model at: " + cpath)
        torch.save(model.state_dict(), cpath)

        if len(self.eval_results_data) > 0:
            df = pd.DataFrame(self.eval_results_data)
            df.to_csv(self.save_eval_csv_path)

            # todo maybe use pandas plotting later?
            groups = df.groupby('policy')
            fig, ax = plt.subplots()
            for name, group in groups:
                ax.plot(group['train_iter'], group['eval/avg_reward'], label=name)
            # 'eval/avg_reward': total_reward / num_eval_ep,
            # 'eval/avg_ep_len': total_timesteps / num_eval_ep

            # Add labels and legend
            ax.set_xlabel('train_iter')
            ax.set_ylabel('eval/avg_reward')
            ax.legend()
        else:
            print('gathered no evaluation data to save')
