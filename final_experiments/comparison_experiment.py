from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from algorithms.evaluate_policy import evaluate_policy
from algorithms.sequence_models.config import TrainConfig
from algorithms.sequence_models.decision_S4.dts4 import DecisionS4
from algorithms.sequence_models.decision_transformer.decision_transformer import DecisionTransformer
from algorithms.sequence_models.decision_sequence_policy import DTPolicy
from experiment import Experiment


class AutomatedComparisonExperiment:
    def __init__(self, env, traj_dataset, config: TrainConfig, device, rtgs_for_train_eval, rtgs_final_test, env_name,
                 experiment_name,
                 DT_Class=DecisionTransformer, S4_Class=DecisionS4):
        rtgs_final_test = list(sorted(rtgs_final_test))
        rtgs_for_train_eval = list(sorted(rtgs_for_train_eval))

        self.traj_dataset = traj_dataset
        self.env = env
        self.config = config
        self.device = device
        self.rtgs_for_train_eval = rtgs_for_train_eval
        self.rtgs_final_test = rtgs_final_test
        self.env_name = env_name
        self.experiment_name = experiment_name

        self.dt_model = DT_Class(
            state_dim=traj_dataset.state_dim(),
            act_dim=traj_dataset.action_dim(),
            n_blocks=config.n_blocks,
            h_dim=config.embed_dim,
            context_len=config.context_len,
            n_heads=config.n_heads,
            drop_p=config.dropout_p,
        ).to(device)

        self.make_dt_policy = partial(DTPolicy, model=self.dt_model, traj_dataset=traj_dataset, device=device,
                                      max_test_ep_len=config.max_eval_ep_len, context_length=config.context_len)

        self.s4_model = S4_Class(
            state_dim=traj_dataset.state_dim(),
            act_dim=traj_dataset.action_dim(),
            h_dim=config.embed_dim,
            drop_p=config.dropout_p,
        ).to(device)

        self.make_s4_policy = partial(DTPolicy, model=self.s4_model, traj_dataset=traj_dataset, device=device,
                                 max_test_ep_len=config.max_eval_ep_len, context_length=config.context_len)

        self.s4_experiment = Experiment(
            model_name='s4',
            model=self.s4_model,
            env_name=env_name,
            env=env,
            experiment_name=experiment_name,
            traj_dataset=traj_dataset,
            dataset_name=f'size={len(traj_dataset)}',
            config=config,
            device=device,
            eval_policies_and_names=[
                (self.make_s4_policy(rtg=rtg), f'S4:rtg={rtg:.2f}')
                for rtg in rtgs_for_train_eval
            ],
            final_eval_policies=[
                self.make_s4_policy(rtg=rtg)
                for rtg in rtgs_final_test
            ]
        )

        self.dt_experiment = Experiment(
            model_name='dt',
            model=self.dt_model,
            env_name=env_name,
            env=env,
            experiment_name=experiment_name,
            traj_dataset=traj_dataset,
            dataset_name=f'size={len(traj_dataset)}',
            config=config,
            device=device,
            eval_policies_and_names=[
                (self.make_dt_policy(rtg=rtg), f'DT:rtg={rtg:.2f}')
                for rtg in rtgs_for_train_eval
            ],
            final_eval_policies=[
                self.make_dt_policy(rtg=rtg)
                for rtg in rtgs_final_test
            ]
        )

    def run_and_report(self, experiment: Experiment, epochs):
        plt.figure()
        print("distribution of rewards in the dataset")
        # rewards:
        plt.hist([traj.returns[0] for traj in self.traj_dataset], bins=50)
        experiment.save_fig("distribution_of_returns_in_trajectories")

        plt.figure()
        report = self.dt_experiment.train_for(epochs)
        experiment.save_fig(f"_rtg_following_learning_process={experiment.custom_callback.iters}")

        plt.figure()
        experiment.plot_loss(report)
        experiment.save_fig(f"loss_after={experiment.custom_callback.iters}")

        rtg_results = [evaluate_policy(policy, self.env, num_eval_ep=self.config.num_eval_ep)['eval/avg_reward']
                       for policy in experiment.final_eval_policies]
        max_in_dataset = max([traj.returns[0] for traj in self.traj_dataset])

        plt.figure()
        plt.plot(self.rtgs_final_test, rtg_results)
        plt.plot(self.rtgs_final_test, self.rtgs_final_test)
        plt.hlines(max_in_dataset, self.rtgs_final_test[0], self.rtgs_final_test[-1])
        plt.xlabel("rtg_command")
        plt.xlabel("rtg_result")
        plt.legend(["agent", "x=y", "max in dataset"])
        experiment.save_fig(f"_rtg_following_iters={experiment.custom_callback.iters}")
