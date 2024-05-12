from abc import abstractmethod

import torch
import torch.nn as nn

from algorithms.random_policy import BasePolicy


class DecisionSequenceModel(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim


# todo add the end we need to train the policy not the model
class SequenceModelPolicy(BasePolicy):
    """
    todo how to avoid the error resulting in not resetting the policy?
    """

    def __init__(self, model: DecisionSequenceModel, traj_dataset, device, rtg):
        super().__init__()
        self.model = model
        self.traj_dataset = traj_dataset
        self.rtg = rtg
        self.state_dim = model.state_dim
        self.act_dim = model.act_dim
        self.device = device

    @abstractmethod
    def add_to_history(self, obs, action, reward, done):
        pass

    def reset(self):
        self.model.eval()


# todo for now use DTPolicy, later we will add S4 policy as well
class DTPolicy(SequenceModelPolicy):
    # todo note: model can be feature extractor + the transformer
    def __init__(self, model, traj_dataset, device, rtg, max_test_ep_len, context_length):
        super().__init__(model=model, traj_dataset=traj_dataset, device=device, rtg=rtg)

        self.eval_batch_size = 1  # todo we can later evaluate in parallel
        self.max_test_ep_len = max_test_ep_len
        self.context_length = context_length

    def reset(self):
        super().reset()

        # same as timesteps used for training the transformer
        # also, crashes if device is passed to arange()
        self.timesteps = torch.arange(start=0, end=self.max_test_ep_len, step=1)
        self.timesteps = self.timesteps.repeat(self.eval_batch_size, 1).to(self.device)

        # zeros place holders
        self.actions = torch.zeros((self.eval_batch_size, self.max_test_ep_len, self.act_dim),
                                   dtype=torch.float32, device=self.device)

        self.states = torch.zeros((self.eval_batch_size, self.max_test_ep_len, self.state_dim),
                                  dtype=torch.float32, device=self.device)

        self.rewards_to_go = torch.zeros((self.eval_batch_size, self.max_test_ep_len, 1),
                                         dtype=torch.float32, device=self.device)
        self.t = 0
        # todo later we might need to make this a tensor if we want to pass feature vector as input
        self.running_rtg = self.traj_dataset.reward_convertor.to_feature_space(self.rtg)

    def add_to_history(self, obs, action, reward, done):
        self.actions[0, self.t] = self.traj_dataset.reward_convertor.to_feature_space(action)
        self.states[0, self.t] = torch.from_numpy(
            self.traj_dataset.state_convertor.to_feature_space(obs)
        ).to(self.device)
        self.rewards_to_go[0, self.t] = self.running_rtg

        self.running_rtg = self.running_rtg - self.traj_dataset.reward_convertor.to_feature_space(reward)
        self.t += 1

    def predict(self, obs):
        # add state in placeholder and normalize
        self.states[0, self.t] = torch.from_numpy(
            self.traj_dataset.state_convertor.to_feature_space(obs)
        ).to(self.device)
        self.rewards_to_go[0, self.t] = self.running_rtg

        if self.t < self.context_length:
            _, act_preds, _ = self.model.forward(self.timesteps[:, :self.context_length],
                                                 self.states[:, :self.context_length],
                                                 self.actions[:, :self.context_length],
                                                 self.rewards_to_go[:, :self.context_length])
            act = act_preds[0, self.t].detach()
        else:
            _, act_preds, _ = self.model.forward(self.timesteps[:, self.t - self.context_length + 1:self.t + 1],
                                                 self.states[:, self.t - self.context_length + 1:self.t + 1],
                                                 self.actions[:, self.t - self.context_length + 1:self.t + 1],
                                                 self.rewards_to_go[:, self.t - self.context_length + 1:self.t + 1])
            act = act_preds[0, -1].detach()
        return self.traj_dataset.action_convertor.from_feature_space(act.cpu().numpy())
