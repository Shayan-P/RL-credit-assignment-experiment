import numpy as np
import torch

from algorithms.sequence_models.trainer import Trainer
from data.trajectory import TrajectoryDataset, TrajectoryData

SPLIT_SEQUENCE_LENGTH = 32
BATCH_SIZE_DEFAULT = 256


class DecisionTransformerTrainer(Trainer):
    def get_batch(self, batch_size=BATCH_SIZE_DEFAULT, max_len=SPLIT_SEQUENCE_LENGTH):
        batch_inds = np.random.choice(
            np.arange(len(self.trajectories)),
            size=batch_size,
            replace=True
            # todo original repo has reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj: TrajectoryData = self.trajectories.get_item(batch_inds[i])  # todo removed sorted_ind as it was not necessary here
            si = np.random.randint(0, traj.rewards.shape[0] - 1)

            # get sequences from dataset
            s.append(traj.observations[si:si + max_len].reshape(1, -1, self.model.state_dim))
            a.append(traj.actions[si:si + max_len].reshape(1, -1, self.model.act_dim))
            r.append(traj.rewards[si:si + max_len].reshape(1, -1, 1))
            d.append(traj.dones[si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.model.max_ep_len] = self.model.max_ep_len-1  # padding cutoff
            rtg.append(traj.returns[si:si + max_len].reshape(1, -1))

                # discount_cumsum(traj.rewards[si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, self.model.state_dim)), s[-1]], axis=1)
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, self.model.act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)

        return s, a, r, d, rtg, timesteps, mask

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        # todo why are we using attention_mask here?
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()

    def evaluate_episode(
            self,
            env,
            max_ep_len=1000,
            target_return=None,
    ):
        self.model.eval()
        self.model.to(device=self.device)  # todo remove this. we don't need it all the time

        state, _ = env.reset()
        state = self.trajectories.state_convertor.to_feature_space(state)
        print(state)
        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, self.model.state_dim).to(device=self.device, dtype=torch.float32)
        actions = torch.zeros((0, self.model.act_dim), device=self.device, dtype=torch.float32)
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        target_return = torch.tensor(target_return, device=self.device, dtype=torch.float32)

        episode_return, episode_length = 0, 0
        for t in range(max_ep_len):

            # add padding
            actions = torch.cat([actions, torch.zeros((1, self.model.act_dim), device=self.device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])

            # todo paused here
            # states, actions, rewards, returns_to_go, timesteps
            action = self.model.get_action(
                states=states.to(dtype=torch.float32),
                actions=actions.to(dtype=torch.float32),
                rewards=rewards.to(dtype=torch.float32),
                returns_to_go=target_return,
            )
            actions[-1] = action

            action = action.detach().cpu().numpy()
            state, reward, termination, truncation, _ = env.step(self.trajectories.action_convertor.from_feature_space(action))
            done = termination or truncation

            state = self.trajectories.state_convertor.to_feature_space(state)
            reward = self.trajectories.reward_convertor.to_feature_space(reward)

            cur_state = torch.from_numpy(state).to(device=self.device).reshape(1, self.model.state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            episode_return += reward
            episode_length += 1

            if done:
                break

        return episode_return, episode_length
