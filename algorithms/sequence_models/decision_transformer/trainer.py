import pandas as pd
import torch

from algorithms.sequence_models.trainer import Trainer


class TrainerDT(Trainer):
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
