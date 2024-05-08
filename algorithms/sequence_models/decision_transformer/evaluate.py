import torch
from data.trajectory import TrajectoryDataset


# trajectory is passed in order to be able to convert state to feature space and normalize the rewards
def evaluate_on_env(model, traj_dataset: TrajectoryDataset, device, context_len, env, rtg_target,
                    num_eval_ep=10, max_test_ep_len=1000, render=False):
    eval_batch_size = 1  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = traj_dataset.state_dim()
    act_dim = traj_dataset.action_dim()

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    with torch.no_grad():

        for _ in range(num_eval_ep):

            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                  dtype=torch.float32, device=device)

            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                 dtype=torch.float32, device=device)

            rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                        dtype=torch.float32, device=device)

            # init episode
            running_state, _ = env.reset()
            running_reward = 0
            running_rtg = traj_dataset.reward_convertor.to_feature_space(rtg_target)

            for t in range(max_test_ep_len):

                total_timesteps += 1

                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(
                    traj_dataset.state_convertor.to_feature_space(running_state)
                ).to(device)

                # calcualate running rtg and add in placeholder
                running_rtg = running_rtg - traj_dataset.reward_convertor.to_feature_space(running_reward)
                rewards_to_go[0, t] = running_rtg

                if t < context_len:
                    _, act_preds, _ = model.forward(timesteps[:, :context_len],
                                                    states[:, :context_len],
                                                    actions[:, :context_len],
                                                    rewards_to_go[:, :context_len])
                    act = act_preds[0, t].detach()
                else:
                    _, act_preds, _ = model.forward(timesteps[:, t - context_len + 1:t + 1],
                                                    states[:, t - context_len + 1:t + 1],
                                                    actions[:, t - context_len + 1:t + 1],
                                                    rewards_to_go[:, t - context_len + 1:t + 1])
                    act = act_preds[0, -1].detach()

                running_state, running_reward, terminated, truncated, _ = env.step(
                    traj_dataset.action_convertor.from_feature_space(act.cpu().numpy())
                )
                done = terminated or truncated
                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward

                if render:
                    env.render()
                if done:
                    break

    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep

    return results
