# # write logs then visualize?
#
# from gym.wrappers.record_video import RecordVideo
#
#
#
# def show_video():
#     """
#     Displays the recorded video of the gym environment.
#     """
#     mp4list = glob.glob('video/*.mp4')
#     if len(mp4list) > 0:
#         # mp4 = mp4list[0]
#         mp4 = max(mp4list, key=os.path.getctime)
#         video = io.open(mp4, 'r+b').read()
#         encoded = base64.b64encode(video)
#         ipythondisplay.display(HTML(data='''<video alt="test" autoplay
#                     loop controls style="height: 400px;">
#                     <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#                 </video>'''.format(encoded.decode('ascii'))))
#     else:
#         print("Could not find video")
#
#
# def wrap_env(env):
#     """
#     Wraps the given gym environment to record videos.
#
#     Parameters:
#         env (gym.Env): The environment to wrap.
#
#     Returns:
#         gym.Env: The wrapped environment.
#     """
#     env = RecordVideo(env, './video',  episode_trigger = lambda episode_number: True)
#     return env
#


from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from collections import deque
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


class EvalLogCallback(BaseCallback):
    def __init__(self, model, eval_env, smooth_window=10, eval_every=10):
        super(EvalLogCallback, self).__init__()
        self.model = model
        self.eval_env = eval_env
        self.smooth_window = smooth_window
        self.reward_window = deque(maxlen=smooth_window)
        self.smooth_rewards = []
        self.pbar = tqdm()
        self.cnt_rollout = 0
        self.eval_every = eval_every
        self.pbar.set_description('number of rollouts')

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        self.pbar.update(1)

    def _on_rollout_end(self) -> None:
        if self.cnt_rollout % self.eval_every == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, render=False, n_eval_episodes=2, deterministic=True, warn=False
            )
            self.reward_window.append(mean_reward)
        self.cnt_rollout += 1
        self.smooth_rewards.append(sum(self.reward_window) / len(self.reward_window))

    def plot_rewards(self):
        plt.plot(self.smooth_rewards)
