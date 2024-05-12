from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class TrainConfig:
    max_eval_ep_len: int  # max len of one evaluation episode
    context_len: int  # K in decision transformer
    rtg_range_check: Tuple[float, float] = (0, 1) # todo useless
    rtg_main: float = 0 # todo useless

    num_eval_ep: int = 10  # num of evaluation episodes per iteration
    batch_size: int = 64  # training batch size
    lr: float = 1e-3  # learning rate
    wt_decay: float = 1e-4  # weight decay
    warmup_steps: int = 10000  # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters: int = 200
    num_updates_per_iter: int = 100

    model_checkpoint_interval: int = 30  # save model every model_eval_callback epochs
    eval_model_interval: int = 30        # evaluate model every interval

    # todo I kinda wanna remove these GPT settings so that this class is general purpose for sequence models...
    # GPT config
    n_blocks: int = 3  # num of transformer blocks
    embed_dim: int = 128  # embedding (hidden) dim of transformer
    n_heads: int = 1  # num of transformer heads
    dropout_p: float = 0.1  # dropout probability
