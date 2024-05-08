import torch
import torch.functional as F

from mingpt.model import GPT


class CustomGPT(GPT):
    def __init__(self):
        # todo later use the config
        model_config = GPT.get_default_config()
        block_size = 64  # todo experiment with block_size. we might need to have block_size >= SPLIT_SEQUENCE_LENGTH but I'm not sure about this

        super().__init__(model_config)
        model_config.model_type = 'gpt-nano'
        model_config.vocab_size = 1
        model_config.block_size = block_size

        # for key, val in config:
        #     # todo check if it has the attribute first
        #     model_config.__setattr__(key, val)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
