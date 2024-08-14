import torch

from .utils import norm_logits, sample, timer


@timer
@torch.no_grad()
def autoregressive_sampling(
    x: torch.Tensor,
    model: torch.nn.Module,
    N: int,
    temperature: float = 1.0,
    top_k: int = 25,
    top_p: float = 0,
    eps: float = 1e-7,
) -> torch.Tensor:

    # prefix length
    # [batch_size, seq_len]
    seq_len = x.shape[1]
    # desired output length
    T = seq_len + N

    while seq_len < T:
        output = model(x)
        # [batch_size, n, vocab_size]
        logits = output.logits
        # [batch_size, vocab_size]
        p = norm_logits(logits[:, -1, :], temperature, top_k, top_p, eps)
        # sample from the distribution
        next_token_id = sample(p)
        # append the sampled token to the input
        x = torch.cat((x, next_token_id), dim=1)
        seq_len += 1

    return x
