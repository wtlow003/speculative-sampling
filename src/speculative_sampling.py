import torch

from .utils import batch_norm_logits, max_fn, norm_logits, sample, timer


# adapted from: https://github.com/feifeibear/LLMSpeculativeSampling/blob/main/sampling/speculative_sampling.py
@timer
@torch.no_grad()
def speculative_sampling(
    x: torch.Tensor,
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    N: int,
    K: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eps: float = 1e-7,
):

    # x = [batch_size, seq_len]
    seq_len = x.shape[1]
    # T = maximum length to generate
    T = seq_len + N

    # we will be increasing input x length until it reaches T
    while x.shape[1] < T:
        prefix = x
        x_len = x.shape[1]

        # step 1: auto-regressive decode K draft tokens and get q probability distribution
        # iterate over number of draft tokens to generate
        for _ in range(K):
            # forward pass through draft model
            outputs = draft_model(prefix)
            # shape: [batch_size, seq_len, vocab_size]
            p = outputs.logits
            # sample next token from logits from the last token position
            next_token = sample(
                norm_logits(p[:, -1, :], temperature, top_k, top_p, eps)
            )
            # [batch_size, prefix_len + K_i + 1]
            prefix = torch.cat([prefix, next_token], dim=1)

        # # [batch_size, prefix_len + K, vocab_size]
        # for i in range(p.shape[1]):  # type: ignore
        #     p[:, i, :] = norm_logits(p[:, i, :], temperature, top_k, top_p, eps)  # type: ignore
        p = batch_norm_logits(p, temperature, top_k, top_p, eps)  # type: ignore

        # step 2: generate K+1 sets of logits from draft with target model
        outputs = target_model(prefix)
        q = outputs.logits
        # normalize logits
        # for i in range(q.shape[1]):
        #     q[:, i, :] = norm_logits(q[:, i, :], temperature, top_k, top_p, eps)
        q = batch_norm_logits(q, temperature, top_k, top_p, eps)

        # append draft tokens based on rejection sampling and resample if not accepted
        is_all_accepted = True
        n = x_len - 1
        for i in range(K):
            # generate r from uniform distribution
            # dim=1
            r = torch.rand(1, device=x.device)
            draft_idx = prefix[:, x_len + i]

            # TODO: enhance understanding
            target_prob_k_i = q[:, x_len + i - 1, draft_idx]
            draft_prob_k_i = p[:, x_len + i - 1, draft_idx]  # type: ignore

            if r < torch.min(
                torch.tensor([1], device=x.device), (target_prob_k_i / draft_prob_k_i)
            ):
                # accepted
                n += 1
            else:
                # rejected
                # resample from the recovered distribution of target model
                t = sample(max_fn(q[:, n, :] - p[:, n, :]))  # type: ignore
                is_all_accepted = False
                break

        x = prefix[:, : n + 1]

        if is_all_accepted:
            # target model generated K+1 logits, we are using the last logit to sample the next token
            t = sample(q[:, -1, :])  # type: ignore

        x = torch.cat([x, t], dim=1)  # type: ignore

    return x
