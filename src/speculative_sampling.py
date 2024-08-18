import torch

from .utils import (
    batch_norm_logits,
    max_fn,
    norm_logits,
    sample,
    timer,
)


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
    eps: float = 1e-10,
):
    """
    Implementation of Algorithm 2 in the paper - Accelerating Large Language Model Decoding
    with Speculative Sampling (https://arxiv.org/abs/2302.01318).
    """
    seq_len = x.shape[1]
    T = seq_len + N

    # we will be increasing input x length until it reaches T
    while x.shape[1] < T:
        prefix = x
        x_len = x.shape[1]

        # -----------------------------------------
        # Step 1: Generate K tokens from draft_model
        # -----------------------------------------
        generated_tokens = []
        for _ in range(K):
            outputs = draft_model(prefix)
            p = outputs.logits
            next_token = sample(
                norm_logits(p[:, -1, :], temperature, top_k, top_p, eps)
            )
            generated_tokens.append(next_token)
            prefix = torch.cat([prefix, next_token], dim=1)

        generated_tokens = torch.cat(generated_tokens, dim=1)
        p = batch_norm_logits(p, temperature, top_k, top_p, eps)  # type: ignore

        # --------------------------------------------
        # Step 2: Evaluate full sequence + K draft tokens using target_model
        # --------------------------------------------
        q = target_model(prefix).logits
        q = batch_norm_logits(q, temperature, top_k, top_p, eps)

        # ------------------------------
        # Step 3: Single Round Rejection Sampling Process
        # ------------------------------
        n = x_len - 1
        target_probs = torch.gather(
            q[:, n : n + K, :], 2, generated_tokens.unsqueeze(-1)
        ).squeeze(-1)
        draft_probs = torch.gather(
            p[:, n : n + K, :], 2, generated_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # acceptance probabilities for all K tokens
        acceptance_probs = torch.minimum(
            torch.ones_like(target_probs), target_probs / draft_probs
        )

        random_vals = torch.rand_like(acceptance_probs)

        # determine which tokens are accepted
        accepted_tokens = random_vals < acceptance_probs

        # ------------------------------
        # Step 4: Combine Results and Resample if Necessary
        # ------------------------------
        # determine where the first rejection occurs for each sequence
        first_rejection_indices = torch.nonzero(~accepted_tokens, as_tuple=True)[1]

        if first_rejection_indices.all():
            # if all tokens are accepted
            x = torch.cat([x, generated_tokens], dim=1)
            next_token = sample(q[:, -1, :])
        else:
            # if there is at least one rejection
            first_rejection_index = first_rejection_indices[0].item()
            x = torch.cat([x, generated_tokens[:, :first_rejection_index]], dim=1)
            # recover probability distribution
            next_token = sample(
                max_fn(
                    q[:, n + first_rejection_index, :]  # type: ignore
                    - p[:, n + first_rejection_index, :]  # type: ignore
                )
            )
            print(
                "rejected at",
                n + first_rejection_index + 1,
                " rejected token:",
                generated_tokens[:, first_rejection_index],  # type: ignore
                " resampled token:",
                next_token.squeeze(-1),
            )

        # add newly generated token to x
        x = torch.cat([x, next_token], dim=1)

    return x
