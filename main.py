import argparse
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.autoregressive_sampling import autoregressive_sampling
from src.speculative_sampling import speculative_sampling
from src.utils import compute_metrics

warnings.filterwarnings("ignore")


def main(args: argparse.Namespace):
    """Entry point for the LLM inference script.

    Args:
        args (argparse.Namespace): Command line arguments
    """
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float32,
        device_map=DEVICE,
    ).eval()
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float32,
        device_map=DEVICE,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model,
        torch_dtype=torch.float32,
        device_map=DEVICE,
    )

    input_ids: torch.Tensor = tokenizer.encode(args.input_str, return_tensors="pt")  # type: ignore

    # skip the first
    ar_timings = []
    ss_timings = []
    # add 1 to account for the first run which is slower but will not account for metrics
    for _ in range(args.num_runs + 1):
        print("#" * 100)
        ar_start = time.perf_counter()
        output = autoregressive_sampling(
            input_ids,
            target_model,
            N=40,
            temperature=0,
            top_k=0,
            top_p=0,
        )
        ar_end = time.perf_counter()
        ar_timings.append(ar_end - ar_start)
        print(f"Generated: {output.shape[1]} tokens")
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Output: ", generated_text)

        print("\n")

        ss_start = time.perf_counter()
        output = speculative_sampling(
            input_ids,
            draft_model,
            target_model,
            N=40,
            K=4,
            temperature=0,
            top_k=0,
            top_p=0,
        )
        ss_end = time.perf_counter()
        ss_timings.append(ss_end - ss_start)
        print(f"Generated: {output.shape[1]} tokens")
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Output: ", generated_text)
        print("#" * 100)
        print("\n")

    compute_metrics(ar_timings, ss_timings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-model", type=str, default="gpt2-xl", help="Target model"
    )
    parser.add_argument("--draft-model", type=str, default="gpt2", help="Draft model")
    parser.add_argument("--input-str", type=str, required=True, help="Input string")
    parser.add_argument(
        "--num-runs", type=int, default=50, help="Number of LLM inference runs"
    )
    parser.add_argument(
        "--N", type=int, default=40, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--K", type=int, default=4, help="Number of tokens to speculate"
    )
    parser.add_argument("--temperature", type=float, default=0, help="Temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Top k sampling")
    parser.add_argument("--top-p", type=float, default=0, help="Top p sampling")
    args = parser.parse_args()

    main(args)
