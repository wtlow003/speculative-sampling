import argparse
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel

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

    # draft_model = AutoModelForCausalLM.from_pretrained(
    #     args.draft_model,
    #     torch_dtype=torch.float32,
    #     device_map=DEVICE,
    #     use_
    # ).eval()

    # target_model = AutoModelForCausalLM.from_pretrained(
    #     args.target_model,
    #     torch_dtype=torch.float32,
    #     device_map=DEVICE,
    # ).eval()

    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.target_model,
    #     torch_dtype=torch.float32,
    #     device_map=DEVICE,
    # )

    draft_model, _ = FastLanguageModel.from_pretrained(
        args.draft_model,
        load_in_4bit=True,
    )

    target_model, tokenizer = FastLanguageModel.from_pretrained(
        args.target_model,
        load_in_4bit=True,
    )

    # disable KV cache
    draft_model.config.use_cache = False
    target_model.config.use_cache = False

    prompt = "<|begin_of_text|>\n{input_str}".format(input_str=args.input_str)
    input_ids: torch.Tensor = tokenizer.encode(args.input_str, return_tensors="pt").to(
        DEVICE
    )  # type: ignore

    if args.sampling_method == "autoregressive":
        # warm-up run
        print("\nStarting warm-up run")
        autoregressive_sampling(
            input_ids,
            target_model,
            N=50,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print("Warm-up complete.")

        print("\nAuto-regressive sampling:")
        ar_output_ids = []
        print(prompt)
        torch.cuda.synchronize() if DEVICE == "cuda" else torch.mps.synchronize()
        ar_start = time.perf_counter()
        for token_id in autoregressive_sampling(
            input_ids,
            target_model,
            N=args.N,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        ):
            ar_output_ids.append(token_id)
            print(
                tokenizer.decode(token_id, skip_special_tokens=True),
                end="",
                flush=True,
            )
        torch.cuda.synchronize() if DEVICE == "cuda" else torch.mps.synchronize()
        ar_end = time.perf_counter()
        ar_time = ar_end - ar_start
        print(
            f"\nTime taken: {ar_time} seconds, {len(ar_output_ids) / ar_time} tokens/s"
        )
        print("\n")
    else:
        # warm-up run
        print("\nStarting warm-up run")
        speculative_sampling(
            input_ids,
            draft_model,
            target_model,
            N=50,
            K=args.K,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print("Warm-up complete.")

        print("\nSpeculative sampling:")
        ss_output_ids = []
        print(prompt)
        torch.cuda.synchronize() if DEVICE == "cuda" else torch.mps.synchronize()
        ss_start = time.perf_counter()
        for token_id, speculated in speculative_sampling(
            input_ids,
            draft_model,
            target_model,
            N=args.N,
            K=args.K,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        ):
            ss_output_ids.append(token_id)
            if speculated:
                print(
                    f"\033[92m{tokenizer.decode(token_id)}\033[0m", end="", flush=True
                )
            else:
                print(
                    tokenizer.decode(token_id, skip_special_tokens=True),
                    end="",
                    flush=True,
                )
        torch.cuda.synchronize() if DEVICE == "cuda" else torch.mps.synchronize()
        ss_end = time.perf_counter()
        ss_time = ss_end - ss_start
        print(
            f"\nTime taken: {ss_time} seconds, {len(ss_output_ids) / ss_time} tokens/s"
        )
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-model",
        type=str,
        default="gpt2-xl",
        help="Target model",
        required=True,
    )
    parser.add_argument("--draft-model", type=str, default="gpt2", help="Draft model")
    parser.add_argument(
        "--sampling-method",
        type=str,
        help="Sampling method",
        choices={"autoregressive", "speculative"},
    )
    parser.add_argument("--input-str", type=str, help="Input string", required=True)
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
