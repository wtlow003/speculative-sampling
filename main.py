import statistics
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.autoregressive_sampling import autoregressive_sampling
from src.speculative_sampling import speculative_sampling

warnings.filterwarnings("ignore")


def main():
    draft_model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float32,
        device_map="mps",
        use_cache=False,
    ).eval()
    target_model = AutoModelForCausalLM.from_pretrained(
        "gpt2-xl",
        torch_dtype=torch.float32,
        device_map="mps",
        use_cache=False,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2-xl",
        torch_dtype=torch.float32,
        device_map="mps",
    )

    input_str = "Once upon a time in Singapore"
    input_ids = tokenizer.encode(input_str, return_tensors="pt").to("mps")  # type: ignore

    # skip the first
    ar_timings = []
    ss_timings = []
    for _ in range(51):
        print("#" * 100)
        ar_start = time.perf_counter()
        output = autoregressive_sampling(
            input_ids,
            target_model,
            N=40,
            temperature=1,
            top_k=20,
            top_p=0.9,
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
            temperature=1,
            top_k=20,
            top_p=0.9,
        )
        ss_end = time.perf_counter()
        ss_timings.append(ss_end - ss_start)
        print(f"Generated: {output.shape[1]} tokens")
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Output: ", generated_text)
        print("#" * 100)
        print("\n")

    ar_avg = sum(ar_timings[1:]) / len(ar_timings[1:])
    ss_avg = sum(ss_timings[1:]) / len(ss_timings[1:])
    ar_std = statistics.stdev(ar_timings[1:])
    ss_std = statistics.stdev(ss_timings[1:])
    print(f"AR Avg: {ar_avg:.4f}, Std: {ar_std:.4f}")
    print(f"SS Avg: {ss_avg:.4f}, Std: {ss_std:.4f}")
    # calculate speedup
    speedup = ar_avg / ss_avg
    speedup_percentage = (speedup - 1) * 100

    print(f"Speedup: {speedup:.2f}x")
    print(f"Percentage improvement: {speedup_percentage:.2f}%")


if __name__ == "__main__":
    main()
