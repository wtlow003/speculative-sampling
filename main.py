import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.autoregressive_sampling import autoregressive_sampling
from src.speculative_sampling import speculative_sampling

warnings.filterwarnings("ignore")


def main():
    draft_model = AutoModelForCausalLM.from_pretrained(
        "gpt2", torch_dtype=torch.float16, device_map="mps"
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        "gpt2-xl", torch_dtype=torch.float16, device_map="mps"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2-xl", torch_dtype=torch.float16, device_map="mps"
    )

    input = "How does Artificial Intelligence impact us Human? The question lies"
    input_ids = tokenizer.encode(input, return_tensors="pt").to("mps")  # type: ignore

    output = autoregressive_sampling(
        input_ids,
        target_model,
        N=20,
        temperature=0,
        top_k=20,
        top_p=0.5,
    )
    print(f"Generated: {output.shape[1]} tokens")
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Output: ", generated_text)

    print("\n")

    output = speculative_sampling(
        input_ids,
        draft_model,
        target_model,
        N=20,
        K=4,
        temperature=0,
        top_k=20,
        top_p=0.5,
    )
    print(f"Generated: {output.shape[1]} tokens")
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Output: ", generated_text)


if __name__ == "__main__":
    main()
