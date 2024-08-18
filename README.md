<h1 align="center">Speculative Sampling for Faster LLM Inference</h1>

<p align="center">
    <img src="https://img.shields.io/badge/python-3.9.10-orange"
         alt="python version">
     <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"
          alt="uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json"
         alt="ruff">
</p>

## About

This repository contains the implementation of the speculative sampling method for faster LLM inference with a draft model.

The implementation is based on the my own interpretation of the paper – [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) by Deepmind.


## Installation

### Setting Up the Environment

This project uses uv for dependency management. To install UV, run the following command:

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
pip install uv

# With pipx.
pipx install uv

# With Homebrew.
brew install uv

# With Pacman.
pacman -S uv
```

Thereafter, install the rest of the dependencies using uv:

```bash
# create a virtual env
uv venv

# install dependencies
uv pip install -r requirements.txt  # Install from a requirements.txt file.
```

## Usage

```bash
# check cli options
python main.py --help

usage: main.py [-h] --target-model TARGET_MODEL --draft-model DRAFT_MODEL --input-str INPUT_STR [--num-runs NUM_RUNS] [--N N] [--K K] [--temperature TEMPERATURE]
               [--top-k TOP_K] [--top-p TOP_P]

optional arguments:
  -h, --help            show this help message and exit
  --target-model TARGET_MODEL
                        Target model
  --draft-model DRAFT_MODEL
                        Draft model
  --input-str INPUT_STR
                        Input string
  --num-runs NUM_RUNS   Number of LLM inference runs
  --N N                 Number of tokens to generate
  --K K                 Number of tokens to speculate
  --temperature TEMPERATURE
                        Temperature
  --top-k TOP_K         Top k sampling
  --top-p TOP_P         Top p sampling
```

Running LLM inference comparison script:

```bash
python main.py --target-model gpt2-xl \
      --draft-model gpt2 \
      --input-str "Alan Turing theorized that computers would one day become" \
      --num-runs 50 \
      --N 40 \
      --K 4 \
      --temperature 0.6 \
      --top-k 25 \
      --top-p 0.9
```

- With `--num-runs 1`, the script will run the LLM inference `num-runs + 1` times to account for the warmup time.


## Results

The following results are obtained on a MacBook Pro M2 Pro Max with 32GB RAM comparing speculative sampling with naive autoregressive sampling in LLM inference:

- `N`: 40
- `K`: 4
- `temperature`: 0.6
- `top-k`: 25
- `top-p`: 0.9

1. **Target Model**: [`gpt2-xl`](https://huggingface.co/openai-community/gpt2-xl) and **Draft Model**: [`gpt2-xl`](https://huggingface.co/openai-community/gpt2-xl)

> [!NOTE]
>
> This serves as a sanity check for the speculative sampling method.
>
> In this case, since the target model and draft model are the same, there should be no rejection of the speculative samples.

| Method                  | num_runs | time       | +/- std  | speedup |
| ----------------------- | -------- | ---------- | -------- | -------- |
| Autoregressive Sampling | 50       | 2.84       |   0.20   | 1.00     |
| Speculative Sampling    | 50       | 2.96       |   0.22   | 0.96     |

2. **Target Model**: [`gpt2-xl`](https://huggingface.co/openai-community/gpt2-xl) and **Draft Model**: [`gpt2`](https://huggingface.co/openai-community/gpt2)

| Method                  | num_runs | time       | +/- std  | speedup |
| ----------------------- | -------- | ---------- | -------- | -------- |
| Autoregressive Sampling | 50       | 2.86       |   0.16   | 1.00     |
| Speculative Sampling    | 50       | 2.17       |   0.32   | 1.31     |

Based on the results above, we observed that the speculative sampling method offers a significant speedup compared to the autoregressive sampling method.

In our sanity check, we confirmed that when the target and draft models are identical, the speculative sampling method does not produce any rejected samples, since the tokens are sampled from the exact same probability distribution. Additionally, because the models are identical in size and we're essentially running the same model twice (more forward passes in the draft model), the speculative sampling method is expected to be slower than the autoregressive sampling method.

## References

```
@misc{chen2023acceleratinglargelanguagemodel,
      title={Accelerating Large Language Model Decoding with Speculative Sampling}, 
      author={Charlie Chen and Sebastian Borgeaud and Geoffrey Irving and Jean-Baptiste Lespiau and Laurent Sifre and John Jumper},
      year={2023},
      eprint={2302.01318},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2302.01318}, 
}
```

## Acknowledgements

The implementation for speculative sampling is build upon the following repository:

1. https://github.com/feifeibear/LLMSpeculativeSampling
2. https://github.com/jaymody/speculative-sampling
3. https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
