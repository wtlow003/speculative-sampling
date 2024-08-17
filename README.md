# Speculative Sampling for Faster LLM Inference

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
python main.py
```

## Results

Using the following models, running for 50 evaluations, the results are as follows:

- Target Model: `gpt2-xl`
- Draft Model: `gpt2`

```
AR Avg: 2.6482, Std: 0.1653
SS Avg: 1.8587, Std: 0.2931
Speedup: 1.42x
Percentage improvement: 42.47%
```


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
