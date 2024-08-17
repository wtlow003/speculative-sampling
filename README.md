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
# example output

####################################################################################################
Finished 'autoregressive_sampling' in 2.6522 secs with 17.344182433651877 tokens/sec
Generated: 46 tokens
Output:  Once upon a time in Singapore, there was an ancient city.

But the city was not called Singapore, it was called Looch, which means city of the blind.

But there was a lot more to


rejected at 15  rejected token: tensor([475], device='mps:0')  resampled token: tensor([780], device='mps:0')
rejected at 21  rejected token: tensor([898], device='mps:0')  resampled token: tensor([1693], device='mps:0')
rejected at 27  rejected token: tensor([2227], device='mps:0')  resampled token: tensor([1422], device='mps:0')
rejected at 38  rejected token: tensor([366], device='mps:0')  resampled token: tensor([880], device='mps:0')
rejected at 44  rejected token: tensor([994], device='mps:0')  resampled token: tensor([287], device='mps:0')
Finished 'speculative_sampling' in 2.0926 secs with 21.982319369168305 tokens/sec
Generated: 46 tokens
Output:  Once upon a time in Singapore, I was a bit of a recluse, because of my age and my job, and because I just didn't have any money, and so I thought, well, let's just live in
####################################################################################################

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
