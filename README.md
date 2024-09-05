# MyoChallenge Analysis

Here we provide the analysis code for our winning solution for the [inaugural MyoChallenge at NeurIPS in 2022](https://github.com/amathislab/myochallenge). 

The analysis is detailed in [Acquiring musculoskeletal skills with curriculum-based reinforcement learning](https://www.biorxiv.org/content/early/2024/01/25/2024.01.24.577123).


## Installation

We use [Poetry](https://python-poetry.org/), a modern python packaging and dependency management software. Hopefully, this will make development a breeze and keep track of new packages that any developer installs in the pyproject.toml file. If you are unfamiliar with it, [here's a quick tutorial](https://www.youtube.com/watch?v=0f3moPe_bhk). To get started, you can use the following commands:

```sh
# if you don't have gh installed, use the git clone command below
# git clone https://github.com/amathislab/MyoChallengeAnalysis.git

# clone the BAMB2024 git repository if you don't already have it
gh repo clone amathislab/MyoChallengeAnalysis

# change branch to main and pull
git checkout main
git pull

# install poetry if you don't have it already
pip install poetry

# configure virtual env to be created within the project
poetry config virtualenvs.in-project true

# create the environment and install dependencies
poetry install

# activate the virtual environment
poetry shell
```

## How to reproduce the figures?

1. **Download the relevant data**
  - The relevant data for models and rollouts is available on [zenodo](https://zenodo.org/records/13332869).
  - Download `datasets.zip`, extract the folder and place it in [`/data`](/data/).
  - Ensure that you have all the relevant folders: 
    - `/data/datasets/csi/`, 
    - `/data/datasets/rollouts/`, and 
    - `/data/datasets/umap/`
2. Run the jupyter notebooks in [`src`](/src/)

## Literature

If you use our code, or ideas please cite:

```
@article {Chiappa2024skills,
	author = {Alberto Silvio Chiappa and Pablo Tano and Nisheet Patel and Abigail Ingster and Alexandre Pouget and Alexander Mathis},
	title = {Acquiring musculoskeletal skills with curriculum-based reinforcement learning},
	elocation-id = {2024.01.24.577123},
	year = {2024},
	doi = {10.1101/2024.01.24.577123},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Efficient, physiologically-detailed musculoskeletal simulators and powerful learning algorithms provide new computational tools to tackle the grand challenge of understanding biological motor control. Our winning solution for the first NeurIPS MyoChallenge leverages an approach mirroring human learning and showcases reinforcement and curriculum learning as mechanisms to find motor control policies in complex object manipulation tasks. Analyzing the policy against data from human subjects reveals insights into efficient control of complex biological systems. Overall, our work highlights the new possibilities emerging at the interface of musculoskeletal physics engines, reinforcement learning and neuroscience.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/01/25/2024.01.24.577123},
	eprint = {https://www.biorxiv.org/content/early/2024/01/25/2024.01.24.577123.full.pdf},
	journal = {bioRxiv}
}
```

Acknowledgments & funding: We thank members of the Mathis Group for helpful feedback. A.M. is appreciative to the Kavli Institute for Theoretical Physics (KITP) in Santa Barbara, where part of the manuscript was written. AM thanks Nicola Hodges and John Krakauer for discussions on skill learning. A.C. and A.M. are funded by Swiss SNF grant (310030_212516). A.I. acknowledges EPFL's Summer in the Lab fellowship to join the Mathis Group. A.M. was supported in part by grants NSF PHY-1748958 and PHY-2309135 to the KITP. P.T. and N.P. were supported by University of Geneva internal funding.
