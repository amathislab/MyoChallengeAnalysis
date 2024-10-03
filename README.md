# MyoChallenge Analysis Code for Musculoskeletal Skill Acquisition

## Overview

This repository contains the analysis code for the second part of our paper [Chiappa, A. S.†, Tano, P.†, Patel, N.†, Ingster, A., Pouget, A., & Mathis, A. (2024). Acquiring musculoskeletal skills with curriculum-based reinforcement learning. Neuron.](https://www.sciencedirect.com/science/article/pii/S0896627324006500). Our work combines musculoskeletal simulation, reinforcement learning, and neuroscience to advance the understanding of biological motor control ([EPFL blog](https://actu.epfl.ch/news/modeling-the-minutia-of-motor-manipulation-with-ai/), [YT Shorts](https://youtube.com/shorts/cHAu9eYfmM4?si=s8tF3DPvzvPo5ygX)).

The analyses in this repository complement our winning solution for the inaugural NeurIPS MyoChallenge, where we trained a recurrent neural network to control a realistic model of the human hand to rotate two Baoding balls. For the code of the winning solution (PPO-LSTM + SDS curriculum to solve the MyoChallenge), please refer to our separate repository, [MyoChallenge](https://github.com/amathislab/myochallenge).

## Getting Started

### Installation and Setup

We use [Poetry](https://python-poetry.org/), a modern Python packaging and dependency management tool. If you're unfamiliar with Poetry, here's a [quick tutorial](https://www.youtube.com/watch?v=0f3moPe_bhk).

To set up the project, follow these steps:

1. Clone the repository:
   ```sh
   # If you have GitHub CLI (gh) installed:
   gh repo clone amathislab/MyoChallengeAnalysis

   # Otherwise, use git clone:
   # git clone https://github.com/amathislab/MyoChallengeAnalysis.git
   ```

2. Navigate to the project directory:
   ```sh
   cd MyoChallengeAnalysis
   ```

3. Ensure you're on the main branch:
   ```sh
   git checkout main
   git pull
   ```

4. Install Poetry (if not already installed):
   ```sh
   pip install poetry
   ```

5. Configure Poetry to create the virtual environment within the project:
   ```sh
   poetry config virtualenvs.in-project true
   ```

6. Create the environment and install dependencies:
   ```sh
   poetry install
   ```

7. Activate the virtual environment:
   ```sh
   poetry shell
   ```

### Reproducing the Figures

1. **Download the Required Data**
   - **Option 1: Manual Download**
     - Access the relevant data for models and rollouts on [Zenodo](https://zenodo.org/records/13332869).
     - Download `datasets.zip`, extract it, and place the folder in the [`/data`](/data/) directory.
   
   - **Option 2: Automatic Download**
     - Run the code in [`src/download_data.ipynb`](src/download_data.ipynb).
     - This will automatically download and extract the dataset to the correct location.
     - Note: Please be patient, as downloading and extracting ~10GB of data may take some time.

2. **Verify Dataset Folders**
   Ensure you have the following dataset folders:
   - `/data/datasets/csi/`
   - `/data/datasets/rollouts/`
   - `/data/datasets/umap/`

3. **Generate Figures**
   Run the Jupyter notebooks in the [`src`](/src/) directory:
   - `fig_1.ipynb` through `fig_6.ipynb`


## Citation

If you use our code or ideas, please cite:

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

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact

For questions, collaborations, or any issues related to this project, we encourage you to:

1. Open an issue in this GitHub repository for bug reports, feature requests, or general questions.
2. Submit a pull request if you have a contribution you'd like to make.
3. For other inquiries or potential collaborations, please reach out to any of the authors through their institutional contact information.

We appreciate your interest in our work and look forward to your feedback and potential contributions!
