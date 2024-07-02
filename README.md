<div align="center">
  <a href="https://flitternie.github.io/ocl/">
    <img style="margin-bottom: 1.5rem" align="center" width="300" src="./docs/logo.png">
  </a>
</div>


Implementation for ICML 2024 Paper "[Online Cascade Learning for Efficient Inference over Streams](https://arxiv.org/pdf/2402.04513)".

## Installation

To use this project, follow these steps:

1. Clone the repository: `git clone https://github.com/flitternie/online_cascade_learning.git`
2. Install the required dependencies: `conda env create -f requirements.yml`

## Usage

1. Prepare your dataset and create `<dataset_name>.py` under `./data/`, specifying your data `preprocess` and `postprocess` functions.
2. Create `.yaml` configuration file to customize your cascade and hyperparameters. Read [specification](./configs/README.md) for details. 
3. Run online cascade learning: `python run.py --config <path_to_config>`.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Citation
```
@inproceedings{nie2024online,
  title={Online Cascade Learning for Efficient Inference over Streams},
  author={Nie, Lunyiu and Ding, Zhimin and Hu, Erdong and Jermaine, Christopher and Chaudhuri, Swarat},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
}
```