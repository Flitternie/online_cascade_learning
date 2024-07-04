<div align="center">
  <a href="https://flitternie.github.io/ocl/">
    <img align="center" width="400" src="./docs/logo.png">
  </a>
</div>
<br>

Implementation for ICML 2024 Paper "[Online Cascade Learning for Efficient Inference over Streams](https://arxiv.org/pdf/2402.04513)".

## Installation

To use this project, follow these steps:

1. Clone the repository: `git clone https://github.com/flitternie/online_cascade_learning.git`
2. Install the required dependencies: `conda env create -f requirements.yml`

## Usage

1. Prepare your dataset using [Huggingface Datasets](https://huggingface.co/docs/datasets/) library. 
2. Create `<dataset_name>.py` under `./data/`, specifying your data `preprocess` and `postprocess` functions. Read [Data Module Specification](./data/README.md) for details.
3. Create `.yaml` configuration file to customize your cascade and hyperparameters. We currently support [Scikit-learn](https://scikit-learn.org/stable/index.html) classification models and [Huggingface transformers](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForSequenceClassification) sequence classification models to compose a cascade. Read [Configuration File Specification](./configs/README.md) for details. 
4. Run online cascade learning: `python run.py --config <path_to_config>`.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

Project contributors: @[Flitternie](https://github.com/Flitternie/) @[Dingz9926](https://github.com/Dingz9926)

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
