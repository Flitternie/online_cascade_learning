# Configuration File Specification

This document provides a comprehensive guide to understanding and using the configuration file for setting up and evaluating online cascade learning framework. A configuration file defines various parameters related to data, models, and hyperparameters, supporting flexible definitions for different implementations. Example configuration files can be found under `./config/`.

## Table of Contents

1. [Log Configuration](#log-configuration)
2. [Data Configuration](#data-configuration)
3. [LLM Configuration](#llm-configuration)
4. [Models Configuration](#models-configuration)
5. [Hyperparameter Evaluation](#hyperparameter-evaluation)

## Log Configuration

```yaml
log: ./logs_refactored/
```

- **log**: The base directory where logs are saved. Logs will be saved to `<log_dir>/<dataset_name>/<model_names>_<llm_name>_<mu>.log`.

## Data Configuration

```yaml
data: 
  name: dataset_name
  env: data.module
  path: ./data/dataset_file.csv
  num_labels: N
```

- **name**: The name of the dataset.
- **env**: The environmental modules for the dataset.
- **path**: The file path to the preprocessed dataset.
- **num_labels**: The number of labels in the dataset.

### Example
```yaml
data: 
  name: imdb
  env: data.imdb
  path: ./data/imdb_preprocessed.csv
  num_labels: 2
```

## LLM Configuration

The configuration for Large Language Models (LLMs) can be defined in two ways and currently supports `openai` and `llama` as sources.

### By Pre-generated File

```yaml
llm:
  name: llm_name
  source: "./llm_results/llm_name/llm_result_file.txt"
```

- **name**: The name of the language model.
- **source**: The path to the file containing pre-generated results from the language model.

### By Source and Model

```yaml
llm:
  name: llm_name
  source: llm_source
  model: llm_model_identifier
```

- **name**: The name of the language model.
- **source**: The provider of the language model (`openai` or `llama`).
- **model**: The specific model to be used (e.g., `gpt-3.5-turbo-1106`).

## Models Configuration

The models section can contain multiple models, each with its own unique configuration. The goal is to learn a cascade of models, starting with lower-capacity models and ending with a powerful LLM. Models currently support `sklearn` and `transformers` as sources.

```yaml
models:
  - name: model_name
    source: model_source_library
    model: model_class
    model_args:
      key1: value1
      key2: value2
      # Additional model-specific arguments
    cache_size: cache_size_value
    cost: computational_cost_value
    wrapper:
      learning_rate: learning_rate_value
      regularization: regularization_value
      decaying_factor: decaying_factor_value
      calibration: calibration_value
```

- **name**: The name of the model.
- **source**: The source library for the model (`sklearn` or `transformers`).
- **model**: The specific model class to be used (e.g., `AutoModelForSequenceClassification`).
- **model_args**: Arguments passed to the model.
  - Customizable key-value pairs specific to the model.
- **cache_size**: The online learning cache size for the model. Reduce this value to allow more frequent updates.
- **cost**: The computational cost of the model. Adjust according to the actual situation.
- **wrapper**: Additional parameters for the model wrapper.
  - **learning_rate**: The learning rate for the wrapper.
  - **regularization**: The regularization parameter to prevent overfitting.
  - **decaying_factor**: The initial decaying factor that controls the probability of jumping directly to the LLM. Set higher for less capable models. The probability decays throughout the learning process as: `decaying_factor ** #model_updates`. Range: [0, 1). 
  - **calibration**: The calibration factor for adjusting the model prediction's confidence. Set higher for less capable models. Range: [0, 0.5).

This section outlines the key parameters for defining each model in the configuration file. Adjust these parameters to optimize the performance and computational efficiency of your model cascade.

### Example

#### Linear Regression (LR) Model

```yaml
- name: LR
  source: sklearn.linear_model
  model: SGDClassifier
  model_args:
    loss: log_loss
  cache_size: 8
  cost: 1
  wrapper:
    learning_rate: 0.0007
    regularization: 0.0001
    decaying_factor: 0.99
    calibration: 0.4
```

#### BERT-base Model

```yaml
- name: BERT-base
  source: transformers
  model: AutoModelForSequenceClassification
  model_args:
    model_name_or_path: bert-base-uncased
    batch_size: 8
    num_epochs: 5
    learning_rate: 1e-5
  cache_size: 16
  cost: 3
  wrapper:
    learning_rate: 0.0007
    regularization: 0.0001
    decaying_factor: 0.97
    calibration: 0.35
```

## Hyperparameter Evaluation

```yaml
mu: 
 - [0.00001, 0.0001, 0.00001]
 - [0.0001, 0.001, 0.00001]
 - 0.002
```

- **mu**: A list of hyperparameters to be evaluated. Each entry can be:
  - A triple in the format `[start, end, step]` specifying the range and step for evaluation.
  - A single number.
  - A mixture of the above formats.
