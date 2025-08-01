# CBFL
## Introduction
Welcome to the project repository for the paper **Tackling Multi-Class Imbalance in Next Activity Prediction with Class-Balanced Focal Loss**.
This repository provides the implementation details and supplementary materials to support the findings presented in the paper.

To help navigate the repository, below is an overview of the contents and related materials referenced in the paper:

- [Data Preprocessing and Feature Encoding](#data-preprocessing-and-feature-encoding)
  - Experimental results comparing selected encoding strategies against alternative approaches.
- [Implemented Architectures](#implemented-architectures)
- [Loss Functions, Training and Evaluation](#loss-functions-training-and-evaluation)
- [Results](#results)
  - Detailed report on overall model performance, including additional metrics such as AUC-PR, accuracy, precision, and recall.
  - Complete set of visualizations illustrating F1-score trends for majority and minority groups across varying majority/minority thresholds.
- [Python Environment Setup](#python-environment-setup)

## Data Preprocessing and Feature Encoding
 
### Comparison of encoding strategies

### Scripts
**`1_data_processing/`** contains scripts for data preprocessing, dataset splitting, and generating traces, prefixes, and next activities:  
- `preprocessing.py` handles data preprocessing
- `train_test_split.py` handles dataset splitting.  
- `create_trace_prefix.py`: generates traces (trace-based encoding), prefixes, log prefixes, and trace suffixes. 

## Implemented Architectures
**`2_models/`** contains the implementation scripts for all models used in this study:
- `create_dl_model.py` defines the deep learning architectures, including LSTM, Transformer, and xLSTM models.
- `xgboost.py` implements the XGBoost model.

## Loss Functions, Training and Evaluation

## Results

## Python Environment Setup
The implementation is based on **Python 3.12.7**. To set up the environment, download `requirements.txt` and run the following command::

```bash
pip install -r requirements.txt
