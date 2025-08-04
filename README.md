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
The table below reports F1-scores on the validation set for both our selected encoding strategy (`Selected`) and a widely used alternative (`Alternative`). For XGBoost, we employ prefix-based index encoding as the selected strategy and compare it against prefix-based aggregation encoding, a common choice in prior work. For deep learning models, we use trace-based encoding as our selected strategy, with prefix-based index encoding as the alternative.

The results show that our selected strategies perform on par with or better than the alternatives, supporting our choices.

|              |**BPIC2017**    |        |**BPIC2019**|            |**BPIC2020**|            |**BAC**     |            |
|--------------|------------|------------|------------|------------|------------|------------|------------|------------|
|              | Selected   | Alternative| Selected   | Alternative| Selected   | Alternative| Selected   | Alternative|
| XGBoost      | 76.2       | 72.6       | 42         |            | 44.3       | 44.4       | 42.7       |  42.1      |
| LSTM         |            |            |            |            |            |            |            |            |
| Transformer  |            |            |            |            |            |            |            |            |
| xLSTM        |            |            |            |            |            |            |            |            |


### Scripts
**`1_data_processing/`** contains scripts for data preprocessing, dataset splitting, and generating traces, prefixes, and next activities:  
- `preprocessing.py` handles data preprocessing
- `train_test_split.py` handles dataset splitting.  
- `create_trace_prefix.py`: generates traces (trace-based encoding), prefixes, log prefixes, and trace suffixes. 

## Implemented Architectures
**`2_models/`** contains the implementation scripts for all models used in this study:
- `create_dl_model.py` defines the deep learning architectures, including LSTM, Transformer, and xLSTM models.
- `xgboost.ipynb` implements the XGBoost model.

## Loss Functions, Training and Evaluation

## Results
### Overall performance with additional metrics
|              |**BPIC2017**|        |     |     |**BPIC2019**|        |     |     |**BPIC2020**|        |     |     |**BAC**     |        |     |     |
|--------------|------------|--------|-----|-----|------------|--------|-----|-----|------------|--------|-----|-----|------------|--------|-----|-----|
|              | Accuracy   | AUC-PR | Pre | Rec | Accuracy   | AUC-PR | Pre | Rec | Accuracy   | AUC-PR | Pre | Rec | Accuracy   | AUC-PR | Pre | Rec |
| XGBoost      |            |        |     |     |            |        |     |     |            |        |     |     |            |        |     |     |
| LSTM         |            |        |     |     |            |        |     |     |            |        |     |     |            |        |     |     |
| Transformer  |            |        |     |     |            |        |     |     |            |        |     |     |            |        |     |     |
| xLSTM        |            |        |     |     |            |        |     |     |            |        |     |     |            |        |     |     |

## Python Environment Setup
The implementation is based on **Python 3.12.7**. To set up the environment, download `requirements.txt` and run the following command::

```bash
pip install -r requirements.txt
