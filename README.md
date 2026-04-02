# ABERTA

This repository contains the implementation of *Audio embeddings and Bidirectional Encoder Representations from Transformers with Attention (ABERTA)* described in "ABERTA: An Explainable End-to-End Approach for Alzheimer’s Disease Detection From Spontaneous Speech".

## About

ABERTA is a deep learning model for Alzheimer’s disease detection from spontaneous speech that seeks to integrate explainability into its architecture. To this end, it combines textual and acoustic information through attention mechanisms. This makes it easier to analyze the model's behavior and predictions, allowing us to identify patterns that help us understand its decision-making process.

### Dataset

For training and evaluating the model, the ADReSSo dataset was used. To access the dataset, you must request permission from [DementiaBank](https://talkbank.org/dementia/ADReSSo-2021/).

### Training

The model can be trained on the classification problem using the command:

```bash
python classify_train.py
```

The model can be trained on the MMSE prediction problem using the command:

```bash
python mmse_train.py
```

### Evaluation

The model checkpoints used for evaluation can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1LXpcHLyiSUnh1pVkYb0kL6MF_qhfR02a?usp=sharing).

## Citation

The paper is currently under review.