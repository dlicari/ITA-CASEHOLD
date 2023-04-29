## Legal Holding Extraction from Italian Case Documents using Italian-LEGAL-BERT Text Summarization
This repository contains the code for the paper [Legal Holding Extraction from Italian Case Documents using Italian-LEGAL-BERT Text Summarization] accepted at [ICAIL 2023].

### Abstract
Legal holdings are used in Italy as a critical component of the legal system, serving to establish legal precedents, provide guidance for future legal decisions, and ensure consistency and predictability in the interpretation and application of the law. They are written by domain experts who describe in a clear and concise manner the principle of law applied in the judgments.

We introduce a legal holding extraction method based on Italian-LEGAL-BERT to automatically extract legal holdings from Italian cases. In addition, we present ITA-CaseHold, a benchmark dataset for Italian legal summarization. We conducted several experiments using this dataset, as a valuable baseline for future research on this topic.

### Instructions to run the experiments

#### Requirements
Please install all requirements from requirements.txt file
    `pip install -r requirements.txt`
We used `torch==1.13.1` and `python 3.8.10` for our experiments

#### We divided the repo into 
    - Datasets
    - Models
    - Outputs

#### Datasets
Contains train, test, validation along with the presented EDA in the paper

To download the dataset from huggingface, please use the following lines:
    
        from datasets import load_dataset
        dataset = load_dataset("itacasehold/itacasehold")


To run the EDA `python eda.py`. The results are updated in the outputs folder

#### Models

This folder contains all the models used in the experiments. We have divided the models into two categories

    - Lexrank (Baseline)
    - BERT Extractive Models

#### Lexrank (Baseline)
This is the baseline model for the experiments.
To run this baseline model

`python lexrank_baseline.py`

The results are stores in the output folder

#### BERT Extractive Models 

The file `run_model.py` contains the training, validation and testing of both the HM-BERT and AM-BERT models. The encoder in this experiment is `Italian-LEGAL-BERT`, you can change it and run it with any Encoder. 

Usage:

     python run_model.py --model arithmetic_mean_bert

or

     python run_model.py --model harmonic_mean_bert


Note:
`BERT Extractive Models also consists Evaluation and testing of the models. The results will be sent to outputs folder in json files. The trained model will be saved in outputs/best_model folder, please change the path in the code to save trained models of Hm-Bert and Am-Bert in different folders.`


