#### 1. Legal Holding Extraction from Italian Case Documents using Italian-LEGAL-BERT Text Summarization

Please install all requirements from requirements.txt file
    `pip install -r requirements.txt`
We used `torch==1.13.1` and `python 3.8.10` for our experiments
### 1. We divided the repo into 
    - Datasets
    - Models
    - Outputs

### 2. Datasets
Contains train, test, validation along with the presented EDA in the paper

To run the EDA `python eda.py`. The results are updated in the outputs folder

### 3. Models
To run baseline lex rank model `python lexrank_baseline.py`
To run Arithmetic Mean BERT model `python arithmetic_mean_bert.py`
To run **Harmonic Mean BERT** model `python harmonic_mean_bert.py`
Note:
 `BERT Extractive Models also consists Evaluation and testing of the models. The results will be sent to outputs folders in json files with the respective model name`
