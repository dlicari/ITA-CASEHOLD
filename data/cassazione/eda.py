import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns


def compression(df):
    '''
    Calculates the token compression ratio
    Args:
      df: dataframe
      Returns:
        compression ratio rounded to 2 decimal places'''
    return round(df['num_doc_tokens']/df['num_sum_tokens'],2)


#nltk tokenizer
def tokenize_text(text):
    '''
    we use nltk tokenizer to tokenize the text
    Args:
      text: string
    Returns:
        len of the tokenized text'''
    return len(nltk.word_tokenize(text, language='italian'))

def generate_eda():
    ''' 
    Generate eda plot, statistics of data and saves them to outputs folder
    '''
    df_train = pd.read_csv('./Dataset/df_train.csv')
    df_val = pd.read_csv('./Dataset/df_val.csv')
    df_test = pd.read_csv('./Dataset/df_test.csv')
    df_train['num_doc_tokens'] = df_train.doc.apply(tokenize_text)
    df_train['num_sum_tokens'] = df_train.summary.apply(tokenize_text)
    df_train['compression'] = compression(df_train)
    df_test['num_doc_tokens'] = df_test.doc.apply(tokenize_text)
    df_test['num_sum_tokens'] = df_test.summary.apply(tokenize_text)
    df_test['compression'] = compression(df_test)
    df_val['num_doc_tokens'] = df_val.doc.apply(tokenize_text)
    df_val['num_sum_tokens'] = df_val.summary.apply(tokenize_text)
    df_val['compression'] = compression(df_val)
    train_describe = pd.DataFrame({'stats':df_train.describe()})
    test_describe = pd.DataFrame({'stats':df_test.describe()})
    val_describe = pd.DataFrame({'stats':df_val.describe()})
    train_describe.to_csv('./outputs/train_data_stats.csv')
    test_describe.to_csv('./outputs/test_data_stats.csv')
    val_describe.to_csv('./outputs/val_data_stats.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    axes[0].hist(df_train['num_doc_tokens'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_title('Train data documents distribution')

    axes[1].hist(df_train['num_sum_tokens'], bins=20, edgecolor='black', alpha=0.7)
    axes[1].set_title('Train data summaries distribution')

    axes[2].hist(df_train['compression'], bins=20, edgecolor='black', alpha=0.7)
    axes[2].set_title('Train data compression distribution')

    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/bert_extractive/train_data_distribution.png')
    plt.show()


if __name__ == '__main__':
    generate_eda()

