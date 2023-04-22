import os
from typing import List
import numpy as numpy
import torch.cuda
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from LexRank import degree_centrality_scores
from rouge import Rouge
from nltk.stem import SnowballStemmer
import numpy as np
import nltk
import re
from datasets import Dataset, DatasetDict
import pandas as pd
nltk.download('punkt')
import json
import sys
sys.path.append('.')

def remove_empty_segments_and_headings(text):
    return [line.strip('\n') for line in text if len(line.split())> 10]


def compute_lexrank_sentences(model, segments, device, num_segments):
    ''' 
    compute lexrank sentences
    args:
      model: sentence transformer model
      segments: list of segments
      device: cuda or cpu
      num_segments: number of segments to return
    returns:
      sorted central_indices: indices of central sentences
    '''

    embeddings = model.encode(segments, convert_to_tensor= True, device = device)

    similarities = cos_sim(embeddings, embeddings).cpu().numpy()

    centrality_scores = degree_centrality_scores(similarities, threshold = None, increase_power= True)

    central_indices = np.argpartition(centrality_scores, -num_segments)[-num_segments:]

    return sorted(list(central_indices))


def get_segmented_text(text, level ='paragraph', filter_length =5):
    ''' 
    segment text into paragraphs or sentences and remove empty segments
    args:
      text: text to segment
      level: paragraph or sentence
      filter_length: minimum length of segment
    returns:  
      segments: list of segments'''

    if level == 'paragraph':
      segments = text.split('\n')
    elif level == 'sentences':
      raise NotImplementedError('not implemented yet')
    else:
      raise ValueError('use Paragraph or sentence')
    segments = remove_empty_segments_and_headings(segments)
    return segments

def to_dataset(df_train):
    return Dataset.from_pandas(df_train)

def compute_median_character_compression_ratio(dataset):
    """
    Use the training data to determine the median compression ratio of articles.
    """
    ratios = []
    for i in dataset:
        ratio = len(get_segmented_text(i["doc"])) / len(get_segmented_text(i["summary"]))
        ratios.append(ratio)
    return np.median(ratios)

class ItaRouge(Rouge):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.lang = 'italian'
      Rouge.STEMMER = stemmer_snowball = SnowballStemmer(self.lang)

    def _preprocess_summary_as_a_whole(self, summary):
          """
          Preprocessing (truncate text if enable, tokenization, stemming if enable, lowering) of a summary as a whole
          Args:
            summary: string of the summary
          Returns:
            Return the preprocessed summary (string)
          """
  

          summary = Rouge.REMOVE_CHAR_PATTERN.sub(' ', summary.lower()).strip()

    
          tokens = super().tokenize_text(summary, self.lang)  

          if self.stemming:
              self.stem_tokens(tokens) # stemming in-place

          preprocessed_summary = [' '.join(tokens)]

          return preprocessed_summary

def get_test_results(df):
    ''' evaluate the test set
    args:
      df: test dataset
    returns:
      scores: dictionary of scores
    '''
    evaluator = ItaRouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=2,
                            apply_avg=True,
                            apply_best=False,
                            alpha=0.5, # Default F1_score
                            weight_factor=1.2,
                            stemming=True)
    all_hypothesis = df['bert_summary'].values
    all_references = df['summary'].values
    scores = evaluator.get_scores(all_hypothesis, all_references)
    return scores 

def compute_summaries():
    ''' 
    Compute lex rank summaries and scores for the test set
    returns:
      scores_test: dictionary of scores to outputs folder
    '''
    df_train = pd.read_csv('./Dataset/df_train.csv')
    df_test = pd.read_csv('./Dataset/df_test.csv')
    variant = 'paragraph'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('dlicari/Italian-Legal-BERT', device = device)
    median_compression_ratio = compute_median_character_compression_ratio(to_dataset(df_train))
    for i, row in df_test.iterrows():
      segments = get_segmented_text(row['doc'])
      expected_length = round(len(segments) / median_compression_ratio)
      most_central_indices = compute_lexrank_sentences(model, segments, device, expected_length)
      bert_summary = [segments[idx] for idx in sorted(most_central_indices)]
      df_test.at[i, 'bert_summary'] = '\n'.join(bert_summary)
    scores_test = get_test_results(df_test)
    with open('./outputs/lexrank_scores_test.json', 'w') as fp:
        json.dump(scores_test, fp)
      
    #remove this to not print results
    for metric, values in scores_test.items():
          precision = round(values['p'] * 100, 2)
          recall = round(values['r'] * 100, 2)
          f1_score = round(values['f'] * 100, 2)
          print(f"\t{metric}: P: {precision:>6.2f} R: {recall:>6.2f} F1: {f1_score:>6.2f}")

if __name__ == '__main__':
    compute_summaries()
