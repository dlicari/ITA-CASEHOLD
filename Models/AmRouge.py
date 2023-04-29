from nltk.stem import SnowballStemmer
from rouge import Rouge
from ItaRouge import *
import pandas as pd
import numpy as np


def calc_am_rouge(df):
    ''' 
    Calculate rouge scores for Arithmetic mean
    Args:
      df: dataframe
    Returns:
      final_scores: array of rouge scores  '''
    evaluator = ItaRouge(
        metrics=["rouge-n"],
        max_n=2,
        apply_avg=False,
        apply_best=False,
        weight_factor=1.2,
        stemming=True,
    )
    all_hypothesis = df.sentence.values
    all_references = df.summary.values

    scores = evaluator.get_scores(all_hypothesis, all_references)
    scores_avg = {}
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        scores_avg[metric] = []
        for hypothesis_id, results_per_ref in enumerate(results):
            nb_references = len(results_per_ref["p"])
            for reference_id in range(nb_references):
                scores_avg[metric].append(results_per_ref["f"][reference_id] * 100.0)
    # we calculate arithmetic mean of R-1 and R-2
    final_scores = (
        np.array(scores_avg["rouge-1"]) + np.array(scores_avg["rouge-2"])
    ) / 2
    return final_scores


def pre_processing(df):
    '''
    Preprocessing the data
    Args: 
      df: dataframe
    Returns:
      df_sentences: dataframe with each sentence as a row
    '''
    df = df.drop_duplicates(subset=["summary"], keep="first")
    df["sentence"] = df.doc.apply(lambda x: x.split("\n"))
    df_sentences = df.explode("sentence")
    df_sentences["sentence"].replace("", np.nan, inplace=True)
    df_sentences.dropna(inplace=True)
    return df_sentences


def generate_am_rouge_scores(df):
    ''' 
    generate rouge scores for each sentence in the dataframe 
    Args:
      df: dataframe
    Returns:
      df_sentences: dataframe with each sentence as a row and rouge scores as a column
    '''
    df_sentences = pre_processing(df)
    scores = calc_am_rouge(df_sentences)
    df_sentences["scores"] = scores
    df_sentences.rename(columns={"sentence": "text", "scores": "label"}, inplace=True)
    df_sentences = df_sentences[["text", "label"]]
    df_sentences["text"].replace("", np.nan, inplace=True)
    df_sentences.dropna(inplace=True)
    return df_sentences