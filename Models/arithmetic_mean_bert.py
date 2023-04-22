import json
import logging
from datasets import metric
import nltk
import numpy as np
import pandas as pd
import tqdm as notebook_tqdm
from nltk.stem import SnowballStemmer
from rouge import Rouge
from simpletransformers.classification import (ClassificationArgs,
                                               ClassificationModel)

nltk.download("punkt")


class ItaRouge(Rouge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang = "italian"
        Rouge.STEMMER = stemmer_snowball = SnowballStemmer(self.lang)

    def _preprocess_summary_as_a_whole(self, summary):
        """
        Preprocessing (truncate text if enable, tokenization, stemming if enable, lowering) of a summary as a whole
        Args:
          summary: string of the summary
        Returns:
          Return the preprocessed summary (string)
        """

        summary = Rouge.REMOVE_CHAR_PATTERN.sub(" ", summary.lower()).strip()

        tokens = super().tokenize_text(summary, self.lang)

        if self.stemming:
            self.stem_tokens(tokens)  # stemming in-place

        preprocessed_summary = [" ".join(tokens)]

        return preprocessed_summary


def prepare_results(p, r, f):
    return "\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}".format(
        metric, "P", 100.0 * p, "R", 100.0 * r, "F1", 100.0 * f
    )

def print_scores(scores_dict):
    for k, v in scores_dict.items():
        print(f"{k}:")
        for metric, values in v.items():
            f1 = round(values['f'] * 100, 2)
            p = round(values['p'] * 100, 2)
            r = round(values['r'] * 100, 2)
            print(f"\t{metric}: P: {p}\tR: {r}\tF1: {f1}")

            
def calc_rouge(df):
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
    return df_sentences


def generate_rouge_scores(df):
    ''' 
    generate rouge scores for each sentence in the dataframe 
    Args:
      df: dataframe
    Returns:
      df_sentences: dataframe with each sentence as a row and rouge scores as a column
    '''
    df_sentences = pre_processing(df)
    scores = calc_rouge(df_sentences)
    df_sentences["scores"] = scores
    df_sentences.rename(columns={"sentence": "text", "scores": "label"}, inplace=True)
    df_sentences = df_sentences[["text", "label"]]
    df_sentences["text"].replace("", np.nan, inplace=True)
    df_sentences.dropna(inplace=True)
    return df_sentences


def generate_summary_eval(row, model):
    '''
    generates bert summaries for k values of 3,5,7
    Args:
      row: row of the dataframe
      model: model
      Returns:
        bert_summary_3: bert summary for k=3
        bert_summary_5: bert summary for k=5
        bert_summary_7: bert summary for k=7
        '''
    doc = row['doc']
    summary = row['summary']
    sentences = doc.split('\n')
    predictions, raw_outputs = model.predict(sentences)
    most_import_sentence_indices = np.argsort(-predictions)
    bert_summary_3 = '\n'.join([sentences[idx].strip() for idx in np.sort(most_import_sentence_indices[0:3])])
    bert_summary_5 = '\n'.join([sentences[idx].strip() for idx in np.sort(most_import_sentence_indices[0:5])])
    bert_summary_7 = '\n'.join([sentences[idx].strip() for idx in np.sort(most_import_sentence_indices[0:7])])
    return bert_summary_3, bert_summary_5, bert_summary_7


def get_scores_eval(df, model):
    '''
    generates bert summaries for k values of 3,5,7 and calculates rouge scores
    Args:
      df: dataframe -> Validation data
      model: model
      Returns:
        scores_dict: dictionary of rouge scores
        '''
    scores_dict = {}
    evaluator = ItaRouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                          max_n=2,
                          apply_avg=True,
                          apply_best=False,
                          alpha=0.5, # Default F1_score
                          weight_factor=1.2,
                          stemming=True)
    df['bert_summary_3'], df['bert_summary_5'], df['bert_summary_7'] = zip(*df.apply(lambda row: generate_summary_eval(row, model), axis=1))
    all_hypothesis = df['bert_summary_3'].values
    all_references = df['summary'].values
    scores_dict[f'bert_summary_3'] = evaluator.get_scores(all_hypothesis, all_references)

    all_hypothesis = df['bert_summary_5'].values
    scores_dict[f'bert_summary_5'] = evaluator.get_scores(all_hypothesis, all_references)

    all_hypothesis = df['bert_summary_7'].values
    scores_dict[f'bert_summary_7'] = evaluator.get_scores(all_hypothesis, all_references)

    return scores_dict


def generate_test_summary(row, model):
    ''' 
    generate summary for each row in the test dataframe
    Args:
      row: row in the dataframe
      model: model (the one that we trained))
    Returns:
      bert_summary: summary generated by the model on k=5 sentences
    '''
    doc = row["doc"]
    summary = row["summary"]
    sentences = doc.split("\n")
    predictions, raw_outputs = model.predict(sentences)
    most_import_sentence_indices = np.argsort(-predictions)
    bert_summary = "\n".join(
        [sentences[idx].strip() for idx in np.sort(most_import_sentence_indices[0:5])]
    )
    return bert_summary


def get_test_results(df):
    ''' 
    generate rouge scores for each row in the test dataframe
    Args:
      df: dataframe (test)
    returns: rouge scores
    '''
    evaluator = ItaRouge(
        metrics=["rouge-n", "rouge-l", "rouge-w"],
        max_n=2,
        apply_avg=True,
        apply_best=False,
        alpha=0.5,  # Default F1_score
        weight_factor=1.2,
        stemming=True,
    )
    all_hypothesis = df["bert_summary"].values
    all_references = df["summary"].values
    return evaluator.get_scores(all_hypothesis, all_references)
    

def training():
    '''
    generate rouge scores for each sentence in the train and validation dataframes
    model training
    evaluating the model for optimal k value
    testing the model
    results are exported to a json file to output folder
    '''
    df_train = pd.read_csv("./Dataset/df_train.csv")
    df_val = pd.read_csv("./Dataset/df_val.csv")
    df_train_sentences = generate_rouge_scores(df_train)
    df_val_sentences = generate_rouge_scores(df_val)
    df_train_sentences["text"].replace("", np.nan, inplace=True)
    df_train_sentences.dropna(inplace=True)
    df_val_sentences["text"].replace("", np.nan, inplace=True)
    df_val_sentences.dropna(inplace=True)
    train_df = df_train_sentences[["text", "label"]]
    eval_df = df_val_sentences[["text", "label"]]
    """ Training starts here"""
    logging.basicConfig(level=logging.DEBUG)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.INFO)
    max_seq_length = 256
    batch_size = 16
    # Enabling regression
    model_args = ClassificationArgs()
    model_args.num_train_epochs = 4
    model_args.evaluate_during_training = True
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.use_early_stopping = True
    model_args.early_stopping_metric = "eval_loss"
    model_args.early_stopping_patience = 15
    model_args.regression = True
    model_args.evaluate_during_training_steps = 1500
    model_args.max_seq_length = max_seq_length
    model_args.manual_seed = 42
    model_args.train_batch_size = batch_size
    model_args.eval_batch_size = batch_size
    model_args.best_model_dir = (
        "./outputs/am_bert/best_model"
    )
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False

    model = ClassificationModel(
        "bert",
        "dlicari/Italian-Legal-BERT",
        num_labels=1,
        args=model_args,
        use_cuda=True,
    )
    # Train model
    _, training_details = model.train_model(train_df, eval_df=eval_df)
    # show training details
    logging.info("training details")
    logging.info(pd.DataFrame(training_details))
    model = ClassificationModel(model.args.model_type, model.args.best_model_dir)

    
    '''
    Evaluating the model on validation dataset on k = 3,5,7, 
    '''
    scores_dict = get_scores_eval(df_val, model)
    print_scores(scores_dict)

    with open('./outputs/scores_eval_arithmetic_bert.json', 'w') as fp:
        json.dump(scores_dict, fp)


    """
    Testing the model on test set and k value set at 5
    """
    df_test = pd.read_csv("./Dataset/df_test.csv")
    df_test["bert_summary"] = df_test.apply(
        generate_test_summary, args=(model,), axis=1
    )
    scores_test_am = get_test_results(df_test)
    with open("./outputs/am_bert_scores_test.json", "w") as fp:
        json.dump(scores_test_am, fp)
    
    #remove this to not print results
    
    for metric, values in scores_test_am.items():
        precision = round(values['p'] * 100, 2)
        recall = round(values['r'] * 100, 2)
        f1_score = round(values['f'] * 100, 2)
        print(f"\t{metric}: P: {precision:>6.2f} R: {recall:>6.2f} F1: {f1_score:>6.2f}")


if __name__ == '__main__':
      training()

