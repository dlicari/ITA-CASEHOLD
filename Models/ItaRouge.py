from rouge import Rouge
import nltk
from nltk.stem.snowball import SnowballStemmer
nltk.download("punkt")
import pandas as pd
import numpy as np

#calculates ItaRouge scores 
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