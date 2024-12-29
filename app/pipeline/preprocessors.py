import re
import json

import spacy
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Regex patterns for handling specific Reddit content
URL_PATTERN = r'http[s]?://\S+|www\.\S+'
MENTION_PATTERN = r'@\w+'
HASHTAG_PATTERN = r'#\w+'

# Load preprocessing parameters
with open('data/preprocessing_parameters.json', 'r') as stream:
    params = json.load(stream)
    slang_dict = params['slang_dict']
    symbol_dict = params['symbol_dict']
    emoji_dict = params['emoji_dict']
    stopword_exceptions = params['stopword_exceptions']
    punctuation_exceptions = params['punctuation_exceptions']
    important_ngrams = params['important_ngrams']


class BasePreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self


class RedditTextCleaner(BasePreprocessor):
    '''An sklearn transformer that cleans reddit comments.'''

    def transform(self, X):
        '''Preprocess a pandas series of strings.'''
        X_copy = pd.DataFrame(X)
        X_copy['text'] = X['text'].apply(self._preprocess_reddit_comment)
        return X_copy

    def _preprocess_reddit_comment(self, text):
        '''Custom tokenizer for Reddit comments'''
        text = re.sub(URL_PATTERN, '<URL>', text)
        text = re.sub(MENTION_PATTERN, '<USER>', text)
        text = re.sub(HASHTAG_PATTERN, '<HASHTAG>', text)
        return text


class Tokenizer(BasePreprocessor):
    '''An sklearn transformer that tokenizes and processes text.'''

    def __init__(self, exceptions=None):
        self.exceptions = exceptions or stopword_exceptions.union(
            punctuation_exceptions
        )
        self.nlp = spacy.load('en_core_web_lg')  # Pre-trained English model

    def transform(self, X):
        '''Tokenize a pandas series of strings.'''
        token_docs = self.nlp.pipe(
            X['text'], disable=["ner", "parser", "tok2vec"], batch_size=1000
        )
        X['tokens'] = [self._process_tokens(doc) for doc in token_docs]
        return X

    def _process_tokens(self, tokens):
        '''Translate slang and symbols and filter punctuation and stopwords.'''
        processed_tokens = []
        for token in tokens:
            if token.lower_ in self.exceptions:
                processed_tokens.append(token.lower_)
            elif token.lower_ in slang_dict:
                processed_tokens.append(slang_dict.get(token.lower_))
            elif token.text in emoji_dict:
                processed_tokens.append(emoji_dict.get(token.text))
            elif not token.is_stop and token.is_alpha:
                processed_tokens.append(token.lemma_.lower())
        return processed_tokens


class FeatureEngineer(BasePreprocessor):
    '''An sklearn transformer that engineers features.'''

    def __init__(self, ngrams):
        self.ngrams = ngrams

    def transform(self, X):
        X['comment_short'] = X['text'].apply(len).lt(50).astype('int8')
        X['comment_long'] = X['text'].apply(len).gt(100).astype('int8')
        X['exclamation_mark_count'] = X['text'].str.count('!')
        X['question_mark_count'] = X['text'].str.count(r'\?')
        X['masked_period_count'] = X['text'].str.count('.').sub(2).clip(0)
        X['double_dot_count'] = X['text'].str.count('..')
        X['quotation_count'] = X['text'].str.count('""')
        X['masked_uppercase_count'] = (
            X['text']
            .apply(
                lambda comment: sum(
                    1 for c in self._remove_artifacts(comment) if c.isupper()
                )
            )
            .sub(6)
            .clip(0)
        )
        X['double_uppercase_count'] = X['text'].apply(self._count_double_uppercase)
        X = self._add_ngrams(X)
        return X

    def _remove_artifacts(self, comment):
        artifacts = {
            '[T]',
            '[ALL]',
            '[NAME]',
            '[RELIGION]',
            '<URL>',
            '<USER>',
            '<HASHTAG>',
        }
        for artifact in artifacts:
            comment = comment.replace(artifact, '')
        return comment

    def _count_double_uppercase(self, comment):
        clean_comment = self._remove_artifacts(comment)
        return sum(
            1
            for i in range(len(clean_comment) - 1)
            if clean_comment[i].isupper() and clean_comment[i + 1].isupper()
        )

    def _add_ngrams(self, X):
        vectorizer = CountVectorizer(ngram_range=(1, 2), vocabulary=self.ngrams)
        ngrams_csrm = vectorizer.fit_transform(X['text'])
        ngram_names = ['ngram_' + ngram for ngram in vectorizer.get_feature_names_out()]
        ngrams_df = pd.DataFrame(
            ngrams_csrm.toarray(), index=X.index, columns=ngram_names
        )
        return pd.concat([X, ngrams_df], axis=1)


class TfidfTransformer(BasePreprocessor):
    '''A sklearn transformer that transforms X["tokens"] to TF-IDF vectors.'''

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.tfidf_vectorizer = TfidfVectorizer(
            preprocessor=self.keep_tokens,
            tokenizer=self.keep_tokens,
            token_pattern=None,
            vocabulary=self.vocabulary,
        )

    @staticmethod
    def keep_tokens(tokens):
        return tokens

    def fit(self, X, y=None):
        self.tfidf_vectorizer.fit(X['tokens'])
        return self

    def transform(self, X):
        tfidf_matrix = self.tfidf_vectorizer.transform(X['tokens'])
        tokens = self.tfidf_vectorizer.get_feature_names_out()
        tokens = ['tfidf_' + token for token in tokens]
        tfidf_df = pd.DataFrame.sparse.from_spmatrix(
            tfidf_matrix, columns=tokens, index=X.index
        )
        return pd.concat([X, tfidf_df], axis=1)


class Scaler(BasePreprocessor):
    '''An sklearn transformer that scales input for machine learning.'''

    def __init__(self):
        self.scaler = StandardScaler()
        self.unscaled_features = [
            'masked_period_count',
            'double_dot_count',
            'quotation_count',
            'exclamation_mark_count',
            'question_mark_count',
            'masked_uppercase_count',
            'double_uppercase_count',
        ]

    def fit(self, X, y=None):
        self.scaler.fit(X[self.unscaled_features])
        return self

    def transform(self, X):
        scaled_data = self.scaler.transform(X[self.unscaled_features])
        X_scaled = pd.DataFrame(
            scaled_data, index=X.index, columns=self.unscaled_features
        )
        X_scaled = pd.concat([X.drop(self.unscaled_features, axis=1), X_scaled], axis=1)
        return X_scaled


class Cleaner(BasePreprocessor):
    '''An sklearn transformer that cleans up input for machine learning.'''

    def transform(self, X):
        '''Drop temporary features and compress features.'''
        X_clean = X.drop(
            ['text', 'tokens', 'processed_tokens'], axis=1, errors='ignore'
        )
        X_clean.columns = X_clean.columns.astype(str)
        X_csrm = csr_matrix(X_clean)
        return X_csrm
