
"""
Disaster Tweet Analysis: EDA, Cleaning, BERT
Author: Mohamed Niyaz
"""

# ==============================
# 1. Library Imports
# ==============================
import gc
import re
import string
import operator
from collections import defaultdict
import random

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import STOPWORDS

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ==============================
# 2. Data Loading
# ==============================
def load_data(train_path: str = "./Data/train.csv", test_path: str = "./Data/test.csv"):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    print(f"Training Set Shape = {df_train.shape}")
    print(f"Training Set Memory Usage = {df_train.memory_usage().sum() / 1024**2:.2f} MB")
    print(f"Test Set Shape = {df_test.shape}")
    print(f"Test Set Memory Usage = {df_test.memory_usage().sum() / 1024**2:.2f} MB")
    return df_train, df_test

# ==============================
# 3. Text Cleaning Utilities
# ==============================
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
HTML_PATTERN = re.compile(r'<.*?>')
MENTION_PATTERN = re.compile(r'@[A-Za-z0-9_]+')
HASHTAG_PATTERN = re.compile(r'#')
PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation)

def clean_text(text: str) -> str:
    """Apply basic cleaning steps to a tweet text."""
    text = URL_PATTERN.sub('', text)
    text = HTML_PATTERN.sub('', text)
    text = MENTION_PATTERN.sub('', text)
    text = HASHTAG_PATTERN.sub('', text)
    text = text.translate(PUNCTUATION_TABLE)
    text = text.lower()
    return text.strip()

# ==============================
# 4. Feature Engineering
# ==============================
METAFEATURES = [
    'word_count',
    'unique_word_count',
    'stop_word_count',
    'url_count',
    'mean_word_length',
    'char_count',
    'punctuation_count',
    'hashtag_count',
    'mention_count',
]

def add_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['unique_word_count'] = df['text'].apply(lambda x: len(set(str(x).split())))
    df['stop_word_count'] = df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    df['url_count'] = df['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
    df['mean_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0)
    df['char_count'] = df['text'].apply(lambda x: len(str(x)))
    df['punctuation_count'] = df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df['hashtag_count'] = df['text'].apply(lambda x: str(x).count('#'))
    df['mention_count'] = df['text'].apply(lambda x: str(x).count('@'))
    return df

# ==============================
# 5. BERT Classifier
# ==============================
class BertTweetClassifier:
    def __init__(self,
                 hub_url: str = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
                 max_seq_length: int = 160,
                 lr: float = 1e-4,
                 epochs: int = 3,
                 batch_size: int = 16,
                 folds: int = 5):
        self.max_seq_length = max_seq_length
        self.bert_layer = hub.KerasLayer(hub_url, trainable=True)
        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        import tokenization  # uses official BERT tokenization script
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.folds = folds
        self.models = []
        self.scores = {}

    def encode(self, texts):
        all_tokens, all_masks, all_segments = [], [], []
        for text in texts:
            text = self.tokenizer.tokenize(text)[:self.max_seq_length - 2]
            input_sequence = ['[CLS]'] + text + ['[SEP]']
            pad_len = self.max_seq_length - len(input_sequence)
            tokens = self.tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * self.max_seq_length
            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)
        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

    def build_model(self):
        input_word_ids = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='input_word_ids')
        input_mask = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='input_mask')
        segment_ids = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='segment_ids')
        pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]
        out = Dense(1, activation='sigmoid')(clf_output)
        model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        optimizer = SGD(learning_rate=self.lr, momentum=0.8)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def train(self, df: pd.DataFrame):
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=SEED)
        for fold, (trn_idx, val_idx) in enumerate(skf.split(df['text_cleaned'], df['target'])):
            print(f"\nFold {fold + 1}/{self.folds}\n{'-'*20}")
            X_trn = self.encode(df.loc[trn_idx, 'text_cleaned'])
            y_trn = df.loc[trn_idx, 'target']
            X_val = self.encode(df.loc[val_idx, 'text_cleaned'])
            y_val = df.loc[val_idx, 'target']

            model = self.build_model()
            es = EarlyStopping(monitor='val_accuracy', patience=1, restore_best_weights=True, verbose=1)
            model.fit(X_trn, y_trn,
                      validation_data=(X_val, y_val),
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      callbacks=[es],
                      verbose=2)
            self.models.append(model)
            val_preds = (model.predict(X_val) > 0.5).astype(int)
            f1 = f1_score(y_val, val_preds)
            self.scores[f"fold_{fold+1}"] = f1
            print(f"Fold {fold+1} F1 Score: {f1:.4f}")

# ==============================
# 6. Main execution
# ==============================
if __name__ == "__main__":
    df_train, df_test = load_data()
    # fill missing keyword & location
    for df in [df_train, df_test]:
        for col in ['keyword', 'location']:
            df[col] = df[col].fillna(f"no_{col}")

    # clean text
    df_train['text_cleaned'] = df_train['text'].apply(clean_text)
    df_test['text_cleaned'] = df_test['text'].apply(clean_text)

    # add meta‑features
    df_train = add_meta_features(df_train)
    df_test = add_meta_features(df_test)

    # Example visualization (uncomment to run)
    sns.countplot(x='target', data=df_train); plt.show()

    # Train BERT model (optional, compute‑heavy)
    classifier = BertTweetClassifier(epochs=10)  # reduce epochs for demo
    classifier.train(df_train)

    # Save processed data
    df_train.to_csv("train_processed.csv", index=False)
    df_test.to_csv("test_processed.csv", index=False)
    print("Preprocessing complete. Processed files saved.")
