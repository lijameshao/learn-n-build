"""
Financial news dataset from:
https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology, 65(4), 782-796.

1. Train baseline model (TF-IDF, Naive Bayes)
2. Compare with Vader Sentiment

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pickle


def get_f1_score(model, X_train_tfidf, y_train):
    """
    Get average f1 score
    """
    kfold = StratifiedKFold(10, shuffle=True, random_state=3)

    f1 = cross_val_score(model, X_train_tfidf, y_train,
                          scoring="f1_micro", cv=kfold)

    return f1.mean()

def get_nb_best_alpha(X_train_tfidf, y_train):
    """
    Get the best alpha for MultinomialNB
    """
    results = pd.Series([
        get_f1_score(MultinomialNB(alpha), X_train_tfidf, y_train)
        for alpha in np.arange(0.05, 5, 0.05)], index=np.arange(0.05, 5, 0.05))

    return results.idxmax()


def handle_class_imbalance(df, undersample=True):
    
    class_label_count = df["label"].value_counts()
    if undersample:
        target_num_labels = class_label_count.min()
    else:
        target_num_labels = class_label_count.max()
    
    df_resampled = pd.DataFrame()
    
    for label in class_label_count.index:
        df_class = df[df["label"]==label]
        df_class = df_class.sample(target_num_labels)
        df_resampled = pd.concat([df_resampled, df_class])

    return df_resampled


financial_news_fp = "data/financial_news_data.csv"
df = pd.read_csv(financial_news_fp, names=["label", "text"], engine="python")
replace_labels = {"neutral": 0, "negative": -1, "positive": 1}

df["text"] = df["text"].str.replace("[^\x00-\x7F]+", " ", regex=True)
df["label"] = df["label"].replace(replace_labels)

# Handle Class imbalance
# df["label"].hist()
df = handle_class_imbalance(df, undersample=True)


X = df["text"]
Y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,
                                                    random_state=3)

tfidf = TfidfVectorizer(lowercase=True, smooth_idf=True, stop_words={"english"})

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

best_alpha = get_nb_best_alpha(X_train_tfidf, y_train)

model = MultinomialNB(alpha=best_alpha)
model.fit(X_train_tfidf, y_train)
pred = model.predict(X_test_tfidf)
print(classification_report(y_test, pred))

df_test_predicted = pd.DataFrame( data={"text": X_test, "ground_truth": y_test,
                                        "pred": pred})


with open("trained_models/baseline_tfidf_nb.pkl", "wb") as f:
    pickle.dump(model, f)


"""
              precision    recall  f1-score   support

          -1       0.64      0.84      0.73       176
           0       0.67      0.62      0.64       188
           1       0.62      0.49      0.55       180

    accuracy                           0.65       544
   macro avg       0.64      0.65      0.64       544
weighted avg       0.64      0.65      0.64       544
"""


# =============================================================================
# 
# Vader sentiment comparison
# 
# =============================================================================

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def categorise_vader_compound_score(score, pos_threshold, neg_threshold):

    if score <= neg_threshold:
        return -1
    elif score > neg_threshold and score <= pos_threshold:
        return 0
    else:
        return 1

analyzer = SentimentIntensityAnalyzer()

vader_sentiment = []

for sentence in X_test:
    vs = analyzer.polarity_scores(sentence)
    sentiment = categorise_vader_compound_score(
        vs["compound"], 0.05, -0.05)
    vader_sentiment.append(sentiment)

df_test_predicted["vader_sentiment"] = vader_sentiment
print(classification_report(y_test, vader_sentiment))

"""
              precision    recall  f1-score   support

          -1       0.67      0.25      0.36       176
           0       0.49      0.51      0.50       188
           1       0.46      0.72      0.56       180

    accuracy                           0.49       544
   macro avg       0.54      0.49      0.47       544
weighted avg       0.54      0.49      0.47       544

"""
