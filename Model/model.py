import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def sentiment_analysis_model(x, y, return_clf=True, test_size=0.20, seed=42,):

    clf = MultinomialNB()
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    clf.fit(X_train, y_train)

    y_prediction = clf.predict(X_test)

    clf_report = classification_report(y_test, y_prediction)

    print(f'Sentiment Analysis Report: {clf_report}')

    if return_clf:
        return clf

reddit_df = pd.read_csv('../Data/Clean_Reddit_Data.csv')
x, y = reddit_df.drop(columns=['category']).copy(), reddit_df['category'].copy()

clf = sentiment_analysis_model(x, y)

