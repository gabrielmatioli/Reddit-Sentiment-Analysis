import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

reddit_df = pd.read_csv('../../Data/Reddit_Data.csv')[:10000]
reddit_df.dropna(inplace=True)
reddit_df['clean_comment'] = reddit_df['clean_comment']

X, y = reddit_df['clean_comment'].copy(), reddit_df['category'].copy()

count_vectorizer = CountVectorizer()
X_transformed = count_vectorizer.fit_transform(X)

tfid = TfidfTransformer()
X_tfid = tfid.fit_transform(X_transformed).toarray()

clean_reddit_df = pd.DataFrame(X_tfid, columns=count_vectorizer.get_feature_names_out())

clean_reddit_df.to_csv('../../Data/Clean_Reddit_Data.csv')