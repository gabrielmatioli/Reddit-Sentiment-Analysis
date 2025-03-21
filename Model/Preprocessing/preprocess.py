import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

reddit_df = pd.read_csv('../../Data/Reddit_Data.csv')[:15000]
reddit_df.dropna(inplace=True)

X, y = reddit_df['clean_comment'].copy(), reddit_df['category'].copy()

count_vectorizer = CountVectorizer()
X_transformed = count_vectorizer.fit_transform(X)

tfid = TfidfTransformer()
X_tfid = tfid.fit_transform(X_transformed).toarray()

sampler = SMOTE()
X_sampled, y_sampled = sampler.fit_resample(X_tfid, y)

clean_reddit_df = pd.DataFrame(X_sampled, columns=[col for col in count_vectorizer.get_feature_names_out() if not col.startswith('unnamed')])
clean_reddit_df['category'] = y_sampled
clean_reddit_df.dropna(inplace=True)

print(y_sampled.value_counts())

clean_reddit_df.to_csv('../../Data/Clean_Reddit_Data.csv', index=False)
