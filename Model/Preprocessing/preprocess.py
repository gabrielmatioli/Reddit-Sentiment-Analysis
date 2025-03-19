import pandas as pd
# from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline

twitter_df = pd.read_csv('../../Data/Twitter_Data.csv')[:70000]
twitter_df.dropna(inplace=True)
X, y = twitter_df['clean_text'].copy(), twitter_df['category'].copy()

# Negative and neutral comments are undersampled in this dataset
# imbalanced_verification = twitter_df['category'].value_counts()

pipe = Pipeline([('count', CountVectorizer()),
                ('tfid', TfidfTransformer())])
X_transformed = pipe.fit_transform(X)

clean_twitter_df = pd.DataFrame(X_transformed.toarray(), columns=pipe.named_steps['count'].get_feature_names_out())
clean_twitter_df['category'] = y

clean_twitter_df.to_csv('../../Data/Clean_Twitter_Data.csv')
