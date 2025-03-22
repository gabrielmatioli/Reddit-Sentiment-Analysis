import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def sentiment_analysis_model(x, y, return_clf=True, test_size=0.20, seed=42, alpha=1):

    if not isinstance(return_clf, bool):
        raise ValueError('return_clf is not an instance of bool')
    
    elif not isinstance(test_size, (float, int)) or not (0 < test_size <= 1):
        raise ValueError('test_size must be a float or int between 0 and 1')
    
    elif not isinstance(seed, (float, int)) or not (seed >= 0):
        raise ValueError('seed must be a non-negative float or int')
    
    elif not isinstance(alpha, (float, int)) or not (alpha >= 0):
        raise ValueError('alpha must be non-negative float or int')

    try:
        
        clf = MultinomialNB(alpha=alpha)
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

        clf.fit(X_train, y_train)

        y_prediction = clf.predict(X_test)

        clf_report = classification_report(y_test, y_prediction)

        print('Sentiment Analysis Report:\n')
        print(clf_report)

        if return_clf:
            return clf
        
    except Exception as e:
        raise RuntimeError(f'Error occurred: {e}')

reddit_df = pd.read_csv('../Data/Clean_Reddit_Data.csv')
x, y = reddit_df.drop(columns=['category']).copy(), reddit_df['category'].copy()

clf = sentiment_analysis_model(x, y, alpha=0.15)

