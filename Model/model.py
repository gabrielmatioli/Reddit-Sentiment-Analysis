import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

reddit_df = pd.read_csv('../Data/Clean_Reddit_Data.csv')
x, y = reddit_df.drop(columns=['category']).copy(), reddit_df['category'].copy()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

clf_report = classification_report(y_test, y_pred)
print(clf_report)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot()
plt.show()