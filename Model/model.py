import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

twitter_df = pd.read_csv('../Data/Clean_Twitter_Data.csv')
X, y = twitter_df.drop(columns=['category']).copy(), twitter_df['category'].copy()
