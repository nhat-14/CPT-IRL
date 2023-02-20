import pandas as pd
import numpy as np

df = pd.read_csv('~/Downloads/IRL_execute/data/1409-20.csv', index_col=None)
df1 = df.iloc[-109:-1]
df.to_csv('~/Downloads/IRL_execute/data/test.csv')