# Import pandas library
import pandas as pd
  
# initialize list elements
data = [1,2,3,4,5,6,7,8,9,10]
  
# Create the pandas DataFrame with column name is provided explicitly
df = pd.DataFrame(data, columns=['Numbers'])
df2= df['Numbers'].rolling(min_periods=1, window=3).sum()
print(df2)