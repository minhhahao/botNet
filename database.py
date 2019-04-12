import pandas as pd

data = '/home/aazasdass/Documents/Coding/tob/RC/RC_2005-12.json'

df = pd.read_json(data, orient = 'columns')
df.head(10)
