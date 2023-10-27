import pandas as pd
df = pd.read_csv('dato.csv')
display(df)

muestreo = df.sample(frac=0.2)
display(muestreo)

dfna=muestreo.dropna()
display(dfna)
