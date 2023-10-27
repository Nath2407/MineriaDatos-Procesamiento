import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('dato.csv')
display(df)

muestreo = df.sample(frac=0.2)
display(muestreo)

dfna=muestreo.dropna()
display(dfna)

scaler = StandardScaler()

dfna['NA_Sales'] = dfna['NA_Sales'].str.replace(',', '.')

# Convertir la columna en números de punto flotante
dfna['NA_Sales'] = dfna['NA_Sales'].astype(float)

dfna['NA_Sales'] = scaler.fit_transform(dfna['NA_Sales'].values.reshape(-1, 1))
