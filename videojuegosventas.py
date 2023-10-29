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

# Reemplazar comas por puntos y convertir a tipo float para columnas específicas
columnas_con_comas = ['EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
for columna in columnas_con_comas:
    dfna[columna] = dfna[columna].str.replace(',', '.').astype(float)

# Normalización de características utilizando StandardScaler
escalador = StandardScaler()
dfna[columnas_con_comas] = escalador.fit_transform(dfna[columnas_con_comas])

# Verificar que las columnas estén normalizadas
print(dfna[columnas_con_comas].head())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Dividir los datos en conjuntos de entrenamiento y prueba
X = dfna.drop('Global_Sales', axis=1)  # Excluimos la variable objetivo 'Global_Sales'
y = dfna['Global_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir las columnas categóricas y numéricas
categorical_columns = ['Platform', 'Genre', 'Publisher', 'Rating', 'Critic_Score_Class']
numeric_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

# Crear transformadores para columnas categóricas y numéricas
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combinar los transformadores utilizando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Crear y entrenar un modelo de regresión lineal en un pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir resultados
print(f"Error cuadrático medio: {mse}")
print(f"Coeficiente de determinación (R^2): {r2}")
