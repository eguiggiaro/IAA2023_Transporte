import pandas as pd
import autokeras as ak

## PROBANDO CON AUTOKERAS

# Leer el archivo CSV
data = pd.read_csv('DATOS_TRANSPORTE.CSV')

data['ES_FERIADO'] = data['ES_FERIADO'].map({'SI': 1, 'NO': 0})
data['HAY_EVENTO'] = data['HAY_EVENTO'].map({'SI': 1, 'NO': 0})

# Dividir los datos en características de entrada (X) y etiquetas de salida (y)
X = data.drop('CANTIDAD', axis=1)
y = data['CANTIDAD']
X = pd.get_dummies(X)

# Crear y entrenar el modelo utilizando Auto-Keras
model = ak.StructuredDataRegressor(max_trials=10)  # número máximo de pruebas aquí
model.fit(X, y, epochs=10)

# Evaluar el modelo
evaluation = model.evaluate(X, y)
print('Pérdida en el conjunto de datos:', evaluation)

# Prompt para probar la red
while True:
    dia_transporte = input('Día del transporte (dd/mm/yyyy): ')
    linea = input('Línea: ')
    tipo_transporte = input('Tipo de transporte (COLECTIVO/LANCHAS/SUBTE/TREN): ')
    provincia = input('Provincia: ')
    municipio = input('Municipio: ')
    es_feriado = input('¿Es feriado? (SI/NO): ')
    hay_evento = input('¿Hay evento? (SI/NO): ')
    
    # Crear un DataFrame con los datos ingresados
    input_data = pd.DataFrame([[dia_transporte, linea, tipo_transporte, provincia, municipio, es_feriado, hay_evento]],
                              columns=['DIA_TRANSPORTE', 'LINEA', 'TIPO_TRANSPORTE', 'PROVINCIA', 'MUNICIPIO', 'ES_FERIADO', 'HAY_EVENTO'])
    
    # Convertir los valores booleanos a 1 o 0 (int)
    input_data['ES_FERIADO'] = input_data['ES_FERIADO'].map({'SI': 1, 'NO': 0})
    input_data['HAY_EVENTO'] = input_data['HAY_EVENTO'].map({'SI': 1, 'NO': 0})
    input_data = pd.get_dummies(input_data)
    
    # Asegurarse de que las columnas coincidan con las columnas utilizadas durante el entrenamiento
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    
    # Predecir la cantidad
    prediction = model.predict(input_data)
    print('Estimación de la cantidad:', prediction[0])

## PROBANDO MANUALMENTE

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Leer el archivo CSV
data = pd.read_csv('DATOS_TRANSPORTE.CSV')

# Convertir los valores de las columnas booleanas a 1 o 0 (int)
data['ES_FERIADO'] = data['ES_FERIADO'].map({'SI': 1, 'NO': 0})
data['HAY_EVENTO'] = data['HAY_EVENTO'].map({'SI': 1, 'NO': 0})

# Dividir los datos en características de entrada (X) y etiquetas de salida (y)
X = data.drop('CANTIDAD', axis=1)
y = data['CANTIDAD']

# Convertir las características de entrada a one-hot encoding
X = pd.get_dummies(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X.astype(float), y.astype(float), test_size=0.2, random_state=42)

# Crear el modelo de red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Pérdida en el conjunto de prueba:', loss)
print('Precisión en el conjunto de prueba:', accuracy)

# Prompt para probar la red
while True:
    dia_transporte = input('Día del transporte (dd/mm/yyyy): ')
    linea = input('Línea: ')
    tipo_transporte = input('Tipo de transporte (COLECTIVO/LANCHAS/SUBTE/TREN): ')
    provincia = input('Provincia: ')
    municipio = input('Municipio: ')
    es_feriado = input('¿Es feriado? (SI/NO): ')
    hay_evento = input('¿Hay evento? (SI/NO): ')
    
    # Crear un DataFrame con los datos ingresados
    input_data = pd.DataFrame([[dia_transporte, linea, tipo_transporte, provincia, municipio, es_feriado, hay_evento]],
                              columns=['DIA_TRANSPORTE', 'LINEA', 'TIPO_TRANSPORTE', 'PROVINCIA', 'MUNICIPIO', 'ES_FERIADO', 'HAY_EVENTO'])
    
    # Convertir los valores booleanos a 1 o 0 (int)
    input_data['ES_FERIADO'] = input_data['ES_FERIADO'].map({'SI': 1, 'NO': 0})
    input_data['HAY_EVENTO'] = input_data['HAY_EVENTO'].map({'SI': 1, 'NO': 0})
    
    # Aplicar one-hot encoding a las características de entrada
    input_data = pd.get_dummies(input_data)
    
    # Asegurarse de que las columnas coincidan con las columnas utilizadas durante el entrenamiento
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    
    # Predecir la cantidad
    prediction = model.predict(input_data)
    print('Estimación de la cantidad:', prediction[0][0])
