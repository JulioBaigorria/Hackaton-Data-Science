# Importar librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

############## LEER DOCUMENTO

df_wdbc = pd.read_csv('wdbc.csv', sep=',')

############## BALANCE DEL DATAFRAME REALIZANDO OVERSAMPLING

X = df_wdbc.drop(columns=['ID', 'Diagnosis'])  # Características
y = df_wdbc['Diagnosis']  # Etiquetas

# Crear una instancia de SMOTE
smote = SMOTE()

# Aplicar SMOTE para generar muestras sintéticas
X_oversampled, y_oversampled = smote.fit_resample(X, y)

# Unir las características y las etiquetas sintéticas en un nuevo DataFrame
df_oversampled = pd.concat([pd.DataFrame(X_oversampled, columns=X.columns),
                            pd.DataFrame(y_oversampled, columns=['Diagnosis'])],
                            axis=1)


############## CORRELACION POR GRUPO

FLOAT_COLUMNS: list[str] = ['radius1', 'texture1', 'perimeter1', 'area1',
                            'smoothness1', 'compactness1', 'concavity1', 'concave_points1',
                            'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2',
                            'area2', 'smoothness2', 'compactness2', 'concavity2', 'concave_points2',
                            'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3',
                            'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3',
                            'symmetry3', 'fractal_dimension3']

df_wdbc_m = df_wdbc[df_wdbc['Diagnosis'] == 'M']

df_wdbc_b = df_wdbc[df_wdbc['Diagnosis'] == 'B']

df_wdbc_m[FLOAT_COLUMNS].corr().to_csv('docs/correlaciones_m.csv', sep=',')

df_wdbc_b[FLOAT_COLUMNS].corr().to_csv('docs/correlaciones_b.csv', sep=',')


############## REGRESIÓN LOGÍSTICA

# Seleccionar las variables predictoras: radius3 y texture3.
X = df_oversampled[['radius3', 'texture3']]

# Seleccionar la variable objetivo: 'Diagnosis'.
y = df_oversampled['Diagnosis']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba, usando un margen de 0.2 reservado.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)

# Inicializar y entrenar el modelo de Regresión Logística.
model = LogisticRegression()
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba.
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo.
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Imprimir la matriz de confusión y el informe de clasificación.
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión')
plt.show()

print('\nInforme de Clasificación:')
print(classification_report(y_test, y_pred))


############## K N N

# Seleccionar variables predictoras
X = df_oversampled.drop(['ID', 'Diagnosis'], axis=1)

# Seleccionar la variable objetivo
y = df_oversampled['Diagnosis']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba. Ajustado Random State = 60
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)

# Inicializar el modelo KNN, ajustando K = 6
knn_model = KNeighborsClassifier(n_neighbors = 4 )

# Entrenar el modelo KNN
knn_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = knn_model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Imprimir la matriz de confusión y el informe de clasificación
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión')
plt.show()

print('\nInforme de Clasificación:')
print(classification_report(y_test, y_pred))


############## S V C

# Seleccionar variables predictoras

X = df_oversampled[['radius3', 'texture3']]
# Seleccionar la variable objetivo
y = df_oversampled['Diagnosis']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)

# Inicializar el modelo SVC
svc_model = SVC(kernel='linear')

# Entrenar el modelo SVC
svc_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = svc_model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Imprimir la matriz de confusión y el informe de clasificación
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=svc_model.classes_, yticklabels=svc_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión')
plt.show()

print('\nInforme de Clasificación:')
print(classification_report(y_test, y_pred))


############## REGRESIÓN LINEAL

X = df_oversampled[['perimeter1']]

# Seleccionar la variable objetivo
y = df_oversampled['concave_points1']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)

model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Visualizar los resultados
plt.scatter(X_test, y_test, color='black', label='Datos reales')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regresión lineal')
plt.xlabel('perimeter1')
plt.ylabel('concave_points1')
plt.legend()
plt.title('Regresión Lineal entre perimeter1 y concave_points1')
plt.show()

sns.scatterplot(x='compactness1', y='concavity1', hue='Diagnosis', data=df_wdbc, palette='viridis')
plt.xlabel('compactness1')
plt.ylabel('concavity1')
plt.title('Scatter Plot entre compactness1 y concavity1 (coloreado por Diagnosis)')
plt.show()


############## PLOTS

# Generar las imagenes de Boxplot.

sns.boxplot(x='Diagnosis', y='radius1', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='radius2', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='radius3', data=df_wdbc)

sns.boxplot(x='Diagnosis', y='texture1', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='texture2', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='texture3', data=df_wdbc)

sns.boxplot(x='Diagnosis', y='perimeter1', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='perimeter2', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='perimeter3', data=df_wdbc)

sns.boxplot(x='Diagnosis', y='area1', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='area2', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='area3', data=df_wdbc)

sns.boxplot(x='Diagnosis', y='smoothness1', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='smoothness2', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='smoothness3', data=df_wdbc)

sns.boxplot(x='Diagnosis', y='compactness1', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='compactness2', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='compactness3', data=df_wdbc)

sns.boxplot(x='Diagnosis', y='concavity1', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='concavity2', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='concavity3', data=df_wdbc)

sns.boxplot(x='Diagnosis', y='concavity1', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='concavity2', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='concavity3', data=df_wdbc)

sns.boxplot(x='Diagnosis', y='concave_points1', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='concave_points2', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='concave_points3', data=df_wdbc)

sns.boxplot(x='Diagnosis', y='symmetry1', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='symmetry2', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='symmetry3', data=df_wdbc)

sns.boxplot(x='Diagnosis', y='fractal_dimension1', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='fractal_dimension2', data=df_wdbc)
sns.boxplot(x='Diagnosis', y='fractal_dimension3', data=df_wdbc)


# Generar imágenes de distribución de datos
sns.histplot(df_wdbc['fractal_dimension1'], kde=True)
plt.title('Distribución de radius1')
plt.show()


# Pairplot de todas las variables cuantitativas 
selected_vars = df_wdbc.drop('ID', axis=1)
sns.pairplot(selected_vars, hue='Diagnosis')
plt.suptitle('Pairplot de variables seleccionadas', y=1.02)
plt.show()