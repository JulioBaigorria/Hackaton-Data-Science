import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df_cultivo = pd.read_csv('estimaciones-agricolas-2023-10.csv',
                         sep=',', encoding='latin-1', low_memory=False)
df_cultivo = df_cultivo[df_cultivo['cultivo'] == 'Cebada forrajera']

df_cultivo[['rendimiento', 'sup_sembrada', 'sup_cosechada', 'produccion']].corr()

X = df_cultivo[['sup_sembrada']]
y = df_cultivo['rendimiento']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# Graficar las predicciones contra los valores reales
plt.scatter(y_test, predicciones)
plt.xlabel('Rendimiento Real')
plt.ylabel('Predicciones de Rendimiento')
plt.title('Predicciones de Rendimiento vs. Rendimiento Real')
plt.show()

