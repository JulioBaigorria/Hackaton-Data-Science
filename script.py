# Importar librerías

import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns

df_wdbc = pd.read_csv('wdbc.csv', sep=',')

df_wdbc.describe().T

df_wdbc.info()

df_wdbc.astype(object).describe().T

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


FLOAT_COLUMNS: list[str] = ['radius1', 'texture1', 'perimeter1', 'area1',
                            'smoothness1', 'compactness1', 'concavity1', 'concave_points1',
                            'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2',
                            'area2', 'smoothness2', 'compactness2', 'concavity2', 'concave_points2',
                            'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3',
                            'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3',
                            'symmetry3', 'fractal_dimension3']

df_wdbc_m = df_wdbc[df_wdbc['Diagnosis'] == 'M']

df_wdbc_b = df_wdbc[df_wdbc['Diagnosis'] == 'B']

df_wdbc_m[FLOAT_COLUMNS].corr().to_csv('correlaciones_m.csv', sep=',')

df_wdbc_b[FLOAT_COLUMNS].corr().to_csv('correlaciones_b.csv', sep=',')