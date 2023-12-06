# Trabajo para Hackaton de Ciencia de Datos.

## Fuente Principal: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## Paper: [NUCLEAR FEATURE EXTRACTION FOR BREAST TUMOR DIAGNOSIS](https://minds.wisconsin.edu/bitstream/handle/1793/59692/TR1131.pdf;jsessionid=A17A68C4112D52DF0267D965E7522D75?sequence=1)

**Tema:** Detección de Cáncer de Mamas a travez de muestras multidimencionales, obtenidas digitalmente mediante la técnica y procedimiento de Biopsia por aspiración con aguja fina.

**Datos:** La fuente de datos tomada representa cada muestra obtenida con el diagnóstico (Benigno - Maligno) y atributos de la muestra obtenida siendo Radio, Textura, Perimetro, Área, Suavidad, Compacidad, Concavidad, Puntos Concavos, Simetria y Dimension Fractal.

El plano de separación que se utiliza para predecir la presencia o ausencia de cáncer de mama se obtuvo utilizando el árbol del método multisuperficie (MSM-T), un método de clasificación que utiliza programación lineal para construir un árbol de decisión.

El MSM-T funciona buscando un conjunto de características que puedan separar los datos de cáncer de los datos de no cáncer. Para ello, el MSM-T realiza una búsqueda exhaustiva en el espacio de 1 a 4 características y 1 a 3 planos de separación.

**Observaciones:**

Archivos:
descripcion.csv -> Descripción del Dataset (Cuenta, Media, Desvío estándar, Minimo, Media, etc.)
correlaciones_b.csv -> Correlaciones de la categoría B
correlaciones_m.csv -> Correlaciones de la categoría M
correlaciones.csv -> Correlaciones del Dataset completo.
wdbc.csv -> Dataset con las muestras.
wdbc.xlsx -> Dataset con las muestras en excel en caso de necesitarse.


1. Responder: ¿Qué tienen en común los datos de cada tipo de diagnóstico? ¿Qué los diferencia de los Malignos y Benignos?
2. Graficar histogramas para revisar la distribución de los distintos features.
3. Graficar con un dotplot para estudiar si hay alguna tendencia.
4. Graficar con Boxplot.

**Objetivo:**

- Leer el Paper. Lo primero que deberíamos hacer todos para entender sobre el asunto. Está en inglés.
- Realizar análisis exploratorio para explicar los casos.
- Redactar un Paper entregable.
- Convenir qué parte expondrá cada uno.
- Armar un modelo predictor. Máximo 2.






