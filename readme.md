Trabajo para Hackaton de Ciencia de Datos.

Fuente Principal: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Tema: Detección de Cáncer de Mamas a travez de muestras multidimencionales, obtenidas digitalmente mediante la técnica y procedimiento de Biopsia por aspiración con aguja fina.

Datos: La fuente de datos tomada representa cada muestra obtenida con el diagnóstico (Benigno - Maligno) y atributos de la muestra obtenida siendo Radio, Textura, Perimetro, Área, Suavidad, Compacidad, Concavidad, Puntos Concavos, Simetria y Dimension Fractal.

El plano de separación que se utiliza para predecir la presencia o ausencia de cáncer de mama se obtuvo utilizando el árbol del método multisuperficie (MSM-T), un método de clasificación que utiliza programación lineal para construir un árbol de decisión.

El MSM-T funciona buscando un conjunto de características que puedan separar los datos de cáncer de los datos de no cáncer. Para ello, el MSM-T realiza una búsqueda exhaustiva en el espacio de 1 a 4 características y 1 a 3 planos de separación.

Observaciones: 

El script genera un csv 'descripcion.csv' para sumarizar las variables cuantitativas, y otro csv 'correlaciones.csv' donde detalla la correlacion entre las distintas variables.

Responder: ¿Qué tienen en común los datos de cada tipo de diagnóstico?
Graficar histogramas para revisar la distribución de los distintos features.
Graficar con un dotplot para estudiar si hay alguna tendencia.

nota: con estos dos gráficos deberían ser suficientes para poder explicar el fenómeno o poder indagar más al respecto.






