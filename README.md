# **ENTREGA: Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución.**
Este proyecto presenta una implementación de una Logistic Regression para la clasificación de flores iris. La implementación es capaz de predecir la clase de flores Iris en función de sus caractrísticas.

## Contenido
El código se compone de la siguiente manera:
1. **Importación de Librerías**: Se importan las librerías necesarias para hacer el manejo de los datos y la visualización.
2. **Carga el Conjunto de Datos**: Se carga un conjunto de datos *"iris.data"*. Se exrtraen las caracteristicas y los objetivos de los datos para su uso en el modelo.
3. **División en el Conjunto de Datos**: Divide el conjunto de datos en training y en test para evaluar el modelo.
4. **Entrenamiento del modelo**: Crea un modelo de Logistic Regression y lo entrena con el conjunto de training. Luego evalúa la presición del modelo en los conjuntos d etraining y test.
5. **Visualización de Curvas de Aprendizaje**: Muestra las curvas de aprendizaje para evaluar cómo el rendimiento del modelo mejora con el aumento del tamaño del conjunto de training.
6. **Hacer Predicciones**: Permite al usuario ingresar valores para predecir la clase de una flor Iris basado en sus características. Esto haciendo uso del modelo previamente entrenado.


## Uso del Código
- **Requisitos previos**
  - Tener las estas librerías instaladas:
  `numpy`
  `pandas`
  `matplotlib`
  `scikit-learn`
  `seaborn`

- **Ejecución del Código**
1. Ejecuta el código en un entorno de Python.
2. El programa realizará los siguientes pasos:
   - Cargar el conjunto de datos *iris.data*
   - Dividir el conjunto de datos en conjuntos de training y test.
   - Entrenar un modelo de Logistic Regression en el conjunto de training.
   - Evaluar la precisión del modelo en los conjuntos de training y test.
   - Mostrar las curvas de aprendizaje para visualizar el rendimiento del modelo.
3. Luego, el programa dará la opción de realizar predicciones ingresando valores para las características de una flor Iris. Las predicciones mostrarán la clase estimada de la flor.
