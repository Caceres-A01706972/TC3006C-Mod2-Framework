# Load the necessary librariess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix


"""
    Load the Iris dataset

    Parameters:
        file_path (str): The path to the CSV file containing the dataset.

    Returns:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target labels.
    """
def load_iris_dataset(file_path='DataSets/iris.data'):
    # Load the dataset from the path
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
    data = pd.read_csv(file_path, names=column_names)
    # Mapear los target labels a valores numericos
    target_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    data['target'] = data['target'].map(target_mapping)

    # features and target labels
    X = data.drop('target', axis=1).values
    y = data['target'].values

    return X, y


"""
    Entrenamiento con logistic regression model.

    Parameters:
        X_train (numpy.ndarray): Feature matrix for training.
        y_train (numpy.ndarray): Target labels for training.
        max_iter (int): Maximum number of iterations for model training.

    Returns:
        model (LogisticRegression): Trained logistic regression model.
    """
def train_logistic_regression(X_train, y_train, max_iter=200):
    # Create a logistic regression model
    model = LogisticRegression(max_iter=max_iter)
    # Entrenamiento 
    model.fit(X_train, y_train)
    return model



"""
    Evaluar el modelo que fue entrenado

    Parameters:
        model (LogisticRegression): Trained logistic regression model.
        X_train (numpy.ndarray): Feature matrix for training.
        y_train (numpy.ndarray): Target labels for training.
        X_test (numpy.ndarray): Feature matrix for testing.
        y_test (numpy.ndarray): Target labels for testing.

    Returns:
        train_accuracy (float): Accuracy on the training dataset.
        test_accuracy (float): Accuracy on the test dataset.
    """
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Make predictions on the training and test datasets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate accuracy scores
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    #Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # Classification report
    classification_rep = classification_report(y_test, y_test_pred)

    return train_accuracy, test_accuracy, conf_matrix, classification_rep



"""
    Plotear las learning curves del mdelo.

    Parameters:
        model (LogisticRegression): Trained logistic regression model.
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target labels.
        train_sizes (numpy.ndarray): Array of training sizes to use for learning curves (default=np.linspace(0.1, 1.0, 10)).
        cv (int): Number of cross-validation folds (default=5).

    Returns:
        nada
    """
def plot_learning_curves(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes=train_sizes, cv=cv)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Precisión en entrenamiento')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Precisión en prueba')
    plt.xlabel('Tamaño del conjunto de entrenamiento')
    plt.ylabel('Precisión')
    plt.title('Curvas de Aprendizaje')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_conf_matrix(conf_matrix):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Reales')
    plt.title('Matriz de Confusión')
    plt.show()

"""
    Si el usuario quiere hacer predicciones usando el modelo que fue entranado.

    Parameters:
        model (LogisticRegression): Trained logistic regression model.

    Returns:
        nada
    """
def make_predictions(model):
    while True:
        user_choice = input("Do you want to make a prediction? (yes/no): ").lower()
        if user_choice == "no":
            print("Exiting the program.")
            break
        elif user_choice == "yes":
            print("\nIngrese valores para predecir la clase de una flor Iris:")
            sepal_length = float(input('Longitud del Sépalo: '))
            sepal_width = float(input('Ancho del Sépalo: '))
            petal_length = float(input('Longitud del Pétalo: '))
            petal_width = float(input('Ancho del Pétalo: '))

            # Hace predccion con lo que metio el usuario
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(features)

            print(f'Clase predicha: {prediction[0]}')


if __name__ == "__main__":
    X, y = load_iris_dataset()
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the logistic regression model
    model = train_logistic_regression(X_train, y_train)
    # Evaluate and print model accuracy
    train_accuracy, test_accuracy, conf_matrix, classification_rep = evaluate_model(model, X_train, y_train, X_test, y_test)
    print(f'Precisión en conjunto de entrenamiento: {train_accuracy * 100:.2f}%')
    print(f'Precisión en conjunto de prueba: {test_accuracy * 100:.2f}%')
    print(classification_rep)
    # Plot learning curves
    plot_learning_curves(model, X, y)
    # Plot conf matrix
    plot_conf_matrix(conf_matrix)
    # Make predictions
    make_predictions(model)
