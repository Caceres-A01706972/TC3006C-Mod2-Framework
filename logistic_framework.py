# Importar las lubrerias necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve



"""
    Load the diabetes dataset and separates the data into training and test

    Parameters:
        file_path (str): The path to the CSV file containing the dataset

    Returns:
        X_train (numpy.ndarray): Features for training.
        X_test (numpy.ndarray): Features for testing.
        y_train (numpy.ndarray): Target labels for training.
        y_test (numpy.ndarray): Target labels for testing.
"""
def load_diabetes_dataset(file_path='DataSets/diabetes2.csv'):
    df = pd.read_csv(file_path)

    # Data preprocessing, quitamos valores NaN
    df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
    df.fillna(df.mean(), inplace=True)

    # Separamos el dataset en las features y el resultado (outcome)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Separamos el data en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test



"""
    Entrenamiento con logistic regression model.

    Parameters:
        X_train (numpy.ndarray): Features for training.
        y_train (numpy.ndarray): Target labels for training.
        max_iter (int): Maximum number of iterations for model training. (tronaba algo si no lo ponia creo...)

    Returns:
        model (LogisticRegression): Trained logistic regression model.
"""
def train_logistic_regression(X_train, y_train, max_iter=1000):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model



"""
    Evaluar el modelo que fue entrenado

    Parameters:
        model (LogisticRegression): Trained logistic regression model.
        X_train (numpy.ndarray): Features for training.
        y_train (numpy.ndarray): Target labels for training.
        X_test (numpy.ndarray): Features for testing.
        y_test (numpy.ndarray): Target labels for testing.

    Returns:
        train_accuracy (float): Accuracy on the training dataset.
        test_accuracy (float): Accuracy on the test dataset.
        conf_matrix (numpy.ndarray): Confusion matrix.
        classification_rep (str): Classification report.
"""
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
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
        Nada
"""
def plot_learning_curves(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes=train_sizes, cv=cv)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Accuracy')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Testing Accuracy')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()



"""
    Plotea la confusion matrix.

    Parameters:
        conf_matrix (numpy.ndarray): Confusion matrix.

    Returns:
        Nada
"""
def plot_confusion_matrix(conf_matrix):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title('Confusion Matrix')
    plt.show()



"""
    Si el usuario quiere hacer predicciones usando el modelo que fue entranado.

    Parameters:
        model (LogisticRegression): Trained logistic regression model.

    Returns:
        Nada
"""
def make_predictions(model):
    while True:
        user_choice = input("Do you want to make a prediction? (yes/no): ").lower()
        if user_choice == "no":
            print("Exiting the program.")
            break
        elif user_choice == "yes":
            print("\nEnter values to predict diabetes:")
            pregnancies = float(input('Pregnancies: '))
            glucose = float(input('Glucose: '))
            blood_pressure = float(input('Blood Pressure: '))
            skin_thickness = float(input('Skin Thickness: '))
            insulin = float(input('Insulin: '))
            bmi = float(input('BMI: '))
            diabetes_pedigree = float(input('Diabetes Pedigree Function: '))
            age = float(input('Age: '))

            features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
            prediction = model.predict(features)

            if prediction[0] == 1:
                result = "has diabetes."
            else:
                result = "does not have diabetes."

            print(f'The person {result}\n')
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")



if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_diabetes_dataset()
    model = train_logistic_regression(X_train, y_train)
    train_accuracy, test_accuracy, conf_matrix, classification_rep = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
    print(f'Testing Accuracy: {test_accuracy * 100:.2f}%')
    print('\nConfusion Matrix:\n', conf_matrix)
    print('\nClassification Report:\n', classification_rep)
    print('\nCoefficients:\n')
    coefficients = model.coef_
    # Print the coefficients for each feature
    for feature, coef in zip(X_train.columns, coefficients[0]):
        print(f'{feature}: {coef}')
    
    plot_learning_curves(model, X_train, y_train)
    plot_confusion_matrix(conf_matrix)
    make_predictions(model)
