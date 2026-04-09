# SLR_sklearn

Python implementation of Simple Linear Regression using scikit-learn. This project demonstrates how to train a simple linear regression model, generate predictions, visualize the regression line, and evaluate model performance using the R² score.

Although the example dataset uses Age and Glucose values, the main purpose of this project is to show what a basic SLR workflow looks like and how R² can be used to measure how well the model fits the data. In real applications, this same process can be used on test data to evaluate how well the model generalizes.

## Features

The project includes:

- training a Simple Linear Regression model with scikit-learn
- reshaping input data for model training
- generating predicted values
- plotting the regression line
- calculating model performance using R²
- demonstrating a basic machine learning workflow

## Example data

```python
Age = np.array([43,21,25,42,57,59,35,15,55,50,65,10,45,35])
Glucose = np.array([99,65,79,75,87,81,80,80,90,70,95,67,90,82])
