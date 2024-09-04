# Titanic Classification Project

This project uses the famous Titanic dataset to build a machine learning model that predicts whether a passenger survived or not, based on various features such as age, sex, passenger class, and fare. The project includes exploratory data analysis (EDA), preprocessing, model training, evaluation, and feature importance analysis.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Building](#model-building)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to apply machine learning techniques to predict the survival of passengers on the Titanic. Using Python and popular libraries like pandas, numpy, sklearn, seaborn, and matplotlib, we build a Random Forest classifier and analyze its performance.

## Dataset

The dataset used in this project is the [Titanic dataset](https://www.kaggle.com/c/titanic/data) from Kaggle. It contains information about the passengers, including their age, gender, class, fare, and whether they survived or not.

## Installation

To run this project locally, you will need Python 3.x installed, along with the following libraries:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```
## Exploratory Data Analysis (EDA)
The EDA phase includes:
- Visualizing the distribution of survival.
- Analyzing relationships between different features and survival rates.
- Understanding the dataset structure and handling missing values.

## Model Building
The machine learning model is built using a Random Forest classifier. The steps involved are:

- Preprocessing: Handling missing values, encoding categorical variables, and scaling numerical features.
- Splitting Data: Dividing the data into training and testing sets.
- Training: Training the Random Forest model on the training data.
- Feature Importance: Identifying the most important features that contribute to predicting survival.
## Evaluation
The model is evaluated using:

- Accuracy: The proportion of correctly predicted instances.
- Classification Report: A summary including precision, recall, F1-score for each class.
- Confusion Matrix: A heatmap visualization of the confusion matrix to understand the performance of the model.

## Results
The model achieved an accuracy of 82% on the test dataset.

The most important features for predicting survival are:

![Accuracy png](https://github.com/user-attachments/assets/4ad64197-3555-4881-bb11-15835ff43189)

![Confusion_Matrix png](https://github.com/user-attachments/assets/f9f3bd8d-8cc1-4775-9e56-4eb458ba7921)

![Feature importance png](https://github.com/user-attachments/assets/a41ef539-1937-4c9c-96ee-8e2a51155982)


## Conclusion
This project demonstrates the use of Random Forests for classification problems and provides insights into the factors affecting the survival of passengers on the Titanic. Further improvements could include testing different models, tuning hyperparameters, and using advanced feature engineering techniques.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any suggestions or improvements.
