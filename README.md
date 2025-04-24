# Data-Science-Projects

# Churn Modeling with Artificial Neural Network

This project implements a churn modeling solution using an Artificial Neural Network (ANN) built with TensorFlow. The goal is to predict customer churn based on a dataset containing customer information.

## Project Description

The project aims to develop a predictive model that can identify customers at high risk of churning. By analyzing various customer attributes, the ANN model learns patterns and predicts the likelihood of churn. This information can be valuable for businesses to take proactive measures to retain customers.

## Data

The dataset used in this project (`Churn_Modelling.csv`) contains customer information relevant to churn prediction. Key features include:

-   `CustomerId`: Unique identifier for each customer.
-   `CreditScore`: Customer's credit score.
-   `Geography`: Customer's geographical location.
-   `Gender`: Customer's gender.
-   `Age`: Customer's age.
-   `Tenure`: Number of years the customer has been with the company.
-   `Balance`: Customer's account balance.
-   `NumOfProducts`: Number of products the customer uses.
-   `HasCrCard`: Whether the customer has a credit card.
-   `IsActiveMember`: Whether the customer is an active member.
-   `EstimatedSalary`: Customer's estimated salary.
-   `Exited`: Whether the customer has churned (target variable).

## Methodology

1.  **Data Loading and Preprocessing**:
    -   Loading the dataset using pandas.
    -   Separating features (independent variables) and the target variable.
    -   Encoding categorical features (e.g., `Gender`) using LabelEncoder.
    -   One-hot encoding categorical features (e.g., `Geography`) to convert them into numerical format.
    -   Splitting the dataset into training and testing sets.
    -   Scaling numerical features using StandardScaler to ensure they have a mean of 0 and a standard deviation of 1.
2.  **ANN Model Building**:
    -   Creating an ANN model using TensorFlow's Keras Sequential API.
    -   Adding multiple dense layers with ReLU activation function.
    -   Adding an output layer with a sigmoid activation function for binary classification.
3.  **Model Compilation and Training**:
    -   Compiling the ANN model with the Adam optimizer, binary cross-entropy loss function, and accuracy as the evaluation metric.
    -   Training the model on the training data for a specified number of epochs.
4.  **Model Evaluation**:
    -   Evaluating the model's performance on the test set.
    -   Making predictions on new data.

## Libraries Used

-   numpy
-   pandas
-   tensorflow
-   scikit-learn (LabelEncoder, OneHotEncoder, train_test_split, StandardScaler)

## Usage

To run this project:

1.  Ensure you have Python and the required libraries installed. You can install the libraries using pip:

    ```bash
    pip install numpy pandas tensorflow scikit-learn
    ```

2.  Place the `Churn_Modelling.csv` file in the same directory as the notebook.
3.  Run the Jupyter Notebook (`churn-modeling.ipynb`) to execute the code and train the model.

## Files

-   `Churn_Modelling.csv`: The dataset used for churn modeling.
-   `churn-modeling.ipynb`: Jupyter Notebook containing the code for building and training the ANN model.

## Key Concepts

-   **Artificial Neural Networks (ANNs)**: A machine learning model inspired by the human brain, used for complex pattern recognition and prediction.
-   **Feature Encoding**: Converting categorical data into numerical format so that it can be used by machine learning models.
-   **Feature Scaling**: Standardizing the range of independent variables to prevent any single variable from dominating the model training.
-   **Binary Classification**: A type of classification problem with two possible outcomes (e.g., churn or no churn).

## Author

\[Naveen Babu Bathula]
