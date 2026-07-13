# Customer Churn Prediction Project

## Project Overview
This project builds a simple machine learning model that predicts whether a customer is likely to cancel their subscription.

The aim of the project is to show a clear and reproducible workflow, not only a working model. The project includes:
- a structured repository
- a Jupyter notebook for experiment documentation
- a decision log explaining key choices
- examples of version control planning and collaboration

## Dataset
The dataset is stored in `data/churn_dataset.csv`.

It contains customer information such as:
- age
- tenure in months
- monthly charges
- contract type
- internet service
- payment method
- support tickets
- average monthly usage
- churn outcome

The target variable is `churn`, where:
- `Yes` means the customer churned
- `No` means the customer did not churn

## Tools Used
- Python
- pandas
- scikit-learn
- Jupyter Notebook

## Project Structure
```text
project/
│
├── data/
│   └── churn_dataset.csv
│
├── notebooks/
│   └── churn_experiment.ipynb
│
├── models/
│   └── model.py
│
├── README.md
├── decision_log.txt
└── requirements.txt
```

## How to Run the Project
1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Python model script:
   ```bash
   python models/model.py
   ```
3. Open the experiment notebook:
   ```text
   notebooks/churn_experiment.ipynb
   ```

## Results Summary
In this example solution, a logistic regression model is trained on the churn dataset after basic preprocessing.

Because the dataset is very small, the result should be treated as a simple teaching example rather than a production result.

## Version Control
The following branches could be used in GitHub:

### branch: decision-tree-experiment
Used for testing a different machine learning model and comparing results.

### branch: update-readme
Used for improving project documentation.

## Pull Request Example
**Title:** Improve churn prediction experiment documentation

**Summary:**  
This update improves the project documentation and adds a worked experiment notebook.

**Changes:**  
- Added a documented Jupyter notebook  
- Added a decision log  
- Improved the README file  

**Results:**  
The project is now easier to understand, reproduce, and review.
