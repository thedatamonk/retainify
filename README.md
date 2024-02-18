# Retainify

<h2>Goal</h2>

In this project, the goal is to demonstrate all the steps of a typical ML workflow using [Kaggle's Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/) dataset.

A typical ML workflow comprises of 2 parts namely **experimentation** and **deployment** - 

<h3>Experimentation</h3>

1. EDA (Exploratory Data Analysis)
2. Data processing
    - Data cleaning
    - Outlier detection
    - Interpolation
    - Feature reduction/derivation
3. Choose the right metrics that are suitable for the problem in hand as well as relate to business metrics.
4. Explore different models
    - train each model on the train dataset
    - validate on validation dataset or perform cross validation
5. Evaluate on test dataset (hold out dataset)
6. Hyperparameter optimization

<h3>Deployment</h3>

1. Serve the model by creating an endpoint
2. Deploy the training so that rapid retraining of the model is possible.
3. Incorporate CI/CD so that deployments are automatic.

<h2>Tasks completed</h2>

*TODO*

<h2>Architecture diagram</h2>

*Insert MIRO board screen shot*

<h2>Folder structure</h2>

<h2>Setup</h2>

<h2>Next Steps</h2>

1. Incorporate Kubeflow/Airflow to build the pipelines and manage the complete workflow
2. Deploy and test the app on cloud
