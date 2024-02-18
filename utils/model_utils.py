from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from utils.data_utils import balance_data
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from joblib import dump, load

def train_model(classifier, x_train, y_train):


    # Fit the classifier on training data
    classifier.fit(x_train, y_train)

    # Apply cross validation and compute mean f1 score (harmonic mean of precision and recall)
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)
    cross_validation_score = cross_val_score(classifier, x_train, y_train, cv = cv, scoring = 'f1').mean()
    print("Cross Validation Score : ",'{0:.2%}'.format(cross_validation_score))

    return classifier, cross_validation_score


def eval_model(classifier, x_test, y_test):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test,classifier.predict(x_test))
    names = ['True Neg','False Pos','False Neg','True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm, annot = labels, cmap = 'Blues', fmt ='', square=True)
    plt.tight_layout()

    ARTIFACTS_DIR = "./artifacts"
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
    
    confusion_matrix_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # Generate Classification Report
    report = classification_report(y_test,classifier.predict(x_test))
    report_path = os.path.join(ARTIFACTS_DIR, "classification_report.txt")

    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Confusion matrix saved to {confusion_matrix_path}")
    print(f"Classification report saved to {report_path}")


async def initiate_training(data, test_size_frac=0.2):

    train_feats = data.drop(columns=['Churn'])
    target_vals = data['Churn']

    # Split data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(train_feats, target_vals, test_size=test_size_frac, random_state=42)
    print(f"X_train: {X_train.shape}\t y_train: {y_train.shape}")
    print (f"X_test: {X_test.shape}\t y_test: {y_test.shape}")

    # Apply SMOTE only on the training set to balance it
    balanced_x_train, balanced_y_train = balance_data(x_train=X_train, y_train=y_train)

    # Initialize the Random Forest classifier
    classifier = RandomForestClassifier(max_depth=4, random_state=42)

    # Perform actual training
    classifier, cross_validation_score = train_model(classifier=classifier, x_train=balanced_x_train, y_train=balanced_y_train)

    # Evaluate model on the test set
    eval_model(classifier=classifier, x_test=X_test, y_test=y_test)

    # Save the model
    ARTIFACTS_DIR = "./artifacts"
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
    
    model_file_path = os.path.join(ARTIFACTS_DIR, "random_forest.joblib")
    dump(classifier, model_file_path)
    print (f"Model saved in {model_file_path}")


async def test_model(data):
    # Load the model
    model = load('artifacts/random_forest.joblib')

    # Make predictions
    predictions = model.predict(data.drop(columns=['customerID']))

    churn_predictions = [{"customerID": customer_id, "churn": int(prediction)} for customer_id, prediction in zip(data['customerID'], predictions)]

    return churn_predictions    








    