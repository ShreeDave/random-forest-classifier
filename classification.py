# Create the DataFrames for both the train and test datasets
import pandas as pd
train_df=pd.read_csv('https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/project-4/gender-voice-train.csv')
test_df=pd.read_csv('https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/project-4/gender-voice-test.csv')
# Display the first and last 5 rows of both sets
train_df.head()
test_df.head()
train_df.tail()
test_df.tail()
# Print the number of rows and columns in both the DataFrames
test_df.shape

# Check for the missing values in both the DataFrames
train_df.isnull().sum()
test_df.isnull().sum()

# Print the count of both classes in both dataframes
train_df["label"].value_counts()
test_df["label"].value_counts()

# Get the feature variables 
x_train =train_df.iloc[:, 1:]
x_test =test_df.iloc[:, 1:]

# Get the target variables
y_train =train_df.iloc[:, 0]
y_test =test_df.iloc[:, 0]

# Import the modules to build the Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Predict the target variable based on the feature variables of the test DataFrame
rf_clf = RandomForestClassifier(n_jobs=-1, n_estimators=50)
rf_clf.fit(x_train, y_train)
predicted = rf_clf.predict(x_test)
predicted

# Print the confusion matrix 
confusion_matrix(y_test,predicted)

# Print the precision, recall and f1-score values for both classes
print(classification_report(y_test,predicted))


