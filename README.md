# random-forest-classifier
 Using a dataset which contains some statistical information about the audio frequencies of different male and female voices, this project sorts each voice by gender using the RandomForestClassifier algorithm.

 In this project, I created a pandas dataframe for the train and test datasets, and started out by displaying the first and last five rows of each set. After this I found the number of rows and columns of each set and checked for null or missing values.

Then I was able to count the number of male and female classes in the train DataFrame and separate the feature variables, and target variable, from both the DataFrames.

After that, I applied the RandomForestClassifier machine learning model to predict the males and female classes in the test DataFrame.

Lastly, I printed the confusion matrix and the classification report to evaluate the prediction model. Also, based on the confusion matrix, precision, using the f1-score values, I was able to determine whether or not the model was making accurate predictions.
