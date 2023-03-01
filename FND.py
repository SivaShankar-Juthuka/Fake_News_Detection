# -*- coding: utf-8 -*-
'''
Created on Tue Feb 14 12:56:52 AM 2023

@author: Siva Shankar
'''

'''
About the Dataset:

id: unique id for a news article
title: the title of a news article
author: author of the news article
text: the text of the article; could be incomplete
label: a label that marks whether the news article is real or fake
1 represents fake news in label 
0 represents real news in label 
'''
# Importing necessary libraries
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Downloading stopwords
import nltk
nltk.download('stopwords')

# Printing the list of stopwords in English
#print(stopwords.words('english'))

# Reading dataset from the CSV file and loading it into a pandas DataFrame
news_dataset = pd.read_csv(r"C:/Users/Siva Shankar/Desktop/Sample/train.csv")

# Printing the description of the dataset
#print("\n\n Description Of the DataSet :\n",news_dataset.describe())

# Printing the dimensions of the dataset
#print("\n\nDimenstion Of the DataSet :",news_dataset.shape)

# Printing the first 5 rows of the dataframe
#print("\n\n First 5 rows  in the SataSet :\n",news_dataset.head())

# Counting the number of missing values in the dataset
#print("\n\n Count Of Null Values in the DataSet :\n",news_dataset.isnull().sum())

# Replacing the null values with an empty string
news_dataset = news_dataset.fillna('')

# Merging the author name and news title to create a new column 'content'
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Separating the data and label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

# Printing dataset without the label columns
#print("\n\n Dataset without the labels columns :\n",X)
# Printing the labels
#print("\n\n Labels extracted to Y :\n", Y)

# Stemming
# Stemming is the process of reducing a word to its root word
# Example: actor, actress, acting -> act
port_stem = PorterStemmer()

# Stemming function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Applying stemming on the 'content' column
news_dataset['content'] = news_dataset['content'].apply(stemming)

# Printing the 'content' column after applying the stemming process
#print("\n\n New Column Content after applying stemming process :\n\n",news_dataset['content'])

# Separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Printing dataset without the labels column after applying stemming
#print("\n\n Dataset without the labels column after applying stemming :\n",X)
# Printing the labels after applying stemming
#print("\n\n Labels after applying stemming :\n ", Y)
      
# Converting the textual data into numerical data because machine learning algorithms don't work on string or character data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

# Converting textual data to numerical data
X = vectorizer.transform(X)

# Printing the transformed data
#print("After Converting textual data to numerical data :\n\n",X)

# Splitting the data for training and testing and passing it to the algorithm to train and test the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

#Create a logistic regression model using Scikit-Learn.
model = LogisticRegression()

#Train the logistic regression model using the training dataset.
model.fit(X_train, Y_train)

#Calculate the accuracy of the model on the training dataset using the accuracy_score function from Scikit-Learn.
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy*100)

#Calculate the accuracy of the model on the testing dataset using the accuracy_score function from Scikit-Learn.
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the testing data : ', test_data_accuracy*100)

#Make a prediction on a new sample from the testing dataset using the trained model.
X_new = X_test[7]
prediction = model.predict(X_new)
#print(prediction)

#Print the prediction as "The news is Real" or "The news is Fake" depending on the predicted label.
if (prediction[0]==0):
    print('The news is Real ')
else:
    print('The news is Fake ')
