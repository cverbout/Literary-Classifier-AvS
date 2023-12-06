### IMPORTS ###

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

### CONSTANTS ###

CV = 10 # Cross Validation folds
TS = 0.2 # Test Size
RS = 18 # Random State for reporducibility

### FUNCTIONS ###

def read_and_split_paragraphs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        # Splitting paragraphs by two newline characters
        paragraphs = text.split('\n\n')
        return paragraphs

### DATA SETUP ###

# Split texts into paragraphs
austen_paragraphs = read_and_split_paragraphs('austen-northanger-abbey.txt') +  read_and_split_paragraphs('austen-pride-and-prejudice.txt')
shelley_paragraphs = read_and_split_paragraphs('shelley-frankenstein.txt') + read_and_split_paragraphs('shelley-the-last-man.txt')

# Create a dataframe with the paragraphs under a column text
austen_df = pd.DataFrame(austen_paragraphs, columns=['text'])
shelley_df = pd.DataFrame(shelley_paragraphs, columns=['text'])

# Adds a column author with the respective authors name as the data
austen_df['author'] = 'Austen'
shelley_df['author'] = 'Shelley'

# Combines these two dataframes
combined_df = pd.concat([austen_df, shelley_df])

# Vectorize the paragraphs under the text column and separate features from the dependent column
vectorizer = CountVectorizer(binary=True)
X_vec = vectorizer.fit_transform(combined_df['text'])
y = combined_df['author']

# Split the vectorized data into training and test sets
X_train_vec, X_test_vec, y_train, y_test = train_test_split(X_vec, y, test_size = TS, random_state=RS)

# Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=RS)

### CROSS VALIDATION ###

# Perform cross-validation on the training set
cv_scores = cross_val_score(clf, X_train_vec, y_train, cv=CV) 
print()
print("CV Scores: ", cv_scores)
print()
print(str(CV) + " fold average CV Score: ", np.mean(cv_scores))
print()

### TRAIN AND TEST ###

# Train the classifier and evaluate it on the test set
clf.fit(X_train_vec, y_train)
test_score = clf.score(X_test_vec, y_test)
print("Test Set Score: ", test_score)