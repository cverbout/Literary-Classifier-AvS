# Authorship

**Chase Verbout | CS 541 Artificial Intelligence**

## Introduction

"Authorship" is a machine learning project aimed at distinguishing the writing styles of Jane Austen and Mary Wollstonecraft Shelley, using their novels from the early 19th century. The task involves classifying paragraphs to identify the respective author.

## Project Description

The project involved creating a machine learning model to classify paragraphs from novels by Austen and Shelley. The novels used, sourced from Project Gutenberg, include Austen's _Pride and Prejudice_ and _Northanger Abbey_, and Shelley's _Frankenstein_ and _The Last Man_. The approach was primarily a 'big bag of words' technique, treating paragraphs as unordered collections of words. A Decision Tree Classifier was implemented for classification, aiming for an accuracy of at least 70% against an independent validation set.

## Status

### What's Completed

- Extraction and preprocessing of text data from the novels.
- Implementation of the Decision Tree Classifier.
- Attainment of the target accuracy for paragraph classification.

### To-Do

- Experiment with other machine learning models for performance comparison.
- Enhance model accuracy and processing efficiency.

## Build and Run Instructions

Implemented in Python, this project utilizes libraries such as NumPy, pandas, and scikit-learn.

### Prerequisites

- Python 3.x
- NumPy
- pandas
- scikit-learn

### Running the Program

1. Clone the project repository.
2. Navigate to the directory containing the project files.
3. Execute the program:
   ```bash
   python Authorship.py
   ```

### Output

The output includes cross-validation scores and the test set score, reflecting the model's ability to differentiate between the writings of Austen and Shelley.
