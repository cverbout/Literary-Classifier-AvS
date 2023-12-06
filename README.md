# Literary Style Classifier: Austen vs. Shelley

## Project Overview

This machine learning project aims to classify paragraphs by authors Jane Austen and Mary Shelley, utilizing their distinct writing styles. By analyzing text from four novels, the classifier discerns between the romantic prose of Austen and the science fiction narratives of Shelley, achieving significant accuracy.

## Features

- **Author Classification**: Identifies paragraphs as either written by Jane Austen or Mary Shelley using their unique writing styles.
- **Machine Learning Model**: Employs a Decision Tree Classifier for effective classification based on textual features.
- **Data Preprocessing**: Involves thorough preprocessing of the novels to extract clean, relevant text for analysis.
- **Cross-Validation**: Utilizes 10-fold cross-validation to ensure model reliability and performance consistency.
- **Feature Selection**: Implements a 'big bag of words' approach with feature selection to distinguish author-specific language patterns.

## Technical Stack

- **Python**: Primary programming language.
- **Pandas**: Used for data manipulation and analysis.
- **Scikit-Learn**: For machine learning model implementation, feature extraction, and evaluation.
- **CountVectorizer**: Converts text data into a matrix of token counts for feature extraction.

## Data Source

The novels used in this project are:
- *Pride and Prejudice* and *Northanger Abbey* by Jane Austen.
- *Frankenstein* and *The Last Man* by Mary Shelley.
  
The texts were sourced from Project Gutenberg and preprocessed to remove extraneous content and format irregularities by Bart Massey.

## Usage

To use the classifier:
1. Ensure Python and necessary libraries (Pandas, Scikit-Learn) are installed.
2. Run the script to preprocess the novels, train the classifier, and evaluate its performance.
3. The script will output cross-validation scores and test set accuracy, demonstrating the classifier's effectiveness.

## Results and Insights

The classifier demonstrates a high level of accuracy in distinguishing between the writing styles of Austen and Shelley. It highlights the potential of machine learning in literary analysis and the identification of author-specific writing patterns.
