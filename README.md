# Toxic Comment Classification Challenge

This repository contains a Jupyter notebook that tackles the problem of classifying toxic comments into six distinct categories. The goal is to develop machine learning models that outperform existing benchmarks, with a focus on state-of-the-art performance metrics like mean column-wise ROC AUC. The challenge can be found here: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview.

## Notebook Overview

### Key Features:

1. <b>Exploratory Data Analysis (EDA)</b>:
- Visualization of word frequencies using word clouds.
- Distribution analysis of the six toxicity categories.
- Correlation analysis between labels.
2. <b>Data Preprocessing</b>:

- Removal of special characters and punctuation using regular expressions.
- Lowercasing text for consistency.
- Removing multiple, leading, and trailing spaces from all comments.
3. <b>Feature Engineering</b>:
- Implementation of text vectorization techniques:
    - TfidfVectorizer: Computes term frequency-inverse document frequency scores and converts all words to their vectorized scores to be used for model training.
- Creation of custom embeddings using SpaCy and pre-trained word vectors.
4. <b>Model Training</b>:
- Evaluation of various machine learning models:
    - Logistic Regression
    - Random Forest Classifier
    - Linear Support Vector Classifier (SVC)
- Application of OneVsRestClassifier for multi-label classification.
5. <b>Evaluation Metrics</b>:
- Accuracy
- Precision, Recall, F1-Score (macro/micro/weighted averages)
- Mean column-wise ROC AUC (primary challenge metric)
- Confusion matrix and detailed classification reports

### Technologies Used:
- Natural Language Processing (NLP)
- Machine Learning classification algorithms
- Data visualization for interpretability

## Requirements
To run the notebook, install the following Python libraries:

- pandas
- matplotlib
- scikit-learn
- spacy
- numpy
- wordcloud
- seaborn

You can install the required libraries with:

`pip install pandas nltk matplotlib scikit-learn spacy numpy wordcloud seaborn`

Additionally, download the NLTK stopwords and SpaCy language models (if not already installed):

`python -m nltk.downloader stopwords` and `python -m spacy download en_core_web_sm`

### How to Use
1. Clone the repository:

    `git clone https://github.com/Vimal-Raghubir/Toxic-Comment-Classification-Challenge.git`

2. Open the Jupyter notebook:

    `jupyter notebook toxic_comment_classification_challenge.ipynb`

3. Follow the structured steps in the notebook to:
- Preprocess the dataset.
- Explore data through visualizations.
- Train and test models.
- Analyze and interpret performance metrics.

## Performance Evaluation

The notebook includes a comprehensive analysis of model performance:

- Detailed metric evaluation for each label.
- Mean column-wise ROC AUC as the key metric to align with the challenge requirements.
- Comparison of model results to identify the best-performing approach.

## Future Enhancements

Potential improvements to the notebook could include:

- Implementing deep learning techniques such as RNNs or Transformers.
- Hyperparameter optimization using grid search or random search.
- Integration of external datasets to enhance model training.

## Acknowledgments
This notebook was inspired by the Toxic Comment Classification Challenge, which aims to improve detection and classification of toxic online content, fostering a safer and more inclusive digital environment.