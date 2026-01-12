Fake News Detector:

A machine learning-based application that predicts whether a news article is real or fake using Natural Language Processing (NLP) and Logistic Regression.


Overview:

Fake news spreads quickly and can mislead people. This project uses text analysis to detect fake news by analyzing the content of news articles. The model classifies news into two categories:
Fake (0)
Real (1)
It can be used as a standalone Python application or integrated into a web application.


Features:

Predicts the authenticity of a news article.
Preprocessing of text using TF-IDF Vectorization.
Trained using Logistic Regression for high accuracy.
Easy integration with Flask for web deployment.
Lightweight and fast for real-time predictions.
The model achieves 98% accuracy on the test dataset, making it highly reliable for distinguishing between real and fake news articles.


Dataset:

This project uses the Fake and Real News Dataset from Kaggle:
Fake News Dataset
Contains labeled news articles: Fake (0) and Real (1).
CSV files: Fake.csv and True.csv.


Tech Stack:

Python 3
Pandas for data manipulation
Scikit-learn for machine learning
Flask for web deployment
TF-IDF Vectorizer for text feature extraction
Logistic Regression for classification


Example:

Input:
"Breaking: Celebrity endorses miracle cure for weight loss."

Output:
Prediction: Fake

Input:
"The government passes a new law to improve public healthcare services."

Output:
Prediction: Real


Future Improvements:

Add Deep Learning models (like LSTM or BERT) for better accuracy.
Build a browser extension to detect fake news on the fly.
Include multi-language support for global news detection.
Deploy as a cloud-based API for scalable access.


License:

This project is open-source and available under the MIT License.
