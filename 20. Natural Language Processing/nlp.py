# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the text
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

corpus = []
for i in range(0, 1000):
#    print(dataset['Review'][i])

    review1 = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # removes the symbols from the sentence
#    print(review1)

    review2 = review1.lower()  # lower the sentence
#    print(review2)

    review3 = review2.split()
#    print(review3)

    ps = PorterStemmer()

    review4 = [ps.stem(word) for word in review3 if not word in set(stopwords.words('english'))]
#    print(review4)

    review5 = ' '.join(review4)
#    print(review5)
    corpus.append(review5)

# creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# Fitting the 12. Naive Bayes to the training set
# Create your classifier here
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set Result
y_pred = classifier.predict(X_test)

# Making The Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
