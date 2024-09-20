import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(1000):
    review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
    review = review.lower().split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words("english"))]
    review = ' '.join(review)
    corpus.append(review)
