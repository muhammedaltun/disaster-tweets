import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy

# Need to load the large model to get the vectors
nlp = spacy.load('en_core_web_lg')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.shape)

# Disabling other pipes because we don't need them and it'll speed up this part a bit
text = train.text[0]
with nlp.disable_pipes():
    vectors = np.array([token.vector for token in  nlp(text)])
print(vectors.shape)

with nlp.disable_pipes():
    doc_vectors = np.array([nlp(text).vector for text in train.text])
    test_vectors = np.array([nlp(text).vector for text in test.text])
print(doc_vectors.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(doc_vectors, train.target,
                                                    test_size=0.1, random_state=1)
                                                    
from sklearn.svm import LinearSVC

# Set dual=False to speed up training, and it's not needed
svc = LinearSVC(random_state=1, dual=False, max_iter=10000)
svc.fit(doc_vectors,train.target)
predic = svc.predict(test_vectors) 

print(predic[:20])

