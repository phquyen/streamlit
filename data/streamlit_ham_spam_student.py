import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

# 1. Read data
data = pd.read_csv("spam.csv", encoding='latin-1')

#--------------
# GUI
st.title("Data Science Project")
st.write("## Ham vs Spam")

# Upload file


# 2. Data pre-processing
source = data['v2']
target = data['v1']
# ham = 0, spam = 1
target = target.replace("ham", 0)
target = target.replace("spam", 1)

text_data = np.array(source)

count = CountVectorizer(max_features=6000)
count.fit(text_data)
bag_of_words = count.transform(text_data)

X = bag_of_words.toarray()

y = np.array(target)

# 3. Build model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 

clf = MultinomialNB()
model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#4. Evaluate model
score_train = model.score(X_train,y_train)
score_test = model.score(X_test,y_test)
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

cr = classification_report(y_test, y_pred)

y_prob = model.predict_proba(X_test)
roc = roc_auc_score(y_test, y_prob[:, 1])

#5. Save models
# luu model classication
pkl_filename = "ham_spam_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(model, file)
  
# luu model CountVectorizer (count)
pkl_count = "count_model.pkl"  
with open(pkl_count, 'wb') as file:  
    pickle.dump(count, file)


#6. Load models 
# Đọc model
# import pickle
with open(pkl_filename, 'rb') as file:  
    ham_spam_model = pickle.load(file)
# doc model count len
with open(pkl_count, 'rb') as file:  
    count_model = pickle.load(file)

# GUI
