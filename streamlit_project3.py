import numpy as np
import pandas as pd
import ujson as json
#from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, plot_roc_curve, plot_confusion_matrix, plot_precision_recall_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay
from time import time


stime = time()

st.set_option('deprecation.showPyplotGlobalUse', False)

df = None
data = None
score_train = None
score_test = None
acc = None
cm = None
cr = None
y_prob = None
roc = None
y_test = None
X_test = None
sentiment_model = None
count_model = None
# 1. Read data
df = pd.read_csv('data_Foody_concat.csv', encoding='utf-8')
df['target'] = 1 # negative
df['target'][df['review_score']>=6.5] = 0
#--------------
# GUI
st.title("DATA SCIENCE PROJECT")
st.write('## SENTIMENT ANALYSIS')
st.write("### POSITIVE OR NEGATIVE")

# Upload file
uploaded_file = st.file_uploader('Choose a file', type=['csv'])

if uploaded_file is not None:
     data = pd.read_csv(uploaded_file, encoding='utf-8')
     # data = pd.read_csv("data_test_streamlit.csv", encoding='utf-8')
     data.to_csv('review_new.csv', index=False)

     # 2. Data pre-processing
     source = data['review_text_t1']
     target = data['target']

     # target = target.replace("Negative", 0)
     # target = target.replace("Neutral", 1)
     # target = target.replace('Positive',2)

     text_data = np.array(source)

     count_model = CountVectorizer(max_features=3500) 
     count_model.fit(text_data)
     bag_of_words = count_model.transform(text_data)

     X = bag_of_words.toarray()

     y = np.array(target)

     # 3. Build model
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 

     clf = LogisticRegression()
     sentiment_model = clf.fit(X_train, y_train)
     y_pred = clf.predict(X_test)

     #4. Evaluate model
     score_train = sentiment_model.score(X_train,y_train)
     score_test = sentiment_model.score(X_test,y_test)
     acc = accuracy_score(y_test,y_pred)
     cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

     cr = classification_report(y_test, y_pred)

     y_prob = sentiment_model.predict_proba(X_test)

     roc = roc_auc_score(y_test, sentiment_model.predict_proba(X_test)[:,1])

     # with open('pretrained_stats.json', 'w') as f:
     #      stats = {
     #           'score_train': score_train,
     #           'score_test': score_test,
     #           'acc': acc,
     #           'cm': cm.tolist(),
     #           'cr': cr,
     #           'y_prob': y_prob.tolist(),
     #           'roc': roc,
     #           'y_test': y_test.tolist(),
     #           'X_test': X_test.tolist()
     
     #      }
     #      json.dump(stats, f)
     #5. Save models
     # luu model classication
     with open('upload_sentiment_model.pkl', 'wb') as file:  
          pickle.dump(sentiment_model, file)
     
     # luu model CountVectorizer (count_model)

     with open('upload_count_model.pkl', 'wb') as file:  
          pickle.dump(count_model, file)


else:
     # load pretrained model and stats
     data = pd.read_csv("data_test_streamlit.csv", encoding='utf-8')
     #6. Load models 
     # Đọc model
     with open('pretrained_sentiment_model.pkl', 'rb') as file:  
          sentiment_model = pickle.load(file)
     # doc model count len
     with open('pretrained_count_model.pkl', 'rb') as file:  
          count_model = pickle.load(file)  
     # Read training stats
     with open('pretrained_stats.json', 'r') as f:
          stats = json.load(f)
          score_train = stats['score_train']
          score_test = stats['score_test']
          acc = stats['acc']
          cm = np.array(stats['cm'])
          cr = stats['cr']
          y_prob = np.array(stats['y_prob'])
          roc = stats['roc']
          y_test = np.array(stats['y_test'])
          X_test = np.array(stats['X_test'])
          del stats

# roc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovo") - Cho mul class (3 class)

print(f'etime: {time() - stime}')



# GUI
menu = ['Business Objective','Build Project','New Prediction']
choice = st.sidebar.selectbox('Menu',menu)  
if choice == 'Business Objective':
     st.subheader('Business Objective')
     st.write("""###### Xây dựng mô hình dự đoán giúp nhà hàng có thể biết được những phản hồi nhanh chóng của khách hàng về sản phẩm hay dịch vụ của họ (tích cực hay tiêu cực). Giúp cải thiện tình hình hoạt động của nhà hàng cũng như mang đến trải nghiệm tốt cho khách hàng.
     """)
     st.write('### Overview:')
     st.dataframe(df[['restaurant','review_text','review_score']].head(3))
     st.dataframe(df[['restaurant','review_text','review_score']].tail(3))

     figov = sns.displot(data = df, x='review_score', palette="Set3")
     st.pyplot(figov)
     st.write('Theo biểu đồ, dữ liệu được đánh giá khá tích cực, chủ yếu từ mức 6.5 hoặc 7.0 trở lên.')
     # figov1 = sns.countplot(data = df[['target']], x='target')
     
     sns.countplot(data=data[['target']], x='target', palette="Set3")
     st.pyplot()
     st.write('Dữ liệu được chia thành 2 class: negative (rating < 6.5) và positive (rating >= 6.5)')
     #st.image("sentiment_analysis.jpg")

elif choice == 'Build Project':
     st.subheader('Build Project')
     st.write("##### Data")
     st.write("Shape of dataset is:", data.shape)
     st.dataframe({'Labels': ['Positive','Negative'],'Target': [0,1]})
     st.dataframe(data[['review_text_t1','target']].head(3))
     st.dataframe(data[['review_text_t1','target']].tail(3))

     st.write('##### Visualize')
     fig1 = sns.countplot(data=data[['target']], x='target',palette="Set3")
     st.pyplot(fig1.figure)

     #st.write('##### 3. Build model ...')

     st.write('##### Evaluations')
     st.code('Score train: ' + str(round(score_train,2)) + '\n''Score test: ' + str(round(score_test,2)))
     st.code('Accuracy: ' + str(round(acc,2)))
     st.write('###### Confusion matrix: ')
     st.code(cm)
     st.write('###### Classication report: ')
     st.code(cr)
     st.code('roc AUC score: '+ str(round(roc,2))) 

     # calculate roc curve
     st.write('##### ROC curve')
     fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
     fig, ax = plt.subplots()
     ax.plot([0,1], [0,1], linestyle='--')
     ax.plot(fpr, tpr, marker='.')
     st.pyplot(fig) 

     ## Positive
     # st.write('##### ROC curve')
     # st.write('#### Positive')
     # fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,0], pos_label=0)
     # fig, ax = plt.subplots()
     # ax.plot([0,1], [0,1], linestyle='--')
     # ax.plot(fpr, tpr, marker='.')
     # st.pyplot(fig)
     class_names = ['target', 'review_text_t1']
     st.subheader("Confusion Matrix") 
     plot_confusion_matrix(sentiment_model, X_test, y_test, display_labels=class_names)
     st.pyplot()

     st.subheader("ROC Curve") 
     plot_roc_curve(sentiment_model, X_test, y_test)
     st.pyplot()

     st.subheader("Precision-Recall Curve")
     plot_precision_recall_curve(sentiment_model, X_test, y_test)
     st.pyplot()

     # ## Negative
     # st.write('##### ROC curve')
     # st.write('#### Begative')
     # fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1], pos_label=1)
     # fig, ax = plt.subplots()
     # ax.plot([0,1], [0,1], linestyle='--')
     # ax.plot(fpr, tpr, marker='.')
     # st.pyplot(fig)
 

     st.write('##### Summary: this model is good enough for Sentiment Analysis classication.')

     # new Prediction
elif choice =='New Prediction':
     st.subheader('Select data')
     flag = False 
     lines = None
     type = st.radio('Upload data or Input data?', options =('Upload', 'Input'))
     if type =='Upload':
         uploaded_file_1 = st.file_uploader('Choose a file', type=['txt','csv'])
         if uploaded_file is not None:
             lines = pd.read_csv(uploaded_file_1, header=None)
             st.dataframe(lines)
             # st.write(lines.columns)
             lines = lines[0]
             flag = True 
     if type == 'Input':
         email = st.text_area(label='Input your content:')
         if email!='':
             lines = np.array([email])
             flag = True
     if flag: 
         st.write('Content:')
         if len(lines)>0:
             st.code(lines)
             x_new = count_model.transform(lines)
             y_pred_new = sentiment_model.predict(x_new)
             st.code('New predictions (0: Positive, 1: Negative): '+ str(y_pred_new))