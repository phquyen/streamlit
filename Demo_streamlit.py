import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Using menu
import streamlit as st
st.title("TRUNG TÂM TIN HỌC")
st.subheader("How to run streamlit app")
menu = ["Home", "About"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':
    st.subheader("Streamlit From Windows")
elif choice == 'Capstone Project':
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Machine-Learning-Capstone-Project_207)")
    st.write("""### Có 3 chủ đề trong khó học:
    - Topic 1: RFM & Clustering
    - Topic 2: Recommendation System
    - Topic 3: Sentiment Analysis
    -...""")
