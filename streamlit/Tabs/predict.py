# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 20:35:53 2023

@author: USER
"""

import streamlit as st
from function import predict
import pandas as pd
from io import StringIO
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

def app():
    st.title("Prediction Pages")
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)
    
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # st.write(stringio)
    
        # To read file as string:
        # string_data = stringio.read()
        # st.write(string_data)
    
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        # st.write(dataframe)
        st.title("Your Data: ")
        df = pd.DataFrame(dataframe)
        st.dataframe(df)
        st.text("")
        dataframe['island'] = dataframe['island'].replace({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})
        dataframe['sex'] = dataframe['sex'].replace({'MALE': 0, 'FEMALE': 1})
        dataframe['species'] = dataframe['species'].replace({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
        
        x_data = dataframe[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
        y_target = dataframe['species']
        
        scaler = MinMaxScaler()
        x_data = scaler.fit_transform(x_data)
        
        X_train, X_test, y_train, y_test = train_test_split(
            x_data, y_target, test_size=0.33, random_state=123
        )
        
        treeClass = tree.DecisionTreeClassifier()
        treeClass.fit(X_train, y_train)
        y_pred = treeClass.predict(X_test)
        treeAccuracy = accuracy_score(y_pred, y_test)
        st.text("Accuracy from your data: "+str(treeAccuracy)+"%")

        st.text("")
        st.text("")
        st.header("Decission Tree from your Data")
        dot_data = tree.export_graphviz(
            decision_tree=treeClass, max_depth=5, out_file=None, filled=True, rounded=True,
            feature_names=['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'],
            class_names=['Species Adelie', 'Species Chinstrap', 'Species Gentoo']
        )
        st.graphviz_chart(dot_data)
        
        st.text("")
        st.text("")
        st.header("Confusion Matrix from your data prediction")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        st.pyplot(plt.gcf())  # Use plt.gcf() to get the current figure
    