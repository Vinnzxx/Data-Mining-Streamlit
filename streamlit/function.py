# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 20:07:02 2023

@author: USER
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import streamlit as st

@st.cache_data()
def load_data1():
    data = pd.read_csv('penguins_size1.csv')
    return data
    
@st.cache_data()
def normalization():
    data = pd.read_csv('penguins_size.csv')
    result = preprocessing.normalize(data, axis=0)
    return result

@st.cache_data()
def load_data():
    data = pd.read_csv('penguins_size.csv')

    x = data[['island','culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','sex']]
    y = data[['species']]

    # sc = StandardScaler()
    # x = sc.fit_transform(x)
    # # y = sc.fit_transform(y)

    
    return data,x,y

@st.cache_data()
def train_model(x, y):
    model = DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=150
    )
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)

    return model, score

def predict(x, y, features):
    model, score = train_model(x, y)
    
    prediction = model.predict(np.array(features).reshape(1, -1))
    
    return prediction, score

