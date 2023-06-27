import streamlit as st
from function import load_data1, load_data, normalization
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

def app():
    st.title("Penguin Species Classification")
    st.text("")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.markdown("The **:green[Dataset]** that using for **Train Model**")
    data = load_data1()
    df = pd.DataFrame(data)
    st.dataframe(df)
    dataset_link = "https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data"
    st.markdown(f"Link for Dataset: [Click Here]({dataset_link})")
    st.text("")
    
    st.title("Why we use this dataset?")
    st.text("1. Real-world application")
    st.text("2. Species diversity")
    st.text("3. Rich attribute information")
    st.text("4. Scientific research")
    
    st.title("Why penguins must be identified?")
    st.text("1. Conservation and Population Monitoring")
    st.text("2. Ecological Studies")
    st.text("3. Behavioral and Reproductive Studies")
    st.text("4. Species-specific Threats and Conservation Measures")
    
    st.title("Objectives from this classification")
    st.text("1. Species Identification")
    st.text("2. Pattern Recognition and Characterization")
    st.text("3. Population Monitoring and Conservation")
    st.text("4. Research and Innovation")    
    
    st.text("")
    st.text("")
    st.text("")
    st.header("Numerical data distribution chart")
    # chart data ori
    numeric_columns = data.select_dtypes(include='number').columns
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the data for each column
    for column in numeric_columns:
        ax.plot(data[column], label=column)
    # Set the x-axis ticks and labels
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data.index)
    # Set the y-axis label
    ax.set_ylabel('Values')
    # Set the chart title
    ax.set_title('Penguin Dataset Chart')
    # Add a legend
    ax.legend()
    # Display the chart
    st.pyplot(fig)
    
    st.text("")
    st.text("")
    st.markdown("because the distribution of the existing data is still less equal, the data is normalized so that the data distribution becomes more equal")
    st.header("Numerical data distribution chart after normalization")
    # chart normalize
    numeric_columns = df.select_dtypes(include='number').columns
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[numeric_columns])
    fig, ax = plt.subplots()
    for i, column in enumerate(numeric_columns):
        ax.plot(normalized_data[:, i], label=column)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Index')
    ax.set_ylabel('Normalized Values')
    ax.set_title('Normalized Dataset Chart')
    ax.legend()
    st.pyplot(fig)
    
    st.text("")
    st.text("")
    st.markdown("For dataset processing, using a decision tree algorithm and get a testing accuracy of")
    data['island'] = data['island'].replace({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})
    data['sex'] = data['sex'].replace({'MALE': 0, 'FEMALE': 1})
    data['species'] = data['species'].replace({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
    
    x_data = data[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    y_target = data['species']
    
    scaler = MinMaxScaler()
    x_data = scaler.fit_transform(x_data)
    
    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_target, test_size=0.33, random_state=123
    )
    
    treeClass = tree.DecisionTreeClassifier()
    treeClass.fit(X_train, y_train)
    y_pred = treeClass.predict(X_test)
    treeAccuracy = accuracy_score(y_pred, y_test)
    st.text("Accuracy From This Model: " + str(treeAccuracy*100) + "%")
    
    st.text("")
    st.text("")
    st.header("Decission Tree Model for this dataset")
    dot_data = tree.export_graphviz(
        decision_tree=treeClass, max_depth=5, out_file=None, filled=True, rounded=True,
        feature_names=['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'],
        class_names=['Species Adelie', 'Species Chinstrap', 'Species Gentoo']
    )
    st.graphviz_chart(dot_data)
    
    st.text("")
    st.text("")
    st.header("Confusion Matrix from testing in this dataset")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    st.pyplot(plt.gcf())  # Use plt.gcf() to get the current figure