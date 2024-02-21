import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

def create_histograms(df, feature_names):
    # Create a histogram for each feature in the dataset
    for feature in feature_names:
        fig, ax = plt.subplots()
        ax.hist(df[feature], bins=20)
        ax.set_title(f'{feature} distribution')
        st.pyplot(fig)

def app():
    st.title('Data Chart')

    st.write("This is the `Data Chart` page of the multi-page app.")

    st.write("The following are the histograms of the `iris` dataset.")

    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns = iris.feature_names)
    Y = pd.Series(iris.target, name = 'class')
    df = pd.concat([X,Y], axis=1)
    df['class'] = df['class'].map({0:"setosa", 1:"versicolor", 2:"virginica"})

    # Call the function to create histograms
    create_histograms(df, iris.feature_names)