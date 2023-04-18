#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler



# Define the Streamlit app
def app():
    #load the dataset
    df = pd.read_csv('titanic.csv', header=0)
    
    #shufle the data
    df = df.sample(frac = 1)
    
    st.dataframe(df, use_container_width=True)
    
#run the app
if __name__ == "__main__":
    app()
