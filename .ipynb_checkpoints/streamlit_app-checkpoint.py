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
    st.title('MLP Neural Network on the Titanic dataset')
    st.subheader('by Louie F. Cervantes M.Eng., WVSU College of ICT')
    st.write('The titanic dataset gives you information about \
    multiple people like their ages, sexes, sibling counts, \
    No of parent or children aboard, embarkment points and \
    whether or not they survived the disaster. Based on these \
    features, you have to predict if an arbitrary passenger on \
    Titanic would survive the sinking. The question of interest \
    for this natural dataset is how survival relates to the \
    other attributes. There is obviously no practical need to \
    predict survival, so the real interest is in interpretation, \
    but success at prediction would appear to be closely related \
    to the discovery of interesting features of the relationship. \
    To simplify the processing, the datarows with missing values \
    so this dataset is not the original dataset available at the \
    machine learning websites.')
    
    #load the dataset
    df = pd.read_csv('titanic.csv', header=0)
    st.dataframe(df, use_container_width=True)  
    #shufle the data
    df = df.sample(frac = 1)
    
   
#run the app
if __name__ == "__main__":
    app()
