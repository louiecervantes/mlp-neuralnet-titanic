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
    st.write('The titanic dataset contains information about multiple \
    people like their ages, sexes, sibling counts, number of parent \
    or children companions on aboard, embarkment points and \
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
    st.subheader('Configuring the Neural Net')
    with st.echo(code_location='below'):
        #set the number of hidden layers
        hidden_layers = st.slider('No. of hidden layers', 5, 15, 10)
        #set the number or iterations
        max_iter = st.slider('Max Iterations', 2000, 4000, 3000, 500)
        if st.button('Run the Neural Net'):
            #load the data and the labels
            X = df.values[:,0:-1]
            y = df.values[:,-1].astype(int)
            
            # Convert string data to numerical data
            label_encoder = [] 
            X_encoded = np.empty(X.shape)

            for i,item in enumerate(X[0]):
                if str(item).replace('.','').isdigit(): 
                    X_encoded[:, i] = X[:, i]
                else:
                    label_encoder = preprocessing.LabelEncoder()
                    X_encoded[:, i] = label_encoder.fit_transform(X[:, i])

            X = X_encoded[:, :].astype(float)
            
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, \
                test_size=0.3, random_state=42)

            # Scale the features using standardization
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Create a neural network classifier using MLPClassifier
            clf = MLPClassifier(hidden_layer_sizes=(hidden_layers), \
                activation='relu', solver='adam', max_iter=max_iter)

            # Train the classifier on the training set
            clf.fit(X_train, y_train)

            # Test the classifier on the testing set
            accuracy = clf.score(X_test, y_test)
            st.write('accuracy = ' + accuracy)
            st.write(classification_report(y_test, clf.predict(X_test)))
            

   
#run the app
if __name__ == "__main__":
    app()
