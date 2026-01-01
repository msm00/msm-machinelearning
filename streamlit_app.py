import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('Machine Learning App 🤖')

# st.write('Hello world!')

st.info('This app builds machine learning model!')

with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
    df

    st.write('**X**')
    X_raw = df.drop('species', axis=1)
    X_raw

    st.write('**Y**')
    y_raw = df.species
    y_raw

with st.expander('Data Visualization'):
    # "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Input Features
with st.sidebar:
    st.header('Input features')
    # "island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"
    island = st.selectbox('Island', df.island.unique())
    gender = st.selectbox('Gender', df.sex.unique())
    bill_length_mm = st.slider('Bill length (mm)', float(df.bill_length_mm.min()), float(df.bill_length_mm.max()), value=float(df.bill_length_mm.mean()))
    bill_depth_mm = st.slider('Bill depth (mm)', float(df.bill_depth_mm.min()), float(df.bill_depth_mm.max()), value=float(df.bill_depth_mm.mean()))
    flipper_length_mm = st.slider('Flipper length (mm)', float(df.flipper_length_mm.min()), float(df.flipper_length_mm.max()), value=float(df.flipper_length_mm.mean()))
    body_mass_g = st.slider('Body mass (g)', float(df.body_mass_g.min()), float(df.body_mass_g.max()), value=float(df.body_mass_g.mean()))

    # Create DataFrame for the input feature
    input_df = pd.DataFrame({
        'island': [island],
        'bill_length_mm': [bill_length_mm],
        'bill_depth_mm': [bill_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [gender]
    })
    input_penguins = pd.concat([input_df, X_raw], axis=0)

    # Data preparation
    # Encode X
    encode = ['sex', 'island']
    df_penguins = pd.get_dummies(input_penguins, prefix=encode)
    X = df_penguins[1:]
    input_row = df_penguins[:1]

    # Encode Y
    # target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
    target_mapper = {y_raw.unique()[0]: 0, y_raw.unique()[1]: 1, y_raw.unique()[2]: 2}

    def target_encode(val):
        return target_mapper[val]

    y = y_raw.apply(target_encode)

with st.expander('Input Features'):
    st.write('**Input Features**')
    input_df
    # st.write('**Input Features with encoded values**')
    # input_row
    # df_penguins
    # df_penguins[1:]

with st.expander('Data Preparation'):
    st.write('**Encoded X**')
    input_row
    st.write('**Encoded Y**')
    y

# Model training and interference
## Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make prediction
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# prediction
prediction_proba
