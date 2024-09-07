import streamlit as st
import xgboost as xgb
import sklearn
import pandas as pd
import pickle
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import plotly.express as px

st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.title("🎈 Streamlit App")

st.write('Es importante verificar la versión sobre la que están construidos los modelos, de otra forma estos\
         no funcionarán en la app.')

st.write("XGBoost version:", xgb.__version__)
st.write("Scikit-learn version:", sklearn.__version__)

st.write('Exploremos algunas opciones que nos ofrece streamlit.')

@st.cache_data
def read_data():
    data = pd.read_csv('./diamonds.csv')
    try:
        data = data.drop(columns=['Unnamed: 0'])
    except:
        pass
    return data

@st.cache_data
def clean_data(data):
    data = data.drop(data[data["x"]==0].index)
    data = data.drop(data[data["y"]==0].index)
    data = data.drop(data[data["z"]==0].index)
    data = data[(data["depth"]<75)&(data["depth"]>45)]
    data = data[(data["table"]<80)&(data["table"]>40)]
    data = data[(data["x"]<30)]
    data = data[(data["y"]<30)]
    data = data[(data["z"]<30)&(data["z"]>2)]
    return data

@st.cache_data
def label_data(data):
    s = (data.dtypes =="object")
    label_encoder = LabelEncoder()
    object_cols = list(s[s].index)
    for col in object_cols:
        data[col] = label_encoder.fit_transform(data[col])
    return data

@st.cache_data
def predict_data(data):
    # load the model from disk
    x = data.drop(["price"],axis =1)
    loaded_model = pickle.load(open('./pipeline_xgb.sav', 'rb'))
    data['price_prediction'] = loaded_model.predict(x)
    return data

@st.cache_data
def predict_values(values_array, cols):
    # load the model from disk
    x = pd.DataFrame([values_array], columns=cols)
    loaded_model = pickle.load(open('./pipeline_xgb.sav', 'rb'))
    prediction = loaded_model.predict(x)
    return prediction[0]

#===============SIDEBAR===============
sidebar = st.sidebar
sidebar.header('Sección de Filtros')

carat = sidebar.number_input('Valor de carat', value = 0.21)
cut = sidebar.number_input('Valor de cut', value = 3,step = 1)
color = sidebar.number_input('Valor de color', value = 1, step = 1)
clarity = sidebar.number_input('Valor de clarity', value = 2, step = 1)
depth = sidebar.number_input('Valor de depth', value = 59.8)
table = sidebar.number_input('Valor de table', value = 61)
x = sidebar.number_input('Valor de x', value = 3.89)
y = sidebar.number_input('Valor de y', value = 3.84)
z = sidebar.number_input('Valor de z', value = 2.31)

valores_input = [carat, cut, color, clarity, depth, table, x, y, z]
#=====================================

df = read_data()
clean_df = clean_data(df)
label_df = label_data(clean_df)
predict_df = predict_data(label_df)
predict_df['cut_cat'] = clean_df['cut']
predict_df['color_cat'] = clean_df['color']
predict_df['clarity_cat'] = clean_df['clarity']

show_clean_df = st.checkbox('Mostrar dataframe limpio', False)
if show_clean_df:
    st.write(clean_df)
show_label_df = st.checkbox('Mostrar dataframe etiquetado', False)
if show_label_df:
    st.write(label_df)
show_predict_df = st.checkbox('Mostrar dataframe con predicciones', True)
if show_predict_df:
    st.write(predict_df)

st.header('Predicciones nuevas')

st.write('En esta sección podrás explorar los resultados del modelo ante una nueva predicción basado en los datos\
         de entrada que le coloques.')

prediccion = predict_values(valores_input, list(label_df.drop(["price"],axis =1).columns))
st.write(prediccion)

fig = px.scatter(label_df, x="price", y="y", title="Regression Line on Price vs 'y'")
st.plotly_chart(fig)

