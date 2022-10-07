# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:05:02 2022

@author: 39333
"""
import numpy as np
import pandas as pd
import pickle
import json
import time
data=time.strftime("%d/%m/%Y")
#data = pd.read_csv(r"C:\Users\39333\desktop\ANACONDA\FuelConsumption.csv")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import yfinance as yf
import streamlit as st
from sklearn.linear_model import SGDRegressor
from PIL import Image
from lineartree import LinearTreeRegressor,LinearBoostRegressor
image=Image.open('sfera.JPG')
image = image.resize((1000, 400))
st.image(image)
#st.subheader('..........................MERLIN.........................')
st.subheader('..........................MERLIN_STOCK_FORECAST.......................')
st.markdown('(CL=F)=CRUDE_OIL(WTI)', unsafe_allow_html=False)
st.markdown('AAPL= APPLE', unsafe_allow_html=False)
st.markdown('MSFT=MICROSOFT', unsafe_allow_html=False)
#st.header=('CLF  e^ CRUDE OIL')
#st.header=('AAPL  e^ APPLE')
#st.header=('MSFT  e^ MICROSOFT')
#st.title('i simboli degli stocks sono di yhaoo finance')
tickers=st.selectbox('SCEGLI UN SIMBOLO.....' ,('CL=F', 'AAPL','MSFT'))
#st.header('i simboli degli stocks sono di yhaoo finance')
#st.header=('CL=F  e^ CRUDE OIL'),st.header=('AAPL  e^ APPLE')
#st.header=('AAPL  e^ APPLE')
#st.header=('MSFT  e^ MICROSOFT')
#tickers=st.selectbox('SCEGLI UN SIMBOLO..CLF=CRUDE_OIL..AAPL=APPLE...MSFT=MICROSOFT ' ,('CL=F', 'AAPL','MSFT'))
              
#st.write('HAI SELEZIONATO:' ,tickers)                      
#tickers=('CL=F')
def new_data():
    #tickers=st.text_input('SIMBOLO')
    data1=yf.download(tickers = tickers,period="12d",interval='1d',auto_adjust=True)
    #data#=data.T#=data['Close']#[:-1]
    data1=(data1['Close'])
    data1=data1.reset_index()
    data1['Date'] = pd.to_datetime(data1['Date'],format='%Y%m%d').dt.date
    data1=data1.set_index('Date')
    data1.fillna(data1.mean(),inplace=True)
    if tickers=='CL=F':
        data1=data1[:-1]
    else:
        data1=data1
       
    return data1#[:-1]
model = pickle.load(open('stocks.pk','rb'))
x1=(new_data()[-7:].values.flatten()).reshape(1,-1)
yhat=model.predict(x1).round(2)
#st.title('........STOCK_FORECAST.........')
#st.sidebar.subheader('ULTIMO DATO IN ARCHIVIO DEL SIMBOLO SCELTO PER IL CALCOLO')
st.sidebar.subheader('AGGIORNAMENTO AUTOMATICO DA YAHOO FINANCE ORARIO NEW YORK')
#st.sidebar.subheader('ORARIO NEW YORK')
st.sidebar.subheader('ULTIMI DATI IN ARCHIVIO DEL SIMBOLO SCELTO PER IL CALCOLO')
st.sidebar.write(new_data()[-7:])#(f'ULTIMi DATO IN ARCHIVIO {new_data[-1:]:.2f}')
#st.sidebar.write('PER INFORMAZIONI.....')
st.sidebar.info('gianfranco.fa@gmail.com')
   
st.markdown(
    """
    <style>
    textarea {
        font-size: 1rem !important;
    }
    input {
        font-size: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
if st.button('FORECAST'):
   prediction=yhat
   st.subheader(f' FORECAST PER IL GIORNO DOPO ... +- 2% ..   {prediction[0]:.2f} USD')
#st.subheader(f' OGGI è ...   {data} ')
