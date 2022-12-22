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
#from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
data=time.strftime("%d/%m/%Y")
#data = pd.read_csv(r"C:\Users\39333\desktop\ANACONDA\FuelConsumption.csv")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import yfinance as yf
import streamlit as st
from sklearn.linear_model import SGDRegressor
from PIL import Image
from lineartree import LinearTreeRegressor,LinearBoostRegressor
from datetime import datetime,date,timedelta
#@st.cache
image=Image.open('petrolio.JPG')#('sfera.JPG')
image = image.resize((1000, 400))
st.image(image)
#st.markdown('')
#st.subheader('..........................MERLIN.........................')
st.subheader('.........................CRUDE_OIL_WTI_FORECAST.....................')
#st.markdown('(CL=F)=CRUDE_OIL(WTI)', unsafe_allow_html=False)
#st.markdown('AAPL= APPLE', unsafe_allow_html=False)
#st.markdown('MSFT=MICROSOFT', unsafe_allow_html=False)
#st.header=('CLF  e^ CRUDE OIL')
#st.header=('AAPL  e^ APPLE')
#st.header=('MSFT  e^ MICROSOFT')
#st.title('i simboli degli stocks sono di yhaoo finance')
tickers=('CL=F')#st.selectbox('SCEGLI UN SIMBOLO.....' ,('CL=F','CL=F'))#, 'AAPL','MSFT'))
#st.header('i simboli degli stocks sono di yhaoo finance')
#st.header=('CL=F  e^ CRUDE OIL'),st.header=('AAPL  e^ APPLE')
#st.header=('AAPL  e^ APPLE')
#st.header=('MSFT  e^ MICROSOFT')
#tickers=st.selectbox('SCEGLI UN SIMBOLO..CLF=CRUDE_OIL..AAPL=APPLE...MSFT=MICROSOFT ' ,('CL=F'))#('CL=F', 'AAPL','MSFT'))
              
#st.write('HAI SELEZIONATO:' ,tickers)                      
#tickers=('CL=F')
now = date.today()
#@st.cache
def new_data():
    #tickers=st.text_input('SIMBOLO')
    data1=yf.download(tickers = tickers,period="2000d",interval='1d',auto_adjust=True)
    #data#=data.T#=data['Close']#[:-1]
    data1=(data1['Close'])
    data1=data1.reset_index()
    data1['Date'] = pd.to_datetime(data1['Date'],format='%Y%m%d').dt.date
    data1=data1.set_index('Date')
    data1.fillna(data1.mean(),inplace=True)
    #if tickers=='CL=F':
        #data1=data1[:-1]
    #else:
        #data1=data1
#return data1### extra  
    return data1#[:-1]       #####  potrebbe essere [:-1] occhio
if new_data()[-1:].index.values==np.array(now):        #######
   new_data=new_data()[:-1]    ####
else:      #####
   new_data=new_data() 
model = pickle.load(open('stocks_RF.pk','rb'))
x1=(new_data[-7:].values.flatten()).reshape(-1,7)
yhat=model.predict(x1).round(5)

#st.title('........STOCK_FORECAST.........')
#st.sidebar.subheader('ULTIMO DATO IN ARCHIVIO DEL SIMBOLO SCELTO PER IL CALCOLO')
st.sidebar.subheader('AGGIORNAMENTO AUTOMATICO DA YAHOO FINANCE ORARIO NEW YORK')
#st.sidebar.subheader('ORARIO NEW YORK')
st.sidebar.subheader('ULTIMI 7 DATI IN ARCHIVIO DEL WTI PER IL CALCOLO')
st.sidebar.write(new_data[-7:])#(f'ULTIMi DATO IN ARCHIVIO {new_data[-1:]:.2f}')
#st.sidebar.write('PER INFORMAZIONI.....')
st.sidebar.info('gianfranco.fa@gmail.com')
#filter=   abs((yhat[-1:]/yhat[-2:-1])-1)
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
###################################################################
new_data3=new_data
lookback_window = 7
x_new_data3 = []
for i in range(lookback_window, len(new_data3)):
    x_new_data3.append(new_data3.shift(-1)[i-lookback_window:i])
x_new_data3 = np.array(x_new_data3)
st.write('........................................MATRICE_DI_CALCOLO.....................................')
st.write(((x_new_data3.flatten().reshape(-1,7)))[-7:])




#new_data3=(new_data[-21:].values.flatten()).reshape(-1,7)
yhat2=model.predict(x_new_data3.flatten().reshape(-1,7))
dfyhat = pd.DataFrame(data=yhat2 )
#change=abs(dfyhat.pct_change().dropna())
#yhat2=np.where(change>0.015,'TRADE','STAY_FLAT')###  ORIGINALE
yhat2=np.where(dfyhat.pct_change()>0.014999,'TRADE_LONG',(np.where(dfyhat.pct_change()<-0.014999,'TRADE_SHORT','FLAT')))
st.subheader(yhat2[-1:])
#st.subheader(new_data3)
if st.button('FORECAST_CRUDE-OIL'):
   prediction=yhat#np.where('change'>0.015,yhat,0)
   st.subheader(f' FORECAST + un giorno in archivio ... +- 2% ..   {prediction[0]:.2f} USD')
#st.subheader(f' OGGI è ...   {data} ')
now2 = datetime.now()
server_time = now2.strftime("DATE_SERVER_%d/%m/%y_TIME_%H:%M:%S")
image2=Image.open('trading_days.JPG')#('sfera.JPG')
st.image(image2)
st.write('MERLIN SYSTEM FROM 15 NOVEMBER 2022 (publication date)')
st.write(server_time)
