# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:05:02 2022

@author: 39333
"""
import streamlit as st
st.set_page_config(page_title="AI side hustle")
st.subheader(' Artificial Intelligence_Trading Signal Generator on WTI')
import numpy as np
import pandas as pd
import pickle
import json
import time
#from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
data=time.strftime("%d/%m/%Y")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import yfinance as yf
#import streamlit as st
from sklearn.linear_model import SGDRegressor
from PIL import Image
#from lineartree import LinearTreeRegressor,LinearBoostRegressor
from datetime import datetime,date,timedelta
#@st.cache
image=Image.open('petrolio.JPG')#('sfera.JPG')
image = image.resize((1000, 400))
st.image(image)
import sklearn
#st.write(sklearn.__version__) #sklearn version

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
st.sidebar.subheader('ULTIMI 7 DATI IN ARCHIVIO WTI....')
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
yhat2=model.predict(((x_new_data3.flatten().reshape(-1,7))[-10:]))##
#st.write(yhat2)
#st.write((x_new_data3.flatten().reshape(-1,7))[-9:])
#st.write((x_new_data3[-5:]).reshape(-1,7))
dfyhat = pd.DataFrame(data=yhat2 )
#st.write(dfyhat)
#change=abs(dfyhat.pct_change().dropna())
#yhat2=np.where(change>0.015,'TRADE','STAY_FLAT')###  ORIGINALE
yhat2=np.where(dfyhat.pct_change()>0.019999,'TRADE_LONG',(np.where(dfyhat.pct_change()<-0.019999,'TRADE_SHORT','FLAT')))
#st.write(yhat2) ############################ ultimi segnali
#st.subheader(yhat2[-1:])
st.subheader(f' TODAY MACHINE LEARNING  FILTERED POSITION...... {yhat2[-1:]}')######
            
#st.subheader(new_data3)
if st.button('FORECAST_CRUDE-OIL'):
   prediction=yhat#np.where('change'>0.0199,yhat,0)
   st.subheader(f' FORECAST + un giorno in archivio ... +- 2% ..   {prediction[0]:.4f} USD')
#st.subheader(f' OGGI è ...   {data} ')
now2 = datetime.now()
server_time = now2.strftime("DATE_SERVER_%d/%m/%y_TIME_%H:%M:%S")
#image2=Image.open('trading_days.JPG')#('sfera.JPG')
#st.image(image2)
#st.write('MERLIN SYSTEM FROM 15 NOVEMBER 2022....investment 10k USD (from publication date__today not included)')
#st.write(server_time)
#st.write(yhat2.reshape(-1,len(yhat2)))
##############################################################   EQUITY #############################################################################
import matplotlib.pyplot as plt
tickers=('CL=F')#,AAPL,MSFT,NG=F')
def data():
    data=yf.download(tickers =tickers,period="2000d",interval='1d',auto_adjust=True)
    data=(data['Close'])
    data=data[data>0]
    data=data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d').dt.date
    data=data.set_index('Date').dropna()
    return data
data=data()
#st.write(data)
if data[-1:].index.values==np.array(now):
    data=data[:-1]
else:
    data=data.round(2)
#st.write(data)    
#data['Target']=data['CL=F'].shift(-1)     #  ORIGINALE
data['Target']=data['Close'].shift(-1)  
data['Target']=data['Target'].fillna(data['Target'].shift(1))
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
data=data['Close'].values# se si usa il multicross al posto di Close mettere il Ticker
data=data[data>0]#.copy()
lookback_window = 7
x, y = [], []
for i in range(lookback_window, len(data)):
    x.append(data[i - lookback_window:i])
    y.append(data[i])
x = np.array(x)
y = np.array(y)
#x.reshape(-1,1),y.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)
model = pickle.load(open('stocks_RF.pk','rb'))
pred=model.predict(x_test)
df_roll=pd.DataFrame(y_test.reshape(-1,1)[-227:],columns=['real_roll'])#,pred.reshape(-1,1)[-10:-1]
df_roll['pred_roll']=pd.DataFrame(pred.reshape(-1,1)[-227:],columns=['pred_roll'])
df_roll['pred_change']=np.where(df_roll['pred_roll'].pct_change()>0,1,-1)
df_roll['real_change']=np.where(df_roll['real_roll'].pct_change()>0,1,-1)
df_roll['real_roll_%change']=(df_roll['real_roll'].pct_change())
df_roll['pred_roll_%change']=(df_roll['pred_roll'].pct_change())
df_roll['equity']=df_roll['pred_change']*df_roll['real_roll_%change']
#df_roll
commission=0.005 ### 1 = 100%    deve essere semprePOSITIVO
stoploss=-0.05 ### -1 = 100%    deve essere sempre col SEGNO MENO DAVANTI
df_roll['equity_sl']=np.where(df_roll['equity']<stoploss,stoploss,df_roll['equity'])
#data['Equity_com']=np.where(data['Equity']>0,(data['Equity']-commission),(data['Equity']-commission))
df_roll['equity_com_sl']=np.where(df_roll['equity_sl']>0,(df_roll['equity_sl']-commission),(df_roll['equity_sl']-commission))#ORIG
#data['Equity_com_sl']=np.where(data['Equity_sl']>0,(data['Equity_sl']-commission),(data['Equity_sl']-commission))
df_roll['commission']=df_roll['real_roll']*commission 
        
df_roll_filtered = df_roll[abs(df_roll['pred_roll_%change']) > 0.01999999 ]# portafoglio filtrato per predict > di tot per cento

from matplotlib.pyplot import figure

import seaborn as sns
#equity=((1000*df_roll['equity_sl']).cumsum()+10000)
df_roll_filtered['system']=(((10000*df_roll_filtered['equity_com_sl']).cumsum()+10000)+150)-10000#,color='red',label='MERLIN_SYSTEM')
df_roll_filtered['real']=((df_roll['real_roll_%change']*10000).cumsum()+10000+50)-10000#,color='black',label='REALE')
chart_data = pd.DataFrame(
    df_roll_filtered['system'].values,
    columns=['merlin_system'])
chart_data2 = pd.DataFrame(df_roll_filtered['real'],columns=['real'])
st.subheader('MERLIN_SYSTEM..10k USD...INVESTED..from 15 NOVEMBER')

st.write('merlin doesn^t trade every day it has a trend filter')
st.subheader('GAIN : ')
st.line_chart(chart_data)
st.subheader('WTI..BUY.and..HOLD ..10k USD...INVESTED')
st.line_chart(chart_data2)
st.write(server_time)
