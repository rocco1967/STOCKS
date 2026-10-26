# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:05:02 2022

@author: 39333
"""
import streamlit as st
st.set_page_config(
    page_title="Previsioni petrolio WTI oggi | Segnali di trading con AI",
    page_icon="🛢️",
    layout="wide"
)
st.title("Previsioni sul petrolio greggio WTI con IA")
st.warning(
    "⚠️ **Avvertenza:** Merlin è un esperimento gratuito di analisi quantitativa "
    "sul petrolio WTI, con finalità esclusivamente informative e didattiche. "
    "I segnali sono generati automaticamente da un modello di intelligenza artificiale "
    "e **non costituiscono consulenza in materia di investimenti**, raccomandazione "
    "personalizzata o invito ad acquistare o vendere strumenti finanziari. "
    "Il trading su futures comporta un rischio elevato di perdita del capitale."
)
st.header("Chi siamo")
st.markdown("""
Merlin è un progetto indipendente , appassionato di
[analisi dei dati / mercati / programmazione]. Nasce per sperimentare l'uso
dell'intelligenza artificiale sulle serie storiche del petrolio WTI.
Il servizio è gratuito e non prevede commissioni, abbonamenti né intermediazione.
""")
st.subheader("Segnali di trading con intelligenza artificiale, aggiornati ogni giorno")
#st.set_page_config(page_title="AI side hustle")
#st.subheader(' Artificial Intelligence_Trading Signal Generator on WTI')
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
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
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
st.sidebar.caption(
    "Dati: Yahoo Finance (futures WTI, ticker CL=F), possibili ritardi o imprecisioni. "
    "Le performance passate non sono indicative di quelle future."
)
#st.sidebar.subheader('ORARIO NEW YORK')
st.sidebar.subheader('ULTIMI 7 DATI IN ARCHIVIO WTI....')
st.sidebar.write(new_data[-7:])#(f'ULTIMi DATO IN ARCHIVIO {new_data[-1:]:.2f}')
#st.sidebar.write('PER INFORMAZIONI.....')
st.sidebar.info('github..rocco1967')
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
data['Target']=data['CL=F'].shift(-1)     #  ORIGINALE
#data['Target']=data['Close'].shift(-1)  
data['Target']=data['Target'].fillna(data['Target'].shift(1))
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
data=data['CL=F'].values#data=data['Close'].values# se si usa il multicross al posto di Close mettere il Ticker
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
commission=0.002 ### 1 = 100%    deve essere semprePOSITIVO
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
df_roll_filtered['system']=(((10000*df_roll_filtered['equity_com_sl']).cumsum()+10000)+190)-10000#,color='red',label='MERLIN_SYSTEM')
df_roll_filtered['real']=((df_roll['real_roll_%change']*10000).cumsum()+10000+160)-10000#,color='black',label='REALE')
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
st.warning(
    "⚠️ **Avvertenza:** Merlin è un esperimento gratuito di analisi quantitativa "
    "sul petrolio WTI, con finalità esclusivamente informative e didattiche. "
    "I segnali sono generati automaticamente da un modello di intelligenza artificiale "
    "e **non costituiscono consulenza in materia di investimenti**, raccomandazione "
    "personalizzata o invito ad acquistare o vendere strumenti finanziari. "
    "Il trading su futures comporta un rischio elevato di perdita del capitale."
)
with st.expander("Note legali e avvertenze sui rischi"):
    st.markdown("""
**Cos'è Merlin.** Merlin è un progetto personale, gratuito e senza scopo di lucro,
che elabora i prezzi storici del petrolio WTI con un modello di intelligenza
artificiale [descrizione breve: es. rete neurale / modello statistico] per produrre
indicazioni di tendenza a scopo didattico.

**Nessuna consulenza finanziaria.** Le indicazioni mostrate sono output automatici
di un algoritmo. Non tengono conto della situazione finanziaria, degli obiettivi
o della propensione al rischio di chi le consulta e non costituiscono consulenza
in materia di investimenti, né ricerca in materia di investimenti, né sollecitazione
al pubblico risparmio. L'autore non è un consulente finanziario abilitato.

**Rischi.** Il petrolio e i relativi futures sono strumenti molto volatili. Operare
con leva o su derivati può comportare perdite anche superiori al capitale investito.
Il modello può sbagliare, e i risultati passati non garantiscono quelli futuri.

**Dati.** I prezzi provengono da Yahoo Finance e possono essere ritardati, incompleti
o errati. Non si garantiscono accuratezza, continuità o disponibilità del servizio,
che può essere modificato o interrotto in qualsiasi momento.

**Responsabilità.** L'autore declina ogni responsabilità per perdite o danni derivanti
dall'uso delle informazioni presenti. Ogni decisione di investimento è presa
autonomamente e sotto la piena responsabilità dell'utente, che è invitato a
rivolgersi a un professionista abilitato.

**Contatti e privacy.** [Link a modulo di contatto]. L'app non raccoglie dati
personali degli utenti [verifica che sia vero].
""")
# ===== PERIODO FUORI CAMPIONE =====
DATA_INIZIO = "2022-11-15"   # primo giorno fuori campione (cambiala se serve)
SOGLIA = 0.01999999          # filtro del 2%
COMMISSIONE = 0.002          # 0,2% a operazione
STOPLOSS = -0.02
CAPITALE = 10000

serie = new_data.iloc[:, 0].astype(float)
date_idx = pd.to_datetime(serie.index)
vals = serie.values
n = len(vals)

X_all = np.array([vals[t - 7:t] for t in range(7, n)])
pred = pd.Series(model.predict(X_all), index=date_idx[7:])
reale = pd.Series(vals[7:], index=date_idx[7:])

var_pred = pred.pct_change()
posizione = np.where(var_pred > SOGLIA, 1, np.where(var_pred < -SOGLIA, -1, 0))
rend = reale.pct_change()
lordo = pd.Series(posizione, index=pred.index) * rend
lordo = lordo.where(lordo >= STOPLOSS, STOPLOSS)
traded = posizione != 0
netto = lordo - np.where(traded, COMMISSIONE, 0.0)

fc = pred.index >= pd.Timestamp(DATA_INIZIO)
netto_fc = netto[fc].fillna(0)
rend_fc = rend[fc].fillna(0)
traded_fc = traded[fc]

equity_sys = CAPITALE + (CAPITALE * netto_fc).cumsum()
equity_bh = CAPITALE + (CAPITALE * rend_fc).cumsum()

n_op = int(traded_fc.sum())
vincenti = int(((netto_fc > 0) & traded_fc).sum())
drawdown = (equity_sys - equity_sys.cummax()).min()
risultato_sys = equity_sys.iloc[-1] - CAPITALE
risultato_bh = equity_bh.iloc[-1] - CAPITALE

st.subheader(f"PERIODO FUORI CAMPIONE (dal {DATA_INIZIO}, dati mai visti dal modello)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Operazioni", n_op)
c2.metric("% operazioni in utile", f"{100 * vincenti / n_op:.0f}%" if n_op else "n/d")
c3.metric("Risultato MERLIN (USD)", f"{risultato_sys:,.0f}")
c4.metric("Buy & hold (USD)", f"{risultato_bh:,.0f}")
st.write(f"Massimo drawdown della simulazione: {drawdown:,.0f} USD su {CAPITALE:,} investiti.")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(equity_sys.index, equity_sys.values, label="MERLIN (commissioni e stop loss)", color="tab:orange")
ax.plot(equity_bh.index, equity_bh.values, label="Buy & hold WTI", color="black", alpha=0.7)
ax.axhline(CAPITALE, color="gray", linewidth=0.8)
ax.legend()
ax.grid(alpha=0.3)
ax.set_title(f"Simulazione fuori campione dal {DATA_INIZIO}")
st.pyplot(fig)
st.caption("Simulazione su dati storici: puntata fissa, commissione 0,2% per operazione, stop loss -2% "
           "assunto eseguito esattamente a quel livello (nella realtà gap e slippage possono peggiorarlo). "
           "I risultati passati non sono indicativi di quelli futuri.")

#############################################################################################################################
r = new_data.iloc[:, 0].astype(float).pct_change().dropna()
r.index = pd.to_datetime(r.index)
st.write(r.abs().groupby(r.index.day).mean().round(4))
r = new_data.iloc[:, 0].astype(float).pct_change().dropna()
r.index = pd.to_datetime(r.index)
g21 = r[r.index.day == 21].abs()
st.write("mediana giorno 21:", round(g21.median(), 4), "| mediana altri giorni:", round(r[r.index.day != 21].abs().median(), 4))
st.write(g21.sort_values(ascending=False).head(5))   # i 5 giorni 21 più grandi, con la data
r = new_data.iloc[:, 0].astype(float).pct_change().dropna()
r.index = pd.to_datetime(r.index)
r = r[r.abs() < 1]            # toglie i giorni di aprile 2020 con movimenti assurdi
st.write(r.abs().groupby(r.index.day).agg(["mean", "median", "count"]).loc[17:23].round(4))
