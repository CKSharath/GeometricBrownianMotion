import yfinance as yf
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

ticker='RITES.NS'
start=dt.datetime.now()-dt.timedelta(days=365)
end=dt.datetime.now()

df=yf.download(ticker,start,end)

LogReturn=df['Close'].apply(np.log).diff().dropna()
DriftRate=np.mean(LogReturn)
Volatility=np.std(LogReturn)

N=50 #100 simulations
T=250 #250 days predition

dt=1

S=df['Close'].iloc[-1]

pricePath=np.zeros((T+1,N))
pricePath[0]=S


for i in range(N):
    Scurr=S
    for t in range(1,T+1):
        z=np.random.randn()
        drift_adj = (DriftRate - (Volatility**2) / 2) * dt
        shock = Volatility * z * np.sqrt(dt)
        Snext = Scurr * np.exp(drift_adj + shock)
        pricePath[t, i] = Snext
        Scurr=Snext

plt.figure(figsize=(15,8))
plt.plot(pricePath)
plt.xlabel("Future dates from today (30/10/2025)")
plt.ylabel("Predicted prices")
plt.grid(True)
plt.tight_layout()
plt.show()
