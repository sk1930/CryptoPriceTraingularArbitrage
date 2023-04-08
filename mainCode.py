'''

# To run this code in jupyter notebook use the command %run mainCode
# to run this code in command prompt enter python mainCode.py

This Code generates BTC_Indicators.csv

This Code is for technical indicators for BTCprice.
The indicators covered in this code are :
Accumulation/Distribution Line,Aroon, Aroon Oscillator, Average Directional Index (ADX), Average True Range (ATR),
%B Indicator, Chaikin Money Flow (CMF), Chaikin Oscillator, Commodity Channel Index (CCI), Coppock Curve, Ease of Movement (EMV), 
Force Index, Mass Index, M ACD, MACD Histogram, Money Flow Index (MFI), 24. Percentage Price Oscillator (PPO), Percentage Volume Oscillator (PVO), 
Pring’s Know Sure Thing (KST), Rate of Change (ROC) and  Momentum,  Relative Strength Index (RSI), Standard Deviation (Volatility), 
Stochastic Oscillator, TRIX, Ulcer Index, Ultimate Oscillator,  Vortex Indicator , Williams %R


'''

import functions as f
import numpy as np
import pandas as pd

from math import floor
from ta.momentum import PercentagePriceOscillator as PPO
from ta.momentum import PercentageVolumeOscillator as PVO
from ta.trend import TRIXIndicator
from ta.trend import VortexIndicator
from ta.momentum import UltimateOscillator

from ta.momentum import WilliamsRIndicator
from ta.trend import ADXIndicator
from ta.momentum import ROCIndicator

import warnings
warnings.filterwarnings("ignore")
'''
'''

def runCode():
    pd.set_option('display.max_columns', None)


    df = f.fetch_data(end_date= '2022-11-23')
    #print(df)
    # f.test()


    df=f.acc_dist(df,trend_periods=21,open_col='Open', high_col='High', low_col='Low', close_col='Close', vol_col='Volume')
    
    df=f.aroon(df,lb=25)
    df=f.adx(df,window=14,fillna=False)  # averag directional index
    df=f.atr(df,window=14) # average true range
    df=f.bb(df,window=20)  # %B Indicator Bolinger bands
    df = f.calculate_chaikin_money_flow(df,n=20) # # Chaikin Money Flow (CMF)
    df=f.chaikin_oscillator(df, periods_short=3, periods_long=10, high_col='High',low_col='Low', close_col='Close', vol_col='Volume')
    df = f.CCI(df,ndays=14) #commodity channel index
    df = f.get_cc(df, roc1_n=14, roc2_n=11, wma_lookback=10)
    df = f.EMV(df, ndays=14)
    df = f.ForceIndex(df,ndays=15)
    df = f.mass_index(df, period=25, ema_period=9, high_col='High', low_col='Low')
    df = f.get_macd(df, slow=26, fast=12, smooth=9)
    data = f.mfi(df,n=14)
    # PercentagePriceOscillator
    ppo1 = PPO(df['Close'])
    df['PercentagePriceOscillator']=ppo1.ppo()
    df['PercentagePriceOscillator_hist']= ppo1.ppo_hist()
    df['PercentagePriceOscillator_signal']= ppo1.ppo_signal()
    # Percentage Volume Oscillator (PVO)
    pvo = PVO(df['Volume'])
    df['PercentageVolumeOscillator']=pvo.pvo()
    df['PercentageVolumeOscillator_hist']=pvo.pvo_hist()
    df['PercentageVolumeOscillator_signal']=pvo.pvo_signal()
    
    
    df = f.get_kst(df,sma1=10, sma2=10, sma3=10, sma4=15, roc1=10, roc2=15, roc3=20, roc4=30, signal=9)
    
    df['Rate of change'] = ROCIndicator(df['Close'], window = 12, fillna = False).roc()
    df=f.get_rsi(df ,lookback=14)
    
    df = f.volatility(df,TRADING_DAYS = 365)
    df = f.add_stochastic_oscillator(df, periods=14)
    
    # TRIX: triple exponential average (TRIX) indicator
    
    data['Trix']=TRIXIndicator(close=data['Close'], window = 15, fillna = False).trix()
    df = f.calc_ulcer_index(df, periods=14)
    df['UltimateOscillator']= UltimateOscillator(df['High'], df['Low'], df['Close'], window1 = 7, window2 = 14, window3 = 28, weight1 = 4.0, weight2 = 2.0, weight3 = 1.0, fillna = False).ultimate_oscillator()

    
    VI = VortexIndicator(data['High'], data['Low'], data['Close'], window = 14, fillna = False)
    
    df['VortexIndicator_diff']=VI.vortex_indicator_diff()

    df['VortexIndicator_neg']=VI.vortex_indicator_neg()

    df['VortexIndicator_pos']=VI.vortex_indicator_pos()
    
    WI = WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp = 14, fillna = False)
    df['WilliamsR'] = WI.williams_r()

    # Remove timezone from columns
    df['Date'] = df['Date'].dt.tz_localize(None)

    # Export to excel
    df.to_csv("BTC_Indicators.csv")
    
    
    
    print(df)
    


if __name__=="__main__":
    runCode()
    