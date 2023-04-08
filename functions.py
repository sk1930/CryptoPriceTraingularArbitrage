
import yfinance as yf
import numpy as np
import pandas as pd

from ta.trend import ADXIndicator
from ta.trend import TRIXIndicator
from ta.trend import VortexIndicator
from ta.momentum import WilliamsRIndicator
from ta.momentum import PercentagePriceOscillator as PPO

def retDate(x):
    y=x.strftime("%d%b%Y").lower()
    return y



def fetch_data(start_date='2014-09-17',end_date= '2022-11-23'):
    data = yf.download(tickers='BTC-USD',start =  start_date ,end=end_date,  interval = '1d')
    print(data.head())
    data1= data.reset_index()
    data1['dateFormatted'] = data1['Date'].apply(retDate)
    print(data1.head())
    return data1

'''
#  accumulation/distribution and accumulation/distribution  exponential moving avergae
def acc_dist(data, trend_periods=21,open_col='<OPEN>',high_col='<HIGH>',low_col='<LOW>',close_col='<CLOSE>', vol_col='<VOL>'):

    x=[]
    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            ac = ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            ac = 0
        x.append(ac)
    data['accumulation_distribution']= x
    data['accumulation_distribution_EMA' + str(trend_periods)] = data['accumulation_distribution'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    
    return data
'''
#  accumulation/distribution 
def acc_dist(data, trend_periods=21, open_col='<OPEN>', high_col='<HIGH>', low_col='<LOW>', close_col='<CLOSE>', vol_col='<VOL>'):
    x=[]
    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            ac = ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            ac = 0
        x.append(ac)
    data['MoneyFlowVolume']= x
    A_D=[]
    P_A_D = 0
    for i in range(len(x)):
        A_D.append(P_A_D+x[i])
        P_A_D = P_A_D+x[i]
    data['accumulation_distribution_indicator']= A_D
    
    return data

 # Aroon: Uses Aroon Up and Aroon down Aroon Oscillator
def aroon(data, lb=25):

    data['Aroon_up'] = 100 * data.High.rolling(lb + 1).apply(lambda x: x.argmax()) / lb
    data['Aroon_down'] = 100 * data.Low.rolling(lb + 1).apply(lambda x: x.argmin()) / lb
    data['Aroon_oscillator']=data['Aroon_up']-data['Aroon_down']

    return data

# Average Directional Index (ADX)
def adx(data,window=14,fillna=False):
    data['Adj Open'] = data.Open * data['Adj Close']/data['Close']
    data['Adj High'] = data.High * data['Adj Close']/data['Close']
    data['Adj Low'] = data.Low * data['Adj Close']/data['Close']

    adxI = ADXIndicator(data['Adj High'],data['Adj Low'],data['Adj Close'],window,fillna)
    data['adx_pos_directional_indicator'] = adxI.adx_pos()
    data['adx_neg_directional_indicator'] = adxI.adx_neg()
    data['avg. Directional index'] = adxI.adx()
    return data


#  Average True Range (ATR)
def atr(data,window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    data['avg.True Range'] = true_range.rolling(window).sum()/window
    return data

# %B Indicator: Shows the relationship :

def sma(data, window):
    sma = data.rolling(window = window).mean()
    return sma

def bb(data, window=20):
    sma_20 = sma(data['Close'], 20)
    std = data['Close'].rolling(window = window).std()
    data['upper_Bollinger_Bands'] = upper_bb = sma_20 + std * 2
    data['lower_Bollinger_Bands'] = sma_20 - std * 2
    data['simple_moving_avg_20'] = sma_20
    
    return data



## Chaikin Money Flow (CMF):
def calculate_money_flow_volume_series(df: pd.DataFrame) -> pd.Series:
    """
    Calculates money flow series
    """
    mfv = df['Volume'] * (2*df['Close'] - df['High'] - df['Low']) / \
                                    (df['High'] - df['Low'])
    return mfv

def calculate_money_flow_volume(df: pd.DataFrame, n: int=21) -> pd.Series:
    """
    Calculates money flow volume, or q_t in our formula
    """
    return calculate_money_flow_volume_series(df).rolling(n).sum()

def calculate_chaikin_money_flow(df: pd.DataFrame, n: int=20) -> pd.Series:
    """
    Calculates the Chaikin money flow
    """
    df['Chaikin money flow'] = calculate_money_flow_volume(df, n) / df['Volume'].rolling(n).sum()
    return df


# chaikin_oscillator
def chaikin_oscillator(data, periods_short=3, periods_long=10, high_col='High',
                       low_col='Low', close_col='Close', vol_col='Volume'):
    ac = []
    val_last = 0

    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            val = val_last + ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            val = val_last
        ac.append(val)
        val_last = val
    #print(ac)
    ac=pd.Series(ac)
    ema_long = ac.ewm(ignore_na=False, min_periods=0, com=periods_long, adjust=True).mean()
    ema_short = ac.ewm(ignore_na=False, min_periods=0, com=periods_short, adjust=True).mean()
    #print("long is",ema_long)
    #print("long is",ema_short)
    
    data['chaikin_oscillator'] = ema_short - ema_long
    #print(ch_osc)
    return data

# Compute the Commodity Channel Index (CCI) based on the 14-day moving average
def CCI(df, ndays=14): 
    TP = (df['High'] + df['Low'] + df['Close']) / 3 
    sma = TP.rolling(ndays).mean()
    mad = TP.rolling(ndays).apply(lambda x: pd.Series(x).mad())
    df['Commodity Channel Index'] = (TP - sma) / (0.015 * mad) 
    return df


# Coppock Curve

def wma(data, lookback):
    weights = np.arange(1, lookback + 1)
    val = data.rolling(lookback)
    wma = val.apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw = True)
    return wma

def get_roc(close, n):
    difference = close.diff(n)
    nprev_values = close.shift(n)
    roc = (difference / nprev_values) * 100
    return roc

def get_cc(data, roc1_n=14, roc2_n=11, wma_lookback=10):
    longROC = get_roc(data['Close'], roc1_n)
    shortROC = get_roc(data['Close'], roc2_n)
    ROC = longROC + shortROC
    data['Coppock Curve'] = wma(ROC, wma_lookback)
    return data


# Ease of Movement (EMV)
def EMV(data, ndays=14): 
    dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
    br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    EMV = dm / br 
    EMV_MA = pd.Series(EMV.rolling(ndays).mean(), name = 'EMV') 
    data['Ease of Movement'] =EMV
    data['Ease of Movement MA'] =EMV_MA
    return data
    

# ForceIndex
def ForceIndex(data, ndays=15): 
    FI = pd.Series(data['Close'].diff(ndays) * data['Volume'], name = 'ForceIndex') 
    data['ForceIndex']=FI
    return data

# mass_index
def mass_index(data, period=25, ema_period=9, high_col='High', low_col='Low'):
    high_low = data[high_col] - data[low_col] + 0.000001 #this is to avoid division by zero below
    ema = high_low.ewm(ignore_na=False, min_periods=0, com=ema_period, adjust=True).mean()
    ema_ema = ema.ewm(ignore_na=False, min_periods=0, com=ema_period, adjust=True).mean()
    div = ema / ema_ema
    mi=[]

    for index, row in data.iterrows():
        if index >= period:
            val = div[index-period:index].sum()
        else:
            val = 0
        mi.append(val)
    data['mass_index'] = mi  
    return data

# Moving Average Convergence/Divergence indicator
def get_macd(data, slow=26, fast=12, smooth=9):
    price=data['Close']
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'macd'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    data['MovingAverageConvergenceDivergence'] =  macd
    data['MovingAverageConvergenceDivergence_signal'] = signal
    data['MovingAverageConvergenceDivergence_hist']= hist
    #df = pd.concat(frames, join = 'inner', axis = 1)
    return data



#  money flow index

    
def gain(x):
    return ((x > 0) * x).sum()

def loss(x):
    return ((x < 0) * x).sum()

def mfi(data, n=14):
    high = data['High']
    low=data['Low']
    close = data['Close']
    volume = data['Volume']
    typical_price = (high + low + close)/3
    money_flow = typical_price * volume
    mf_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
    signed_mf = money_flow * mf_sign
    mf_avg_gain = signed_mf.rolling(n).apply(gain, raw=True)
    mf_avg_loss = signed_mf.rolling(n).apply(loss, raw=True)
    data['money flow index'] = (100 - (100 / (1 + (mf_avg_gain / abs(mf_avg_loss))))).to_numpy()
    
    return data


# Pring’s Know Sure Thing (KST)
def get_roc(close, n):
    difference = close.diff(n)
    nprev_values = close.shift(n)
    roc = (difference / nprev_values) * 100
    return roc

# Pring’s Know Sure Thing (KST)
def get_kst(data, sma1=10, sma2=10, sma3=10, sma4=15, roc1=10, roc2=15, roc3=20, roc4=30, signal=9):
    close = data['Close']
    rcma1 = get_roc(close, roc1).rolling(sma1).mean()
    rcma2 = get_roc(close, roc2).rolling(sma2).mean()
    rcma3 = get_roc(close, roc3).rolling(sma3).mean()
    rcma4 = get_roc(close, roc4).rolling(sma4).mean()
    data['Know Sure Thing'] = (rcma1 * 1) + (rcma2 * 2) + (rcma3 * 3) + (rcma4 * 4)
    data['KST_signal'] = data['Know Sure Thing'].rolling(signal).mean()
    return data

# Relative Strength Index
def get_rsi(data, lookback=14):
    close = data['Close']
    ret = close.diff()
    #print("ret is", ret)
    up = []
    
    down = []
    for i in range(len(ret)):
        if ret[i] < 0:
            up.append(0)
            down.append(ret[i])
        else:
            up.append(ret[i])
            down.append(0)
    up_series = pd.Series(up)
    down_series = pd.Series(down)#.abs()
    #print("up series is ",up_series)
    #print("down series is ",down_series)

    up_ewm = up_series.ewm(com = lookback - 1, min_periods=lookback,adjust = True).mean()
    down_ewm = down_series.ewm(com = lookback - 1,min_periods=lookback, adjust = True).mean()
    #print("up_ewm is",up_ewm)
    #print("down_ewm is",down_ewm)

    rs = abs(up_ewm/down_ewm)
    rsi = 100 - (100 / (1 + rs))
    #print("rsi is",rsi)
    data['Relative Strength Index'] = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
    return data


def volatility(data,TRADING_DAYS = 365):

    returns = np.log(data['Close']/data['Close'].shift(1))
    returns.fillna(0, inplace=True)
    data['volatility/sd'] = returns.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)
    return data


# Stochastic_oscillator
def add_stochastic_oscillator(df, periods=14):    
    high_roll = df["High"].rolling(periods).max()
    low_roll = df["Low"].rolling(periods).min()
    
    # Fast stochastic indicator
    num = df["Close"] - low_roll
    denom = high_roll - low_roll
    df["%K-FAST Stochastic indicator"] = (num / denom) * 100
    
    # Slow stochastic indicator
    df["%D- slow stochastic indicator"] = df["%K-FAST Stochastic indicator"].rolling(3).mean()
    
    return df



 # Ulcer Index

def calc_ulcer_index(df, periods=14):
    # Returns a dataframe read from filepath with ulcer index as an added column
    # Ulcer Index formula:
    # 1) Percentage Drawdown = [(Close - N-period High Close)/N-period High Close] x 100
    # 2) Squared Average = (N-period Sum of Percent-Drawdown Squared)/N
    # 3) Ulcer Index = Square Root of Squared Average
     
    period_high_close = df['Close'].rolling(periods + 1).apply(lambda x: np.amax(x), raw=True)
    percentage_drawdown = df['Close']
    percentage_drawdown = (percentage_drawdown - period_high_close)/period_high_close * 100
    percentage_drawdown = np.clip(percentage_drawdown, a_min=None, a_max=0)
    percentage_drawdown = percentage_drawdown ** 2
    percentage_drawdown = percentage_drawdown.fillna(0)
    period_sum = percentage_drawdown.rolling(periods+1).sum()
    squared_average = round((period_sum / periods), 2)
    df['ulcer_index'] = round(squared_average ** 0.5, 2)
    return df