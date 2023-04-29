import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta import add_all_ta_features


def add_all_indicators(csv, drop_duplicates=True, drop_na=True):
    data =pd.read_csv(csv)
    if drop_na:
        data.dropna(inplace=True)
    if drop_duplicates:
        data.drop_duplicates(inplace=True)
    data = add_all_ta_features(data, 'open', 'high', 'low', 'close', 'volume', fillna=True)
    # Adding Talib Functions
    bban1, bban2, bban3 = talib.BBANDS(data['close'])
    dema = talib.DEMA(data['close'])
    ema = talib.EMA(data['close'])
    hilbert_trendline = talib.HT_TRENDLINE(data['close'])
    kama =talib.KAMA(data['close'])
    ma = talib.MA(data['close'])
    mama1, mama2 = talib.MAMA(data['close'])
    midpoint = talib.MIDPOINT(data['close'])
    midpoint_price = talib.MIDPRICE(data['high'], data['low'])
    sar = talib.SAR(data['high'], data['low'])
    sarext = talib.SAREXT(data['high'], data['low'])
    sma = talib.SMA(data['close'])
    t3 = talib.T3(data['close'])
    tema = talib.TEMA(data['close'])
    wma = talib.WMA(data['close'])
    adx = talib.ADX(data['high'], data['low'], data['close'])
    adxr = talib.ADXR(data['high'], data['low'], data['close'])
    apo = talib.APO(data['close'])
    aroon1 , aroon2 = talib.AROON(data['high'], data['low'])
    aroonsc = talib.AROONOSC(data['high'], data['low'])
    bop = talib.BOP(data['open'] ,data['high'], data['low'], data['close'])
    cci = talib.CCI(data['high'], data['low'], data['close'])
    cmo = talib.CMO(data['close'])
    dx = talib.DX(data['high'], data['low'], data['close'])
    macd1, macd2, macd3 = talib.MACD(data['close'])
    macdext1, macdext2, macdext3 = talib.MACDEXT(data['close'])
    macdfix1, macdfix2, macdfix3 = talib.MACDFIX(data['close'])
    mfi = talib.MFI(data['high'], data['low'], data['close'], data['volume'])
    minus_di = talib.MINUS_DI(data['high'], data['low'], data['close'])
    minus_dm = talib.MINUS_DM(data['high'], data['low'])
    mom = talib.MOM(data['close'])
    plus_di = talib.PLUS_DI(data['high'], data['low'], data['close'])
    plus_dm = talib.PLUS_DM(data['high'], data['low'])
    pro = talib.PPO(data['close'])
    roc = talib.ROC(data['close'])
    rocp = talib.ROCP(data['close'])
    rocr = talib.ROCR(data['close'])
    rocr100 = talib.ROCR100(data['close'])
    rsi = talib.RSI(data['close'])
    stoch1, stoch2 = talib.STOCH(data['high'], data['low'], data['close'])
    stochf1, stochf2 = talib.STOCHF(data['high'], data['low'], data['close'])
    stochrsi1, stochrsi2 = talib.STOCHRSI(data['close'])
    trix = talib.TRIX(data['close'])
    ultosc = talib.ULTOSC(data['high'], data['low'], data['close'])
    willr = talib.WILLR(data['high'], data['low'], data['close'])
    ad = talib.AD(data['high'], data['low'], data['close'], data['volume'])
    adosc = talib.ADOSC(data['high'], data['low'], data['close'], data['volume'])
    obv = talib.OBV(data['close'], data['volume'])
    ht_dcperiod = talib.HT_DCPERIOD(data['close'])
    ht_dcphase = talib.HT_DCPHASE(data['close'])
    ht_phasor1, ht_phasor2 = talib.HT_PHASOR(data['close'])
    ht_sine1, ht_sine2 = talib.HT_SINE(data['close'])
    ht_trendmode = talib.HT_TRENDMODE(data['close'])
    avrg_price = talib.AVGPRICE(data['open'] ,data['high'], data['low'], data['close'])
    atr = talib.ATR(data['high'], data['low'], data['close'])
    natr = talib.NATR(data['high'], data['low'], data['close'])
    trange = talib.TRANGE(data['high'], data['low'], data['close'])
    
    # Adding functions to data frame
    data['trange'] = trange
    data['natr'] = natr
    data['atr'] = atr
    data['sarext'] = sarext
    data['sar'] = sar
    data['dx'] = dx
    data['dema'] = dema
    data['macd1'] = macd1
    data['macd2'] = macd2
    data['macd3'] = macd3
    data['midpoint'] = midpoint
    data['wma'] = wma
    data['apo'] = apo
    data['mama1'] = mama1
    data['mama2'] = mama2
    data['bop'] = bop
    data['tema'] = tema
    data['avgprice'] = avrg_price
    data['adxr'] = adxr
    data['ma'] = ma
    data['midprice'] = midpoint_price
    data['ht_trendmode'] = ht_trendmode
    data['ht_sine1'] = ht_sine1
    data['ht_sine2'] = ht_sine2
    data['ht_phasor1'] = ht_phasor1
    data['ht_phasor2'] = ht_phasor2
    data['ht_dcphase'] = ht_dcphase
    data['ht_dcperiod'] = ht_dcperiod
    data['obv'] = obv
    data['sma'] = sma
    data['adosc'] = adosc
    data['ad'] = ad
    data['t3'] = t3
    data['cci'] = cci
    data['willr'] = willr 
    data['ultosc'] = ultosc
    data['adx'] = adx
    data['cmo'] = cmo
    data['ht_trendline'] = hilbert_trendline
    data['stochrsi1'] = stochrsi1
    data['stochrsi2'] = stochrsi2
    data['stochf1'] = stochf1
    data['stochf2'] = stochf2
    data['stoch1'] = stoch1
    data['stoch2'] = stoch2
    data['rsi'] = rsi
    data['trix'] = trix
    data['rocr100'] = rocr100
    data['rocr'] = rocr
    data['rocp'] = rocp
    data['roc'] = roc
    data['aroonsc'] = aroonsc
    data['mom'] = mom
    data['kama'] = kama
    data['plus_di'] = plus_di
    data['pro'] = pro
    data['plus_dm'] = plus_dm
    data['minus_dm'] = minus_dm
    data['minus_di'] = minus_di
    data['mfi'] = mfi
    data['macdfix1'] = macdfix1
    data['macdfix2'] = macdfix2
    data['macdfix3'] = macdfix3
    data['aroon1'] = aroon1
    data['aroon2'] = aroon2
    data['bban1'] = bban1
    data['bban2'] = bban2
    data['bban3'] = bban3
    data['ema'] = ema
    data['macdext1']= macdext1
    data['macdext2'] = macdext2
    data['macdext3'] = macdext3
    if drop_na:
        data = data.dropna()
    if drop_duplicates:
        data = data.drop_duplicates()

    return data

data = add_all_indicators('data.csv')

data = data.iloc[:, 1:]
data.set_index(pd.RangeIndex(len(data)),inplace=True)
data.to_csv('filename.csv')



