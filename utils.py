import pandas as pd
import numpy as np

def calculate_slope(prices):
    return prices.diff()


def calculate_rsi(prices, period=14):
    changes = prices.diff()
    gains = changes.where(changes > 0, 0)
    losses = -changes.where(changes < 0, 0)
    average_gain = gains.rolling(window=period).mean()
    average_loss = losses.rolling(window=period).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength)) - 50
    return rsi


def calculate_ema(prices, period):
    ema = prices.ewm(span=period, adjust=False).mean() - prices
    return ema


def calculate_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def calculate_ma(prices, period):
    ma = prices.rolling(window=period).mean() - prices
    return ma


def ema_diff(prices, ma1, ma2):
    ema1 = prices.ewm(span=ma1, adjust=False).mean()
    ema2 = prices.ewm(span=ma2, adjust=False).mean()

    return ema1-ema2


def calculate_atr(prices, period=14):
    high_prices = prices["High"]
    low_prices = prices["Low"]
    close_prices = prices["Close"]

    tr1 = high_prices - low_prices
    tr2 = abs(high_prices - close_prices.shift())
    tr3 = abs(low_prices - close_prices.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_kdj(prices, period=14):
    high_prices = prices["High"]
    low_prices = prices["Low"]
    close_prices = prices["Close"]

    lowest_low = low_prices.rolling(window=period).min()
    highest_high = high_prices.rolling(window=period).max()

    rsv = (close_prices - lowest_low) / (highest_high - lowest_low) * 100

    k = rsv.ewm(span=3, adjust=False).mean()
    d = k.ewm(span=3, adjust=False).mean()
    j = 3 * k - 2 * d

    return k-d, d-j, k-j


def calculate_vma(prices, period):
    volumes = prices["Volume"]
    vma = volumes.rolling(window=period).mean()
    return vma - volumes


def calculate_wma(prices, period):
    weights = pd.Series(range(1, period+1))
    wma = prices.rolling(window=period).apply(
        lambda x: (x * weights).sum() / weights.sum(), raw=True)
    return wma - prices


def calculate_cci(prices, high_prices, low_prices, period=20):
    tp = (high_prices + low_prices + prices) / 3
    tp_mean = tp.rolling(window=period).mean()
    tp_std = tp.rolling(window=period).std()
    cci = (tp - tp_mean) / (0.015 * tp_std)
    return cci - tp_std


def calculate_mfi(prices, volumes, period=14):
    typical_price = (prices + prices.shift(1) + prices.shift(-1)) / 3
    money_flow = typical_price * volumes
    positive_money_flow = money_flow.where(typical_price.diff() > 0, 0)
    negative_money_flow = money_flow.where(typical_price.diff() < 0, 0)
    positive_money_flow_sum = positive_money_flow.rolling(window=period).sum()
    negative_money_flow_sum = negative_money_flow.rolling(window=period).sum()
    mfi = 100 - (100 / (1 + positive_money_flow_sum /
                 negative_money_flow_sum))-50
    return mfi


def calculate_sma(prices, period):
    sma = prices.rolling(window=period).mean()
    return sma - prices


# 读取本地CSV文件
data = pd.read_csv("data\BNBUSDT5m2023-06-162023-07-17.csv")

# 提取需要计算的价格数据
close_prices = data["Close"]

# 计算RSI指标
rsi = calculate_rsi(close_prices, period=14)

# 计算EMA200指标
ema200 = calculate_ema(close_prices, period=200)
ema20 = calculate_ma(close_prices, period=20)

# 计算MACD指标
macd = calculate_macd(close_prices)

# 计算MA200指标

ma200 = calculate_ma(close_prices, period=200)

# 计算MA200指标
ema20_ema200_diff = ema_diff(close_prices, 21, 200)
atr = calculate_atr(data, period=14)
kd, dj, kj = calculate_kdj(data)
close_prices = data["Close"]
high_prices = data["High"]
low_prices = data["Low"]
volumes = data["Volume"]

# 计算WMA指标
wma = calculate_wma(close_prices, period=20)


# 计算CCI指标
cci = calculate_cci(close_prices, high_prices, low_prices, period=20)

# 计算MFI指标
mfi = calculate_mfi(close_prices, volumes, period=14)

# 计算vma
vma = calculate_vma(data, 20)

# 将计算结果添加到原始数据中

data["RSI"] = calculate_slope(rsi)
data["EMA200"] = calculate_slope(ema200)
data["EMA20"] = calculate_slope(ema20)
data["EMA20_EMA200_DIFF"] = calculate_slope(ema20_ema200_diff)
data["MACD"] = calculate_slope(macd)
data["atr"] = calculate_slope(atr)
data["CCI"] = calculate_slope(cci)
data["kd"] = calculate_slope(kd)
data["kj"] = calculate_slope(kj)
data["dj"] = calculate_slope(dj)


data = data[200:-1]
data = data.dropna()
# 需要去除的列名列表
columns_to_drop = ["Close time", "Quote asset volume", "Number of trades",
                   "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]

# 去除指定的列
data = np.array(data.drop(columns=columns_to_drop))
print(data)