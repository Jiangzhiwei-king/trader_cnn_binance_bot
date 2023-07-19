import requests
import csv
import time
import pandas as pd
from cnn import Cnn
from config import time_intervals

def calculate_slope(prices):
    return prices.diff()


def ema_diff(prices, ma1, ma2):
    ema1 = prices.ewm(span=ma1, adjust=False).mean()
    ema2 = prices.ewm(span=ma2, adjust=False).mean()

    return ema1-ema2


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


def calculate_atr(high_prices, low_prices, close_prices, period=14):
    tr1 = high_prices - low_prices
    tr2 = abs(high_prices - close_prices.shift())
    tr3 = abs(low_prices - close_prices.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_kdj(high_prices, low_prices, close_prices, period=14):
    lowest_low = low_prices.rolling(window=period).min()
    highest_high = high_prices.rolling(window=period).max()
    rsv = (close_prices - lowest_low) / (highest_high - lowest_low) * 100
    k = rsv.ewm(span=3, adjust=False).mean()
    d = k.ewm(span=3, adjust=False).mean()
    j = 3 * k - 2 * d
    return k - d, d - j, k - j


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


def get_binance_future_kline_data(symbol, interval, start_time, end_time):
    base_url = "https://fapi.binance.com"

    start_time_ms = int(time.mktime(
        time.strptime(start_time, "%Y-%m-%d"))) * 1000
    end_time_ms = int(time.mktime(time.strptime(end_time, "%Y-%m-%d"))) * 1000

    kline_data = []
    total_items = (end_time_ms - start_time_ms) // (499 * time_intervals[interval]*1000) + 1
    current_item = 0

    while start_time_ms < end_time_ms:
        url = f"{base_url}/fapi/v1/klines?symbol={symbol}&interval={interval}&startTime={start_time_ms}&limit=499"

        try:
            response = requests.get(url)
            data = response.json()

            if response.status_code == 200:
                if len(data) == 0:
                    break
                kline_data.extend(data)

                last_data_time = data[-1][0]
                start_time_ms = last_data_time + 1
            else:
                print(f"请求失败：{response.status_code}, {data}")
                break
        except requests.exceptions.RequestException as e:
            print(f"请求异常：{e}")
            break

        current_item += 1
        print(
            f"Fetching data for {symbol}, Progress: {current_item}/{total_items}")

        time.sleep(10)

    return kline_data


def save_to_csv(file_path, kline_data):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume",
                        "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"])

        for kline in kline_data:
            writer.writerow(kline)


def process_data(file_path):
    try:
        data = pd.read_csv(file_path)
        close_prices = data["Close"]
        ema200 = calculate_ema(close_prices, period=200)
        ema20 = calculate_ma(close_prices, period=20)
        macd = calculate_macd(close_prices)

        # 计算RSI指标
        rsi = calculate_rsi(close_prices, period=14)

        # 计算MA200指标
        ma200 = calculate_ma(close_prices, period=200)

        # 计算MA200指标
        ema20_ema200_diff = ema_diff(close_prices, 21, 200)
        atr = calculate_atr(data["High"], data["Low"],
                            data["Close"], period=14)
        kd, dj, kj = calculate_kdj(data["High"], data["Low"], data["Close"])
        volumes = data["Volume"]

        # 计算WMA指标
        wma = calculate_wma(close_prices, period=20)

        # 计算CCI指标
        cci = calculate_cci(close_prices, data["High"], data["Low"], period=20)

        # 计算MFI指标
        mfi = calculate_mfi(close_prices, volumes, period=14)

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
        columns_to_drop = ["Close time", "Quote asset volume",
                           "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]

        # 去除指定的列
        data = data.drop(columns=columns_to_drop)

        # 保存更新后的数据
        data.to_csv(file_path, index=False)

        print("处理完成")
    except FileNotFoundError:
        print(f"文件未找到：{file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空：{file_path}")
    except Exception as e:
        print(f"处理数据时出现异常：{e}")


if __name__ == "__main__":
    # 示例用法
    listS = [
        # "RVNUSDT", "BTCUSDT",
          "XRPUSDT", "ETHUSDT",
             "BNBUSDT", "OPUSDT", "LTCUSDT", "COMPUSDT", "BCHUSDT","DOTUSDT","AVAXUSDT","1INCHUSDT"]
    interval = "5m"
    start_time = "2023-06-04"
    end_time = "2023-07-17"

    total_items = len(listS)
    for idx, symbol in enumerate(listS, 1):
        kline_data = get_binance_future_kline_data(
            symbol, interval, start_time, end_time)
        save_to_csv("./data/" + symbol + interval +
                    start_time + end_time + '.csv', kline_data)
        print(f"Processing {symbol}, {idx}/{total_items}")
        process_data("./data/" + symbol + interval +
                     start_time + end_time + '.csv')
    Cnn()
