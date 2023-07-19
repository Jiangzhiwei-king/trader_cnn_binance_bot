import datetime
import hashlib
import hmac
import json
import logging
import time
from threading import Thread

import numpy as np
import pandas as pd
import requests
import mysql.connector
from sklearn.preprocessing import MinMaxScaler
import keras
from binance.client import Client
# Assuming you have a `config.py` file with the required variables:
from config import API_KEY, API_SECRET, time_intervals


def timestamp_to_time(timestamp):
    # 将时间戳转换为datetime对象
    dt_object = datetime.fromtimestamp(timestamp)

    # 将datetime对象格式化为可读的时间字符串
    time_string = dt_object.strftime('%Y-%m-%d %H:%M:%S')

    return time_string


class TechnicalIndicators:
    @staticmethod
    def multiply_rows(matrix):
        matrix = np.array([matrix])
        return np.multiply(matrix, matrix.T)

    @staticmethod
    def calculate_slope(prices):
        return prices.diff()

    @staticmethod
    def ema_diff(prices, ma1, ma2):
        ema1 = prices.ewm(span=ma1, adjust=False).mean()
        ema2 = prices.ewm(span=ma2, adjust=False).mean()
        return ema1 - ema2

    @staticmethod
    def calculate_rsi(prices, period=14):
        changes = prices.diff()
        gains = changes.where(changes > 0, 0)
        losses = -changes.where(changes < 0, 0)
        average_gain = gains.rolling(window=period).mean()
        average_loss = losses.rolling(window=period).mean()
        relative_strength = average_gain / average_loss
        rsi = 100 - (100 / (1 + relative_strength)) - 50
        return rsi

    @staticmethod
    def calculate_ema(prices, period):
        ema = prices.ewm(span=period, adjust=False).mean() - prices
        return ema

    @staticmethod
    def calculate_macd(prices):
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal

    @staticmethod
    def calculate_ma(prices, period):
        ma = prices.rolling(window=period).mean() - prices
        return ma

    @staticmethod
    def calculate_atr(high_prices, low_prices, close_prices, period=14):
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift())
        tr3 = abs(low_prices - close_prices.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    @staticmethod
    def calculate_kdj(high_prices, low_prices, close_prices, period=14):
        lowest_low = low_prices.rolling(window=period).min()
        highest_high = high_prices.rolling(window=period).max()
        rsv = (close_prices - lowest_low) / (highest_high - lowest_low) * 100
        k = rsv.ewm(span=3, adjust=False).mean()
        d = k.ewm(span=3, adjust=False).mean()
        j = 3 * k - 2 * d
        return k - d, d - j, k - j

    @staticmethod
    def calculate_wma(prices, period):
        weights = pd.Series(range(1, period + 1))
        wma = prices.rolling(window=period).apply(
            lambda x: (x * weights).sum() / weights.sum(), raw=True)
        return wma - prices

    @staticmethod
    def calculate_cci(prices, high_prices, low_prices, period=20):
        tp = (high_prices + low_prices + prices) / 3
        tp_mean = tp.rolling(window=period).mean()
        tp_std = tp.rolling(window=period).std()
        cci = (tp - tp_mean) / (0.015 * tp_std)
        return cci - tp_std

    @staticmethod
    def calculate_mfi(prices, volumes, period=14):
        typical_price = (prices + prices.shift(1) + prices.shift(-1)) / 3
        money_flow = typical_price * volumes
        positive_money_flow = money_flow.where(typical_price.diff() > 0, 0)
        negative_money_flow = money_flow.where(typical_price.diff() < 0, 0)
        positive_money_flow_sum = positive_money_flow.rolling(
            window=period).sum()
        negative_money_flow_sum = negative_money_flow.rolling(
            window=period).sum()
        mfi = 100 - (100 / (1 + positive_money_flow_sum /
                     negative_money_flow_sum)) - 50
        return mfi


class BinanceFuturesAPI:
    BASE_URL = 'https://fapi.binance.com'

    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = Client(api_key, api_secret)
        logger.info("FutureApiBox create complete")

    def get_server_time(self):
        data = self.client.get_server_time()
        return data['serverTime']

    def get_ticker_price(self, symbol):
        data = self.client.get_ticker(symbol=symbol)
        return float(data['lastPrice'])

    def get_klines(self, symbol, interval, limit=500):
        data = self.client.futures_klines(
            symbol=symbol, interval=interval, limit=limit)
        return data

    def place_order(self, symbol, side, order_type, quantity, price=None):
        recvWindow = 5000
        timestamp = int(self.get_server_time())

        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'recvWindow': recvWindow,
            'timestamp': timestamp
        }

        if order_type == 'LIMIT':
            params['timeInForce'] = 'GTC'
            params['price'] = price

        try:
            response = self.client.futures_create_order(**params)
            logger.info(json.dumps(response, indent=4))  # Better readability
            return response
        except Exception as e:
            logger.error(f'Error occurred during placing order: {e}')
            raise Exception(f'Error occurred: {e}')

    def get_account_balance(self):
        timestamp = int(time.time() * 1000)
        params = {'timestamp': timestamp}

        query_string = '&'.join(
            [f'{key}={value}' for key, value in sorted(params.items())])
        signature = hmac.new(self.api_secret.encode(
            'utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        params['signature'] = signature

        try:
            data = self.client.futures_account_balance(**params)
            return data
        except Exception as e:
            logger.error(f'Error occurred during API request: {e}')
            raise Exception(f'Error occurred: {e}')


class BinanceDataProcessor:
    def __init__(self, host, port, user, password, database, symbols):
        self.db_connection = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        self._ensure_tables_exist(symbols)

    def __del__(self):
        self.db_connection.close()

    def _create_table_if_not_exists(self, symbol):
        cursor = self.db_connection.cursor()
        table_name = f'kline_data_{symbol.upper()}'

        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        if not cursor.fetchone():
            cursor.execute(f'''
                CREATE TABLE {table_name} (
                    open_time BIGINT,
                    open_price DOUBLE,
                    high_price DOUBLE,
                    low_price DOUBLE,
                    close_price DOUBLE,
                    volume DOUBLE,
                    close_time BIGINT,
                    quote_asset_volume DOUBLE,
                    number_of_trades INT,
                    taker_buy_base_asset_volume DOUBLE,
                    taker_buy_quote_asset_volume DOUBLE,
                    PRIMARY KEY (open_time)
                )
            ''')
            logger.info(f"Table '{table_name}' created.")
            self.db_connection.commit()
            return False
        else:
            logger.info(f"Table '{table_name}' already exists.")
            return True

    def _ensure_tables_exist(self, symbols):
        for symbol in symbols:
            self._create_table_if_not_exists(symbol)

    def process_klines(self, symbol, interval, limit=500):
        cursor = self.db_connection.cursor()

        query = f'SELECT MAX(open_time) FROM kline_data_{symbol.lower()}'
        cursor.execute(query)
        result = cursor.fetchone()
        max_open_time = result[0] if result[0] else 0
        try:
            klines = binance_api.get_klines(symbol, interval, limit)
            for kline in klines:
                open_time = kline[0]

                if int(open_time) > int(max_open_time)-time_intervals[interval]*1000*5:
                    # 删除开盘时间相同的数据
                    delete_query = f'DELETE FROM kline_data_{symbol.lower()} WHERE open_time = %s'
                    cursor.execute(delete_query, (open_time,))

                    row = (
                        open_time,
                        float(kline[1]),
                        float(kline[2]),
                        float(kline[3]),
                        float(kline[4]),
                        float(kline[5]),
                        kline[6],
                        float(kline[7]),
                        kline[8],
                        float(kline[9]),
                        float(kline[10])
                    )
                    insert_query = f'INSERT INTO kline_data_{symbol.lower()} VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
                    cursor.execute(insert_query, row)
            logger.info(
                f"Processed kline for {symbol}...")
        except Exception as e:
            # 回滚事务，以便撤销之前的操作
            self.db_connection.rollback()
            logger.error(
                f'Error occurred during kline processing for {symbol}: {e}')

    def fetch_historical_klines(self, symbol, limit=250):
        cursor = self.db_connection.cursor()

        try:
            query = f'SELECT * FROM kline_data_{symbol.lower()} ORDER BY open_time DESC LIMIT {limit}'
            cursor.execute(query)
            historical_klines = cursor.fetchall()
            historical_klines.reverse()  # Reverse the list to get the oldest data first
            return historical_klines
        except Exception as e:
            logger.error(
                f'Error occurred during fetching historical klines for {symbol}: {e}')
            return []

    def data_preprocess_matrix(self, data):
        data = np.array(data)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[:, 6:])

        input_data = []
        window_size = 23
        for i in range(len(scaled_data) - (window_size)):
            templatebox = []
            for j in range(window_size):
                templatebox.append(
                    TechnicalIndicators.multiply_rows(scaled_data[i + j, :]))
            input_data.append(templatebox)
        input_data = np.array(input_data)
        return input_data

    def process_data(self, data):
        try:
            # 将原始数据转换为DataFrame，指定列名
            data = pd.DataFrame(data, columns=["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume",
                                               "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume"])
            # 计算EMA200指标
            close_prices = data["Close"]
            ema200 = TechnicalIndicators.calculate_ema(
                close_prices, period=200)

            # 计算EMA20指标
            ema20 = TechnicalIndicators.calculate_ma(close_prices, period=20)

            # 计算MACD指标
            macd = TechnicalIndicators.calculate_macd(close_prices)

            # 计算RSI指标
            rsi = TechnicalIndicators.calculate_rsi(close_prices, period=14)

            # 计算MA200指标
            ma200 = TechnicalIndicators.calculate_ma(close_prices, period=200)

            # 计算EMA20和EMA200差值指标
            ema20_ema200_diff = TechnicalIndicators.ema_diff(
                close_prices, 21, 200)

            # 计算ATR指标
            atr = TechnicalIndicators.calculate_atr(
                data["High"], data["Low"], data["Close"], period=14)

            # 计算KDJ指标
            kd, dj, kj = TechnicalIndicators.calculate_kdj(
                data["High"], data["Low"], data["Close"])

            # 获取交易量数据
            volumes = data["Volume"]

            # 计算WMA指标
            wma = TechnicalIndicators.calculate_wma(close_prices, period=20)

            # 计算CCI指标
            cci = TechnicalIndicators.calculate_cci(
                close_prices, data["High"], data["Low"], period=20)

            # 计算MFI指标
            mfi = TechnicalIndicators.calculate_mfi(
                close_prices, volumes, period=14)

            # 将计算结果添加到原始数据中
            data["RSI"] = TechnicalIndicators.calculate_slope(rsi)
            data["EMA200"] = TechnicalIndicators.calculate_slope(ema200)
            data["EMA20"] = TechnicalIndicators.calculate_slope(ema20)
            data["EMA20_EMA200_DIFF"] = TechnicalIndicators.calculate_slope(
                ema20_ema200_diff)
            data["MACD"] = TechnicalIndicators.calculate_slope(macd)
            data["atr"] = TechnicalIndicators.calculate_slope(atr)
            data["CCI"] = TechnicalIndicators.calculate_slope(cci)
            data["kd"] = TechnicalIndicators.calculate_slope(kd)
            data["kj"] = TechnicalIndicators.calculate_slope(kj)
            data["dj"] = TechnicalIndicators.calculate_slope(dj)

            # 去除前200行数据和最后一行数据
            data = data[200:]

            # 去除包含空值的行
            data = data.dropna()

            # 需要去除的列名列表
            columns_to_drop = ["Close time", "Quote asset volume",
                               "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume"]

            # 去除指定的列
            data = data.drop(columns=columns_to_drop)

            # 返回处理后的数据
            return data
        except Exception as e:
            logger.error(e)


class DataProcessingThread(Thread):
    def __init__(self, binance_api):
        super(DataProcessingThread, self).__init__()
        self.binance_api = binance_api
        # Dictionary to store the current direction and quantity for each symbol
        self.directions = {}
        # Dictionary to store the previous predictions for each symbol
        self.previous_predictions = {}

    def run(self):
        symbols = ['SUIUSDT','OPUSDT',"XRPUSDT","EOSUSDT"]
        interval = '5m'
        limit = 500  # Initial limit for the first iteration
        data_processor = BinanceDataProcessor(
            host='localhost', port=3306, user='root', password='root', database='kline', symbols=symbols)
        model = keras.models.load_model("./cnn_model.h5")

        while True:
            try:
                current_server_time = self.binance_api.get_server_time()
                # Round to the start of the current minute
                current_server_minute = (current_server_time // 300000) * 300000
                next_trigger_time = current_server_minute + \
                    300000  # Add 1 minute to the current minute
                time_to_wait = max(0, next_trigger_time -
                                   current_server_time) / 1000

                # Wait until the next trigger time (start of the next one-minute interval)
                time.sleep(time_to_wait)

                for symbol in symbols:
                    data_processor.process_klines(symbol, interval, limit)
                    res = data_processor.data_preprocess_matrix(
                        data_processor.process_data(
                            data_processor.fetch_historical_klines(symbol, 225)
                        )
                    )
                    predictions = model.predict(res)[0]
                    logger.info("Probility of increase is:" +
                                str(predictions[0]))
                    logger.info("Probility of decrease is:" +
                                str(predictions[1]))

                    # Check if the symbol has been processed before
                    if symbol in self.directions:
                        direction, quantity = self.directions[symbol]
                        previous_prediction = self.previous_predictions[symbol]
                    else:
                        direction, quantity = None, 15
                        previous_prediction = None
                    if previous_prediction is None:
                        if predictions[0] > predictions[1]:
                            logger.info(f"{symbol}: up trend")
                            # Place a market order to go long (做多)
                            order_response = self.binance_api.place_order(
                                symbol, 'BUY', 'MARKET', quantity)
                            logger.info("Order response:", order_response)
                            # Update the direction and quantity for the symbol
                            self.directions[symbol] = ('long', 30)
                        elif predictions[0] < predictions[1]:
                            logger.info(f"{symbol}: down trend")
                            # Place a market order to go short (做空)
                            order_response = self.binance_api.place_order(
                                symbol, 'SELL', 'MARKET', quantity)
                            logger.info("Order response:", order_response)
                            # Update the direction and quantity for the symbol
                            self.directions[symbol] = ('short', 30)
                        else:
                            logger.info(f"{symbol}: keep position")
                    # Check if the prediction direction has changed
                    elif previous_prediction is not None and (previous_prediction[0] > previous_prediction[1]) == (predictions[0] > predictions[1]):
                        logger.info(f"{symbol}: keep position")
                    elif previous_prediction is not None and (previous_prediction[0] > previous_prediction[1]) != (predictions[0] > predictions[1]):
                        if predictions[0] > predictions[1]:
                            logger.info(f"{symbol}: up trend")
                            # Place a market order to go long (做多)
                            order_response = self.binance_api.place_order(
                                symbol, 'BUY', 'MARKET', quantity)
                            logger.info("Order response:", order_response)
                            # Update the direction and quantity for the symbol
                            self.directions[symbol] = ('long', 30)
                        elif predictions[0] < predictions[1]:
                            logger.info(f"{symbol}: down trend")
                            # Place a market order to go short (做空)
                            order_response = self.binance_api.place_order(
                                symbol, 'SELL', 'MARKET', quantity)
                            logger.info("Order response:", order_response)
                            # Update the direction and quantity for the symbol
                            self.directions[symbol] = ('short', 30)
                        else:
                            logger.info(f"{symbol}: keep position")
                    else:
                        logger.info(f"{symbol}: keep position")
                    # Update the previous predictions for the current symbol
                    self.previous_predictions[symbol] = predictions

                limit = 10

            except Exception as e:
                logger.error(f'Error occurred during data processing: {e}')
                logger.info('Retrying in 5 seconds...')
                time.sleep(5)


class TimeSyncThread(Thread):
    def __init__(self, binance_api):
        super(TimeSyncThread, self).__init__()
        self.binance_api = binance_api

    def run(self):
        binance_api = self.binance_api
        while True:
            try:
                server_time = binance_api.get_server_time()
                local_time = int(time.time() * 1000)
                time_difference = server_time - local_time
                logger.info(f"Time difference (ms): {time_difference}")
                time.sleep(3600)  # Sync time every hour

            except Exception as e:
                logger.error(
                    f'Error occurred during time synchronization: {e}')
                logger.info('Retrying in 5 seconds...')
                time.sleep(5)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
log_file_handler = logging.FileHandler(
    'data_processor.log')  # Save log messages to a file
log_file_handler.setFormatter(logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(message)s'))
logger.addHandler(log_file_handler)


# Main function
# Main function
if __name__ == "__main__":
    # Create an instance of the BinanceFuturesAPI class
    binance_api = BinanceFuturesAPI(API_KEY, API_SECRET)

    # Start the time synchronization thread with the Binance API instance
    time_sync_thread = TimeSyncThread(binance_api)
    time_sync_thread.start()

    # Start the data processing thread with the Binance API instance
    data_processing_thread = DataProcessingThread(binance_api)
    data_processing_thread.start()
