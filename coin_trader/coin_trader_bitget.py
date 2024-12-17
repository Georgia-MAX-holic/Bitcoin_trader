import os
import json
import requests
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI
import ta
from pybitget.client import Client
import schedule
import time

# Load environment variables
load_dotenv()

# 전역 변수로 현재 포지션 상태 저장
current_position = None  # 'long', 'short', 또는 None

# Helper Functions
def add_indicators(df):
    """Add technical indicators to the DataFrame."""
    if 'close' not in df.columns:
        raise ValueError("Missing required 'close' column in DataFrame")

    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna()

    # Add Bollinger Bands with window=20 for 15m timeframe
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()

    # Add RSI with window=14 for 15m timeframe
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    # Add MACD with standard settings for 15m timeframe
    macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # Add Moving Averages adjusted for 15m timeframe
    df['sma_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['ema_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()

    return df

def get_fear_and_greed_index():
    """Fetch the current Fear and Greed Index."""
    url = "https://api.alternative.me/fng/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('data', [{}])[0]
    else:
        print(f"Failed to fetch Fear and Greed Index. Status code: {response.status_code}")
        return None

def get_latest_eth_news():
    """Fetch the latest Ethereum-related news from the database."""
    conn = sqlite3.connect('Crypto_news.db')
    cursor = conn.cursor()
    query = """
        SELECT title, published_date 
        FROM crypto_news 
        WHERE crypto_type = 'ETH' 
        ORDER BY published_date DESC 
        LIMIT 5
    """
    cursor.execute(query)
    news = cursor.fetchall()
    conn.close()

    return [{"title": item[0], "date": item[1]} for item in news]

def fetch_candles(client, symbol, granularity, max_candles):
    """캔들 데이터를 가져오고 인디케이터를 추가한 DataFrame 반환"""
    # 서버 시간 가져오기 및 정수로 변환
    server_time_response = client.spot_get_server_time()
    current_time = int(server_time_response['data'])

    # 타임 범위 계산
    granularity_to_seconds = {
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1H": 3600,
        "4H": 14400,
        "1D": 86400,
        "1W": 604800,
        "1M": 2592000
    }

    if granularity not in granularity_to_seconds:
        raise ValueError(f"Invalid granularity: {granularity}")

    time_interval = max_candles * granularity_to_seconds[granularity] * 1000  # 밀리초 단위
    start_time = current_time - time_interval

    # 캔들 데이터 가져오기
    candles = client.mix_get_candles(
        symbol=symbol,
        granularity=granularity,  # 수정: 문자열 그대로 사용
        startTime=start_time,
        endTime=current_time
    )

    # 캔들 데이터 처리
    if candles:
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "extra"])
        df = df.drop(columns=["extra"], errors="ignore")
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return add_indicators(df)
    else:
        raise ValueError("캔들 데이터가 반환되지 않았습니다.")

def ai_trading():
    """Main function to perform AI-assisted trading."""
    global current_position  # 전역 변수 사용 선언

    # Initialize API clients
    client = Client(
        os.getenv("BITGET_API_KEY"), 
        os.getenv("BITGET_API_SECRET"), 
        os.getenv("BITGET_PASSPHRASE")
    )

    try:
        # Fetch market data with 15m as main and 30m as auxiliary
        df_15m = fetch_candles(client, "ETHUSDT_UMCBL", granularity="15m", max_candles=200)
        df_30m = fetch_candles(client, "ETHUSDT_UMCBL", granularity="30m", max_candles=200)
    except Exception as e:
        print(f"캔들 데이터 가져오기 실패: {e}")
        return

    # Convert timestamp to string for JSON serialization
    df_15m['timestamp'] = df_15m['timestamp'].astype(str)
    df_30m['timestamp'] = df_30m['timestamp'].astype(str)

    # Fetch additional data
    fear_greed_index = get_fear_and_greed_index()
    news_headlines = get_latest_eth_news()
    account_info = client.mix_get_accounts(productType="umcbl")
    my_USDT = float(account_info['data'][0]['available']) if account_info['data'] else 0

    # AI Decision
    openai_client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """
                        You are an expert in ETH futures trading. Analyze the provided data, 
                        including technical indicators, market data, and the Fear and Greed Index. 
                        Provide a JSON response indicating whether to open a long position, 
                        open a short position, or hold. 
                        Consider that the leverage is 20x.
                        Transactions are carried out in fifteen-minute increments.
                        
                        Please answer in Korean.
                        Response in JSON format.
                        Response Example:
                        {"decision": "open_long", "reason": "some technical, fundamental, and sentiment-based reason"}
                        {"decision": "open_short", "reason": "some technical, fundamental, and sentiment-based reason"}
                        {"decision": "hold", "reason": "some technical, fundamental, and sentiment-based reason"}
                    """
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "investment_status": account_info,
                        "ohlcv_15m": df_15m.to_dict(orient='records'),
                        "ohlcv_30m": df_30m.to_dict(orient='records'),
                        "fear_greed_index": fear_greed_index,
                        "news": news_headlines,
                        "wallet_balance": my_USDT
                    })
                }
            ],
            response_format={"type": "json_object"},
            timeout=30
        )
    except Exception as e:
        print(f"AI 요청 실패: {e}")
        return

    try:
        result = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print("AI 응답을 JSON으로 파싱하는 데 실패했습니다.")
        return

    print("### AI Decision: ", result.get("decision", "UNKNOWN").upper(), "###")
    print(f"### Reason: {result.get('reason', 'No reason provided')} ###")

    # 레버리지 조정
    try:
        client.mix_adjust_leverage(symbol="ETHUSDT_UMCBL", marginCoin="USDT", leverage=20)
    except Exception as e:
        print(f"레버리지 조정 실패: {e}")

 

    trade_size = 0.06

    if result.get("decision") == "open_long" and my_USDT > 10:
        if current_position == "short":
            # 전량 매도 (숏 청산)
            try:
                close_short_response = client.mix_place_order(
                    symbol="ETHUSDT_UMCBL", 
                    side="close_short", 
                    marginCoin="USDT", 
                    size=trade_size, 
                    orderType="market"
                )
                print(close_short_response)
                print("Closed short position")
            except Exception as e:
                print(f"숏 포지션 청산 실패: {e}")
        # 롱 포지션 열기
        try: 
            open_long_response = client.mix_place_order(
                symbol="ETHUSDT_UMCBL", 
                side="open_long", 
                marginCoin="USDT", 
                size=trade_size, 
                orderType="market"
            )
            print(open_long_response)
            current_position = "long"
            print("Opened long position")
        except Exception as e:
            print(f"롱 포지션 열기 실패: {e}")

    elif result.get("decision") == "open_short" and my_USDT > 10:
        if current_position == "long":
            # 전량 매도 (롱 청산)
            try:
                close_long_response = client.mix_place_order(
                    symbol="ETHUSDT_UMCBL", 
                    side="close_long", 
                    marginCoin="USDT", 
                    size=trade_size, 
                    orderType="market"
                )
                print(close_long_response)
                print("Closed long position")
            except Exception as e:
                print(f"롱 포지션 청산 실패: {e}")
        # 숏 포지션 열기
        try:
            open_short_response = client.mix_place_order(
                symbol="ETHUSDT_UMCBL", 
                side="open_short", 
                marginCoin="USDT", 
                size=trade_size, 
                orderType="market"
            )
            print(open_short_response)
            current_position = "short"
            print("Opened short position")
        except Exception as e:
            print(f"숏 포지션 열기 실패: {e}")

    elif result.get("decision") == "hold":
        print("### Hold Position ###")
        # 포지션이 없는 상태에서 "hold" 시그널이 나온 경우
        if current_position is None:
            print("### No position to hold ###")

    # 포지션 상태 출력 (디버깅용)
    print(f"현재 포지션 상태: {current_position}")

# Run the trading bot

schedule.every().hour.at(":00").do(ai_trading)
schedule.every().hour.at(":15").do(ai_trading)
schedule.every().hour.at(":30").do(ai_trading)
schedule.every().hour.at(":45").do(ai_trading)

print("트레이딩 봇이 시작되었습니다. 15분마다 실행됩니다.")
while True:
    schedule.run_pending()
    time.sleep(1)  # CPU 부하를 방지하기 위해 1초 대기
#ai_trading()