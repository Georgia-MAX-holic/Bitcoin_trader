import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import json
from openai import OpenAI
import ta
from ta.utils import dropna
import requests
from datetime import datetime
import schedule
import time
import sqlite3
import crypto_crawler
from chart_capture import main as chart_capture_main
import base64
load_dotenv()

def add_indicators(df):
    # 볼린저 밴드: 윈도우 20, 표준편차 2로 설정
    indicator_bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    # RSI: 기간 14로 설정
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # MACD: 긴 주기 26, 짧은 주기 12, 시그널 주기 9
    macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # 이동평균선: EMA(20), SMA(50)로 설정
    df['sma_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['ema_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
    
    return df

def get_fear_and_greed_index():
    url = "https://api.alternative.me/fng/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['data'][0]
    else:
        print(f"Failed to fetch Fear and Greed Index. Status code: {response.status_code}")
        return None

def get_latest_eth_news():
    # SQLite 데이터베이스 연결
    conn = sqlite3.connect('Crypto_news.db')
    cursor = conn.cursor()
    
    # 최신 published_date 기준으로 5개의 뉴스 추출
    cursor.execute("SELECT title, published_date FROM crypto_news WHERE crypto_type = 'ETH' ORDER BY published_date DESC LIMIT 5")
    news = cursor.fetchall()
    
    # 연결 종료
    conn.close()
    
    return [{"title": item[0], "date": item[1]} for item in news]

def ai_trading():
    base64_image = chart_capture_main()
    crypto_crawler.main()
    # Upbit 객체 생성
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    upbit = pyupbit.Upbit(access, secret)

    # 1. 현재 투자 상태 조회
    all_balances = upbit.get_balances()
    filtered_balances = [balance for balance in all_balances if balance['currency'] in ['ETH', 'KRW']]
    
    # 2. 오더북(호가 데이터) 조회
    orderbook = pyupbit.get_orderbook("KRW-ETH")
    
    # 3. 1시간 차트와 4시간 차트 데이터 조회 및 보조지표 추가
    df_hourly = pyupbit.get_ohlcv("KRW-ETH", interval="minute60", count=48)  # 최근 48시간 데이터
    df_hourly = dropna(df_hourly)
    df_hourly = add_indicators(df_hourly)
    
    df_4hour = pyupbit.get_ohlcv("KRW-ETH", interval="minute240", count=30)  # 최근 30개의 4시간 봉 데이터
    df_4hour = dropna(df_4hour)
    df_4hour = add_indicators(df_4hour)

    # 4. 공포 탐욕 지수 가져오기
    fear_greed_index = get_fear_and_greed_index()

    # 5. 최신 뉴스 헤드라인 가져오기
    news_headlines = get_latest_eth_news()

    my_ETH = upbit.get_balance("ETH")
    
    # AI에게 데이터 제공하고 판단 받기
    client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": """You are an expert in ETH investing. Analyze the provided data including technical indicators, market data, and the Fear and Greed Index. Tell me whether to buy, sell, or hold at the moment. Consider the following in your analysis:
        - Technical indicators and market data from both 1-hour and 4-hour charts
        - The Fear and Greed Index and its implications
        - Recent news headlines and their potential impact on ETH price
        - Overall market sentiment
        - Image of current 1-hourly candle chart for pattern analysis

        please answer in korean 
        Response in json format.

        Response Example:
        {"decision": "buy", "reason": "some technical, fundamental, and sentiment-based reason"}
        {"decision": "sell", "reason": "some technical, fundamental, and sentiment-based reason"}
        {"decision": "hold", "reason": "some technical, fundamental, and sentiment-based reason"}"""
        },
        {
        "role": "user",
        "content": f"""Current investment status: {json.dumps(filtered_balances)}
Orderbook: {json.dumps(orderbook)}
Hourly OHLCV with indicators (48 hours): {df_hourly.to_json()}
4-Hour OHLCV with indicators (30 periods): {df_4hour.to_json()}
Fear and Greed Index: {json.dumps(fear_greed_index)}
Recent news headlines: {json.dumps(news_headlines)}
my wallet status: {my_ETH}
chart image: {base64_image}"""
        }
    ],
    response_format={
        "type": "json_object"
    }
    )
    result = response.choices[0].message.content

    # AI의 판단에 따라 실제로 자동매매 진행하기
    result = json.loads(result)

    print("### AI Decision: ", result["decision"].upper(), "###")
    print(f"### Reason: {result['reason']} ###")
    print(datetime.now())
    
    if result["decision"] == "buy":
        my_krw = upbit.get_balance("KRW")
        if my_krw * 0.9995 > 5000:
            print("### Buy Order Executed ###")
            print(upbit.buy_market_order("KRW-ETH", my_krw * 0.9995))
        else:
            print("### Buy Order Failed: Insufficient KRW (less than 5000 KRW) ###")
    elif result["decision"] == "sell":
        current_price = pyupbit.get_orderbook(ticker="KRW-ETH")['orderbook_units'][0]["ask_price"]
        if my_ETH * current_price > 5000:
            print("### Sell Order Executed ###")
            print(upbit.sell_market_order("KRW-ETH", my_ETH))
        else:
            print("### Sell Order Failed: Insufficient ETH (less than 5000 KRW worth) ###")
    elif result["decision"] == "hold":
        print("### Hold Position ###")



schedule.every().hour.at(":00").do(ai_trading)

while True:
    schedule.run_pending()
    time.sleep(1)
