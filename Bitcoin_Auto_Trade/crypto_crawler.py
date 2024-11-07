
import requests
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime

class CryptoNewsCrawler:
    def __init__(self, db_path="crypto_news.db"):
        """
        초기화 함수
        
        Args:
            db_path (str): SQLite 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.initialize_database()
    
    def initialize_database(self):
        """데이터베이스 및 테이블 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crypto_news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crypto_type TEXT NOT NULL,
                    title TEXT NOT NULL UNIQUE,
                    link TEXT NOT NULL,
                    description TEXT,
                    published_date DATETIME NOT NULL,
                    crawled_date DATETIME NOT NULL
                )
            ''')
            conn.commit()

    def crawl_crypto_news(self):
        """뉴스 크롤링 및 데이터베이스 저장"""
        base_url = "https://news.google.com/rss/search?q={}&hl=ko&gl=KR&ceid=KR:ko"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        # 검색 키워드
        searches = {
#            'Bitcoin': ['비트코인', 'bitcoin', 'btc', 'BTC'],
            'DOGE': ['도지코인', 'DOGE', 'doge', '도지']
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for crypto_type, keywords in searches.items():
                    for keyword in keywords:
                        url = base_url.format(keyword)
                        response = requests.get(url, headers=headers)
                        response.raise_for_status()
                        
                        soup = BeautifulSoup(response.content, 'xml')
                        news_items = soup.find_all('item')
                        
                        for item in news_items[:5]:  # 최신 뉴스 5개만 처리
                            title = item.title.text
                            link = item.link.text
                            description = item.description.text if item.description else ""
                            pub_date = datetime.strptime(
                                item.pubDate.text, 
                                '%a, %d %b %Y %H:%M:%S %Z'
                            )
                            
                            try:
                                cursor.execute('''
                                    INSERT INTO crypto_news (
                                        crypto_type, title, link, description,
                                        published_date, crawled_date
                                    ) VALUES (?, ?, ?, ?, ?, ?)
                                ''', (
                                    crypto_type, title, link, description,
                                    pub_date, datetime.now()
                                ))
                            except sqlite3.IntegrityError:
                                # 중복된 뉴스는 건너뛰기
                                continue
                            
                        # 각 유형의 최신 5개만 남기고 이전 뉴스 삭제
                        cursor.execute('''
                            DELETE FROM crypto_news 
                            WHERE id NOT IN (
                                SELECT id FROM crypto_news
                                WHERE crypto_type = ?
                                ORDER BY published_date DESC
                                LIMIT 5
                            ) AND crypto_type = ?
                        ''', (crypto_type, crypto_type))
                
                conn.commit()
            return True
            
        except requests.RequestException as e:
            print(f"크롤링 중 오류 발생: {e}")
            return False
    
    def get_latest_news(self, crypto_type=None, limit=5):
        """
        최신 뉴스 조회
        
        Args:
            crypto_type (str): 'DOGE' 또는 'Bitcoin' (None이면 모두 조회)
            limit (int): 조회할 뉴스 개수
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if crypto_type:
                cursor.execute('''
                    SELECT crypto_type, title, link, published_date
                    FROM crypto_news
                    WHERE crypto_type = ?
                    ORDER BY published_date DESC
                    LIMIT ?
                ''', (crypto_type, limit))
            else:
                cursor.execute('''
                    SELECT crypto_type, title, link, published_date
                    FROM crypto_news
                    ORDER BY published_date DESC
                    LIMIT ?
                ''', (limit,))
            
            return cursor.fetchall()

def main():
    # 크롤러 인스턴스 생성
    crawler = CryptoNewsCrawler()
    
    # 뉴스 크롤링
    print("뉴스 크롤링 시작...")
    crawler.crawl_crypto_news()
    
    # 최신 뉴스 확인
    print("\n=== Bitcoin 최신 뉴스 ===")
    for news in crawler.get_latest_news('Bitcoin', 5):
        print(f"제목: {news[1]}")
        print(f"링크: {news[2]}")
        print(f"작성일: {news[3]}")
        print("-" * 80)
    
    print("\n=== DOGE 최신 뉴스 ===")
    for news in crawler.get_latest_news('DOGE', 5):
        print(f"제목: {news[1]}")
        print(f"링크: {news[2]}")
        print(f"작성일: {news[3]}")
        print("-" * 80)

if __name__ == "__main__":
    main()