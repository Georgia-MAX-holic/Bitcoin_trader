from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
import time
import base64
import io
import os
from datetime import datetime
import logging

# 로거 설정
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

class UpbitChartCapture:
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.driver.maximize_window()

    def capture_and_encode_screenshot(self):
        try:
            # 스크린샷 캡처
            png = self.driver.get_screenshot_as_png()
            
            # PIL Image로 변환
            img = Image.open(io.BytesIO(png))
            
            # 이미지 리사이즈 (OpenAI API 제한에 맞춤)
            img.thumbnail((700, 700))  # 리사이즈로 이미지 크기 축소
            
            # JPEG 형식으로 저장하여 압축 적용 (품질을 85로 설정하여 용량 줄이기)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"upbit_chart_{current_time}.jpg"  # JPEG 형식 사용
            
            # 현재 스크립트의 경로를 가져옴
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 파일 저장 경로 설정
            file_path = os.path.join(script_dir, filename)
            
            # 이미지 파일로 저장 (JPEG, quality 설정으로 압축)
            img.save(file_path, "JPEG", quality=85)  # 품질을 85로 설정하여 용량 줄이기
            logger.info(f"스크린샷이 저장되었습니다: {file_path}")
            
            # 이미지를 바이트로 변환
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)  # 품질을 85로 설정하여 압축
            buffered.seek(0)
            
            # base64로 인코딩
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return base64_image, file_path
        except Exception as e:
            logger.error(f"스크린샷 캡처 및 인코딩 중 오류 발생: {e}")
            return None, None

    def capture_upbit_chart(self):
        try:
            # 업비트 페이지 로드
            self.driver.get('https://upbit.com/exchange?code=CRIX.UPBIT.KRW-ETH')
            time.sleep(4)  # 페이지 로딩 시간 확보

            # 차트 옵션 메뉴 클릭
            self.driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[3]/div/section[1]/article[1]/div/span[2]/div/div/div[1]/div[1]/div/cq-menu[1]').click()
            time.sleep(1)  # 메뉴가 열릴 시간을 확보

            # 차트 옵션 선택
            self.driver.find_element(By.XPATH,'/html/body/div[1]/div[2]/div[3]/div/section[1]/article[1]/div/span[2]/div/div/div[1]/div[1]/div/cq-menu[1]/cq-menu-dropdown/cq-item[8]').click()
            time.sleep(1)  # 옵션 반영 시간 확보

            # 스크린샷 캡처 및 Base64 인코딩
            base64_image = self.capture_and_encode_screenshot()

            if base64_image:
                return base64_image  # 앞부분만 출력
            else:
                logger.error("스크린샷 캡처 및 인코딩 실패")
        except Exception as e:
            logger.error(f"차트 캡처 중 오류 발생: {e}")
        finally:
            # 드라이버 종료
            self.driver.quit()

def main():
    # UpbitChartCapture 클래스의 인스턴스 생성
    capture = UpbitChartCapture()
    base64_image = capture.capture_upbit_chart()
    return base64_image

if __name__ == "__main__":
    main()
