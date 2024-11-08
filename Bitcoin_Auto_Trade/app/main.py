from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.requests import Request
from fastapi import WebSocket

# FastAPI 앱 생성
app = FastAPI()

# 템플릿 디렉토리 설정
templates = Jinja2Templates(directory="app/templates")

# 정적 파일 디렉토리 설정 (CSS, JS 파일 등)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "My FastAPI Web Page"})

# DashBoard 페이지 라우트
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "title": "Dashboard"})

# News 페이지 라우트
@app.get("/news", response_class=HTMLResponse)
async def news(request: Request):
    return templates.TemplateResponse("news.html", {"request": request, "title": "News"})

# Log 페이지 라우트
@app.get("/log", response_class=HTMLResponse)
async def log(request: Request):
    return templates.TemplateResponse("log.html", {"request": request, "title": "Log"})

