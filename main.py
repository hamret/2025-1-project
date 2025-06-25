from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, BartForConditionalGeneration
import uvicorn
import torch
import fitz  # PyMuPDF
#
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# KoBART 요약 모델 로드
tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")
model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")

@app.get("/")
def root():
    return {"message": "논문 요약기 서버가 실행 중입니다."}

@app.post("/summarize")
async def summarize(request: Request):
    data = await request.json()
    input_text = data.get("text", "")

    if not input_text.strip():
        return {"summary": "입력 텍스트가 없습니다."}

    return {"summary": summarize_text(input_text)}

@app.post("/summarize_pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"summary": "PDF 파일만 지원됩니다."}

    # PDF에서 텍스트 추출
    text = ""
    pdf = fitz.open(stream=await file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()

    if not text.strip():
        return {"summary": "PDF에서 텍스트를 추출할 수 없습니다."}

    return {"summary": summarize_text(text)}

# 요약 함수 공통 처리
def summarize_text(input_text):
    inputs = tokenizer(
        input_text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=300,  # 더 많은 토큰 허용
        min_length=100,  # 최소 요약 길이 설정 (선택사항)
        num_beams=5,  # 빔 수 증가 → 다양하고 풍부한 출력
        repetition_penalty=1.8,  # 반복 줄임
        length_penalty=1.0,  # 길이 제한 완화
        early_stopping=False  # 너무 일찍 멈추지 않도록
    )

    summary = tokenizer.decode(
        summary_ids.squeeze(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return summary

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
