from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, BartForConditionalGeneration
import uvicorn
import torch

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
tokenizer = AutoTokenizer.from_pretrained("kobart-summarization")
model = BartForConditionalGeneration.from_pretrained("kobart-summarization")

@app.get("/")
def root():
    return {"message": "논문 요약기 서버가 실행 중입니다."}

@app.post("/summarize")
async def summarize(request: Request):
    data = await request.json()
    input_text = data.get("text", "")

    if not input_text.strip():
        return {"summary": "입력 텍스트가 없습니다."}

    # 입력 토큰화
    inputs = tokenizer(
        input_text,
        max_length=2048,
        truncation=True,
        return_tensors="pt"
    )

    # 요약 생성
    summary_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=150,
        num_beams=4,
        repetition_penalty=2.0,
        early_stopping=True
    )

    # 요약 디코딩
    summary = tokenizer.decode(
        summary_ids.squeeze(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return {"summary": summary}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)