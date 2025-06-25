document.getElementById("generateBtn").addEventListener("click", function () {
    const question = document.getElementById("questionInput").value.trim();

    if (!question) {
        alert("질문을 입력해주세요!");
        return;
    }

    document.getElementById("answerText").textContent = "답변을 생성 중입니다...";

    fetch("http://localhost:7001/generate", { // FastAPI 서버 URL과 포트 확인
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: question })
    })
    .then(response => {
        if (!response.ok) {
            // 서버에서 보낸 에러 메시지를 파싱하여 사용자에게 보여줍니다.
            return response.json().then(errorData => {
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        document.getElementById("answerText").textContent = data.answer;
    })
    .catch(error => {
        document.getElementById("answerText").textContent = "⚠️ 서버 오류 발생! " + error.message;
        console.error(error);
    });
});