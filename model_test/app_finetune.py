from flask import Flask, render_template, request
from openai import OpenAI
from openai_api_key import OPENAI_API_KEY
import re

app = Flask(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

def use_fine_model(message):
    completion = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:personal::9op509tW",
        messages=[
            {"role": "system", "content": "학습한 내용을 바탕으로 occupation/gender/experienced 여부에 따라 적절한 질문과 그에 따른 모범답안을 생성해줘 질문:~~, 답변:~~ 형식으로 답해줘"},
            {"role": "user", "content": message}
        ]
    )
    content = completion.choices[0].message.content
    
    # 정규 표현식을 사용하여 질문과 답변 추출
    match = re.search(r'질문:\s*(.*?)\s*/?\s*답변:\s*(.*)', content, re.DOTALL)
    
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        return f"질문: {question}\n\n답변: {answer}"
    else:
        return f"질문과 답변을 찾을 수 없습니다.\n원본 응답: {content}"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        input_text = request.form['input']
        result = use_fine_model(input_text)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)