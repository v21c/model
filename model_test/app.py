from flask import Flask, render_template, request
from openai import OpenAI
from openai_api_key import OPENAI_API_KEY
import re
import pandas as pd

app = Flask(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)
# CSV 파일 읽기
df = pd.read_csv('/Users/kbsoo/coding/codes/python/model_test/all_temp.csv')

def use_prompt(message):
    occupation, gender, experience = message.split('/')
    
    # 데이터 필터링
    filtered_df = df[(df['occupation'] == occupation) & 
                     (df['gender'] == gender) & 
                     (df['experience'] == experience.upper())]
    
    # 기존 질문과 답변 추출
    existing_qa = filtered_df[['question_text', 'answer_text']].values.tolist()
    
    # 기존 Q&A를 바탕으로 프롬프트 생성
    qa_prompt = f"""
    Based on the following questions and answers for {occupation} professionals who are {gender} and {experience}:

    {existing_qa}

    Generate a new
    The question and answer should not directly repeat any of the existing ones, but should be inspired by the themes and insights present in them.
    
    Format the output as:
    질문: [Your generated question]
    답변: [Your generated answer]
    """
    
    # OpenAI API 호출
    completion = client.chat.completions.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": "You are a career advisor specializing in tech industry trends and challenges."},
            {"role": "user", "content": qa_prompt}
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
        # result = use_fine_model(input_text)
        result = use_prompt(input_text)
    return render_template('/templates/index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)