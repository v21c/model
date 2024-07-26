from flask import Flask, render_template, request, session, redirect, url_for
from openai import OpenAI
from openai_api_key import OPENAI_API_KEY
import re
import pandas as pd

app = Flask(__name__)
app.secret_key = 'secretkey123'  # 세션을 위한 비밀 키 설정

client = OpenAI(api_key=OPENAI_API_KEY)
df = pd.read_csv('/Users/kbsoo/coding/codes/python/model_test2/all_temp.csv')

def get_question(message):
    occupation, gender, experience = message.split('/')
    
    filtered_df = df[(df['occupation'] == occupation) & 
                     (df['gender'] == gender) & 
                     (df['experience'] == experience.upper())]
    
    existing_qa = filtered_df[['question_text', 'answer_text']].values.tolist()
    
    qa_prompt = f"""
    Based on the following questions and answers for {occupation} professionals who are {gender} and {experience}:

    {existing_qa}

    Generate a new, relevant question about career development, industry trends, or challenges faced by this group. 
    The question should not directly repeat any of the existing ones, but should be inspired by the themes and insights present in them.
    And The non-first question also allows you to ask questions related to the content and answer of the previous question.
    
    Format the output as:
    질문: [Your generated question]
    
    The question should be in Korean.
    """
    
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": "You are a career advisor specializing in tech industry trends and challenges."},
            {"role": "user", "content": qa_prompt}
        ]
    )
    content = completion.choices[0].message.content

    match = re.search(r'질문:\s*(.*)', content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return "질문을 생성할 수 없습니다."

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    if request.method == 'POST':
        if 'start_chat' in request.form:
            input_text = request.form['input']
            session['initial_input'] = input_text  # 초기 입력값 저장
            question = get_question(input_text)
            session['chat_history'] = [{'type': 'question', 'content': question}]
        elif 'user_answer' in request.form:
            user_answer = request.form['user_answer']
            session['chat_history'].append({'type': 'answer', 'content': user_answer})
            if 'initial_input' in session:
                question = get_question(session['initial_input'])
            else:
                question = "죄송합니다. 새로운 질문을 생성할 수 없습니다. 채팅을 다시 시작해주세요."
            session['chat_history'].append({'type': 'question', 'content': question})
        elif 'restart_chat' in request.form:
            session.clear()
            return redirect(url_for('index'))
        
        session.modified = True
    
    return render_template('index.html', chat_history=session.get('chat_history', []))

if __name__ == '__main__':
    app.run(debug=True)
