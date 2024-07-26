from flask import Flask, render_template, request, session, redirect, url_for
from openai import OpenAI
from openai_api_key import OPENAI_API_KEY, URI
import re
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId

app = Flask(__name__)
app.secret_key = 'secretkey123'  # 세션을 위한 비밀 키 설정

uri=URI
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['intheview']
userId = ObjectId('66a3324a1cc4ca962e4c9afe')

client = OpenAI(api_key=OPENAI_API_KEY)
df = pd.read_csv('/Users/kbsoo/coding/codes/python/model_test2/all_temp.csv')

def get_question(message, chat_history):
    occupation, gender, experience = message.split('/')
    
    filtered_df = df[(df['occupation'] == occupation) & 
                     (df['gender'] == gender) & 
                     (df['experience'] == experience.upper())]
    
    existing_qa = filtered_df[['question_text', 'answer_text']].values.tolist()
    
    chat_history_text = "\n".join([f"질문: {item['content']}" if item['type'] == 'question' else f"답변: {item['content']}" for item in chat_history])
    
    qa_prompt = f"""
    Based on the following questions and answers for {occupation} professionals who are {gender} and {experience}:

    {existing_qa}

    Additionally, consider the previous conversation:
    {chat_history_text}

    Generate a new, relevant question about career development, industry trends, or challenges faced by this group. 
    The question should be inspired by the themes and insights present in the previous questions and answers, and should follow naturally from the previous answer.
    
    For example, if the previous question was about self-development and the answer was about reading books, the next question could be about what books they have recently read.
    
    Format the output as:
    질문: [Your generated question]
    
    The question should be in Korean.
    """
    
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": "You are an interviewer who refers to the given files and asks questions according to job occupation/gender/experience"},
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

    session_collection = db['sessions']
    user_collection = db['users']
    
    if request.method == 'POST':
        if 'start_chat' in request.form:
            input_text = request.form['input']
            session['initial_input'] = input_text  # 초기 입력값 저장
            
            # 새로운 세션 생성
            new_session = {"chat_history": []}
            result = session_collection.insert_one(new_session)
            session_id = result.inserted_id
            
            # 사용자 세션에 추가
            user_collection.update_one(
                {"_id": userId},
                {"$push": {"sessions": session_id}}
            )
            
            session['session_id'] = str(session_id)
            question = get_question(input_text, session['chat_history'])
            session['chat_history'] = [{'type': 'question', 'content': question}]
            
            # 새로 생성된 질문을 세션 데이터베이스에 추가
            session_collection.update_one(
                {"_id": session_id},
                {"$push": {"chat_history": {"question": question, "answer": ""}}}
            )
            
        elif 'user_answer' in request.form:
            user_answer = request.form['user_answer']
            session['chat_history'].append({'type': 'answer', 'content': user_answer})
            
            session_id = ObjectId(session['session_id'])
            # 마지막 질문 업데이트
            session_collection.update_one(
                {"_id": session_id, "chat_history.answer": ""},
                {"$set": {"chat_history.$.answer": user_answer}}
            )
            
            if 'initial_input' in session:
                question = get_question(session['initial_input'], session['chat_history'])
            else:
                question = "죄송합니다. 새로운 질문을 생성할 수 없습니다. 채팅을 다시 시작해주세요."
            session['chat_history'].append({'type': 'question', 'content': question})
            
            # 새로운 질문 추가
            session_collection.update_one(
                {"_id": session_id},
                {"$push": {"chat_history": {"question": question, "answer": ""}}}
            )
            
        elif 'restart_chat' in request.form:
            session.clear()
            return redirect(url_for('index'))
        
        session.modified = True
    
    return render_template('index.html', chat_history=session.get('chat_history', []))

if __name__ == '__main__':
    app.run(debug=True)
