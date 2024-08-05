import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import openai
import requests

# CSV 파일 URL
csv_url = "https://huggingface.co/datasets/kbsoo/InTheview/resolve/main/scored_answers.csv"

# CSV 파일 다운로드 및 데이터 로드
response = requests.get(csv_url)
with open("scored_answers.csv", "wb") as f:
    f.write(response.content)
data = pd.read_csv("scored_answers.csv")

# 데이터 확인
print(data.head())
print(data.columns)

# OpenAI API 키

# 데이터 전처리
X = data[['question_text', 'answer_text']]
y = data['score']

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X['question_text'] + " " + X['answer_text'])

# 머신러닝 모델 훈련
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# GPT-4를 사용하여 상세 점수 계산
def get_detailed_score(question, answer):
    prompt = f"""
    다음 질문에 대한 답변을 아래 기준에 따라 평가하세요:
    1. 관련성 (0-25): 답변이 질문을 얼마나 잘 다루고 있는가?
    2. 명확성 (0-25): 답변이 얼마나 명확하고 이해하기 쉬운가?
    3. 깊이 (0-25): 답변이 얼마나 깊이 있고 포괄적인가?
    4. 실용성 (0-25): 답변이 실제 지식이나 경험을 얼마나 잘 보여주는가?
    질문: {question}
    답변: {answer}
    각 기준에 대한 점수만 제공하세요. 총점은 자동으로 계산됩니다.
    형식:
    관련성: [점수]
    명확성: [점수]
    깊이: [점수]
    실용성: [점수]
    """
    completion = openai.ChatCompletion.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": "당신은 전문 면접 평가자입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    content = completion.choices[0].message['content']
    
    # 점수 추출 및 합산
    scores = re.findall(r'(\w+): (\d+)', content)
    total_score = sum(int(score) for _, score in scores)
    
    return total_score  # 0-100 사이의 총점 반환

# 최종 점수 계산 함수
def get_final_score(question, answer):
    detailed_score = get_detailed_score(question, answer)
    
    # 머신러닝 모델을 사용하여 점수 예측
    features = vectorizer.transform([question + " " + answer])
    predicted_score = model.predict(features)[0]
    
    # 두 점수를 결합 (각각 50%)
    final_score = (0.5 * detailed_score) + (0.5 * predicted_score)
    
    return final_score

while True:
    question = input("Q : ")
    answer = input("A : ")
    final_score = get_final_score(question, answer)

    print("final_score : ", final_score)
