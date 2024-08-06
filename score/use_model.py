import re
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
import os
import json
import joblib
import time
from datetime import datetime
import pytz
import glob


def print_step(step):
    kst = pytz.timezone('Asia/Seoul')
    current_time = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] {step} 시작")


# 전체 프로세스 시작 시간
overall_start_time = time.time()

# OpenAI API 키 설정
openai.api_key = ''

# 현재 디렉토리의 .joblib 파일들 출력
print("현재 디렉토리의 .joblib 파일들:")
joblib_files = glob.glob("*.joblib")
for file in joblib_files:
    print(file)

# 모델 및 벡터라이저 로드
print_step("모델 및 벡터라이저 로드")
start_time = time.time()
model_file = "scored_model_10.joblib"

if os.path.exists(model_file):
    pipeline = joblib.load(model_file)
    model = pipeline.named_steps['randomforestregressor']
    vectorizer = pipeline.named_steps['tfidfvectorizer']
    print(f"모델 로드 완료. 경과 시간: {time.time() - start_time:.2f}초")
else:
    print(f"오류: {model_file}을 찾을 수 없습니다.")
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print("파일이 올바른 위치에 있는지 확인하세요.")
    exit(1)

# GPT-4 결과 캐싱
cache_file = "score_cache.json"
try:
    with open(cache_file, "r") as f:
        score_cache = json.load(f)
except FileNotFoundError:
    score_cache = {}


# GPT-4를 사용하여 상세 점수 계산
def get_detailed_score(question, answer):
    cache_key = f"{question}|{answer}"
    if cache_key in score_cache:
        return score_cache[cache_key]

    # print_step("GPT-4o API 호출")
    start_time = time.time()
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
    completion = openai.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": "당신은 전문 면접 평가자입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    content = completion.choices[0].message.content

    # 점수 추출 및 합산
    scores = re.findall(r'(\w+): (\d+)', content)
    total_score = sum(int(score) for _, score in scores)

    score_cache[cache_key] = total_score
    with open(cache_file, "w") as f:
        json.dump(score_cache, f)

    # print(f"GPT-4 API 호출 완료. 경과 시간: {time.time() - start_time:.2f}초")
    return total_score  # 0-100 사이의 총점 반환


# 최종 점수 계산 함수
# 최종 점수 계산 함수
def get_final_score(question, answer):
    # print_step("상세 점수 계산")
    start_time = time.time()
    detailed_score = get_detailed_score(question, answer)
    # print(f"상세 점수 계산 완료. 경과 시간: {time.time() - start_time:.2f}초")

    # print_step("머신러닝 모델 예측")
    start_time = time.time()
    features = vectorizer.transform([question + " " + answer])
    predicted_score = model.predict(features)[0]
    # print(f"머신러닝 모델 예측 완료. 경과 시간: {time.time() - start_time:.2f}초")

    # 두 점수를 결합 (각각 50%)
    final_score = (0.5 * detailed_score) + (0.5 * predicted_score)

    return detailed_score, predicted_score


print(f"\n모든 준비가 완료되었습니다. 총 경과 시간: {time.time() - overall_start_time:.2f}초")
print("질문과 답변을 입력해주세요.")

while True:
    question = input("\nQ : ")
    answer = input("A : ")

    # start_time = time.time()
    d_score, p_score = get_final_score(question, answer)
    f_score = (0.5 * d_score) + (0.5 * p_score)
    # end_time = time.time()

    print(f"\nML_score : {p_score}")
    print(f"\nGPT_score : {d_score}")
    print(f"\nfinal_score : {f_score}")
    # print(f"처리 시간: {end_time - start_time:.2f}초")
