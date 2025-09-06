import streamlit as st
import pandas as pd
import json
import time
import os
from typing import Dict, List
from openai import OpenAI

class ExplicitFormEvaluator:
    
    def __init__(self):
        # OpenAI API 키는 Streamlit secrets에서 가져오거나 환경변수 사용
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("OpenAI API 키가 설정되지 않았습니다. Streamlit secrets 또는 환경변수에 설정해주세요.")
            st.stop()
        
        self.client = OpenAI(api_key=api_key)
        self.rubric = {
            5: {
                "score": 5,
                "description": "시각적 형태 완벽 보존",
                "criteria": [
                    "번안문의 전체적인 시각적 배치가 원문과 동일",
                    "원문에 존재하는 모든 형태적 구분이 그대로 유지",
                    "텍스트 덩어리의 크기와 배열이 원문과 일치",
                    "눈으로 봤을 때 원문과 구별하기 어려운 수준"
                ]
            },
            4: {
                "score": 4,
                "description": "시각적 형태 거의 보존",
                "criteria": [
                    "원문의 기본적인 시각적 구조가 명확히 유지",
                    "미세한 형태 조정은 있으나 전체 외형 보존",
                    "원문의 시각적 의도가 번안문에서도 명확히 인식",
                    "장르적 특성이 시각적으로 그대로 유지"
                ]
            },
            3: {
                "score": 3,
                "description": "시각적 형태 상당 변경",
                "criteria": [
                    "원문과 명확히 다른 시각적 구조로 재구성",
                    "텍스트 덩어리의 개수나 크기가 크게 변화",
                    "원문에 없던 구분 요소 추가 또는 기존 요소 제거",
                    "전체적 외형이 원문과 구별되는 수준"
                ]
            },
            2: {
                "score": 2,
                "description": "시각적 형태 대부분 손실",
                "criteria": [
                    "원문의 시각적 특성이 대부분 사라짐",
                    "텍스트가 소수의 큰 덩어리로 압축 또는 과도 분할",
                    "장르적 시각적 특성이 크게 훼손",
                    "원문의 시각적 의도 파악이 어려운 수준"
                ]
            },
            1: {
                "score": 1,
                "description": "시각적 형태 완전 붕괴",
                "criteria": [
                    "모든 내용이 하나 또는 소수의 연속 텍스트로 변환",
                    "원문의 모든 시각적 구분이 완전 제거",
                    "장르를 구별할 수 없을 정도로 형태 손실",
                    "원문과 완전히 다른 시각적 장르로 변환"
                ]
            }
        }        
        self.prompt_template = self._create_evaluation_prompt()
    
    def _create_evaluation_prompt(self) -> str:
        """SemScore G-Eval 기반 평가 프롬프트 생성"""
        rubric_text = "\n".join([
            f"**{score}점 - {data['description']}**\n" + 
            "\n".join([f"  - {criterion}" for criterion in data['criteria']]) + "\n"
            for score, data in sorted(self.rubric.items(), reverse=True)
        ])
        
        return f"""# 느린학습자용 번안문 Explicit Form 평가

## 평가 목적
느린학습자를 위한 쉬운말쉬운글 번안기의 품질 향상을 위해, 번안문이 원문의 시각적 형태(Explicit Form)를 얼마나 잘 보존하는지 평가합니다.

## Explicit Form 정의 (시각적 외형 전용)
Explicit Form은 텍스트의 눈으로 보이는 시각적 형태를 의미합니다:
- 문단의 개수와 배치
- 텍스트 덩어리의 크기와 분할
- 시각적 구분 요소 (번호, 기호, 들여쓰기 등)
- 전체적인 레이아웃과 외형
- 장르별 시각적 특성

## 평가 기준 (루브릭)
{rubric_text}

## 평가 지침 (시각적 형태만 평가)
1. 원문과 번안문을 눈으로 비교하여 시각적 유사성만 평가
2. 내용이나 논리는 무시하고 순수 외형만 판단
3. 번안문이 원문과 시각적으로 동일하면 5점
4. 시각적 형태가 크게 바뀌면 낮은 점수
5. 장르를 불문하고 시각적 변화 정도만 측정

## 평가 절차
다음 원문과 번안문을 비교하여 시각적 형태만 평가하세요:

**원문:**
{{original_text}}

**번안문:**
{{translated_text}}

## 응답 형식
반드시 다음 JSON 형식으로 응답하세요:

{{"score": [1-5 점수], "reasoning": "시각적 형태 변화에 대한 구체적 관찰"}}
"""

    def evaluate_pair(self, original: str, translated: str) -> Dict:
        """단일 원문-번안문 쌍에 대한 explicit form 평가"""
        
        prompt = self.prompt_template.replace('{original_text}', original).replace('{translated_text}', translated)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": """당신은 느린학습자를 위한 쉬운글 텍스트 번안 품질 평가 전문가입니다. 

주요 역할:
- 원문과 번안문의 구조적 형태(Explicit Form) 보존도를 정확히 측정
- 제목, 섹션, 문단, 리스트 등 시각적 구조 요소를 체계적으로 분석
- 1-5점 척도로 객관적이고 일관된 평가 수행

평가 원칙:
- 구조가 완전히 동일하면 5점, 완전히 무너지면 1점
- 느린학습자를 위한 합리적 구조 개선은 긍정적 평가
- 눈으로 명확히 구분되는 구조적 차이에 집중

반드시 JSON 형식으로 정확한 점수와 구체적 근거를 제시하세요."""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # JSON 파싱 시도
            try:
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                json_text = result_text[json_start:json_end]
                
                result = json.loads(json_text)
                
                # 스코어 유효성 검사
                if not (1 <= result.get('score', 0) <= 5):
                    raise ValueError("Invalid score range")
                    
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                # 백업: 텍스트에서 점수 추출
                score = self._extract_score_from_text(result_text)
                return {
                    "score": score,
                    "reasoning": result_text
                }
                
        except Exception as e:
            return {
                "score": 0,
                "reasoning": f"평가 중 오류 발생: {str(e)}"
            }    

    def _extract_score_from_text(self, text: str) -> int:
        """텍스트에서 점수 추출 (백업용)"""
        import re
        
        patterns = [
            r'score["\s]*:\s*(\d)',
            r'점수[:\s]*(\d)',
            r'(\d)점',
            r'Score:\s*(\d)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if 1 <= score <= 5:
                    return score
        
        return 3  # 기본값


def main():
    st.set_page_config(
        page_title="Explicit Form Evaluator",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Explicit Form Evaluator")
    st.markdown("**느린학습자용 번안문의 시각적 형태 보존도 평가 도구**")
    
    # 사이드바 설정을 맨 위로 이동
    st.sidebar.header("설정")
    
    # 평가 모드 선택
    eval_mode = st.sidebar.radio(
        "평가 모드:",
        ["개별 쌍 평가", "동일 원문 평가"],
        key="eval_mode_radio"  # 키 추가
    )
    
    # 나머지 설정들도 여기서 처리
    if eval_mode == "개별 쌍 평가":
        num_pairs = st.sidebar.selectbox(
            "평가할 쌍의 개수를 선택하세요:",
            options=[1, 2, 3, 4, 5],
            index=2
        )
        num_translations = None
    else:
        num_translations = st.sidebar.selectbox(
            "번안문 개수를 선택하세요:",
            options=[2, 3, 4, 5], 
            index=1
        )
        num_pairs = None
    # 루브릭 표시
    with st.expander("📋 평가 기준 (루브릭)"):
        evaluator = ExplicitFormEvaluator()
        for score, data in sorted(evaluator.rubric.items(), reverse=True):
            st.write(f"**{score}점 - {data['description']}**")
            for criterion in data['criteria']:
                st.write(f"  • {criterion}")
            st.write("")
    
    # 세션 상태 초기화
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # 입력 섹션
    st.header("📝 텍스트 입력")
    
    pairs_data = []
    
    # 각 쌍에 대한 입력 창 생성
    for i in range(num_pairs):
        st.subheader(f"쌍 {i+1}")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**원문**")
            original = st.text_area(
                f"원문을 입력하세요 (쌍 {i+1})",
                height=200,
                key=f"original_{i}",
                label_visibility="collapsed"
            )
        
        with col2:
            st.write("**번안문**")
            translated = st.text_area(
                f"번안문을 입력하세요 (쌍 {i+1})",
                height=200,
                key=f"translated_{i}",
                label_visibility="collapsed"
            )
        
        pairs_data.append((original, translated))
        st.divider()
    
    # 평가 실행 버튼
    if st.button("🔍 평가 시작", type="primary", use_container_width=True):
        if not all(original.strip() and translated.strip() for original, translated in pairs_data):
            st.error("모든 원문과 번안문을 입력해주세요.")
            return
        
        # 결과 초기화
        st.session_state.results = []
        
        # 평가 진행
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        evaluator = ExplicitFormEvaluator()
        
        for i, (original, translated) in enumerate(pairs_data):
            status_text.text(f"쌍 {i+1}/{num_pairs} 평가 중...")
            progress_bar.progress((i) / num_pairs)
            
            result = evaluator.evaluate_pair(original, translated)
            result['pair_number'] = i + 1
            result['original'] = original
            result['translated'] = translated
            
            st.session_state.results.append(result)
            
            # API 호출 제한 고려
            if i < len(pairs_data) - 1:  # 마지막이 아니면 대기
                time.sleep(1)
        
        progress_bar.progress(1.0)
        status_text.text("평가 완료!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
    
    # 결과 표시
    if st.session_state.results:
        st.header("평가 결과")
        
        # 전체 요약
        scores = [r['score'] for r in st.session_state.results if r['score'] > 0]
        if scores:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("평균 점수", f"{sum(scores)/len(scores):.2f}")
            with col2:
                st.metric("최고 점수", max(scores))
            with col3:
                st.metric("최저 점수", min(scores))
        
        st.divider()
        
        # 모드에 따른 결과 표시
        eval_mode = st.session_state.results[0]['eval_mode']
        
        if eval_mode == "개별 쌍 평가":
            # 각 쌍별 결과
            for result in st.session_state.results:
                st.subheader(f"쌍 {result['pair_number']} 결과")
                
                # 점수 표시
                score = result['score']
                if score > 0:
                    score_color = {5: "🟢", 4: "🔵", 3: "🟡", 2: "🟠", 1: "🔴"}
                    st.markdown(f"### {score_color.get(score, '⚪')} **{score}점** - {evaluator.rubric[score]['description']}")
                else:
                    st.markdown("### ❌ **평가 실패**")
                
                # 이유 표시
                st.write("**평가 근거:**")
                st.write(result['reasoning'])
                
                # 원문/번안문 미리보기
                with st.expander(f"쌍 {result['pair_number']} 텍스트 보기"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**원문:**")
                        st.text(result['original'][:300] + "..." if len(result['original']) > 300 else result['original'])
                    with col2:
                        st.write("**번안문:**")
                        st.text(result['translated'][:300] + "..." if len(result['translated']) > 300 else result['translated'])
                
                st.divider()
        
        else:  # 동일 원문 평가 모드
            # 공통 원문 표시
            st.subheader("공통 원문")
            with st.expander("원문 보기"):
                st.text(st.session_state.results[0]['original'])
            
            st.divider()
            
            # 각 번안문별 결과
            for result in st.session_state.results:
                st.subheader(f"번안문 {result['translation_number']} 결과")
                
                # 점수 표시
                score = result['score']
                if score > 0:
                    score_color = {5: "🟢", 4: "🔵", 3: "🟡", 2: "🟠", 1: "🔴"}
                    st.markdown(f"### {score_color.get(score, '⚪')} **{score}점** - {evaluator.rubric[score]['description']}")
                else:
                    st.markdown("### **평가 실패**")
                
                # 이유 표시
                st.write("**평가 근거:**")
                st.write(result['reasoning'])
                
                # 번안문 미리보기
                with st.expander(f"번안문 {result['translation_number']} 전체 보기"):
                    st.text(result['translated'])
                
                st.divider()
        
        # 결과 다운로드
        if st.button("결과 다운로드 (CSV)"):
            # DataFrame 생성
            if eval_mode == "개별 쌍 평가":
                df_results = pd.DataFrame([
                    {
                        '쌍 번호': r['pair_number'],
                        'EXP FORM 점수': r['score'],
                        '이유': r['reasoning'],
                        '원문': r['original'],
                        '번안문': r['translated']
                    }
                    for r in st.session_state.results
                ])
            else:  # 동일 원문 평가
                df_results = pd.DataFrame([
                    {
                        '번안문 번호': r['translation_number'],
                        'EXP FORM 점수': r['score'],
                        '이유': r['reasoning'],
                        '원문': r['original'],
                        '번안문': r['translated']
                    }
                    for r in st.session_state.results
                ])
            
            # CSV 다운로드
            csv = df_results.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="CSV 파일 다운로드",
                data=csv,
                file_name="explicit_form_results.csv",
                mime="text/csv"
            )
       

if __name__ == "__main__":

    main()



