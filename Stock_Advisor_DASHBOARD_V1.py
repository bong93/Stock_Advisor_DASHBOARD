import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import requests
import urllib.parse
import ta
from sklearn.preprocessing import RobustScaler
from pykrx import stock
import joblib
import lightgbm as lgb
import time
import os
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import networkx as nx
import pytz
from groq import Groq

# 🌟 최상단 배치 (Streamlit 설정)
st.set_page_config(page_title="AI Quant Master", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# 🌟 [신규] 모바일 반응형 UI 강제 최적화 (CSS Injection)
st.markdown("""
<style>
    /* 1. 모바일 환경에서 탭(Tab) 버튼이 한 줄로 예쁘게 나오도록 수정 */
    div[data-baseweb="tab-list"] {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
    }
    
    /* 2. 스마트폰 세로 모드일 때 메트릭(숫자) 크기 자동 조절 */
    @media (max-width: 600px) {
        h1 { font-size: 24px !important; }
        h2 { font-size: 20px !important; }
        h3 { font-size: 18px !important; }
        div[data-testid="stMetricValue"] { font-size: 22px !important; }
        
        /* 3. 모바일에서 데이터프레임 폰트 및 여백 최소화 (렌더링 속도 개선) */
        .dataframe { font-size: 12px !important; }
        .stDataFrame { padding: 0 !important; }
        
        /* 4. 모바일에서 버튼을 화면 꽉 차게 만들어 터치하기 쉽게 변경 */
        .stButton>button { width: 100% !important; height: 50px !important; font-size: 16px !important; font-weight: bold !important; }
    }
</style>
""", unsafe_allow_html=True)

# 🌟 주린이(초보자)용 용어 해설 사전
JURIN_DICT = {
    "쌍끌이 매수": "외국인과 기관 투자자가 동시에 주식을 대량으로 사들이는 아주 강력한 호재 신호입니다.",
    "양매도": "외국인과 기관이 동시에 주식을 대량으로 팔고 있는 악재 신호입니다.",
    "과열권": "주가가 단기간에 너무 많이 올라 조만간 떨어질(조정받을) 가능성이 높은 위험한 상태입니다.",
    "낙폭 과대": "주가가 단기간에 너무 많이 떨어져서 조만간 다시 오를(반등할) 가능성이 높은 상태입니다.",
    "눌림목": "주가가 상승 추세에서 잠시 숨을 고르며 살짝 떨어지는 시점(좋은 매수 타이밍)입니다.",
    "수급": "시장에서 주식을 사려는 사람과 팔려는 사람 간의 힘 겨루기(자금 흐름)를 말합니다.",
    "센티먼트": "뉴스를 통해 느껴지는 사람들의 투자 심리나 분위기(호재/악재)를 뜻합니다."
}

def apply_jurin_help(text):
    for term, desc in JURIN_DICT.items():
        html_tag = f'<span title="{desc}" style="cursor: help; border-bottom: 2px dotted #00E676; color: #00E676; font-weight: bold;">{term}</span>'
        text = text.replace(term, html_tag)
    return text.replace('\n', '<br>')

# --- 1. V6 앙상블 마스터 AI 모델 구조 (24 Features) ---
class SwingBinaryMasterGRU(nn.Module):
    def __init__(self, input_size=24, hidden_size=128, num_layers=2):
        super(SwingBinaryMasterGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))
    def forward(self, x):
        out, _ = self.gru(x)
        w = torch.softmax(torch.tanh(self.attention(out)), dim=1)
        c = torch.sum(w * out, dim=1)
        return self.fc(c)

@st.cache_data(ttl=1800) # 30분 동안 분석 결과를 기억하여 중복 과금을 막습니다!
def analyze_with_llama3(ticker_name, custom_event): # 👈 news_text 매개변수 삭제
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            return 0.0, "스트림릿 Secrets에 GROQ_API_KEY가 설정되지 않았습니다."

        client = Groq(api_key=api_key)

        prompt = f"""
        당신은 월스트리트의 수석 매크로 전략가이자 퀀트 애널리스트입니다.
        입력된 [이슈]가 한국 증시 '{ticker_name}'의 단기 주가와 펀더멘털에 미칠 영향을 다각도로 분석하세요.

        [분석 지침]
        1. **직접/간접 영향 분석**: 이슈가 해당 기업의 매출, 공급망, 원가에 주는 직접적 영향뿐만 아니라 인플레이션, 금리, 환율 등을 통한 매크로적 간접 영향까지 입체적으로 고려하십시오.
        2. **논리적 엄밀성**: 단순히 '좋다/나쁘다'가 아니라, 어떤 경로(Transmission Mechanism)를 통해 주가에 반영되는지 설명하십시오.
        3. **노이즈 판단**: 글로벌 거시 경제나 해당 기업의 비즈니스와 완전히 무관한 이슈인 경우에만 0점을 부여하십시오.
        4. **형식 준수**: 아래 포맷에 맞춰 정확히 답변하십시오.

        [분석 대상]
        이슈: {custom_event if custom_event else "특이사항 없음"}
        종목: {ticker_name}
        
        [출력 포맷]
        SCORE: (정수 범위 -10 ~ +10)
        REASON: (구체적인 수혜/피해 경로를 포함하여 3줄 이내 요약)
        """

        # 🌟 현존 가장 빠르고 가벼운 최신 Llama 3.1 8B 모델 호출
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", 
            temperature=0.2,
        )
        
        text = chat_completion.choices[0].message.content

        import re
        score_match = re.search(r"SCORE:\s*([+-]?\d+)", text)
        reason_match = re.search(r"REASON:\s*(.+)", text, re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.0
        reason = reason_match.group(1).strip() if reason_match else "분석 결과를 파싱할 수 없습니다."

        return score, reason

    except Exception as e:
        return 0.0, f"LLM 분석 중 오류 발생: {e}"

def generate_unified_market_briefing(up_df, th_df):
    if up_df.empty or th_df.empty:
        return "데이터가 부족하여 종합 브리핑을 생성할 수 없습니다."

    # 평균치 계산
    up_avg = up_df['등락률'].mean()
    th_avg = th_df['등락률'].mean()

    # 상/하위 데이터 추출
    top_up = up_df.iloc[0]
    bot_up = up_df.iloc[-1]
    top_th = th_df.iloc[0]

    # 🌟 AI 시장 상태 판별 로직 (업종과 테마의 상관관계 분석)
    if up_avg > 0.3 and th_avg > 0.3:
        status, icon = "전방위 강세장 (Risk-On)", "🔥"
        desc = "업종의 큰 돈(기관/외인)과 테마의 빠른 돈(개인)이 모두 상승을 가리키고 있습니다. 주도주 중심의 적극적인 비중 확대가 유효합니다."
    elif up_avg < -0.3 and th_avg < -0.3:
        status, icon = "전방위 약세장 (Risk-Off)", "❄️"
        desc = "거시적(업종) 하방 압력과 투심(테마) 악화가 겹쳤습니다. 현금 비중을 높이고 하락 방어력이 좋은 대형주 위주로 짧게 대응하십시오."
    elif up_avg > 0 and th_avg <= 0:
        status, icon = "실적/대형주 주도장", "🏢"
        desc = "테마성 투기 자금은 빠지고 있으나, 굵직한 업종 사이클은 버티고 있습니다. 펀더멘털이 튼튼한 우량주 중심의 시장입니다."
    else: 
        status, icon = "테마/개별주 장세 (순환매)", "🎯"
        desc = "시장 전체의 지수(업종)는 부진하나, 특정 테마로 돈이 쏠리고 있습니다. 지수보다는 이슈 중심의 트레이딩(단타/스윙)이 유리합니다."

    briefing = f"""
<div style="background-color: #1E1E2E; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #FF4B4B;">
<h3 style="margin-top: 0;">🧠 AI 머니플로우 종합 피드백</h3>
현재 시장은 <strong>{icon} {status}</strong> 구간입니다. <br><span style="color: #A0A0B0; font-size: 14px;">{desc}</span>

<hr style="border-color: #333;">

<h4 style="margin-bottom: 10px;">📊 1. 시장 온도 비교 (업종 vs 테마)</h4>
<ul style="margin-bottom: 15px;">
    <li><strong>업종 평균 (큰 파도):</strong> <span style="color: {'#FF4B4B' if up_avg > 0 else '#1C83E1'}; font-weight: bold;">{up_avg:+.2f}%</span> (현재 주도: <strong>{top_up['이름']}</strong> <code>{top_up['등락률']:+.2f}%</code>)</li>
    <li><strong>테마 평균 (빠른 물결):</strong> <span style="color: {'#FF4B4B' if th_avg > 0 else '#1C83E1'}; font-weight: bold;">{th_avg:+.2f}%</span> (현재 주도: <strong>{top_th['이름']}</strong> <code>{top_th['등락률']:+.2f}%</code>)</li>
</ul>

<h4 style="margin-bottom: 10px;">💡 2. 트레이딩 전략 인사이트</h4>
<ul style="margin-bottom: 0;">
    <li><strong>🎯 롱(Long) 전략:</strong> 현재 시장의 큰 자금은 [{top_up['이름']}] 업종으로, 빠른 단기 자금은 [{top_th['이름']}] 테마로 쏠리고 있습니다. 우량주 스윙을 원하신다면 업종 대장주를, 단기 변동성을 노리신다면 테마 대장주 중 AI 타점이 높은 종목을 분리하여 공략하십시오.</li>
    <li><strong>⚠️ 숏(Short) 회피:</strong> <strong>[{bot_up['이름']}]</strong> 업종은 현재 자금 이탈이 가장 심각합니다. (<code>{bot_up['등락률']:+.2f}%</code>). 해당 섹터의 매매는 당분간 보류하십시오.</li>
</ul>
</div>
"""
    return briefing

# --- 2. 뉴스 감성 분석 (FinBERT) 엔진 ---
@st.cache_resource
def load_finbert_model():
    MODEL_NAME = "snunlp/KR-FinBERT-SC"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        return tokenizer, model
    except Exception: return None, None

fin_tokenizer, fin_model = load_finbert_model()

@st.cache_data(ttl=3600*24)
def get_stock_tags_mapping():
    """KRX 종목 상세 정보를 이용해 업종과 테마(주요제품) 꼬리표를 생성합니다."""
    try:
        df_desc = fdr.StockListing('KRX-DESC')
        mapping = {}
        for _, row in df_desc.iterrows():
            sector = str(row.get('Sector', '')).strip()
            industry = str(row.get('Industry', '')).strip()
            
            if sector == 'nan' or not sector: sector = "분류없음"
            if industry == 'nan' or not industry: industry = "특징없음"
            
            # 너무 긴 테마(Industry) 텍스트는 가독성을 위해 자름
            if len(industry) > 15:
                industry = industry[:15] + "..."
                
            mapping[row['Code']] = f"🏢 {sector} | 🏷️ {industry}"
        return mapping
    except:
        return {}
    
# --- [신규/수정] 네이버 금융 직접 크롤링 엔진 ---
@st.cache_data(ttl=1800)
def get_naver_market_data(group_type="upjong", count=50):
    """
    업종/테마별 상세 페이지를 정밀 파싱하여 
    최고 상승률(1등)과 최저 상승률(꼴등)을 동시에 찾습니다.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
    
    data = []
    page = 1
    seen_codes = set() # 🌟 중복 검사용 저장소
    
    # 1. 리스트 수집
    while True:
        if group_type == "upjong":
            list_url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
        else:
            list_url = f"https://finance.naver.com/sise/theme.naver?page={page}"
        
        try:
            res = requests.get(list_url, headers=headers, timeout=10)
            res.encoding = 'euc-kr'
            soup = BeautifulSoup(res.text, 'html.parser')
            
            table = soup.select_one('table.type_1')
            if not table: break

            rows = table.select('tr')
            page_item_count = 0
            is_duplicate_page = False # 🌟 중복 페이지 탈출 플래그
            
            for row in rows:
                cols = row.select('td')
                if len(cols) >= 2:
                    link_tag = cols[0].find('a')
                    if link_tag and 'no=' in link_tag.get('href', ''):
                        code = link_tag['href'].split('no=')[-1].split('&')[0] 
                        
                        # 🌟 이미 수집한 코드라면, 마지막 페이지를 넘어서 중복 페이지를 돌고 있는 것!
                        if code in seen_codes:
                            is_duplicate_page = True
                            break
                            
                        seen_codes.add(code) # 새로운 코드는 저장소에 등록
                        
                        name = link_tag.text.strip()
                        change_text = cols[1].text.strip().replace('%', '').replace('+', '')
                        try: change_val = float(change_text)
                        except: change_val = 0.0
                        
                        data.append({"이름": name, "등락률": change_val, "code": code})
                        page_item_count += 1
            
            # 업종이거나, 데이터가 없거나, 중복 페이지가 감지되면 즉시 루프 탈출
            if group_type == "upjong" or page_item_count == 0 or is_duplicate_page:
                break
                
            if page > 15: # 혹시 모를 록인(Lock-in) 방지
                break
                
            page += 1
            
        except Exception as e:
            break

    full_df = pd.DataFrame(data).sort_values("등락률", ascending=False).reset_index(drop=True)
    
    # 2. 상세 페이지 크롤링
    final_list = []
    for i in range(min(count, len(full_df))):
        row = full_df.iloc[i]
        detail_url = f"https://finance.naver.com/sise/sise_group_detail.naver?type={group_type}&no={row['code']}"
        
        try:
            d_res = requests.get(detail_url, headers=headers, timeout=5)
            d_res.encoding = 'euc-kr'
            d_soup = BeautifulSoup(d_res.text, 'html.parser')
            
            stock_table = d_soup.select_one('table.type_5')
            max_change, min_change = -999.0, 999.0
            top_name, bottom_name = "-", "-"
            
            target_td_idx = 3 if group_type == "upjong" else 4
            
            if stock_table:
                stock_rows = stock_table.select('tr')
                for s_row in stock_rows:
                    name_cell = s_row.select_one('td.name a')
                    tds = s_row.select('td')
                    
                    if name_cell and len(tds) > target_td_idx:
                        s_name = name_cell.text.strip()
                        change_text = tds[target_td_idx].text.strip().replace('%', '').replace('+', '').replace(',', '')
                        if not change_text: continue 
                        
                        try:
                            s_change = float(change_text)
                            if s_change > max_change:
                                max_change = s_change
                                top_name = s_name
                            if s_change < min_change:
                                min_change = s_change
                                bottom_name = s_name
                        except:
                            continue
                            
            if max_change == -999.0: max_change = 0.0
            if min_change == 999.0: min_change = 0.0
                
            final_list.append({
                "이름": row['이름'],
                "등락률": row['등락률'],
                "1등주(대장)": top_name,
                "1등 수익률": max_change,
                "꼴등주(부진)": bottom_name,
                "꼴등 수익률": min_change
            })
            
        except Exception:
            continue

    detail_df = pd.DataFrame(final_list)
    return full_df, detail_df
    
def generate_live_sector_briefing(df, g_type="업종"):
    if df.empty: return "현재 분석 가능한 데이터가 없습니다."
    
    top = df.iloc[0]
    bottom = df.iloc[-1]
    avg_change = df['등락률'].mean()
    
    # 시장 온도 상태 판별
    if avg_change > 0.5:
        status_icon, status_text = "🔥", "강세 (불장)"
        status_desc = "대부분의 산업에 돈이 유입되고 있는 활기찬 상태입니다. 적극적인 종목 발굴이 유리합니다."
    elif avg_change < -0.5:
        status_icon, status_text = "❄️", "약세 (냉장고)"
        status_desc = "시장 전체의 엔진이 식어가는 중입니다. 무리한 매수보다는 현금 비중을 늘리고 관망할 때입니다."
    else:
        status_icon, status_text = "☁️", "혼조세 (안개)"
        status_desc = "오르는 곳과 내리는 곳이 팽팽합니다. 방향성이 정해질 때까지 방망이를 짧게 잡아야 합니다."
    
    briefing = f"### 🚀 실시간 {g_type} 수급 브리핑\n\n"
    briefing += f"현재 시장의 {g_type} 기류는 평균 **{avg_change:+.2f}%** 변동하며 **{status_icon} {status_text}**를 기록 중입니다.\n\n"
    
    # 상세 분석 카드 (HTML/Markdown 활용)
    briefing += f"""
---
#### 🌡️ 시황 온도계: "이 수치는 어떤 의미인가요?"
* **평균 변동률 ({avg_change:+.2f}%):** {status_desc}
* **주도 섹터 ({top['이름']}):** 남들보다 **{top['등락률'] - avg_change:+.2f}%p** 더 강한 에너지를 보입니다. 현재 시장의 '주인공'입니다.
* **하락 섹터 ({bottom['이름']}):** 시장의 하락세보다 훨씬 더 깊게 눌리고 있습니다. 자금이 빠르게 이탈 중인 '위험지역'입니다.

#### 💡 주린이를 위한 투자 가이드
1. **상대적 강세에 주목:** 전체 평균({avg_change:+.2f}%)보다 높은 수익률을 기록 중인 섹터는 하락장에서도 누군가 계속 사고 있다는 증거입니다.
2. **엇박자 주의:** 평균은 마이너스인데 혼자 폭등하는 섹터는 '테마성 급등'일 확률이 높으니 추격 매수에 주의하세요.
3. **바닥 확인:** 하락 섹터가 며칠째 최하위라면, 투매가 끝나고 반등이 나올 '눌림목' 후보가 될 수 있습니다.
---
"""
    # 기존 특이사항 로직 유지
    if "조선" in top['이름'] or "운수장비" in top['이름']:
        briefing += "\n> 🔔 **특이사항:** 조선/중공업 사이클에 강한 수급이 포착되었습니다. 대형 수주 뉴스나 환율 효과를 점검하십시오."
    elif "건설" in top['이름']:
        briefing += "\n> 🔔 **특이사항:** 건설/인프라 섹터에 온기가 돌고 있습니다. 정책 변화나 금리 동향이 반영되었을 가능성이 높습니다."
        
    return briefing

def get_google_news_titles(keyword, display=100, days=30):
    search_query = f"{keyword} 주식 when:{days}d"
    enc_keyword = urllib.parse.quote(search_query)
    url = f"https://news.google.com/rss/search?q={enc_keyword}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        soup = BeautifulSoup(res.content, 'xml')
        items = soup.find_all('item')
        results = []
        for item in items[:display]:
            title = item.title.get_text() if item.title else "제목 없음"
            link = item.link.get_text() if item.link else "#"
            pub_date_str = item.pubDate.get_text() if item.pubDate else ""
            if pub_date_str:
                try:
                    dt_utc = pd.to_datetime(pub_date_str, utc=True).tz_convert('Asia/Seoul')
                    date_str = dt_utc.strftime('%Y-%m-%d %H:%M')
                except: date_str = pub_date_str
            else: date_str = "날짜 미상"
            results.append({"title": title, "link": link, "date": date_str})
        return results
    except Exception: return []

def analyze_sentiment(text):
    if fin_model is None or fin_tokenizer is None: return 0.0
    inputs = fin_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = fin_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    return probs[2].item() - probs[0].item()

def get_news_sentiment_details(ticker_name, display=100):
    news_items = get_google_news_titles(ticker_name, display=display)
    if not news_items: return 0.0, []
    for item in news_items: item['score'] = analyze_sentiment(item['title'])
    return sum(item['score'] for item in news_items) / len(news_items), news_items

# --- 3. 보안 접속 ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "dlghdud121!":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else: st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🛡️ AI Quant Radar 접속 보안")
        st.text_input("보안 코드를 입력하고 Enter를 누르세요", type="password", on_change=password_entered, key="password")
        st.info("승인된 트레이더만 접속 가능합니다.")
        st.stop()
        return False
    elif not st.session_state["password_correct"]:
        st.error("보안 코드가 일치하지 않습니다.")
        st.stop()
        return False
    return True

# --- 4. 데이터 및 분석 엔진 ---
@st.cache_data(ttl=3600*24)
def get_all_stock_list():
    try:
        df_krx = fdr.StockListing('KRX')
        df_etf = fdr.StockListing('ETF/KR').rename(columns={'Symbol':'Code'})
        return pd.concat([df_krx['Name'] + " (" + df_krx['Code'] + ")", df_etf['Name'] + " (" + df_etf['Code'] + ")"]).dropna().tolist()
    except: return ["삼성전자 (005930)"]
    
@st.cache_resource
def load_ensemble_models(gru_path, lgb_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_gru = SwingBinaryMasterGRU(input_size=24)
    try:
        if not os.path.exists(gru_path) or not os.path.exists(lgb_path): return None, None, device
        model_gru.load_state_dict(torch.load(gru_path, map_location=device, weights_only=True))
        model_gru.to(device).eval()
        return model_gru, joblib.load(lgb_path), device
    except: return None, None, device

@st.cache_data(ttl=1800)
def get_macro_dashboard_data():
    data = {}

    # 🌟 1. 네이버 금융 실시간 크롤링 (환율, 유가, 달러인덱스)
    try:
        url = "https://finance.naver.com/marketindex/"
        res = requests.get(url, headers={'User-agent': 'Mozilla/5.0'}, timeout=5)
        res.encoding = 'euc-kr'
        soup = BeautifulSoup(res.text, 'html.parser')

        # 크롤링 타겟 CSS 선택자 매핑
        targets = {
            "USD/KRW": "a.head.usd .head_info",
            "달러인덱스": "a.head.usd_idx .head_info",
            "WTI유": "a.head.wti .head_info"
        }

        for name, selector in targets.items():
            info = soup.select_one(selector)
            if info:
                # 현재 값 및 변동폭 추출
                val = float(info.select_one('.value').text.replace(',', ''))
                change_text = info.select_one('.change').text.replace(',', '')
                change = float(change_text)

                # 상승/하락 여부 확인 (클래스에 point_dn이 있으면 하락, point_up이면 상승)
                if 'point_dn' in info.get('class', []):
                    change = -change

                # 등락률(%) 역산: 변동분 / (현재값 - 변동분) * 100
                prev_val = val - change
                pct = (change / prev_val) * 100 if prev_val != 0 else 0.0

                data[name] = (val, pct)
    except Exception as e:
        print(f"네이버 매크로 크롤링 에러: {e}")

    # 🌟 2. FDR 기반 실시간 지수 데이터 (나스닥, KOSPI, KOSDAQ, VIX)
    indices = {"NASDAQ": "IXIC", "KOSPI": "KS11", "KOSDAQ": "KQ11", "VIX": "VIX"}
    for name, code in indices.items():
        try:
            # 빈칸(NaN) 데이터 완벽 제거
            df = fdr.DataReader(code).dropna(subset=['Close']).tail(2)
            if len(df) >= 2:
                curr = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2]
                pct = ((curr - prev) / prev) * 100
                data[name] = (curr, pct)
            elif len(df) == 1:
                data[name] = (df['Close'].iloc[-1], 0.0)
        except:
            continue

    return data

@st.cache_data(ttl=3600*6)
def load_macro_feature_data():
    end_dt = datetime.today().strftime('%Y-%m-%d')
    start_dt = (datetime.today() - timedelta(days=500)).strftime('%Y-%m-%d')
    macro_df = pd.concat([
        fdr.DataReader('USD/KRW', start_dt, end_dt)['Close'].rename('usd_krw'),
        fdr.DataReader('IXIC', start_dt, end_dt)['Close'].rename('nasdaq'),
        fdr.DataReader('KS11', start_dt, end_dt)['Close'].rename('kospi'),
        fdr.DataReader('KQ11', start_dt, end_dt)['Close'].rename('kosdaq'),
        fdr.DataReader('VIX', start_dt, end_dt)['Close'].rename('vix')
    ], axis=1).ffill().dropna()
    for col in ['usd_krw', 'nasdaq', 'kospi', 'kosdaq', 'vix']: macro_df[f'{col}_ret'] = macro_df[col].pct_change()
    return macro_df[['usd_krw_ret', 'nasdaq_ret', 'kospi_ret', 'kosdaq_ret', 'vix_ret']]

def get_naver_supply_demand_history(code, pages=4):
    records = []
    for p in range(1, pages + 1):
        try:
            res = requests.get(f"https://finance.naver.com/item/frgn.naver?code={code}&page={p}", headers={'User-agent': 'Mozilla/5.0'}, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            tables = soup.find_all('table', class_='type2')
            if len(tables) < 2: continue
            for row in tables[1].find_all('tr'):
                cols = row.find_all('td')
                if len(cols) >= 9 and cols[0].text.strip():
                    try:
                        records.append({
                            'Date': pd.to_datetime(cols[0].text.strip().replace('.', '-')),
                            'inst_net': int(cols[5].text.strip().replace(',', '').replace('+', '')),
                            'foreigner_net': int(cols[6].text.strip().replace(',', '').replace('+', ''))
                        })
                    except: pass
        except: continue
    return pd.DataFrame(records).set_index('Date').sort_index() if records else pd.DataFrame()

def prepare_master_features(ticker, df_chart, macro_df):
    if len(df_chart) < 60: return pd.DataFrame(), None, 0, 0, None
    sd_df = get_naver_supply_demand_history(ticker, pages=4)
    df = df_chart.copy().join(macro_df, how='left')
    df = df.join(sd_df, how='left') if not sd_df.empty else df.assign(inst_net=0, foreigner_net=0)
    df.ffill(inplace=True); df.bfill(inplace=True)

    high_9, low_9 = df['High'].rolling(9).max(), df['Low'].rolling(9).min()
    high_26, low_26 = df['High'].rolling(26).max(), df['Low'].rolling(26).min()
    high_52, low_52 = df['High'].rolling(52).max(), df['Low'].rolling(52).min()
    
    tenkan, kijun = (high_9 + low_9) / 2, (high_26 + low_26) / 2
    span_a_raw, span_b_raw = (tenkan + kijun) / 2, (high_52 + low_52) / 2

    df_plot = pd.concat([df, pd.DataFrame(index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=26, freq='B'))])
    tmp_span_a = pd.Series(index=df_plot.index, dtype=float); tmp_span_a.loc[df.index] = span_a_raw
    tmp_span_b = pd.Series(index=df_plot.index, dtype=float); tmp_span_b.loc[df.index] = span_b_raw
    df_plot['senkou_span_a'], df_plot['senkou_span_b'] = tmp_span_a.shift(26), tmp_span_b.shift(26)
    df_plot['tenkan_sen'], df_plot['kijun_sen'] = tenkan, kijun

    close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']
    feats = pd.DataFrame(index=df.index)
    feats['ret'], feats['dist_ma'] = close.pct_change(), close / (close.rolling(20).mean() + 1e-8)
    feats['macd_hist'] = ta.trend.MACD(close).macd_diff()
    feats['adx'], feats['rsi'] = ta.trend.ADXIndicator(high, low, close).adx() / 100.0, ta.momentum.RSIIndicator(close).rsi() / 100.0
    feats['stoch'] = ta.momentum.StochasticOscillator(high, low, close).stoch() / 100.0
    feats['bb_pband'], feats['atr_pct'] = ta.volatility.BollingerBands(close).bollinger_pband(), ta.volatility.AverageTrueRange(high, low, close).average_true_range() / (close + 1e-8)
    feats['obv_ret'], feats['mfi'] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume().pct_change(), ta.volume.MFIIndicator(high, low, close, vol).money_flow_index() / 100.0
    feats['bb_width'], feats['cci'] = ta.volatility.BollingerBands(close).bollinger_wband() / 100.0, ta.trend.CCIIndicator(high, low, close).cci() / 100.0
    feats['roc'], feats['cmf'] = ta.momentum.ROCIndicator(close).roc() / 100.0, ta.volume.ChaikinMoneyFlowIndicator(high, low, close, vol).chaikin_money_flow()
    feats['will_r'] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r() / -100.0
    
    feats['inst_ratio'], feats['foreigner_ratio'] = df['inst_net'] / (vol + 1e-8), df['foreigner_net'] / (vol + 1e-8)
    feats['inst_ratio_5d'] = df['inst_net'].rolling(5).sum() / (vol.rolling(5).sum() + 1e-8)
    feats['foreigner_ratio_5d'] = df['foreigner_net'].rolling(5).sum() / (vol.rolling(5).sum() + 1e-8)
    for col in ['usd_krw_ret', 'nasdaq_ret', 'kospi_ret', 'kosdaq_ret', 'vix_ret']: feats[col] = df[col]
    
    feats.replace([np.inf, -np.inf], np.nan, inplace=True); feats.dropna(inplace=True)
    
    feature_cols = ['ret', 'dist_ma', 'macd_hist', 'adx', 'rsi', 'stoch', 'bb_pband', 'atr_pct', 'obv_ret', 'mfi', 'bb_width', 'cci', 'roc', 'cmf', 'will_r', 'inst_ratio', 'foreigner_ratio', 'inst_ratio_5d', 'foreigner_ratio_5d', 'usd_krw_ret', 'nasdaq_ret', 'kospi_ret', 'kosdaq_ret', 'vix_ret']
    
    if len(feats) == 0: return feats, None, 0, 0, df_plot
    return feats[feature_cols], df.index[-1].strftime('%Y-%m-%d'), feats['inst_ratio'].iloc[-1], feats['foreigner_ratio'].iloc[-1], df_plot

@st.cache_data(ttl=3600)
def get_v6_market_rankings(market_type="KOSPI", top_n=50):
    try:
        df_list = fdr.StockListing(market_type)
        if market_type == "ETF/KR":
             tickers, names = df_list.sort_values('Volume', ascending=False).head(top_n)['Symbol'].tolist(), df_list.sort_values('Volume', ascending=False).head(top_n)['Name'].tolist()
        else:
             tickers, names = df_list.sort_values('Marcap', ascending=False).head(top_n)['Code'].tolist(), df_list.sort_values('Marcap', ascending=False).head(top_n)['Name'].tolist()
        ticker_name_map = dict(zip(tickers, names))
    except: return pd.DataFrame()
    
    results = []
    prog = st.progress(0)
    macro_df = load_macro_feature_data()
    tag_map = get_stock_tags_mapping() # 🌟 꼬리표 매핑 데이터 로드
    
    for i, ticker in enumerate(tickers):
        try:
            df_chart = fdr.DataReader(ticker, (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d'))
            
            # 🌟 [신규 방어막] 실시간 스캔 시 거래정지 종목 완벽 차단!
            if df_chart.empty or df_chart['Volume'].iloc[-1] == 0 or df_chart['Open'].iloc[-1] == 0:
                continue
                
            feats_df, _, _, _, _ = prepare_master_features(ticker, df_chart, macro_df)
            if feats_df.empty or len(feats_df) < 60: continue
            
            scaled_feat = RobustScaler().fit_transform(feats_df.tail(60).values)
            inp = torch.FloatTensor(scaled_feat).unsqueeze(0).to(device)
            with torch.no_grad(): gru_prob = torch.softmax(model_gru(inp), dim=1).cpu().numpy()[0][1]
            lgb_prob = model_lgb.predict_proba(scaled_feat[-1].reshape(1, -1))[0][1]
            base_prob = (gru_prob * 0.5 + lgb_prob * 0.5) * 100
            
            t_name = ticker_name_map[ticker]
            tag_info = tag_map.get(ticker, "🏢 분류없음 | 🏷️ -") # 🌟 해당 종목의 꼬리표 추출
            news_score, _ = (0.0, []) if market_type == "ETF/KR" else get_news_sentiment_details(t_name, display=15)
            final_prob = max(0.0, min(100.0, base_prob + (news_score * 5.0)))
            
            results.append({"종목명": t_name, "코드": ticker,"업종/테마 태그": tag_info, "현재가": int(df_chart['Close'].iloc[-1]), "기본확률(AI)": base_prob, "뉴스점수": news_score, "최종확률": final_prob})
        except: continue
        prog.progress((i + 1) / len(tickers))
    prog.empty()
    return pd.DataFrame(results)

def generate_ai_briefing(name, base_prob, news_score, final_prob, for_ratio, inst_ratio, rsi, stoch, valid_date, news_items=None):
    date_str = f"({valid_date} 장마감 기준)" if valid_date else "(수급 정보 없음)"
    briefing = f"[{name} 트레이딩 브리핑] {date_str}\n\n"
    briefing += f"AI 기본 판단: 차트와 수급을 분석한 기술적 상승 확률은 {base_prob:.1f}% 입니다. "
    
    news_impact = news_score * 5.0
    if news_score > 0.3: briefing += f"여기에 최근 시장의 호재 뉴스가 반영되어 +{news_impact:.1f}%의 강력한 확률 가산점을 받았습니다.\n\n"
    elif news_score < -0.3: briefing += f"다만, 최근 발생한 악재 뉴스로 인해 시장 센티먼트가 위축되어 {news_impact:.1f}%의 확률 감점이 발생했습니다.\n\n"
    else: briefing += f"최근 두드러진 호재나 악재 뉴스가 없어 감성 점수는 중립({news_score:+.2f}점)을 유지 중입니다.\n\n"

    if news_items:
        pos_news = max(news_items, key=lambda x: x['score'])
        neg_news = min(news_items, key=lambda x: x['score'])
        if pos_news['score'] >= 0.25 or neg_news['score'] <= -0.25:
            briefing += f"AI 이슈 요약:**\n"
            if pos_news['score'] >= 0.25: briefing += f"- 🔥 강력한 호재: [{pos_news['title']}]\n"
            if neg_news['score'] <= -0.25: briefing += f"- 🛑 주의할 악재: [{neg_news['title']}]\n"
            briefing += "\n"

    briefing += f"🎯 종합 타점 등급: "
    if final_prob >= 70: briefing += f"🔥 [S급] 초고도 확신 구간 ({final_prob:.1f}%):** 차트와 재료(뉴스)가 완벽히 일치하는 강력한 매수 타이밍입니다.\n\n"
    elif final_prob >= 60: briefing += f"🚀 [A급] 강한 확신 구간 ({final_prob:.1f}%):** 상승 에너지가 긍정적인 매수 우위 자리입니다.\n\n"
    elif final_prob >= 50: briefing += f"✅ [B급] 관망 권장 구간 ({final_prob:.1f}%):** 반반의 확률을 가진 애매한 구간입니다.\n\n"
    else: briefing += f"🛑 [패스] 매수 금지 구간 ({final_prob:.1f}%):** 하방 압력이 강해 관망을 권장합니다.\n\n"

    if for_ratio > 0.001 and inst_ratio > 0.001: briefing += "💡 수급/기술적 코멘트: 현재 외국인과 기관의 쌍끌이 매수가 유입 중이며, "
    elif for_ratio > 0.001: briefing += "💡 수급/기술적 코멘트: 외국인 자금이 유입되며 하방을 방어 중이며, "
    elif inst_ratio > 0.001: briefing += "💡 수급/기술적 코멘트: 기관의 저가 매수세가 들어오는 가운데, "
    elif for_ratio < -0.001 and inst_ratio < -0.001: briefing += "💡 수급/기술적 코멘트: 현재 메이저 양매도가 출회 중이므로 접근에 주의해야 하며, "
    else: briefing += "💡 수급/기술적 코멘트: 메이저 수급의 뚜렷한 특징은 없으며, "
    
    if rsi > 0.7 or stoch > 0.8: briefing += "차트가 단기 과열권에 진입했습니다. 추격 매수보다는 눌림목을 대기하세요."
    elif rsi < 0.3 or stoch < 0.2: briefing += "단기 낙폭 과대 구간입니다. 기술적 반등을 노린 분할 매수가 유효합니다."
    else: briefing += "기술적 지표는 안정적인 적정 구간에 위치해 있습니다."
    return briefing

def draw_ichimoku_chart(df_plot):
    fig = go.Figure()
    
    # 🌟 [버그 수정] 화면에 보여줄 150일(약 6개월)치 데이터만 먼저 잘라냅니다!
    start_idx = max(0, len(df_plot) - 146)
    df_visible = df_plot.iloc[start_idx:].copy()
    
    # 🌟 [신규 기능 1] 매물대 (Volume Profile) 계산 및 추가
    df_valid = df_visible.dropna(subset=['Close', 'Volume'])
    if not df_valid.empty:
        min_p, max_p = df_valid['Close'].min(), df_valid['Close'].max()
        bins = np.linspace(min_p, max_p, 25) 
        df_valid['bin'] = pd.cut(df_valid['Close'], bins=bins)
        vp = df_valid.groupby('bin', observed=False)['Volume'].sum()
        
        customdata = [f"{b.left:,.0f}원 ~ {b.right:,.0f}원" for b in vp.index]
        
        fig.add_trace(go.Bar(
            x=vp.values, y=[b.mid for b in vp.index], orientation='h',
            xaxis='x2', marker=dict(color='rgba(150, 150, 150, 0.4)', line=dict(width=0)),
            customdata=customdata,
            hovertemplate="<b>매물대 구간:</b> %{customdata}<br><b>누적 거래량:</b> %{x:,.0f}주<extra></extra>",
            showlegend=False, name='매물대'
        ))

    # 기존 일목균형표 및 캔들 차트
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['senkou_span_a'], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['senkou_span_b'], line=dict(width=0), fill='tonexty', fillcolor='rgba(150, 150, 150, 0.2)', name='Kumo Cloud'))
    fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['tenkan_sen'], line=dict(color='orange', width=1), name='전환선'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['kijun_sen'], line=dict(color='dodgerblue', width=1), name='기준선'))
    
    start_idx = max(0, len(df_plot) - 146)
    fig.update_layout(
        height=550, template="plotly_dark", xaxis_rangeslider_visible=False, 
        xaxis=dict(type='date', range=[df_plot.index[start_idx], df_plot.index[-1]], tickformat="%Y/%m"), 
        # 🌟 매물대가 차트 왼쪽 1/3 지점까지만 표시되도록 스케일 조정
        xaxis2=dict(overlaying='x', side='top', showgrid=False, showticklabels=False, range=[0, vp.max() * 3] if not df_valid.empty else [0,1]),
        yaxis=dict(tickformat=",", ticksuffix="원"),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

def calculate_realtime_volume_burst(df_chart):
    if len(df_chart) < 20: return 0.0, 0, 0
    
    curr_vol = df_chart['Volume'].iloc[-1]
    vma_20 = df_chart['Volume'].iloc[-21:-1].mean()
    if vma_20 == 0: return 0.0, curr_vol, 0
    
    import pytz
    now = datetime.now(pytz.timezone('Asia/Seoul'))
    today_date = now.date()
    
    # 🌟 [버그 수정] 불러온 데이터의 마지막 날짜 확인
    last_row_date = df_chart.index[-1].date()
    
    market_open = now.replace(hour=9, minute=0, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)
    
    # 🌟 데이터가 오늘 날짜가 아니거나(어제 종가), 장전/장마감인 경우 뻥튀기(시간비례)를 하지 않습니다!
    if last_row_date != today_date or now < market_open or now > market_close:
        return (curr_vol / vma_20) * 100, curr_vol, vma_20
        
    elapsed_mins = (now - market_open).total_seconds() / 60
    if elapsed_mins < 5: elapsed_mins = 5
    
    expected_daily_vol = curr_vol * (390.0 / elapsed_mins)
    burst_ratio = (expected_daily_vol / vma_20) * 100
    
    return burst_ratio, expected_daily_vol, vma_20
def check_seasonality(df_chart):
    """과거 5년간 이번 달(Month)에 상승했던 확률(계절성)을 분석"""
    curr_month = datetime.now().month
    monthly_df = df_chart.resample('ME').last()
    monthly_df['ret'] = monthly_df['Close'].pct_change()
    
    # 이번 달과 같은 달의 데이터만 추출 (최대 5년치)
    target_months = monthly_df[monthly_df.index.month == curr_month].tail(5)
    if len(target_months) < 3: return "데이터 부족"
    
    win_rate = (len(target_months[target_months['ret'] > 0]) / len(target_months)) * 100
    avg_ret = target_months['ret'].mean() * 100
    return f"최근 5년간 {curr_month}월 상승 확률 **{win_rate:.0f}%** (평균 수익률 {avg_ret:+.1f}%)"

def draw_correlation_network(market="KOSPI", top_n=30):
    try:
        df_list = fdr.StockListing(market)
        tickers = df_list.sort_values('Marcap', ascending=False).head(top_n)['Code'].tolist()
        names = df_list.sort_values('Marcap', ascending=False).head(top_n)['Name'].tolist()
        t_map = dict(zip(tickers, names))
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        df_prices = pd.DataFrame()
        
        prog = st.progress(0)
        for i, t in enumerate(tickers):
            try:
                p = fdr.DataReader(t, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))['Close']
                df_prices[t_map[t]] = p.pct_change()
            except: pass
            prog.progress((i+1)/len(tickers))
        prog.empty()
        
        df_prices.dropna(inplace=True)
        corr_matrix = df_prices.corr()
        
        G = nx.Graph()
        THRESHOLD = 0.7 
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= THRESHOLD:
                    G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=corr)
                    
        pos = nx.spring_layout(G, k=1.2, seed=42) 
        
        # --- 1. 엣지(선) 데이터 생성 ---
        edge_x, edge_y = [], []
        # 🌟 엣지 툴팁용 중앙점 데이터
        edge_mid_x, edge_mid_y, edge_hover_text = [], [], []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # 선의 중앙 지점 계산 및 툴팁 텍스트 생성
            # 🌟 툴팁이 더 잘 걸리도록 중앙뿐만 아니라 여러 지점에 포인트를 심습니다.
            for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: # 30%, 50%, 70% 지점에 포인트 배치
                edge_mid_x.append(x0 + (x1 - x0) * ratio)
                edge_mid_y.append(y0 + (y1 - y0) * ratio)
                edge_hover_text.append(f"🔗 연결: {edge[0]} ↔ {edge[1]}<br>📊 상관계수: {edge[2]['weight']:.4f}")
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, 
                                line=dict(width=0.7, color='rgba(150, 150, 150, 0.4)'), 
                                hoverinfo='none', mode='lines')

        # 🌟 엣지 중앙에 보이지 않는 점을 배치하여 툴팁 구현
        edge_info_trace = go.Scatter(
            x=edge_mid_x, y=edge_mid_y, mode='markers',
            marker=dict(size=5, color='rgba(0,0,0,0)'), # 투명한 점
            text=edge_hover_text, hoverinfo='text'
        )
        
        # --- 2. 노드(종목) 데이터 생성 ---
        node_x, node_y, node_labels, node_hover_text, node_size = [], [], [], [], []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(node)
            
            # 해당 종목의 최근 누적 수익률 계산 (정보 제공용)
            cum_ret = (df_prices[node] + 1).prod() - 1
            degree = G.degree(node)
            
            # 🌟 노드 툴팁 텍스트 고도화
            node_hover_text.append(
                f"🏢 <b>{node}</b><br>" +
                f"📈 60일 누적수익률: {cum_ret*100:+.2f}%<br>" +
                f"🕸️ 연결된 종목 수: {degree}개<br>" +
                f"💡 {node}와 동조화된 종목들을 확인하세요."
            )
            node_size.append(18 + (degree * 3)) 
            
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', 
            text=node_labels, textposition="bottom center",
            hovertext=node_hover_text, hoverinfo='text', 
            marker=dict(
                showscale=True, colorscale='Viridis', size=node_size,
                color=[G.degree(n) for n in G.nodes()], 
                line_width=2, colorbar=dict(title="연결도", thickness=15)
            ))
            
        fig = go.Figure(data=[edge_trace, edge_info_trace, node_trace],
             layout=go.Layout(
                title=dict(text=f'🕸️ {market} 핵심 테마 동조화 맵 (상계 {THRESHOLD}↑)', font=dict(size=16)),
                showlegend=False, hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template='plotly_dark'
             ))
        return fig
    except Exception as e:
        st.error(f"네트워크 맵 생성 중 오류: {e}")
        return go.Figure()

def get_market_signal_lamp(macro_df):
    """KOSPI, KOSDAQ 지수를 종목처럼 분석하여 시장의 온도를 측정합니다."""
    indices = {"KOSPI": "KS11", "KOSDAQ": "KQ11"}
    lamp_results = {}
    
    for name, ticker in indices.items():
        try:
            df_idx = fdr.DataReader(ticker, (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d'))
            # 지수 데이터는 수급(sd_df)이 종목과 형식이 다르므로 수급 없이 매크로만으로 분석
            feats_df, _, _, _, _ = prepare_master_features(ticker, df_idx, macro_df)
            if feats_df.empty: continue
            
            scaled_feat = RobustScaler().fit_transform(feats_df.tail(60).values)
            inp = torch.FloatTensor(scaled_feat).unsqueeze(0).to(device)
            with torch.no_grad(): gru_prob = torch.softmax(model_gru(inp), dim=1).cpu().numpy()[0][1]
            lgb_prob = model_lgb.predict_proba(scaled_feat[-1].reshape(1, -1))[0][1]
            prob = ((gru_prob * 0.5) + (lgb_prob * 0.5)) * 100
            
            # 확률에 따른 상태 지정
            if prob >= 60: status, color, icon = "매수 우위", "#FF4B4B", "☀️"
            elif prob <= 40: status, color, icon = "하락 경계", "#1C83E1", "🌧️"
            else: status, color, icon = "중립/관망", "#AAAAAA", "☁️"
            
            lamp_results[name] = {"prob": prob, "status": status, "color": color, "icon": icon}
        except: continue
    return lamp_results
    
# --- 5. 메인 실행부 ---
GRU_PATH = r"weather_advisor_v7_sniper.pt"
LGB_PATH = r"weather_advisor_v7_sniper_lgb.pkl"
RESULT_CSV = r"morning_scan_result.csv" # 🌟 사전 분석 결과 파일명
# 🌟 [추가] 섹터, 테마, ETF CSV 파일 경로
SECTOR_UP_CSV = r"sector_upjong.csv"
SECTOR_TH_CSV = r"sector_theme.csv"
ETF_CSV = r"etf_scanner_result.csv"

model_gru, model_lgb, device = load_ensemble_models(GRU_PATH, LGB_PATH)

if check_password():
    menu = st.sidebar.radio("메뉴 선택", ["단일 종목 스캐너", "섹터 주도주 레이더", "스윙 타점 스캐너", "자금 흐름 네트워크 맵", "ETF 스캐너", "내 관심종목", "자동 모의투자"], horizontal=True, label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.warning(
            "⚠️ **투자 경고 및 면책 조항**\n\n"
            "본 시스템(AI Quant V6)이 제공하는 모든 AI 예측 확률, 타점, "
            "그리고 모의투자 결과는 **단순 참고 및 교육/연구용**입니다.\n\n"
            "절대적인 수익을 보장하지 않으며, "
            "**실제 투자에 대한 최종 판단과 모든 책임은 오직 투자자 본인에게 있습니다.**\n\n"
            "본 프로그램의 개발자 및 제공자는 사용자의 투자 손실에 대해 "
            "**어떠한 민·형사상 법적 책임도 지지 않습니다.**"
        )
        
    # 1. 상단 매크로 지표 출력
    idx_data = get_macro_dashboard_data()
    st.subheader("📊 글로벌 실시간 지표")
    
    # 🌟 늘어난 데이터 개수만큼 자동으로 화면(Column) 분할
    idx_cols = st.columns(len(idx_data))
    
    for i, (k, v) in enumerate(idx_data.items()):
        # 환율, VIX, 달러인덱스, WTI유가 상승은 증시에 부담(악재)이므로 역색상(inverse) 적용
        color = "inverse" if k in ["USD/KRW", "VIX", "달러인덱스", "WTI유"] else "normal"
        idx_cols[i].metric(k, f"{v[0]:,.2f}", f"{v[1]:+.2f}%", delta_color=color)
    
    st.markdown("---")

    # 🌟 2. [신규] 시장 AI 신호등 섹터 (모바일 최적화)
    macro_df_for_lamp = load_macro_feature_data()
    lamp_data = get_market_signal_lamp(macro_df_for_lamp)
    
    if lamp_data:
        st.subheader("🚦 AI 시장 온도계 (오늘의 방향성)")
        l_col1, l_col2 = st.columns(2)
        for i, (name, res) in enumerate(lamp_data.items()):
            target_col = l_col1 if i == 0 else l_col2
            with target_col:
                st.markdown(f"""
                <div style="background-color: #1E1E2E; padding: 15px; border-radius: 10px; border-top: 5px solid {res['color']}; text-align: center;">
                    <span style="font-size: 14px; color: #AAAAAA;">{name} 예측 확률</span><br>
                    <span style="font-size: 28px; font-weight: bold; color: {res['color']};">{res['prob']:.1f}%</span><br>
                    <span style="font-size: 16px;">{res['icon']} {res['status']}</span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
        
    if menu == "단일 종목 스캐너":
        st.title("단일 종목 스캐너 (AI + News + 시뮬레이터)")
        
        all_stocks = get_all_stock_list()
        default_idx = all_stocks.index("삼성전자 (005930)") if "삼성전자 (005930)" in all_stocks else 0
        selected_item = st.sidebar.selectbox("종목 검색 (초성/이름/코드 입력)", options=all_stocks, index=default_idx)
        
        if model_gru is None or model_lgb is None: st.error(f"❌ AI 모델 로드 실패. 파일 경로를 확인하세요.")
        else:
            import re
            match = re.match(r"(.*) \((.*)\)", selected_item)
            if match: name, ticker = match.group(1), match.group(2)
            else: name, ticker = "삼성전자", "005930"
            
            with st.spinner(f"[{name}] 데이터 분석 중..."):
                try:
                    news_items = []
                    # 🌟 [수정포인트 1] 데이터 수집 기간 연장 (월봉 출력을 위해 1000일(약 3년)치 데이터 수집)
                    df_chart = fdr.DataReader(ticker, (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d'))
                    macro_df = load_macro_feature_data()
                    feats_df, v_date, i_r, f_r, df_plot_daily = prepare_master_features(ticker, df_chart, macro_df)
                    
                    if not feats_df.empty and len(feats_df) >= 60:
                        curr_p, prev_p = df_chart['Close'].iloc[-1], df_chart['Close'].iloc[-2]
                        st.metric(label=f"{name} ({ticker})", value=f"{int(curr_p):,}원", delta=f"{int(curr_p-prev_p):+}원 ({(curr_p-prev_p)/prev_p*100:+.2f}%)")
                        
                        briefing_container = st.container()
                        
                        # 🌟 [신규 기능 3] 실시간 거래량 폭발 및 계절성 지표 화면 출력
                        burst_ratio, exp_vol, vma_20 = calculate_realtime_volume_burst(df_chart)
                        seasonality_text = check_seasonality(df_chart)
                    
                        st.markdown("---")
                        st.subheader("🔥 실시간 거래량 및 계절성 (Seasonality)")
                        vol_col1, vol_col2, vol_col3 = st.columns(3)
                    
                        vol_color = "normal" if burst_ratio > 100 else "off"
                        vol_col1.metric("실시간 거래량 폭발 지수", f"{burst_ratio:.0f}%", "20일 평균 돌파!" if burst_ratio > 100 else "거래량 미달", delta_color=vol_color)
                        vol_col2.metric("오늘 예상 마감 거래량", f"{int(exp_vol):,} 주")
                        vol_col3.info(f"📅 **계절성:** {seasonality_text}")
                    
                        with st.expander("🎛️ 매크로 스트레스 테스트 (What-If 시뮬레이터)", expanded=False):
                            st.info("만약 오늘 밤 나스닥이 폭락하거나 환율이 치솟는다면, 이 종목의 내일 상승 확률은 어떻게 변할지 테스트해보세요.")
                            col_s1, col_s2, col_s3 = st.columns(3)
                            sim_nasdaq = col_s1.slider("🇺🇸 나스닥 변동 (%)", -5.0, 5.0, 0.0, 0.5,
                                                       help="간밤에 미국 기술주(나스닥)가 폭락하거나 폭등했을 때, 다음 날 해당 종목에 미칠 충격을 시뮬레이션합니다.")
                            sim_usdkrw = col_s2.slider("💵 환율 변동 (%)", -3.0, 3.0, 0.0, 0.1,
                                                       help="원/달러 환율이 급등(원화 가치 하락)하면 외국인 자금 이탈 우려가 커져 증시에 악재로 작용하는 경향이 있습니다.")
                            sim_vix = col_s3.slider("😨 VIX 공포지수 변동 (%)", -20.0, 20.0, 0.0, 1.0,
                                                    help="시장의 공포지수입니다. VIX가 치솟으면 전 세계적인 투자 심리가 얼어붙어 증시에 강한 하방 압력을 줍니다.")
                        
                        # 시뮬레이션 데이터 복사 및 변동치 적용
                        sim_feats_df = feats_df.copy()
                        sim_feats_df.loc[sim_feats_df.index[-1], 'nasdaq_ret'] += (sim_nasdaq / 100.0)
                        sim_feats_df.loc[sim_feats_df.index[-1], 'usd_krw_ret'] += (sim_usdkrw / 100.0)
                        sim_feats_df.loc[sim_feats_df.index[-1], 'vix_ret'] += (sim_vix / 100.0)

                        # 시뮬레이션 적용된 데이터로 스케일링 및 AI 예측 (고정된 일봉 모델 사용)
                        scaled_feat = RobustScaler().fit_transform(sim_feats_df.tail(60).values)
                        inp = torch.FloatTensor(scaled_feat).unsqueeze(0).to(device)
                        
                        with torch.no_grad(): gru_prob = torch.softmax(model_gru(inp), dim=1).cpu().numpy()[0][1]
                        lgb_prob = model_lgb.predict_proba(scaled_feat[-1].reshape(1, -1))[0][1]
                        base_prob_pct = ((gru_prob * 0.5) + (lgb_prob * 0.5)) * 100
                        
                        # 🌟 [ETF 에러 해결 핵심] 변수 안전 초기화 및 ETF 뉴스 검색 스킵
                        sentiment_score, news_items = 0.0, [] 
                        etf_brands = ['KODEX', 'TIGER', 'KBSTAR', 'ACE', 'ARIRANG', 'KOSEF', 'HANARO', 'SOL']
                        
                        # 종목명에 위 ETF 브랜드가 포함되어 있지 않을 때만 뉴스를 긁어옵니다.
                        if not any(brand in name for brand in etf_brands):
                            sentiment_score, news_items = get_news_sentiment_details(name, display=100)
                                
                        news_impact = sentiment_score * 5.0
                        final_prob_pct = max(0.0, min(100.0, base_prob_pct + news_impact))
                        
                        st.markdown("---")
                        # 🌟 [신규 UI] 노이즈 차단 & Llama 3 타겟 분석
                        st.subheader("⚡ Llama 3 매크로 역학 분석 (Groq API)")
                        custom_event = st.text_input("💡 주가에 영향을 줄 수 있는 글로벌 매크로 이슈를 직접 입력하세요", placeholder="예: 이란-미국 갈등 격화로 인한 국제 유가 급등 등")
                        use_llama = st.checkbox("🚀 입력한 이슈만으로 타겟 수혜/악재 역학 분석 실행")

                        llama_score = 0.0
                        llama_reason = ""
                        
                        if use_llama:
                            if not custom_event.strip():
                                st.warning("⚠️ 분석할 글로벌 이슈를 텍스트 박스에 먼저 입력해 주세요!")
                                news_impact = 0.0
                            else:
                                with st.spinner(f"'{custom_event}' 이슈가 {name}에 미칠 영향을 추론 중입니다..."):
                                    # 뉴스 텍스트 없이 오직 종목명과 이슈만 넘깁니다.
                                    llama_score, llama_reason = analyze_with_llama3(name, custom_event)
                                    
                                    # 일반 뉴스 점수는 완전히 버리고, LLM의 이슈 판단 점수만 반영합니다.
                                    news_impact = llama_score * 0.5 
                                    
                                    st.success(f"**[Llama 3 매크로 분석 결과: {llama_score:+.0f}점]**\n\n{llama_reason}")
                        else:
                            # 🌟 체크박스를 켜지 않으면 외부 뉴스/이슈 노이즈를 0%로 완벽 차단합니다.
                            news_impact = 0.0 

                        final_prob_pct = max(0.0, min(100.0, base_prob_pct + news_impact))
                        
                        # 🌟 [수정포인트 2] 타임프레임 선택 라디오 버튼 (UI 배치 변경)
                        tf_col, _ = st.columns([1, 2])
                        with tf_col:
                            timeframe = st.radio("📊 차트 및 AI 분석 기준 선택", ["일봉 (단기 5일)", "주봉 (중기 4주)", "월봉 (장기 3개월)"], horizontal=True)
                        
                        # 🌟 [수정포인트] 타임프레임 선택 및 데이터 리샘플링 로직
                        df_chart_plot = df_plot_daily.copy()
                        
                        if timeframe != "일봉 (단기 5일)":
                            # 주봉/월봉 변환
                            rule = 'W-FRI' if timeframe == "주봉 (중기 4주)" else 'ME'
                            # 🌟 [에러 해결 핵심] 주봉/월봉으로 묶을 때 거래량(Volume)도 모두 합치도록 명령을 추가합니다!
                            df_chart_plot = df_chart.resample(rule).agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
                            
                            # 일목균형표 기본선 계산
                            h9, l9 = df_chart_plot['High'].rolling(9).max(), df_chart_plot['Low'].rolling(9).min()
                            h26, l26 = df_chart_plot['High'].rolling(26).max(), df_chart_plot['Low'].rolling(26).min()
                            h52, l52 = df_chart_plot['High'].rolling(52).max(), df_chart_plot['Low'].rolling(52).min()
                            
                            tenkan_sen = (h9 + l9) / 2
                            kijun_sen = (h26 + l26) / 2
                            span_a_raw = (tenkan_sen + kijun_sen) / 2
                            span_b_raw = (h52 + l52) / 2
                            
                            # 🌟 [핵심 해결] 미래 날짜(26칸)를 만들어 차트 꼬리에 붙여줍니다.
                            # 주봉이면 26주, 월봉이면 26개월 치 빈 껍데기(Index)를 만듭니다.
                            future_dates = pd.date_range(start=df_chart_plot.index[-1] + pd.Timedelta(days=1), periods=26, freq=rule)
                            future_df = pd.DataFrame(index=future_dates)
                            df_chart_plot = pd.concat([df_chart_plot, future_df])
                            
                            # 값을 넣고 26칸을 앞(미래)으로 밀어냅니다.
                            tmp_span_a = pd.Series(index=df_chart_plot.index, dtype=float)
                            tmp_span_a.loc[df_chart.resample(rule).last().dropna().index] = span_a_raw
                            
                            tmp_span_b = pd.Series(index=df_chart_plot.index, dtype=float)
                            tmp_span_b.loc[df_chart.resample(rule).last().dropna().index] = span_b_raw
                            
                            df_chart_plot['tenkan_sen'] = tenkan_sen
                            df_chart_plot['kijun_sen'] = kijun_sen
                            df_chart_plot['senkou_span_a'] = tmp_span_a.shift(26)
                            df_chart_plot['senkou_span_b'] = tmp_span_b.shift(26)
                            
                            st.caption(f"💡 **안내:** 현재 보여지는 차트는 **{timeframe}** 추세 확인용입니다. 하단의 AI 타점 확률은 최적화된 **'일봉(단기 스윙)'** 데이터를 기준으로 계산되었습니다.")
                        
                        # 🌟 [수정포인트 4] AI 브리핑 동적 생성 로직 (함수 대신 인라인으로 교체하여 유연성 확보)
                        date_str = f"({v_date} 장마감 기준)" if v_date else "(수급 정보 없음)"
                        briefing = f"[{name} {timeframe} 트레이딩 브리핑] {date_str}\n\n"
                        briefing += f"AI 모델 및 뉴스 센티먼트를 종합한 최종 기술적 상승 확률은 **{final_prob_pct:.1f}%** 입니다.\n\n"
                        
                        # 🌟 [업그레이드 완료] 실전 매매 관점을 포함한 매물대 해석 가이드
                        with st.expander("💡 차트 왼쪽의 '매물대(회색 막대)' 실전 200% 활용법"):
                            st.markdown("""
                            **매물대(Volume Profile)**는 특정 가격대에서 '얼마나 많은 주식이 거래되었는가'를 보여주는 지표입니다. 막대가 길수록 해당 가격에 주식을 사서 들고 있는 주주(평단가)가 많다는 뜻입니다.

                            🔥 **매물대에 숨겨진 3가지 투자 심리**
                            * **1️⃣ 주가가 매물대 '아래'에 있을 때 (콘크리트 저항선)**
                                * 가장 긴 막대에 물려있는 주주들의 강렬한 **'본전 탈출 심리'**가 발동하여 매도 폭탄이 쏟아집니다. 이를 뚫고 올라가려면 평소보다 훨씬 거대한 거래량이 필요합니다.
                            * **2️⃣ 주가가 매물대 '위'에 있을 때 (든든한 지지선)**
                                * 주가가 하락하더라도 과거 그 가격대에서 수익을 냈던 사람들이나 대기자들의 매수세가 받쳐주어 **쉽게 빠지지 않고 반등할 확률**이 높습니다. (최적의 스윙 진입 타점)
                            * **3️⃣ 매물대 막대가 거의 없는 구간 (고속도로 진공 구간)**
                                * 과거에 거래 없이 순식간에 지나간 가격대로, 묶여있는 주주가 거의 없습니다. 악성 매물도 없고 바닥 지지도 없어서 **위든 아래든 순식간에 급등/급락**해 버리는 특징이 있습니다.

                            💡 **실전 핵심 팁:** AI가 제시한 목표가까지 가는 길에 거대한 매물대 막대가 없다면(진공 구간), 저항 없이 고속도로처럼 시원하게 목표가에 도달할 확률이 매우 높습니다! (회색 막대에 마우스를 올리면 가격 구간과 누적 거래량을 볼 수 있습니다.)
                            """)

                        if timeframe == "일봉 (단기 5일)":
                            # 기존 일봉 백테스트 결과 유지
                            if final_prob_pct >= 70: briefing += f"[S급] 초고도 확신 (승률 84.8%): 강력한 매수 타이밍입니다. (TP 4% 목표)\n"
                            elif final_prob_pct >= 60: briefing += f"[A급] 강한 확신 (승률 57.7%): 단기 상승 에너지가 긍정적인 자리입니다.\n"
                            else: briefing += f"[일반 매수] (승률 50.9%): 확률적 우위가 크지 않은 애매한 구간이므로 관망을 권장합니다.\n"
                        
                        elif timeframe == "주봉 (중기 4주)":
                            # 🌟 업데이트: 20일 예측 팩트 데이터 적용
                            if final_prob_pct >= 70: briefing += f"[S급] 중기 대세 상승 (승률 75.8%): 4주간 홀딩해도 매우 안전한 최상위 스윙 타점입니다. (TP 8% 목표)\n"
                            elif final_prob_pct >= 60: briefing += f"[A급] 단기 대응 요망 (승률 50.0%): 중기(4주)로 끌고 가기엔 리스크가 있습니다. 5일 이내 단기 수익 실현을 권장합니다.\n"
                            else: briefing += f"[관망]: 중기 추세의 방향성이 불확실합니다.\n"
                            
                        elif timeframe == "월봉 (장기 3개월)":
                            # 🌟 업데이트: 60일 예측 팩트 데이터 적용
                            if final_prob_pct >= 70: briefing += f"[S급] 역사적 변곡점 (승률 76.9%): 시장에 극히 드물게 나타나는 초장기 바닥/대세 상승 초입입니다! 적극적인 비중 확대를 고려하십시오. (TP 12% 목표)\n"
                            elif final_prob_pct >= 60: briefing += f"[A급] 장기 우상향 (승률 55.9%): 거시적으로 무난한 상승장 흐름에 탑승하고 있습니다.\n"
                            else: briefing += f"[시장 평균] (승률 58.5%): 시장 평균적인 움직임을 보이고 있습니다. 개별 종목의 펀더멘털 분석이 추가로 필요합니다.\n"
                            
                        briefing += "\n💡 **수급/기술적 코멘트:**\n"
                        if f_r > 0.001 and i_r > 0.001: briefing += "현재 외국인과 기관의 쌍끌이 매수가 유입 중입니다. "
                        elif f_r > 0.001: briefing += "외국인 자금이 유입되며 하방을 방어 중입니다. "
                        elif i_r > 0.001: briefing += "기관의 저가 매수세가 들어오고 있습니다. "
                        elif f_r < -0.001 and i_r < -0.001: briefing += "현재 메이저 양매도가 출회 중이므로 접근에 주의하십시오. "
                        else: briefing += "메이저 수급의 뚜렷한 이탈이나 유입은 감지되지 않습니다. "

                        rsi = feats_df['rsi'].iloc[-1]
                        stoch = feats_df['stoch'].iloc[-1]
                        if rsi > 0.7 or stoch > 0.8: briefing += "차트가 단기 과열권에 진입했습니다. 급등 시 추격 매수보다는 조정을 대기하세요.\n"
                        elif rsi < 0.3 or stoch < 0.2: briefing += "단기 낙폭 과대 구간입니다. 기술적 반등을 노린 분할 매수가 유효합니다.\n"
                        else: briefing += "기술적 지표는 안정적인 적정 구간에 위치해 있습니다.\n"
                        
                        if news_items:
                            pos_news = max(news_items, key=lambda x: x['score'])
                            neg_news = min(news_items, key=lambda x: x['score'])
                            if pos_news['score'] >= 0.25 or neg_news['score'] <= -0.25:
                                briefing += f"\n🗣️ **AI 이슈 요약:**\n"
                                if pos_news['score'] >= 0.25: briefing += f"- 🔥 강력한 호재: [{pos_news['title']}]\n"
                                if neg_news['score'] <= -0.25: briefing += f"- 🛑 주의할 악재: [{neg_news['title']}]\n"

                        # 브리핑 출력
                        briefing_html = apply_jurin_help(briefing)
                        with briefing_container:
                            st.markdown(f"""
                            <div style="background-color: rgba(0, 230, 118, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #00E676; margin-bottom: 20px; line-height: 1.6;">
                                {briefing_html}
                            </div>
                            """, unsafe_allow_html=True)

                        # 차트 및 점수판 출력
                        col1, col2 = st.columns([2, 1])
                        with col1: 
                            st.plotly_chart(draw_ichimoku_chart(df_chart_plot), use_container_width=True)
                        
                        with col2:
                            is_simulated = sim_nasdaq != 0 or sim_usdkrw != 0 or sim_vix != 0
                            title_prefix = "🔬 [시뮬레이션 적용됨]" if is_simulated else "1차: 2 AI 앙상블"
                            
                            st.subheader(title_prefix)
                            cA, cB, cC = st.columns(3)
                            cA.metric("GRU", f"{gru_prob*100:.1f}%")
                            cB.metric("LGBM", f"{lgb_prob*100:.1f}%")
                            cC.metric("기본 확률", f"{base_prob_pct:.1f}%")
                            
                            st.markdown("---")
                            st.subheader("📰 2차: 뉴스 센티먼트 융합")
                            cD, cE, cF = st.columns(3)
                            news_emoji = "🔥" if sentiment_score > 0 else ("🛑" if sentiment_score < 0 else "➖")
                            cD.metric(f"뉴스 ({news_emoji})", f"{sentiment_score:+.2f}점")
                            cE.metric("가산점", f"{news_impact:+.1f}%p")
                            
                            delta_str = "시뮬레이션!" if is_simulated else None
                            cF.metric("최종 확신도", f"{final_prob_pct:.1f}%", delta=delta_str, delta_color="inverse")
                            
                            st.markdown("---")
                            st.subheader(f"🎯 {timeframe.split(' ')[0]} 매매 가이드")
                            
                            if timeframe == "일봉 (단기 5일)":
                                tp_rate, sl_rate = 4.0, -3.0
                            elif timeframe == "주봉 (중기 4주)":
                                tp_rate, sl_rate = 8.0, -5.0
                            else: # 월봉 (장기 3개월)
                                tp_rate, sl_rate = 12.0, -10.0
                                
                            target_price = curr_p * (1 + (tp_rate / 100))
                            stop_loss = curr_p * (1 + (sl_rate / 100))
                            
                            # 🌟 [수정 완료] 거대한 st.metric 대신 깔끔한 텍스트 리스트로 통일
                            st.write(f"- 적정 매수가: `{int(curr_p):,}원`")
                            st.write(f"- 목표가 (+{tp_rate}%): `{int(target_price):,}원`")
                            st.write(f"- 손절가 ({sl_rate}%): `{int(stop_loss):,}원`")
                                                                            
                            st.markdown("---")
                            st.write(f"📊 실시간 수급 (비중)")
                            st.write(f"- 외국인: {f_r * 100:+.2f}%")
                            st.write(f"- 기  관: {i_r * 100:+.2f}%")

                    st.markdown("---")
                    st.subheader(f"📰 {name} 주요 최신 뉴스 (표본 100개 중 최신 10개)")
                    if news_items:
                        for n in news_items[:10]: 
                            score = n['score']
                            emoji = "🔥" if score > 0.3 else ("🛑" if score < -0.3 else "➖")
                            st.markdown(f"• `{n['date']}` | {emoji} **[{score:+.2f}점]** [{n['title']}]({n['link']})")
                    else: st.write("최근 30일간 검색된 뉴스가 없습니다.")
                except Exception as e: st.error(f"분석 중 오류 발생: {e}")

    # 🌟 [수정완료] 섹터 주도주 레이더 (CSV에서 데이터 로드)
    elif menu == "섹터 주도주 레이더":
        st.title("🏢 섹터 & 테마 머니플로우 (사전 분석)")
        st.info("시장의 큰 파도(업종)와 빠른 물결(테마)을 요약한 새벽 분석 리포트입니다.")
        
        if os.path.exists(SECTOR_UP_CSV) and os.path.exists(SECTOR_TH_CSV):
            detail_up = pd.read_csv(SECTOR_UP_CSV)
            detail_th = pd.read_csv(SECTOR_TH_CSV)
            
            st.markdown(generate_unified_market_briefing(detail_up, detail_th), unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["업종별 사이클 (Industry)", "테마별 수급 (Theme)"])
            with tab1:
                st.markdown(generate_live_sector_briefing(detail_up, "업종"))
                st.dataframe(
                    detail_up.style.format({"등락률": "{:+.2f}%", "1등 수익률": "{:+.2f}%", "꼴등 수익률": "{:+.2f}%"}).map(
                        lambda x: 'color: #FF4B4B; font-weight: bold' if x > 0 else ('color: #1C83E1' if x < 0 else 'color: gray'), 
                        subset=["등락률", "1등 수익률", "꼴등 수익률"]
                    ),
                    use_container_width=True
                )
            with tab2:
                st.markdown(generate_live_sector_briefing(detail_th, "테마"))
                st.dataframe(
                    detail_th.style.format({"등락률": "{:+.2f}%", "1등 수익률": "{:+.2f}%", "꼴등 수익률": "{:+.2f}%"}).map(
                        lambda x: 'color: #FF4B4B; font-weight: bold' if x > 0 else ('color: #1C83E1' if x < 0 else 'color: gray'), 
                        subset=["등락률", "1등 수익률", "꼴등 수익률"]
                    ),
                    use_container_width=True
                )
        else:
            st.warning("⚠️ 아직 섹터 분석 데이터(CSV)가 생성되지 않았습니다. 깃허브 액션 실행을 확인하세요.")

    # 🌟 자금 흐름 네트워크 맵
    elif menu == "자금 흐름 네트워크 맵":
        st.title("시총 상위 자금 흐름 네트워크")
        st.info("KOSPI/KOSDAQ 시장의 대형주들이 어떻게 묶여서 같이 오르고 내리는지 상관관계를 시각화합니다. (연결선이 굵고 많을수록 시장의 주도 테마입니다.)")
        
        m_type = st.radio("타겟 시장 선택", ["KOSPI", "KOSDAQ"], horizontal=True)
        if st.button("네트워크 맵 분석 시작 (약 10초 소요)"):
            with st.spinner(f"{m_type} 시총 상위 50개 종목의 최근 60일 상관관계를 분석 중입니다..."):
                fig = draw_correlation_network(market=m_type, top_n=50)
                st.plotly_chart(fig, use_container_width=True)
                
                # 🌟 [신규 추가] 초보자를 위한 매물대 해석 가이드
                with st.expander("💡 차트 왼쪽의 '매물대(회색 막대)' 보는 법"):
                    st.markdown("""
                        * **매물대란?** 과거 해당 가격대에서 주식이 얼마나 많이 거래되었는지를 나타내는 지표입니다.
                        * **가장 긴 막대 (매물 집중 구간):** 이 가격대에서 사고판 사람이 가장 많다는 뜻입니다. 
                        * 주가가 이 구간보다 **아래**에 있다면 뚫기 힘든 **강력한 저항선(악성 매물)**이 됩니다.
                        * 주가가 이 구간보다 **위**에 있다면 떨어질 때 받쳐주는 **든든한 지지선** 역할을 합니다.
                        * **마우스 활용:** 회색 막대에 마우스를 올리시면 정확한 '가격대 구간'과 '누적 거래량'을 확인하실 수 있습니다.
                        """)

    # 🌟 [수정완료] ETF 스캐너 (CSV에서 데이터 로드)
    elif menu == "ETF 스캐너":
        st.title("🔥 ETF AI 방향성 레이더 (사전 분석)")
        st.write("시장 전체의 자금 흐름과 분위기를 파악합니다.")
        st.markdown("---")
        
        if os.path.exists(ETF_CSV):
            mod_time = datetime.fromtimestamp(os.path.getmtime(ETF_CSV)).strftime('%Y-%m-%d %H:%M')
            st.success(f"📅 마지막 업데이트: {mod_time} (KST)")
            etf_df = pd.read_csv(ETF_CSV)
            
            # 문자열이 아닐 경우 % 포맷팅 처리
            if etf_df['최종확률'].dtype in [float, int]:
                etf_df['최종확률'] = etf_df['최종확률'].apply(lambda x: f"{x:.1f}%")
                
            display_cols = ['시장', '종목명', '코드', '예측시점가격', '목표가', '손절가', '최종확률']
            st.dataframe(etf_df[display_cols].sort_values("최종확률", ascending=False).reset_index(drop=True), use_container_width=True)
        else:
            st.warning("⚠️ 아직 ETF 스캔 데이터(CSV)가 생성되지 않았습니다. 깃허브 액션 실행을 확인하세요.")

    # 🌟 스윙 타점 스캐너
    elif menu == "스윙 타점 스캐너":
        st.title("🎯 저격수 스캐너 (사전 분석 완료 모드)")
        
        if os.path.exists(RESULT_CSV):
            mod_time = datetime.fromtimestamp(os.path.getmtime(RESULT_CSV)).strftime('%Y-%m-%d %H:%M')
            st.success(f"📅 마지막 데이터 업데이트: {mod_time} (KST)")
            
            full_df = pd.read_csv(RESULT_CSV)
            
            if '업종/테마 태그' not in full_df.columns:
                tag_map = get_stock_tags_mapping()
                full_df['업종/테마 태그'] = full_df['코드'].apply(lambda x: tag_map.get(str(x).zfill(6), "🏢 분류없음"))

            sniper_s_df = full_df[full_df['최종확률'] >= 70.0].sort_values(by='최종확률', ascending=False).reset_index(drop=True)
            sniper_a_df = full_df[(full_df['최종확률'] >= 60.0) & (full_df['최종확률'] < 70.0)].sort_values(by='최종확률', ascending=False).reset_index(drop=True)

            if not sniper_s_df.empty or not sniper_a_df.empty:
                from collections import Counter
                high_prob_df = pd.concat([sniper_s_df, sniper_a_df])
                sectors = [tag.split('|')[0].replace('🏢', '').strip() for tag in high_prob_df['업종/테마 태그'] if '분류없음' not in tag]
                if sectors:
                    top_sector = Counter(sectors).most_common(1)[0][0]
                    st.error(f"🧠 **[AI 바텀업 분석]** 오늘의 주도 업종은 **[{top_sector}]** 입니다!")

            st.markdown("---")
            if not sniper_s_df.empty:
                st.error(f"🔥 **[S급] 초고도 확신 타점 ({len(sniper_s_df)}건)**")
                st.dataframe(sniper_s_df, use_container_width=True)
                
            if not sniper_a_df.empty:
                st.info(f"🚀 **[A급] 강한 확신 타점 ({len(sniper_a_df)}건)**")
                st.dataframe(sniper_a_df, use_container_width=True)

            with st.expander("📋 전체 분석 데이터 보기 (B급 이하 포함)"):
                st.dataframe(full_df.sort_values(by='최종확률', ascending=False), use_container_width=True)

        else:
            st.warning("⚠️ 아직 사전 분석 결과 파일(CSV)이 없습니다. 깃허브 액션이 실행되었는지 확인하세요.")

        if st.sidebar.button("🔄 실시간 스캔 강제 실행"):
            st.cache_data.clear()
            st.rerun()
    
    # 🌟 [신규 기능 2] 내 관심종목 (Watchlist) 모니터링
    elif menu == "내 관심종목":
        st.title("⭐️ 나만의 관심종목 모니터링")
        st.info("내가 보유 중이거나 눈여겨보는 종목들만 모아서 실시간 수급과 AI 확률을 비교합니다.")
        
        if "watchlist" not in st.session_state:
            st.session_state["watchlist"] = ["삼성전자 (005930)", "SK하이닉스 (000660)"] # 기본값
            
        all_stocks = get_all_stock_list()
        selected_stocks = st.multiselect("📌 관심종목 추가/삭제", all_stocks, default=st.session_state["watchlist"])
        st.session_state["watchlist"] = selected_stocks
        
        if st.button("🚀 선택한 관심종목 AI 즉시 분석"):
            if not selected_stocks:
                st.warning("선택된 종목이 없습니다.")
            else:
                watch_results = []
                prog_bar = st.progress(0)
                macro_df = load_macro_feature_data()
                
                for i, item in enumerate(selected_stocks):
                    import re
                    match = re.match(r"(.*) \((.*)\)", item)
                    if match: name, ticker = match.group(1), match.group(2)
                    else: continue
                    
                    try:
                        df_chart = fdr.DataReader(ticker, (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d'))
                        feats_df, _, i_r, f_r, _ = prepare_master_features(ticker, df_chart, macro_df)
                        burst_ratio, _, _ = calculate_realtime_volume_burst(df_chart)
                        
                        if not feats_df.empty and len(feats_df) >= 60:
                            scaled = RobustScaler().fit_transform(feats_df.tail(60).values)
                            inp = torch.FloatTensor(scaled).unsqueeze(0).to(device)
                            with torch.no_grad(): gru_prob = torch.softmax(model_gru(inp), dim=1).cpu().numpy()[0][1]
                            lgb_prob = model_lgb.predict_proba(scaled[-1].reshape(1, -1))[0][1]
                            base_prob = (gru_prob * 0.5 + lgb_prob * 0.5) * 100
                            
                            curr_price = df_chart['Close'].iloc[-1]
                            watch_results.append({
                                "종목명": name, "현재가": int(curr_price),
                                "AI 확률": base_prob, "거래량 폭발": burst_ratio,
                                "외인수급": f_r * 100, "기관수급": i_r * 100
                            })
                    except: pass
                    prog_bar.progress((i + 1) / len(selected_stocks))
                
                prog_bar.empty()
                if watch_results:
                    w_df = pd.DataFrame(watch_results)
                    w_df['AI 확률'] = w_df['AI 확률'].apply(lambda x: f"{x:.1f}%")
                    w_df['거래량 폭발'] = w_df['거래량 폭발'].apply(lambda x: f"{x:.0f}%")
                    w_df['외인수급'] = w_df['외인수급'].apply(lambda x: f"{x:+.2f}%")
                    w_df['기관수급'] = w_df['기관수급'].apply(lambda x: f"{x:+.2f}%")
                    
                    st.dataframe(w_df.sort_values("AI 확률", ascending=False).reset_index(drop=True), use_container_width=True)
    
    # 🌟 [업그레이드 완결판] 누적식 자동 모의투자 및 연간 수익률 차트
    elif menu == "자동 모의투자":
        st.title("🤖 데이트레이딩 자동 모의투자 (복리 누적 시스템)")
        st.info("매일 장이 열리면 AI 추천 주도주를 매매하며, 매일의 수익이 누적되어 다음 날의 투자 원금이 됩니다. (매년 1월 1일 1,000만 원으로 리셋)")

        import os
        import pandas as pd
        from datetime import datetime
        import pytz
        import plotly.graph_objects as go

        # 📂 누적 기록을 저장할 비밀 장부 파일
        HISTORY_CSV = "mock_invest_history.csv"
        now = datetime.now(pytz.timezone('Asia/Seoul'))
        today_str = now.strftime('%Y-%m-%d')
        curr_year = now.year

        # 🌟 [지능형 시작일 설정] 
        # 2026년이면 5월 1일 시작, 2027년부터는 해당 연도 1월 1일 시작
        if curr_year == 2026:
            START_DATE = pd.to_datetime("2026-05-01")
        else:
            START_DATE = pd.to_datetime(f"{curr_year}-01-01")
            
        # 1. 누적 데이터 로드 및 연도별 초기화 로직
        if os.path.exists(HISTORY_CSV):
            hist_df = pd.read_csv(HISTORY_CSV)
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
            hist_df = hist_df[hist_df['Date'] >= START_DATE].copy()
            
            # 만약 새해가 되었는데 첫 기록(1월 1일)이 없다면 초기화
            if hist_df.empty:
                initial_setup = pd.DataFrame([{'Date': START_DATE, 'Invested': 0, 'PnL': 0, 'Balance': 10000000}])
                hist_df = initial_setup
        else:
            hist_df = pd.DataFrame(columns=['Date', 'Invested', 'PnL', 'Balance'])
            initial_setup = pd.DataFrame([{'Date': START_DATE, 'Invested': 0, 'PnL': 0, 'Balance': 10000000}])
            hist_df = pd.concat([hist_df, initial_setup], ignore_index=True)

        # 🌟 [NameError 해결 핵심!] 하단 로직과 차트가 고장나지 않도록 변수명 연결
        hist_this_year = hist_df.copy()
        
        # '오늘' 이전까지의 기록 중 가장 마지막 잔고를 오늘의 '시작 원금'으로 설정
        hist_before_today = hist_this_year[hist_this_year['Date'].dt.strftime('%Y-%m-%d') < today_str]
        if hist_before_today.empty:
            INITIAL_CAPITAL = 10000000
        else:
            INITIAL_CAPITAL = hist_before_today.iloc[-1]['Balance']

        # 2. 오늘 타점 분석 및 실시간 수익률 계산
        total_invested = 0
        total_current_value = 0
        total_pnl_krw = 0
        final_balance = INITIAL_CAPITAL
        portfolio = []

        if os.path.exists(RESULT_CSV):
            scan_df = pd.read_csv(RESULT_CSV)
            target_df = scan_df[scan_df['최종확률'] >= 60.0].copy()

            if target_df.empty:
                st.warning("오늘 장은 AI 확률 60% 이상의 타점이 없어 매매를 쉬어갑니다 (이전 잔고 유지).")
            else:
                target_df = target_df.reset_index(drop=True)
                num_stocks = len(target_df)
                alloc_per_stock = INITIAL_CAPITAL // num_stocks # 누적된 원금으로 N빵 분할
                
                st.write("**실시간 시장 데이터로 포트폴리오 수익률을 추적 중입니다...**")
                prog = st.progress(0)
                
                for i, row in target_df.iterrows():
                    ticker = str(row['코드']).zfill(6)
                    try:
                        df_today = fdr.DataReader(ticker, today_str)
                        # 🌟 [버그 수정] 거래량(Volume)이 0이거나 시가(Open)가 0원이면 거래정지로 판별!
                        if not df_today.empty and df_today['Open'].iloc[-1] > 0 and df_today['Volume'].iloc[-1] > 0:
                            buy_price = df_today['Open'].iloc[-1]
                            curr_price = df_today['Close'].iloc[-1]
                        else:
                            # 거래정지 종목은 살 수 없으므로, 돈을 현금으로 남겨두고 다음 종목으로 패스(Skip)
                            prog.progress((i + 1) / num_stocks)
                            continue 
                    except:
                        prog.progress((i + 1) / num_stocks)
                        continue
                    
                    quantity = alloc_per_stock // buy_price
                    invested = quantity * buy_price
                    current_val = quantity * curr_price
                    pnl_pct = ((curr_price - buy_price) / buy_price) * 100
                    
                    total_current_value += current_val
                    total_invested += invested
                    
                    portfolio.append({
                        "시장": row.get('시장', '-'), "종목명": row['종목명'], "AI확률": f"{row['최종확률']:.1f}%",
                        "매수가(09:00)": int(buy_price), "현재가": int(curr_price),
                        "보유수량": int(quantity), "투자금액": int(invested),
                        "평가금액": int(current_val), "수익률": pnl_pct
                    })
                    prog.progress((i + 1) / num_stocks)
                
                prog.empty()
                total_pnl_krw = total_current_value - total_invested
                final_balance = total_current_value + (INITIAL_CAPITAL - total_invested) 
        else:
            st.warning("아직 사전 분석 결과 파일(CSV)이 없습니다.")

        # 3. ⏰ 실시간 시뮬레이션 및 장 마감 판별 (스트림릿은 더 이상 파일을 저장하지 않음!)
        if now.hour >= 15:
            if not target_df.empty: st.error(f"**15:00 장 마감 (청산 완료):** 모든 포트폴리오가 자동 매도되어 오늘 잔고가 확정되었습니다.")
            status_text = "최종 실현 손익"
            
            # 장 마감 후에는 스트림릿이 임의로 차트에 오늘 데이터를 그리지 않고 깃허브 봇이 만든 파일만 읽습니다.
        else:
            if not target_df.empty: st.success(f"🟢 **장중 실시간 추적 중 ({now.strftime('%H:%M')}):** 오후 4시에 깃허브 봇이 장부에 확정 기록합니다.")
            status_text = "실시간 평가 손익"

        # 4. 📈 연간 누적 수익률 차트 및 메트릭 출력
        ytd_pnl = final_balance - 10000000
        ytd_pnl_pct = (ytd_pnl / 10000000) * 100

        st.markdown("---")
        st.subheader(f"📈 {curr_year}년 누적 자산 성장 곡선 (YTD: {ytd_pnl_pct:+.2f}%)")
        
        plot_df = hist_this_year.copy()
        
        # 🌟 장중(15시 이전)이거나, 깃허브 봇이 아직 장부를 업데이트하기 전이라면 화면에만 가상으로 오늘 점을 찍어줌
        if today_str not in plot_df['Date'].dt.strftime('%Y-%m-%d').values:
            temp_record = pd.DataFrame([{'Date': pd.to_datetime(today_str), 'Balance': final_balance}])
            plot_df = pd.concat([plot_df, temp_record], ignore_index=True)

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=plot_df['Date'], y=plot_df['Balance'], mode='lines+markers', line=dict(color='#00ffcc', width=3),
            marker=dict(size=6, color='white'), hovertemplate="%{x|%Y-%m-%d}<br>잔고: %{y:,.0f}원<extra></extra>"
        ))
        
        # Y축 범위를 현재 잔고 수준에 맞게 다이나믹하게 조절
        min_bal, max_bal = plot_df['Balance'].min(), plot_df['Balance'].max()
        if min_bal == max_bal: min_bal, max_bal = 9000000, 11000000
        
        fig_eq.update_layout(
            height=350, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(tickformat=",", ticksuffix="원", range=[min_bal * 0.98, max_bal * 1.02]),
            xaxis=dict(tickformat="%m/%d")
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # 요약 상황판
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("오늘의 시작 원금", f"{int(INITIAL_CAPITAL):,} 원")
        col2.metric("오늘 배팅된 금액", f"{int(total_invested):,} 원")
        col3.metric("현재 총 자산", f"{int(final_balance):,} 원", f"{ytd_pnl_pct:+.2f}% (연 누적)")
        total_pnl_pct = (total_pnl_krw / total_invested * 100) if total_invested > 0 else 0
        col4.metric(f"오늘 {status_text}", f"{int(total_pnl_krw):,} 원", f"{total_pnl_pct:+.2f}% (일일)")

        if portfolio:
            st.markdown("---")
            st.subheader("📋 오늘 포트폴리오 실시간 내역")
            pf_df = pd.DataFrame(portfolio)
            formatted_df = pf_df.style.format({
                "매수가(09:00)": "{:,}원", "현재가": "{:,}원", "투자금액": "{:,}원",
                "평가금액": "{:,}원", "수익률": "{:+.2f}%"
            }).map(lambda x: 'color: #FF4B4B; font-weight: bold' if x > 0 else ('color: #1C83E1' if x < 0 else 'color: gray'), subset=["수익률"])
            st.dataframe(formatted_df, use_container_width=True)
