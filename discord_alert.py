import os
import argparse
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import pytz
import ta
from sklearn.preprocessing import RobustScaler
import FinanceDataReader as fdr
from bs4 import BeautifulSoup
import joblib
import warnings
import concurrent.futures
import random
import time
import sys
import holidays
import lightgbm as lgb

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# 🌟 V7 다중 분류 마스터 AI 모델 구조 (3 Class)
class SwingMasterGRU_V7(nn.Module):
    # 🌟 [수정 완료] 입력 사이즈 14로 변경
    def __init__(self, input_size=14, hidden_size=128, num_layers=2):
        super(SwingMasterGRU_V7, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 3))
        
    def forward(self, x):
        out, _ = self.gru(x)
        w = torch.softmax(torch.tanh(self.attention(out)), dim=1)
        c = torch.sum(w * out, dim=1)
        return self.fc(c)
        
# 🌟 설정 변수
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
GRU_PATH = "weather_advisor_v7_sniper.pt" 
LGB_PATH = "weather_advisor_v7_sniper_lgb.pkl"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_ensemble_models():
    model_gru = SwingMasterGRU_V7(input_size=24)
    try:
        model_gru.load_state_dict(torch.load(GRU_PATH, map_location=device, weights_only=True))
        model_gru.to(device).eval()
        model_lgb = joblib.load(LGB_PATH)
        return model_gru, model_lgb
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return None, None

def send_discord(title, fields_data, color):
    if not DISCORD_WEBHOOK_URL:
        print("❌ 웹훅 주소가 없습니다!")
        return
        
    payload = {
        "content": "📢 **[AI Quant V7 Sniper]**",
        "embeds": [{
            "title": title,
            "description": f"기준일시: {datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M')} (KST)",
            "color": color, 
            "fields": fields_data,
            "footer": {"text": "V7 앙상블 스나이퍼 (다중분류 필터링 적용)"}
        }]
    }
    requests.post(DISCORD_WEBHOOK_URL, json=payload, headers={"Content-Type": "application/json"})

# --- 데이터 파이프라인 ---
def load_macro_feature_data():
    end_dt = datetime.today().strftime('%Y-%m-%d')
    start_dt = (datetime.today() - timedelta(days=200)).strftime('%Y-%m-%d')
    usdkrw = fdr.DataReader('USD/KRW', start_dt, end_dt)['Close'].rename('usd_krw')
    nasdaq = fdr.DataReader('IXIC', start_dt, end_dt)['Close'].rename('nasdaq')
    kospi = fdr.DataReader('KS11', start_dt, end_dt)['Close'].rename('kospi')
    kosdaq = fdr.DataReader('KQ11', start_dt, end_dt)['Close'].rename('kosdaq')
    vix = fdr.DataReader('VIX', start_dt, end_dt)['Close'].rename('vix')
    
    macro_df = pd.concat([usdkrw, nasdaq, kospi, kosdaq, vix], axis=1).ffill().dropna()
    macro_df['usd_krw_ret'] = macro_df['usd_krw'].pct_change()
    macro_df['nasdaq_ret'] = macro_df['nasdaq'].pct_change()
    macro_df['kospi_ret'] = macro_df['kospi'].pct_change()
    macro_df['kosdaq_ret'] = macro_df['kosdaq'].pct_change()
    macro_df['vix_ret'] = macro_df['vix'].pct_change()
    return macro_df[['usd_krw_ret', 'nasdaq_ret', 'kospi_ret', 'kosdaq_ret', 'vix_ret']]

def get_naver_supply_demand_history(code, pages=4):
    records = []
    for p in range(1, pages + 1):
        try:
            url = f"https://finance.naver.com/item/frgn.naver?code={code}&page={p}"
            res = requests.get(url, headers={'User-agent': 'Mozilla/5.0'}, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            tables = soup.find_all('table', class_='type2')
            if len(tables) < 2: continue
            
            rows = tables[1].find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 9 and cols[0].text.strip():
                    date_str = cols[0].text.strip().replace('.', '-')
                    inst_str = cols[5].text.strip().replace(',', '').replace('+', '')
                    for_str = cols[6].text.strip().replace(',', '').replace('+', '')
                    try:
                        records.append({
                            'Date': pd.to_datetime(date_str),
                            'inst_net': int(inst_str),
                            'foreigner_net': int(for_str)
                        })
                    except: pass
        except: continue
        
    if records:
        return pd.DataFrame(records).set_index('Date').sort_index()
    return pd.DataFrame()

def get_naver_market_data(group_type="upjong", count=50):
    headers = {'User-Agent': 'Mozilla/5.0'}
    data, page, seen_codes = [], 1, set()
    while True:
        list_url = "https://finance.naver.com/sise/sise_group.naver?type=upjong" if group_type == "upjong" else f"https://finance.naver.com/sise/theme.naver?page={page}"
        try:
            res = requests.get(list_url, headers=headers, timeout=10)
            res.encoding = 'euc-kr'
            soup = BeautifulSoup(res.text, 'html.parser')
            table = soup.select_one('table.type_1')
            if not table: break

            rows = table.select('tr')
            page_item_count, is_duplicate_page = 0, False
            for row in rows:
                cols = row.select('td')
                if len(cols) >= 2:
                    link_tag = cols[0].find('a')
                    if link_tag and 'no=' in link_tag.get('href', ''):
                        code = link_tag['href'].split('no=')[-1].split('&')[0] 
                        if code in seen_codes:
                            is_duplicate_page = True; break
                        seen_codes.add(code)
                        name = link_tag.text.strip()
                        change_text = cols[1].text.strip().replace('%', '').replace('+', '')
                        try: change_val = float(change_text)
                        except: change_val = 0.0
                        data.append({"이름": name, "등락률": change_val, "code": code})
                        page_item_count += 1
            if group_type == "upjong" or page_item_count == 0 or is_duplicate_page: break
            if page > 15: break
            page += 1
        except: break

    full_df = pd.DataFrame(data).sort_values("등락률", ascending=False).reset_index(drop=True)
    final_list = []
    for i in range(min(count, len(full_df))):
        row = full_df.iloc[i]
        detail_url = f"https://finance.naver.com/sise/sise_group_detail.naver?type={group_type}&no={row['code']}"
        try:
            d_res = requests.get(detail_url, headers=headers, timeout=5)
            d_res.encoding = 'euc-kr'
            d_soup = BeautifulSoup(d_res.text, 'html.parser')
            stock_table = d_soup.select_one('table.type_5')
            max_change, min_change, top_name, bottom_name = -999.0, 999.0, "-", "-"
            target_td_idx = 3 if group_type == "upjong" else 4
            
            if stock_table:
                for s_row in stock_table.select('tr'):
                    name_cell = s_row.select_one('td.name a')
                    tds = s_row.select('td')
                    if name_cell and len(tds) > target_td_idx:
                        s_name = name_cell.text.strip()
                        change_text = tds[target_td_idx].text.strip().replace('%', '').replace('+', '').replace(',', '')
                        if not change_text: continue 
                        try:
                            s_change = float(change_text)
                            if s_change > max_change: max_change, top_name = s_change, s_name
                            if s_change < min_change: min_change, bottom_name = s_change, s_name
                        except: continue
            if max_change == -999.0: max_change = 0.0
            if min_change == 999.0: min_change = 0.0
            final_list.append({"이름": row['이름'], "등락률": row['등락률'], "1등주(대장)": top_name, "1등 수익률": max_change, "꼴등주(부진)": bottom_name, "꼴등 수익률": min_change})
        except: continue
    return full_df, pd.DataFrame(final_list)

def extract_features_v7(ticker, df_chart, macro_df):
    if len(df_chart) < 60: return pd.DataFrame()
    
    sd_df = get_naver_supply_demand_history(ticker, pages=4)
    df = df_chart.copy().join(macro_df, how='left')
    
    if not sd_df.empty: df = df.join(sd_df, how='left')
    else: df['inst_net'] = 0; df['foreigner_net'] = 0
        
    df.ffill(inplace=True); df.bfill(inplace=True)

    close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']
    feats = pd.DataFrame(index=df.index)
    
    # 🌟 [수정 완료] 정확히 14개의 피처만 남김!
    feats['ret'] = close.pct_change()
    feats['dist_ma'] = close / (close.rolling(20).mean() + 1e-8)
    feats['macd_hist'] = ta.trend.MACD(close).macd_diff()
    feats['rsi'] = ta.momentum.RSIIndicator(close).rsi() / 100.0
    feats['stoch'] = ta.momentum.StochasticOscillator(high, low, close).stoch() / 100.0
    feats['bb_pband'] = ta.volatility.BollingerBands(close).bollinger_pband()
    feats['atr_pct'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range() / (close + 1e-8)
    feats['mfi'] = ta.volume.MFIIndicator(high, low, close, vol).money_flow_index() / 100.0
    feats['cci'] = ta.trend.CCIIndicator(high, low, close).cci() / 100.0
    feats['will_r'] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r() / -100.0
    
    for col in ['usd_krw_ret', 'nasdaq_ret', 'kospi_ret', 'vix_ret']: 
        if col in df.columns: feats[col] = df[col]
        else: feats[col] = 0.0
    
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats.dropna(inplace=True)
    
    feature_cols = [
        'ret', 'dist_ma', 'macd_hist', 'rsi', 'stoch', 'bb_pband', 'atr_pct', 'mfi', 'cci', 'will_r',
        'usd_krw_ret', 'nasdaq_ret', 'kospi_ret', 'vix_ret'
    ]
    return feats[feature_cols]
def process_single_ticker(ticker, name, market, mode, macro_df, model_gru, model_lgb):
    try:
        time.sleep(random.uniform(0.1, 0.5))
        df = fdr.DataReader(ticker, (datetime.now() - pd.Timedelta(days=150)).strftime('%Y-%m-%d'))
        
        # 🌟 거래정지 방어막
        if df.empty or df['Volume'].iloc[-1] == 0 or df['Open'].iloc[-1] == 0:
            return None
        
        if mode == "afternoon" and len(df) > 2: pred_df = df.iloc[:-1] 
        else: pred_df = df
            
        f_df = extract_features_v7(ticker, pred_df, macro_df)
        if f_df.empty or len(f_df) < 60: return None
        
        scaled = RobustScaler().fit_transform(f_df.tail(60).values)
        input_t = torch.FloatTensor(scaled).unsqueeze(0).to(device)
        
        # 🌟 V7 다중분류 해석 (Class 2: 강력매수 확률 추출)
        with torch.no_grad():
            gru_probs = torch.softmax(model_gru(input_t), dim=1).cpu().numpy()[0]
        lgb_probs = model_lgb.predict(scaled[-1].reshape(1, -1))[0]
        
        ensemble_probs = (gru_probs * 0.5) + (lgb_probs * 0.5)
        prob_sell = ensemble_probs[0] * 100
        prob_hold = ensemble_probs[1] * 100
        buy_prob = ensemble_probs[2] * 100 # 대시보드와 맞추기 위해 '최종확률'로 사용
        
        curr_price = int(pred_df['Close'].iloc[-1])
        tp_price = int(curr_price * 1.04)
        sl_price = int(curr_price * 0.97)
        
        res_dict = {
            "시장": market, "종목명": name, "코드": ticker,
            "최종확률": buy_prob, # 스윙타점 스캐너 호환용
            "절대금지확률": prob_sell, "관망확률": prob_hold, 
            "예측시점가격": curr_price, "목표가": tp_price, "손절가": sl_price
        }
        if mode == "afternoon": res_dict["오늘종가"] = int(df['Close'].iloc[-1])
            
        return res_dict
    except: return None

# --- 메인 스캐너 및 알람 실행 로직 ---
def run_scanner(mode="morning_scan"):
    if mode == "morning_alert":
        print("🚀 [08:33] 장 시작 전 디스코드 알람 전송 시작...")
        try:
            rank_df = pd.read_csv("morning_scan_result.csv")
            s_class = rank_df[rank_df["최종확률"] >= 70.0].sort_values("최종확률", ascending=False)
            a_class = rank_df[(rank_df["최종확률"] >= 60.0) & (rank_df["최종확률"] < 70.0)].sort_values("최종확률", ascending=False)
            
            fields = []
            if not s_class.empty:
                fields.append({"name": "🔥 **[S급] 초고도 확신 타점 (승률 최상위)**", "value": "적극적인 비중 베팅을 고려할 만한 강력한 상승 신호입니다.", "inline": False})
                fields.extend([{"name": f"🎯 [{row['시장']}] {row['종목명']}", "value": f"강력매수 확률: **{row['최종확률']:.1f}%**\n💵 적정가: `{row['예측시점가격']:,}원`\n🚀 목표가: `{row['목표가']:,}원` (+4%)\n🛑 손절가: `{row['손절가']:,}원` (-3%)", "inline": False} for _, row in s_class.head(5).iterrows()])
                
            if not a_class.empty:
                fields.append({"name": "🚀 **[A급] 강한 확신 타점 (승률 60%↑)**", "value": "매수 우위 구간입니다. 수급과 호가를 체크하며 진입하세요.", "inline": False})
                fields.extend([{"name": f"✅ [{row['시장']}] {row['종목명']}", "value": f"강력매수 확률: **{row['최종확률']:.1f}%**\n💵 적정가: `{row['예측시점가격']:,}원`\n🚀 목표가: `{row['목표가']:,}원` (+4%)\n🛑 손절가: `{row['손절가']:,}원` (-3%)", "inline": False} for _, row in a_class.head(5).iterrows()])
                
            if not fields: fields.append({"name": "🛑 **관망 권장**", "value": "오늘 장은 60% 이상 확신할 만한 S급/A급 매수 타점이 포착되지 않았습니다.", "inline": False})
                
            send_discord("🌅 [08:33] V7 전 종목 스캔 주도주 브리핑", fields, 15158332)
            print("✅ 알람 전송 완료")
        except Exception as e: print(f"❌ 알람 전송 실패: {e}")
        return 

    model_gru, model_lgb = load_ensemble_models()
    if not model_gru or not model_lgb: return
    
    try:
        df_krx = fdr.StockListing('KRX')
        df_kospi = df_krx[df_krx['Market'] == 'KOSPI'].sort_values('Amount', ascending=False).head(500)
        df_kosdaq = df_krx[df_krx['Market'] == 'KOSDAQ'].sort_values('Amount', ascending=False).head(500)
        df_list = pd.concat([df_kospi, df_kosdaq])
        tickers, names, markets = df_list['Code'].tolist(), df_list['Name'].tolist(), df_list['Market'].tolist()
    except Exception as e: 
        print(f"종목 리스트 로드 에러: {e}"); return
        
    macro_df, results = load_macro_feature_data(), []
    print(f"🚀 총 {len(tickers)}개 종목 (상위 1,000개) 멀티스레드 딥스캔 시작...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single_ticker, t, n, m, mode, macro_df, model_gru, model_lgb): t for t, n, m in zip(tickers, names, markets)}
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: results.append(res)
            completed_count += 1
            if completed_count % 100 == 0: print(f"🔄 병렬 분석 진행 중... [{completed_count} / {len(tickers)}] 완료", flush=True)

    rank_df = pd.DataFrame(results)
    if rank_df.empty: return

    # 🌟 [새벽 모드] 데이터 수집 및 CSV 굽기
    if mode == "morning_scan":
        rank_df.to_csv("morning_scan_result.csv", index=False, encoding='utf-8-sig')
        print("✅ [1/3] V7 스캔 결과 CSV 저장 완료")

        try:
            _, detail_up = get_naver_market_data("upjong", 76)
            detail_up.to_csv("sector_upjong.csv", index=False, encoding='utf-8-sig')
            _, detail_th = get_naver_market_data("theme", 264)
            detail_th.to_csv("sector_theme.csv", index=False, encoding='utf-8-sig')
            print("✅ [2/3] 섹터/테마 CSV 저장 완료")
        except: pass

        try:
            etf_list = fdr.StockListing('ETF/KR')
            etf_tickers = etf_list.sort_values('Volume', ascending=False).head(20)['Symbol'].tolist()
            etf_names = etf_list.sort_values('Volume', ascending=False).head(20)['Name'].tolist()
            etf_markets = ["ETF"] * len(etf_tickers)
            etf_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as etf_executor:
                etf_futures = [etf_executor.submit(process_single_ticker, t, n, m, mode, macro_df, model_gru, model_lgb) for t, n, m in zip(etf_tickers, etf_names, etf_markets)]
                for future in concurrent.futures.as_completed(etf_futures):
                    res = future.result()
                    if res: etf_results.append(res)
            pd.DataFrame(etf_results).to_csv("etf_scanner_result.csv", index=False, encoding='utf-8-sig')
            print("✅ [3/3] ETF CSV 저장 완료")
        except: pass

    # 🌟 [오후 모드] 장 마감 복기 및 모의투자 장부 업데이트
    elif mode == "afternoon":
        picks = rank_df[rank_df["최종확률"] >= 60.0].sort_values("최종확률", ascending=False)
        bottom_picks = rank_df.sort_values("최종확률", ascending=True).head(10)
        fields = []

        # 모의투자 장부 작성
        HISTORY_CSV = "mock_invest_history.csv"
        now = datetime.now(pytz.timezone('Asia/Seoul'))
        today_str, curr_year = now.strftime('%Y-%m-%d'), now.year
        START_DATE = pd.to_datetime("2026-05-01") if curr_year == 2026 else pd.to_datetime(f"{curr_year}-01-01")

        if os.path.exists(HISTORY_CSV):
            hist_df = pd.read_csv(HISTORY_CSV)
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
            if hist_df.empty: hist_df = pd.DataFrame([{'Date': START_DATE, 'Invested': 0, 'PnL': 0, 'Balance': 10000000}])
        else: hist_df = pd.DataFrame([{'Date': START_DATE, 'Invested': 0, 'PnL': 0, 'Balance': 10000000}])
        
        hist_before_today = hist_df[hist_df['Date'].dt.strftime('%Y-%m-%d') < today_str]
        INITIAL_CAPITAL = 10000000 if hist_before_today.empty else hist_before_today.iloc[-1]['Balance']

        total_invested, total_pnl_krw = 0, 0
        if not picks.empty:
            alloc_per_stock = INITIAL_CAPITAL // len(picks)
            for _, row in picks.iterrows():
                buy_price, curr_price = row['예측시점가격'], row['오늘종가']
                if buy_price > 0:
                    quantity = alloc_per_stock // buy_price
                    invested = quantity * buy_price
                    current_val = quantity * curr_price
                    total_invested += invested
                    total_pnl_krw += (current_val - invested)
                    
        final_balance = INITIAL_CAPITAL + total_pnl_krw
        if today_str not in hist_df['Date'].dt.strftime('%Y-%m-%d').values:
            hist_df = pd.concat([hist_df, pd.DataFrame([{'Date': pd.to_datetime(today_str), 'Invested': total_invested, 'PnL': total_pnl_krw, 'Balance': final_balance}])], ignore_index=True)
        else:
            idx = hist_df.index[hist_df['Date'].dt.strftime('%Y-%m-%d') == today_str].tolist()[0]
            hist_df.loc[idx, ['Invested', 'PnL', 'Balance']] = [total_invested, total_pnl_krw, final_balance]
        
        hist_df.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')
        print("✅ [자동 모의투자] V7 수익률 장부(CSV) 기록 완료")
        
        if picks.empty:
            top_pick = rank_df.sort_values("최종확률", ascending=False).head(1)
            row = top_pick.iloc[0]
            change_pct = ((row['오늘종가'] - row['예측시점가격']) / row['예측시점가격']) * 100
            emoji = "🔴 상승 (놓침)" if change_pct > 0 else ("🔵 하락 (방어성공)" if change_pct < 0 else "⚪ 보합")
            fields.append({"name": "💤 오늘 아침 추천 타점 없음 (관망 채점)", "value": "60% 이상 확신 종목이 없어 매수를 쉬었습니다. 가장 점수가 높았던 종목 결과를 복기합니다.", "inline": False})
            fields.append({"name": f"📝 [{row['시장']}] {row['종목명']} (강력매수 확률: {row['최종확률']:.1f}%)", "value": f"시작가: {row['예측시점가격']:,}원 ➡️ 마감가: {row['오늘종가']:,}원\n관망 결과: {emoji} **({change_pct:+.2f}%)**", "inline": False})
        else:
            hit_count, total_profit = 0, 0
            for i, row in picks.head(10).iterrows():
                change_pct = ((row['오늘종가'] - row['예측시점가격']) / row['예측시점가격']) * 100
                total_profit += change_pct
                if change_pct > 0: hit_count += 1
                emoji = "🔴 적중" if change_pct > 0 else ("🔵 실패" if change_pct < 0 else "⚪ 보합")
                fields.append({"name": f"📝 [{row['시장']}] {row['종목명']} (확률: {row['최종확률']:.1f}%)", "value": f"시작가: {row['예측시점가격']:,}원 ➡️ 마감가: {row['오늘종가']:,}원\n결과: {emoji} **({change_pct:+.2f}%)**", "inline": False})
                
            avg_profit = total_profit / len(picks.head(10))
            win_rate = (hit_count / len(picks.head(10))) * 100
            bottom_profit = sum([((r['오늘종가'] - r['예측시점가격']) / r['예측시점가격']) * 100 for _, r in bottom_picks.iterrows()])
            avg_bottom_profit = bottom_profit / len(bottom_picks) if not bottom_picks.empty else 0
            spread = avg_profit - avg_bottom_profit
            spread_text = "🔥 V7 모델 변별력 우위 (초과수익 달성)" if spread > 0 else "⚠️ 시장 베타(지수) 우위장"

            fields.insert(0, {
                "name": "📊 **[V7 모델 변별력 성적표]**", 
                "value": f"🚀 상위 10픽 평균 수익률: **{avg_profit:+.2f}%** (승률 {win_rate:.0f}%)\n🐢 하위 10픽 평균 수익률: **{avg_bottom_profit:+.2f}%**\n📈 상/하위 수익률 격차: **{spread:+.2f}%p**\n👉 **평가: {spread_text}**\n---", 
                "inline": False
            })
            
        send_discord("🏁 [16:11] 장 마감 복기 및 V7 스나이퍼 채점", fields, 10181046)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["morning_scan", "morning_alert", "afternoon", "monthly"])
    args = parser.parse_args()
    
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    
    if now.weekday() >= 5:
        print(f"[{now.strftime('%Y-%m-%d')}] 주말 휴장. 봇을 종료합니다.")
        sys.exit(0)
        
    kr_holidays = holidays.KR(years=now.year)
    if now.date() in kr_holidays:
        print(f"[{now.strftime('%Y-%m-%d')}] 법정 공휴일. 봇을 종료합니다.")
        sys.exit(0)
        
    if (now.month == 5 and now.day == 1) or (now.month == 12 and now.day == 31):
        print(f"[{now.strftime('%Y-%m-%d')}] KRX 특수 휴장일. 봇을 종료합니다.")
        sys.exit(0)
        
    run_scanner(args.mode)
