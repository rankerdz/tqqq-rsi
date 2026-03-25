import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_market_calendars as mcal


# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="RSI 기반 예상 종가 앱",
    page_icon="📈",
    layout="wide",
)

NY_TZ = ZoneInfo("America/New_York")
KST_TZ = ZoneInfo("Asia/Seoul")


# =========================
# 유틸
# =========================
@st.cache_data(ttl=300)
def get_us_market_calendar():
    return mcal.get_calendar("XNYS")


def get_now_ny():
    return datetime.now(NY_TZ)


def format_date(d):
    if pd.isna(d) or d is None:
        return ""
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def parse_target_rsis(text: str):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
            if 0 < v < 100:
                values.append(v)
        except Exception:
            pass
    values = sorted(set(values))
    return values


# =========================
# 거래일/장마감 기준 처리
# =========================
@st.cache_data(ttl=300)
def get_schedule(start_date, end_date):
    cal = mcal.get_calendar("XNYS")
    return cal.schedule(start_date=start_date, end_date=end_date).copy()


def get_latest_confirmed_session_date(now_ny: datetime):
    """
    미국 본장 종료 기준으로 '확정된 최신 실제 종가 날짜' 반환.
    - 오늘이 거래일이 아니면: 오늘 이전 마지막 거래일
    - 오늘이 거래일인데 아직 장 종료 전이면: 전 거래일
    - 오늘이 거래일이고 장 종료 후면: 오늘
    """
    today = now_ny.date()
    start = today - timedelta(days=15)
    end = today + timedelta(days=5)

    schedule = get_schedule(start, end)
    if schedule.empty:
        return None

    session_dates = [idx.date() for idx in schedule.index]

    today_schedule = schedule.loc[schedule.index.date == today]

    if len(today_schedule) == 0:
        past_sessions = [d for d in session_dates if d < today]
        return max(past_sessions) if past_sessions else None

    market_close = today_schedule.iloc[0]["market_close"].tz_convert(NY_TZ)

    if now_ny < market_close:
        past_sessions = [d for d in session_dates if d < today]
        return max(past_sessions) if past_sessions else None

    return today


def get_next_session_date(ref_date):
    """
    ref_date 다음 미국 거래일 반환
    """
    start = ref_date
    end = ref_date + timedelta(days=15)
    schedule = get_schedule(start, end)
    if schedule.empty:
        return None

    future_sessions = [idx.date() for idx in schedule.index if idx.date() > ref_date]
    return min(future_sessions) if future_sessions else None


# =========================
# 데이터 로드
# =========================
@st.cache_data(ttl=300)
def load_price_data(ticker: str, period="10y", interval="1d"):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # yfinance가 멀티인덱스로 줄 때 대응
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df.index = pd.to_datetime(df.index)

    # timezone-aware면 뉴욕 기준으로 정리
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(NY_TZ).tz_localize(None)

    return df


# =========================
# RSI 계산 (Wilder 방식)
# =========================
def add_rsi_wilder(df: pd.DataFrame, period: int = 14):
    out = df.copy()
    close = out["Close"].astype(float)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = pd.Series(index=out.index, dtype="float64")
    avg_loss = pd.Series(index=out.index, dtype="float64")

    if len(out) <= period:
        out["RSI"] = np.nan
        out["AvgGain"] = np.nan
        out["AvgLoss"] = np.nan
        return out

    first_avg_gain = gain.iloc[1:period + 1].mean()
    first_avg_loss = loss.iloc[1:period + 1].mean()

    avg_gain.iloc[period] = first_avg_gain
    avg_loss.iloc[period] = first_avg_loss

    for i in range(period + 1, len(out)):
        avg_gain.iloc[i] = ((avg_gain.iloc[i - 1] * (period - 1)) + gain.iloc[i]) / period
        avg_loss.iloc[i] = ((avg_loss.iloc[i - 1] * (period - 1)) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # avg_loss == 0 이면 RSI = 100 처리
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    # avg_gain == 0 and avg_loss == 0 이면 중립 50 처리
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)

    out["AvgGain"] = avg_gain
    out["AvgLoss"] = avg_loss
    out["RSI"] = rsi

    return out


# =========================
# 최신 실제값 찾기
# =========================
def prepare_daily_df(df: pd.DataFrame):
    out = df.copy()
    out["_trade_date"] = pd.to_datetime(out.index).date
    return out


def get_latest_actual_row(df: pd.DataFrame, now_ny: datetime):
    """
    미국 장 종료 기준으로 확정된 최신 실제 행 반환
    """
    latest_confirmed_date = get_latest_confirmed_session_date(now_ny)
    if latest_confirmed_date is None:
        return None

    daily = prepare_daily_df(df)
    eligible = daily[daily["_trade_date"] <= latest_confirmed_date].copy()

    if eligible.empty:
        return None

    return eligible.iloc[-1]


# =========================
# 목표 RSI 달성을 위한 다음 거래일 예상 종가 계산
# Wilder smoothing 역산
# =========================
def price_for_target_rsi_next_day(latest_close, prev_avg_gain, prev_avg_loss, period, target_rsi):
    """
    다음 거래일 종가가 얼마가 되면 target RSI가 되는지 계산.
    반환값: 예상 종가
    """
    latest_close = safe_float(latest_close)
    prev_avg_gain = safe_float(prev_avg_gain)
    prev_avg_loss = safe_float(prev_avg_loss)
    target_rsi = safe_float(target_rsi)

    if None in [latest_close, prev_avg_gain, prev_avg_loss, target_rsi]:
        return None

    if not (0 < target_rsi < 100):
        return None

    n = period
    a = (n - 1) * prev_avg_gain
    b = (n - 1) * prev_avg_loss

    target_rs = target_rsi / (100 - target_rsi)

    # 상승 케이스(delta >= 0)
    # ((a + delta) / n) / (b / n) = target_rs
    # a + delta = target_rs * b
    delta_up = target_rs * b - a

    # 하락 케이스(delta < 0)
    # (a / n) / ((b + (-delta)) / n) = target_rs
    # a / (b - delta) = target_rs
    # b - delta = a / target_rs
    # delta = b - a / target_rs
    delta_down = b - (a / target_rs)

    candidates = []

    if delta_up >= 0:
        candidates.append(delta_up)

    if delta_down < 0:
        candidates.append(delta_down)

    if not candidates:
        # 경계 근처 수치오차 방지
        if abs(delta_up) < 1e-9:
            candidates.append(0.0)
        elif abs(delta_down) < 1e-9:
            candidates.append(0.0)
        else:
            return None

    # 일반적으로 하나만 맞음
    delta = min(candidates, key=lambda x: abs(x))
    predicted_close = latest_close + delta

    return predicted_close


# =========================
# UI
# =========================
st.title("📈 RSI 기반 예상 종가 앱")
st.caption("미국 본장 종료 기준으로 최신 실제 종가를 판단하고, 그 다음 거래일의 목표 RSI별 예상 종가를 계산합니다.")

with st.sidebar:
    st.header("설정")

    default_tickers = "QQQ, TQQQ, SOXL, NVDA, TSLA"
    ticker_text = st.text_input("종목 입력 (쉼표로 구분)", value=default_tickers)
    tickers = [x.strip().upper() for x in ticker_text.split(",") if x.strip()]

    rsi_period = st.number_input("RSI 기간", min_value=2, max_value=50, value=14, step=1)

    target_rsi_text = st.text_input(
        "목표 RSI 목록",
        value="30, 35, 40, 45, 50, 55, 60, 65, 70"
    )
    target_rsis = parse_target_rsis(target_rsi_text)

    data_period = st.selectbox(
        "데이터 조회 기간",
        options=["2y", "5y", "10y", "max"],
        index=2
    )

    st.markdown("---")
    refresh = st.button("새로고침")


if refresh:
    st.cache_data.clear()

if not tickers:
    st.warning("종목을 하나 이상 입력해줘.")
    st.stop()

if not target_rsis:
    st.warning("유효한 목표 RSI를 하나 이상 입력해줘. 예: 30, 40, 50, 60, 70")
    st.stop()

now_ny = get_now_ny()
latest_confirmed_session = get_latest_confirmed_session_date(now_ny)

top_col1, top_col2, top_col3 = st.columns(3)
top_col1.metric("현재 한국시간", datetime.now(KST_TZ).strftime("%Y-%m-%d %H:%M:%S"))
top_col2.metric("현재 뉴욕시간", now_ny.strftime("%Y-%m-%d %H:%M:%S"))
top_col3.metric(
    "최신 실제 확정 거래일",
    latest_confirmed_session.strftime("%Y-%m-%d") if latest_confirmed_session else "-"
)

prediction_rows = []
actual_rows = []
error_tickers = []

for ticker in tickers:
    try:
        raw = load_price_data(ticker, period=data_period, interval="1d")

        if raw.empty or "Close" not in raw.columns:
            error_tickers.append(ticker)
            continue

        df = add_rsi_wilder(raw, period=int(rsi_period))
        df = df.dropna(subset=["Close"]).copy()

        latest_row = get_latest_actual_row(df, now_ny)
        if latest_row is None:
            error_tickers.append(ticker)
            continue

        latest_actual_date = latest_row["_trade_date"]
        latest_actual_close = safe_float(latest_row["Close"])
        latest_actual_rsi = safe_float(latest_row["RSI"])
        latest_avg_gain = safe_float(latest_row["AvgGain"])
        latest_avg_loss = safe_float(latest_row["AvgLoss"])

        next_session = get_next_session_date(latest_actual_date)

        actual_rows.append({
            "종목": ticker,
            "실제 기준일": latest_actual_date,
            "실제 종가": latest_actual_close,
            "실제 RSI": latest_actual_rsi,
        })

        for target_rsi in target_rsis:
            predicted_close = price_for_target_rsi_next_day(
                latest_close=latest_actual_close,
                prev_avg_gain=latest_avg_gain,
                prev_avg_loss=latest_avg_loss,
                period=int(rsi_period),
                target_rsi=target_rsi,
            )

            diff_pct = None
            if predicted_close is not None and latest_actual_close not in [None, 0]:
                diff_pct = ((predicted_close / latest_actual_close) - 1) * 100

            prediction_rows.append({
                "종목": ticker,
                "예측 기준일": next_session,
                "목표 RSI": float(target_rsi),
                "예상 종가": predicted_close,
                "현재가 대비(%)": diff_pct,
            })

    except Exception:
        error_tickers.append(ticker)


prediction_df = pd.DataFrame(prediction_rows)
actual_df = pd.DataFrame(actual_rows)

if actual_df.empty:
    st.error("데이터를 불러오지 못했어. 종목 코드를 확인해줘.")
    st.stop()


# =========================
# 표시용 포맷
# =========================
if not prediction_df.empty:
    prediction_df = prediction_df.sort_values(["종목", "목표 RSI"]).reset_index(drop=True)
    prediction_df["예측 기준일"] = prediction_df["예측 기준일"].apply(format_date)
    prediction_df["목표 RSI"] = prediction_df["목표 RSI"].map(lambda x: f"{x:.0f}")
    prediction_df["예상 종가"] = prediction_df["예상 종가"].map(
        lambda x: f"{x:,.2f}" if pd.notnull(x) else ""
    )
    prediction_df["현재가 대비(%)"] = prediction_df["현재가 대비(%)"].map(
        lambda x: f"{x:+.2f}%" if pd.notnull(x) else ""
    )

actual_df = actual_df.sort_values(["종목"]).reset_index(drop=True)
actual_df["실제 기준일"] = actual_df["실제 기준일"].apply(format_date)
actual_df["실제 종가"] = actual_df["실제 종가"].map(
    lambda x: f"{x:,.2f}" if pd.notnull(x) else ""
)
actual_df["실제 RSI"] = actual_df["실제 RSI"].map(
    lambda x: f"{x:.2f}" if pd.notnull(x) else ""
)


# =========================
# 화면 출력
# =========================
st.markdown("---")
st.subheader("목표 RSI별 예상 종가")
st.caption("예측 기준일은 미국 본장 종료 기준 최신 실제 종가의 다음 거래일입니다.")

if prediction_df.empty:
    st.info("예상 종가 결과가 없습니다.")
else:
    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")
st.subheader("최신 실제 종가 / RSI")
st.caption("실제 기준일은 미국 정규장 종료 기준으로 확정된 최신 거래일입니다.")

st.dataframe(
    actual_df,
    use_container_width=True,
    hide_index=True,
)

if error_tickers:
    st.markdown("---")
    st.warning("일부 종목은 불러오지 못했어: " + ", ".join(sorted(set(error_tickers))))

st.markdown("---")
st.caption(
    "주의: 예상 종가는 RSI 수학식 기준의 역산값이며, 실제 시장 예측치가 아니라 "
    "다음 거래일 종가가 특정 RSI가 되기 위해 필요한 이론적 가격입니다."
)