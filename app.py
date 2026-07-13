"""
app.py — แอปวิเคราะห์หุ้นไทยด้วย AI (Streamlit)
รันบน Streamlit Community Cloud ได้ฟรี มีลิงก์ถาวร
"""
import os
import json
import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

# ============================================================
#  ตั้งค่าหน้าเพจ
# ============================================================
st.set_page_config(page_title="วิเคราะห์หุ้นไทยด้วย AI",
                   page_icon="📊", layout="wide")

st.markdown("""
<style>
  h1, h2, h3 { letter-spacing: .3px; }
  .pill { display:inline-block; padding:4px 12px; border-radius:999px;
          font-size:13px; font-weight:600; }
  .ok  { background:#dcfce7; color:#15803d; }
  .off { background:#fef3c7; color:#b45309; }
</style>
""", unsafe_allow_html=True)

DEFAULT_UNIVERSE = ["PTT.BK", "AOT.BK", "KBANK.BK", "SCB.BK", "CPALL.BK",
                    "ADVANC.BK", "GULF.BK", "DELTA.BK", "BDMS.BK", "SCC.BK"]

# ชุดใหญ่: ดัชนี SET100 อย่างเป็นทางการ (รอบ 1 ก.ค. – 31 ธ.ค. 2026, อัปเดต 17 มิ.ย. 2026)
# = 100 ตัวสภาพคล่อง/มาร์เก็ตแคปสูงสุด (ครอบ SET50 ไว้แล้ว) · ตัด EA ออกตามต้องการ -> เหลือ 99 ตัว
# SET ปรับรายชื่อปีละ 2 ครั้ง (มีผลต้น ม.ค. และ ต้น ก.ค.) — รอบหน้าแค่มาแทนลิสต์นี้
UNIVERSE_LARGE = [
    "AAV.BK", "ADVANC.BK", "AEONTS.BK", "AMATA.BK", "AOT.BK", "AP.BK",
    "AURA.BK", "AWC.BK", "BA.BK", "BAM.BK", "BANPU.BK", "BBL.BK",
    "BCH.BK", "BCP.BK", "BCPG.BK", "BDMS.BK", "BEM.BK", "BGRIM.BK",
    "BH.BK", "BJC.BK", "BLA.BK", "BTG.BK", "BTS.BK", "CBG.BK",
    "CCET.BK", "CENTEL.BK", "CHG.BK", "CK.BK", "COM7.BK", "CPALL.BK",
    "CPF.BK", "CPN.BK", "CRC.BK", "DELTA.BK", "DOHOME.BK", "EGCO.BK",
    "ERW.BK", "GFPT.BK", "GLOBAL.BK", "GPSC.BK", "GULF.BK", "GUNKUL.BK",
    "HANA.BK", "HMPRO.BK", "ICHI.BK", "IRPC.BK", "IVL.BK", "JMT.BK",
    "JTS.BK", "KBANK.BK", "KCE.BK", "KKP.BK", "KTB.BK", "KTC.BK",
    "LH.BK", "M.BK", "MEGA.BK", "MINT.BK", "MOSHI.BK", "MRDIYT.BK",
    "MTC.BK", "OR.BK", "OSP.BK", "PLANB.BK", "PR9.BK", "PRM.BK",
    "PTG.BK", "PTT.BK", "PTTEP.BK", "PTTGC.BK", "QH.BK", "RATCH.BK",
    "RCL.BK", "SAWAD.BK", "SCB.BK", "SCC.BK", "SCGP.BK", "SIRI.BK",
    "SPALI.BK", "SPRC.BK", "STA.BK", "STECON.BK", "STGT.BK", "TASCO.BK",
    "TCAP.BK", "TFG.BK", "THAI.BK", "THCOM.BK", "TIDLOR.BK", "TISCO.BK",
    "TLI.BK", "TOA.BK", "TOP.BK", "TRUE.BK", "TTB.BK", "TU.BK",
    "VGI.BK", "WHA.BK", "WHAUP.BK",
]

# ชุด SET50 (50 ตัว) — สำหรับสแกนแบบกระชับ เฉพาะเมกะแคปสภาพคล่องสูงสุด
UNIVERSE_SET50 = [
    "ADVANC.BK", "AOT.BK", "AWC.BK", "BANPU.BK", "BBL.BK", "BCP.BK",
    "BDMS.BK", "BEM.BK", "BH.BK", "BJC.BK", "CCET.BK", "COM7.BK",
    "CPALL.BK", "CPF.BK", "CPN.BK", "CRC.BK", "DELTA.BK", "EGCO.BK",
    "GPSC.BK", "GULF.BK", "HMPRO.BK", "IVL.BK", "KBANK.BK", "KKP.BK",
    "KTB.BK", "KTC.BK", "LH.BK", "MINT.BK", "MRDIYT.BK", "MTC.BK",
    "OR.BK", "OSP.BK", "PTT.BK", "PTTEP.BK", "PTTGC.BK", "RATCH.BK",
    "SCB.BK", "SCC.BK", "SCGP.BK", "TCAP.BK", "TFG.BK", "THAI.BK",
    "TIDLOR.BK", "TISCO.BK", "TLI.BK", "TOP.BK", "TRUE.BK", "TTB.BK",
    "TU.BK", "WHA.BK",
]


# ============================================================
#  API key (อ่านจาก Streamlit secrets)
# ============================================================
def get_api_key():
    key = None
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    return key or os.environ.get("ANTHROPIC_API_KEY")


API_KEY = get_api_key()


# ============================================================
#  แหล่งข้อมูลราคา: Settrade Open API (real-time) หรือ yfinance (ดีเลย์)
# ============================================================
def get_settrade_creds():
    names = ["SETTRADE_APP_ID", "SETTRADE_APP_SECRET",
             "SETTRADE_BROKER_ID", "SETTRADE_APP_CODE"]
    vals = {}
    for n in names:
        v = None
        try:
            v = st.secrets.get(n)
        except Exception:
            pass
        vals[n] = v or os.environ.get(n)
    return vals if all(vals.values()) else None


SETTRADE = get_settrade_creds()
DATA_SOURCE = "Settrade Open API (real-time)" if SETTRADE else "yfinance (ดีเลย์ ~15 นาที)"


@st.cache_resource(show_spinner=False)
def get_investor():
    from settrade_v2 import Investor
    c = SETTRADE
    return Investor(app_id=c["SETTRADE_APP_ID"], app_secret=c["SETTRADE_APP_SECRET"],
                    app_code=c["SETTRADE_APP_CODE"], broker_id=c["SETTRADE_BROKER_ID"])


def _pick(d, name):
    """หา key แบบไม่สนตัวพิมพ์เล็ก-ใหญ่ (กันรูปแบบ response ต่างกัน)"""
    for k in d:
        if k.lower() == name:
            return d[k]
    raise KeyError(f"ไม่พบฟิลด์ '{name}' ใน response ของ Settrade")


def _fetch_settrade(symbol, limit=250, interval="1d"):
    sym = symbol.replace(".BK", "").replace("^", "").upper()  # PTT.BK -> PTT, ^SET.BK -> SET
    r = get_investor().MarketData().get_candlestick(symbol=sym, interval=interval, limit=limit)
    idx = pd.to_datetime(_pick(r, "time"), unit="s")
    return pd.DataFrame({"Open": _pick(r, "open"), "High": _pick(r, "high"),
                         "Low": _pick(r, "low"), "Close": _pick(r, "close"),
                         "Volume": _pick(r, "volume")}, index=idx).dropna()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(ticker, period="6mo", interval="1d"):
    if SETTRADE:
        df = _fetch_settrade(ticker)
    else:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=False, progress=False)
        if not df.empty and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]] if not df.empty else df
    if df is None or df.empty:
        raise ValueError(f"ไม่พบข้อมูลของ {ticker}")
    return df.dropna()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_batch(tickers, period="6mo", interval="1d"):
    """ดึงหลายตัวพร้อมกันเป็นก้อนเดียว -> ลดจำนวนครั้งที่เรียก yfinance (กัน rate limit + เบา RAM)
    คืน dict {ticker: DataFrame}; ตัวไหนไม่มีข้อมูลจะไม่อยู่ใน dict"""
    out = {}
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if SETTRADE:  # Settrade ไม่มี batch — ดึงทีละตัว
        for t in tickers:
            try:
                out[t] = _fetch_settrade(t)
            except Exception:
                pass
        return out
    try:
        data = yf.download(list(tickers), period=period, interval=interval,
                           auto_adjust=False, progress=False,
                           group_by="ticker", threads=True)
    except Exception:
        return out
    if data is None or data.empty:
        return out
    for t in tickers:
        try:
            sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            sub = sub[cols].dropna()
            if len(sub) >= 30:
                out[t] = sub
        except Exception:
            pass
    return out


def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def sma(s, w): return s.rolling(w).mean()


def rsi(close, period=14):
    d = close.diff(); gain = d.clip(lower=0); loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100/(1 + ag/al))


def macd(close, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow); sig = ema(line, signal)
    return line, sig, line - sig


def rsi_signal(rsi_series, period=14):
    return rsi_series.rolling(period).mean()


def bollinger(close, period=20, mult=2.0):
    basis = close.rolling(period).mean(); sd = close.rolling(period).std()
    return basis, basis + mult*sd, basis - mult*sd


def atr(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def detect_candlestick(df):
    if len(df) < 2: return []
    o, h, l, c = (float(df[x].iloc[-1]) for x in ("Open", "High", "Low", "Close"))
    po, pc = float(df["Open"].iloc[-2]), float(df["Close"].iloc[-2])
    rng = h - l
    if rng <= 0: return []
    body = abs(c-o); upper = h-max(o, c); lower = min(o, c)-l; out = []
    if body <= 0.1*rng: out.append("Doji (ตลาดลังเล อาจกลับตัว)")
    if body > 0 and lower >= 2*body and upper <= body and body <= 0.4*rng: out.append("Hammer (กลับตัวขาขึ้น)")
    if body > 0 and upper >= 2*body and lower <= body and body <= 0.4*rng: out.append("Shooting Star (กลับตัวขาลง)")
    if pc < po and c > o and o <= pc and c >= po: out.append("Bullish Engulfing (สัญญาณซื้อ)")
    if pc > po and c < o and o >= pc and c <= po: out.append("Bearish Engulfing (สัญญาณขาย)")
    return out


def add_indicators(df):
    out = df.copy(); close = out["Close"]
    out["EMA20"] = ema(close, 20); out["EMA50"] = ema(close, 50)
    out["EMA100"] = ema(close, 100); out["EMA200"] = ema(close, 200)
    out["RSI14"] = rsi(close, 14); out["RSI_SIGNAL"] = rsi_signal(out["RSI14"], 14)
    ml, sl, hi = macd(close); out["MACD"] = ml; out["MACD_signal"] = sl; out["MACD_hist"] = hi
    bb_b, bb_u, bb_l = bollinger(close, 20, 2.0)
    out["BB_BASIS"] = bb_b; out["BB_UPPER"] = bb_u; out["BB_LOWER"] = bb_l
    out["ATR14"] = atr(out, 14)
    out["VOL_AVG5"] = out["Volume"].rolling(5).mean(); out["VOL_AVG20"] = out["Volume"].rolling(20).mean()
    return out


# timeframe -> (period, interval) สำหรับ yfinance
TF_MAP = {"1W": ("5y", "1wk"), "1D": ("2y", "1d"), "1H": ("3mo", "1h")}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_tf_indicators(ticker, tf):
    """ดึงข้อมูล 1 timeframe แล้วคำนวณ indicator; คืน None ถ้าไม่มีข้อมูล (เช่น hourly ของบางตัว)"""
    period, interval = TF_MAP[tf]
    try:
        df = fetch_ohlcv(ticker, period=period, interval=interval)
    except Exception:
        return None
    if df is None or len(df) < 30:
        return None
    return add_indicators(df)


def _sr_levels(df, f):
    return {"recent_high_5": f(df.tail(5)["High"].max()),
            "recent_low_5": f(df.tail(5)["Low"].min()),
            "recent_high_20": f(df.tail(20)["High"].max()),
            "recent_low_20": f(df.tail(20)["Low"].min())}


def build_tf_snapshot(df):
    """สรุปค่า indicator ล่าสุดของ 1 timeframe"""
    last = df.iloc[-1]; prev = df.iloc[-2]
    def f(x, n=2): return None if pd.isna(x) else round(float(x), n)
    close = float(last["Close"])
    chgp = (close - float(prev["Close"])) / float(prev["Close"]) * 100
    def side(a, b): return "above" if a > b else "below"
    ema_stack = bool(last["EMA20"] > last["EMA50"] > last["EMA100"])
    bb_pos = ("above_upper" if close > last["BB_UPPER"] else
              "below_lower" if close < last["BB_LOWER"] else
              "upper_half" if close >= last["BB_BASIS"] else "lower_half")
    vol_ratio = (float(last["Volume"]) / float(last["VOL_AVG20"])
                 if pd.notna(last["VOL_AVG20"]) and last["VOL_AVG20"] > 0 else None)
    return {"close": f(close), "change_pct": f(chgp),
            "rsi14": f(last["RSI14"]), "rsi_signal": f(last["RSI_SIGNAL"]),
            "macd": f(last["MACD"], 4), "macd_signal": f(last["MACD_signal"], 4),
            "macd_hist": f(last["MACD_hist"], 4),
            "ema20": f(last["EMA20"]), "ema50": f(last["EMA50"]),
            "ema100": f(last["EMA100"]), "ema200": f(last["EMA200"]),
            "ema_stack_bullish": ema_stack,
            "price_vs_ema20": side(close, last["EMA20"]),
            "price_vs_ema50": side(close, last["EMA50"]),
            "price_vs_ema200": side(close, last["EMA200"]),
            "bb_basis": f(last["BB_BASIS"]), "bb_upper": f(last["BB_UPPER"]),
            "bb_lower": f(last["BB_LOWER"]), "bb_position": bb_pos,
            "atr14": f(last["ATR14"]),
            "volume": int(last["Volume"]),
            "vol_x_avg20": round(vol_ratio, 2) if vol_ratio is not None else None,
            "candlestick": detect_candlestick(df) or ["ไม่พบรูปแบบเด่นชัด"],
            "support_resistance": _sr_levels(df, f),
            "entry_now": entry_now(df)}


def build_multi_payload(ticker):
    """ดึง 3 timeframe (week/day/hour) แล้วรวมเป็น payload เดียวสำหรับ AI"""
    inds = {k: fetch_tf_indicators(ticker, tf)
            for k, tf in [("week", "1W"), ("day", "1D"), ("hour", "1H")]}
    tfs = {k: (build_tf_snapshot(v) if v is not None else None) for k, v in inds.items()}
    as_of = str(inds["day"].index[-1]) if inds["day"] is not None else None
    return {"ticker": ticker, "as_of": as_of,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "timeframes": tfs,
            "note_hour": None if tfs["hour"] else "ไม่มีข้อมูลรายชั่วโมง (yfinance) — วิเคราะห์จาก week/day"}


def momentum_score(df):
    last = df.iloc[-1]; close = float(last["Close"]); score = 0.0; sig = {}
    if close > last["EMA20"]: score += 15
    if close > last["EMA50"]: score += 10
    if last["EMA20"] > last["EMA50"]: score += 10
    if pd.notna(last["EMA100"]) and last["EMA50"] > last["EMA100"]: score += 5
    if last["MACD"] > last["MACD_signal"]: score += 10
    if last["MACD_hist"] > 0: score += 5
    r_ = last["RSI14"]
    if pd.notna(r_):
        if 50 <= r_ <= 70: score += 15
        elif 70 < r_ <= 80: score += 8
        elif r_ > 80: score += 2
        sig["RSI"] = round(float(r_), 1)
    if pd.notna(last["BB_BASIS"]) and close > last["BB_BASIS"]: score += 5
    if pd.notna(last["VOL_AVG20"]) and last["VOL_AVG20"] > 0:
        r = last["Volume"]/last["VOL_AVG20"]
        score += 10 if r >= 1.5 else 6 if r >= 1.2 else 3 if r >= 1.0 else 0
        sig["Vol x avg"] = round(float(r), 2)
    if len(df) > 20:
        r5 = (close/float(df["Close"].iloc[-6])-1)*100
        if r5 > 0: score += 8
        if (close/float(df["Close"].iloc[-21])-1) > 0: score += 7
        sig["5วัน %"] = round(r5, 2)
    return round(score, 1), sig


def entry_now(df):
    """True ถ้าราคาปัจจุบันอยู่ในจุดเข้าซื้อได้เลย (ไม่ต้องรอย่อ)."""
    last = df.iloc[-1]
    close = float(last["Close"])
    e20, e50 = float(last["EMA20"]), float(last["EMA50"])
    r = float(last["RSI14"]) if pd.notna(last["RSI14"]) else 50
    uptrend = close > e20 > e50                     # แนวโน้มขาขึ้นเรียงตัวสวย
    macd_bull = last["MACD"] > last["MACD_signal"]  # โมเมนตัมหนุน
    in_zone = 50 <= r < 72                          # มีแรง แต่ยังไม่ overbought (เกิน 72 ควรรอย่อ)
    ext = (close - e20) / e20 * 100                 # ราคาเหินจาก EMA20 กี่ %
    not_extended = ext <= 7                          # ไม่ยืดเกินไป (ถ้ายืดมากควรรอย่อ)
    hi10 = float(df.tail(10)["High"].max())
    breakout = close >= hi10 * 0.995                 # กำลังทะลุ/ชนไฮ 10 วัน = เข้าได้เลย
    return bool(uptrend and macd_bull and in_zone and (not_extended or breakout))


MULTI_TF_PROMPT = """คุณเป็นผู้เชี่ยวชาญการเทรดหุ้นไทยแบบ swing (ถือ 1-3 วัน เก็บ capital gain)
วิเคราะห์จากค่า indicator จริง 3 timeframe: สัปดาห์ (week) = เทรนด์ใหญ่, วัน (day) = ตั้ง setup, ชั่วโมง (hour) = จังหวะเข้า
ถ้า hour เป็น null ให้ใช้ week + day แทน และระบุว่าไม่มีข้อมูลรายชั่วโมง

indicator ที่ให้มาต่อ timeframe: EMA 20/50/100/200, Bollinger Bands(20,2), RSI14 + เส้น signal, MACD(12/26/9), ATR14, ปริมาณเทียบเฉลี่ย, แนวรับ-แนวต้าน, รูปแบบแท่งเทียน

ให้น้ำหนักการวิเคราะห์: แนวรับ-แนวต้าน + ปริมาณ เป็นหลัก ส่วน RSI/MACD/EMA/BB/ATR เป็นตัวยืนยัน

ตอบสั้น กระชับ เป็นภาษาไทย และต้องตอบ "ครบทุกหัวข้อเสมอ":

【ภาพรวม 3 TF】 สรุปทีละ TF สั้นๆ: week (เทรนด์ใหญ่ ขึ้น/ลง/ออกข้าง) · day (setup) · hour (จังหวะ) — ถ้า TF ขัดแย้งกันให้ชี้ให้เห็น
【สัญญาณรวม】 เลือก 1 อย่าง: ✅ น่าสนใจ / ⏳ รอจังหวะ / ❌ หลีกเลี่ยง
【เหตุผล】 อ้างอิงแนวรับ-แนวต้าน + ปริมาณเป็นหลัก เสริมด้วย RSI/MACD/EMA/BB/ATR

【ถ้ายังไม่มีของ】 (ต้องตอบเสมอ)
  - เข้าอย่างไร: ให้สอดคล้องกับ entry_now ของ TF วัน — ถ้า day.entry_now = true ตอบ "เข้าได้เลยที่ราคาปัจจุบัน" / ถ้า false ระบุโซนราคาที่ควรรอ
  - Stop loss: อิงแนวรับ TF วันที่ใกล้สุด (ราคา + % โดยประมาณ) และใช้ ATR ประกอบการวางระยะ
  - เป้าทำกำไร: แนวต้านถัดไป (ราคา + %) · Risk:Reward โดยประมาณ

【ถ้ามีของอยู่แล้ว】 (ต้องตอบเสมอ)
  - ราคาขายทำกำไร (แนวต้าน) · จุด stop loss (แนวรับ) · คำแนะนำ: "ถือต่อ" / "ทยอยขาย" / "ขายออก" พร้อมเหตุผล 1 บรรทัด

【ความเห็นของ AI】 (ต้องตอบเสมอ) "ถ้าเป็นผมและยึดสัญญาณเทคนิคล้วน ผมจะเข้า/ไม่เข้าตอนนี้" พร้อมเหตุผล 1-2 บรรทัด

ปิดท้ายบรรทัดเดียว: เป็นมุมมองเชิงเทคนิคเพื่อประกอบการตัดสินใจ ไม่ใช่คำแนะนำการลงทุน ความเสี่ยงเป็นของผู้ลงทุน

ข้อมูล:
__DATA__
"""


def analyze_with_claude(payload, prompt=MULTI_TF_PROMPT, model="claude-sonnet-4-6"):
    from anthropic import Anthropic
    client = Anthropic(api_key=API_KEY)
    msg = client.messages.create(model=model, max_tokens=1600,
        messages=[{"role": "user",
                   "content": prompt.replace("__DATA__", json.dumps(payload, ensure_ascii=False, indent=2))}])
    return "".join(b.text for b in msg.content if b.type == "text")


@st.cache_data(ttl=900, show_spinner=False)
def cached_analyze(ticker, as_of, payload):
    """เรียก AI ครั้งเดียวต่อหุ้น/รอบเวลา แล้วเก็บผลไว้ (เปิดซ้ำไม่เสียค่า API เพิ่ม)"""
    return analyze_with_claude(payload)


# ============================================================
#  ส่วนหน้าตา (UI)
# ============================================================
st.title("📊 วิเคราะห์หุ้นไทยด้วย AI")
st.caption(f"แหล่งข้อมูล: {DATA_SOURCE} · เพื่อประกอบการตัดสินใจ ไม่ใช่คำแนะนำการลงทุน")

with st.sidebar:
    st.header("สถานะ")
    if API_KEY:
        st.markdown('<span class="pill ok">✓ เชื่อมต่อ AI พร้อมใช้</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill off">โหมดดูตัวเลข (ยังไม่ได้ตั้ง API key)</span>', unsafe_allow_html=True)
        st.caption("ตั้ง ANTHROPIC_API_KEY ใน Settings → Secrets เพื่อเปิดใช้ AI")
    st.divider()
    st.caption(f"แหล่งข้อมูล: {DATA_SOURCE}")
    st.caption(f"ชุดสแกนหลัก: หุ้นสภาพคล่องสูง ~{len(UNIVERSE_LARGE)} ตัว (หรือกำหนดเอง)")

tab1, tab2 = st.tabs(["📈 วิเคราะห์รายตัว", "🔍 สแกนโมเมนตัม"])

# ---------- แท็บ 1: วิเคราะห์รายตัว ----------
with tab1:
    c1, c2 = st.columns([3, 1])
    ticker = c1.text_input("สัญลักษณ์หุ้น (เช่น PTT.BK, AOT.BK, ^SET.BK)", value="PTT.BK")
    use_ai = c2.toggle("ให้ AI วิเคราะห์", value=bool(API_KEY), disabled=not API_KEY)

    if st.button("วิเคราะห์", type="primary", use_container_width=True):
        tk = ticker.strip().upper()
        try:
            with st.spinner("กำลังดึงข้อมูล 3 timeframe (วัน/สัปดาห์/ชั่วโมง) และคำนวณ..."):
                payload = build_multi_payload(tk)
                day_ind = fetch_tf_indicators(tk, "1D")

            if day_ind is None:
                st.error("ดึงข้อมูลรายวันไม่ได้ ลองใหม่หรือตรวจสัญลักษณ์อีกครั้ง")
            else:
                d = payload["timeframes"]["day"]
                m = st.columns(4)
                m[0].metric("ราคาล่าสุด (วัน)", f"{d['close']:,.2f}", f"{d['change_pct']:+.2f}%")
                m[1].metric("RSI (14)", d["rsi14"])
                m[2].metric("ATR (14)", d["atr14"])
                m[3].metric("วอลุ่ม x เฉลี่ย", d["vol_x_avg20"])

                st.line_chart(day_ind.tail(80)[["Close", "EMA20", "EMA50", "EMA100", "EMA200"]])

                st.markdown("##### สรุป 3 Timeframe")
                cols = st.columns(3)
                for col, (label, key) in zip(cols, [("🗓 สัปดาห์", "week"),
                                                    ("📆 วัน", "day"), ("⏱ ชั่วโมง", "hour")]):
                    tf = payload["timeframes"][key]
                    if tf is None:
                        col.markdown(f"**{label}**\n\n_ไม่มีข้อมูล_")
                    else:
                        srl = tf["support_resistance"]
                        col.markdown(
                            f"**{label}**\n\n"
                            f"- ราคา: {tf['close']} ({tf['change_pct']:+.2f}%)\n"
                            f"- RSI: {tf['rsi14']} / sig {tf['rsi_signal']}\n"
                            f"- EMA: {'ขาขึ้นเรียงตัว' if tf['ema_stack_bullish'] else 'ไม่เรียงตัว'}\n"
                            f"- BB: {tf['bb_position']}\n"
                            f"- รับ/ต้าน (20): {srl['recent_low_20']} / {srl['recent_high_20']}\n"
                            f"- วอลุ่ม x เฉลี่ย: {tf['vol_x_avg20']}\n"
                            f"- เข้าได้เลย: {'✅' if tf['entry_now'] else '—'}")
                if payload.get("note_hour"):
                    st.caption("⚠️ " + payload["note_hour"])

                if use_ai and API_KEY:
                    with st.spinner("AI กำลังวิเคราะห์ 3 timeframe..."):
                        st.divider()
                        st.subheader("ผลวิเคราะห์จาก Claude (Multi-Timeframe)")
                        st.markdown(cached_analyze(tk, payload["as_of"], payload))
                elif not API_KEY:
                    st.info("อยู่ในโหมดดูตัวเลข — ตั้ง API key ใน Secrets เพื่อให้ AI วิเคราะห์อัตโนมัติ")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")

# ---------- แท็บ 2: สแกนโมเมนตัม ----------
with tab2:
    set_choice = st.radio("ชุดหุ้นที่จะสแกน",
                          ["SET50 (50 ตัว)", f"SET100 (~{len(UNIVERSE_LARGE)} ตัว)", "กำหนดเอง"],
                          horizontal=True, index=0)
    if set_choice == "กำหนดเอง":
        custom = st.text_input("พิมพ์สัญลักษณ์ คั่นด้วยจุลภาค เช่น PTT.BK, AOT.BK, KBANK.BK",
                               value="PTT.BK, AOT.BK, KBANK.BK")
        scan_list = [t.strip().upper() for t in custom.split(",") if t.strip()]
    elif set_choice.startswith("SET50"):
        scan_list = UNIVERSE_SET50
    else:
        scan_list = UNIVERSE_LARGE
    st.caption("สแกนจัดอันดับด้วยกราฟรายวัน (เร็ว) · กดเจาะลึกรายตัวแล้ว AI จะอ่าน 3 timeframe (วัน/สัปดาห์/ชั่วโมง)")

    # เลือกจำนวนอันดับที่จะแสดง — ใช้ selectbox (แตะเลือก) แทน slider (ลาก)
    # เพื่อกันนิ้วไปโดนแถบเลื่อนตอนปัดหน้าจอบนมือถือ
    n_total = max(1, len(scan_list))
    _presets = [("10 อันดับแรก", 10), ("20 อันดับแรก", 20),
                ("30 อันดับแรก", 30), ("50 อันดับแรก", 50)]
    top_options = [lbl for lbl, v in _presets if v < n_total] + ["ทั้งหมด"]
    default_idx = top_options.index("10 อันดับแรก") if "10 อันดับแรก" in top_options else len(top_options) - 1
    top_choice = st.selectbox("แสดงกี่อันดับ", top_options, index=default_idx)
    top_n = n_total if top_choice == "ทั้งหมด" else dict(_presets)[top_choice]

    only_buynow = st.checkbox("แสดงเฉพาะหุ้นที่เข้าซื้อได้เลย (ไม่ต้องรอย่อ)")

    # ดึงเป็นก้อน (batch) + เพดานจำนวน + คืนหน่วยความจำ กันแอปล้มบน Streamlit Cloud ฟรี
    BATCH_SIZE = 20
    MAX_SCAN = 100
    if len(scan_list) > 20:
        st.caption(f"⏳ สแกน {min(len(scan_list), MAX_SCAN)} ตัว (ดึงทีละก้อน {BATCH_SIZE} ตัว) ใช้เวลาสักครู่")

    if st.button("เริ่มสแกน", type="primary", use_container_width=True):
        targets = scan_list[:MAX_SCAN]
        if len(scan_list) > MAX_SCAN:
            st.warning(f"สแกนเฉพาะ {MAX_SCAN} ตัวแรก (กันเครื่องล้ม) — ถ้าต้องการดูครบ แบ่งสแกนเป็นรอบ")
        rows = []
        prog = st.progress(0.0, text="กำลังสแกน...")
        done = 0
        for start in range(0, len(targets), BATCH_SIZE):
            chunk = tuple(targets[start:start + BATCH_SIZE])
            batch = fetch_batch(chunk)          # ดึงทั้งก้อนในครั้งเดียว
            for t in chunk:
                done += 1
                prog.progress(done / len(targets),
                              text=f"กำลังสแกน {t.replace('.BK','')} ({done}/{len(targets)})")
                df = batch.get(t)
                if df is None or len(df) < 30:
                    continue
                try:
                    ind = add_indicators(df); sc, sig = momentum_score(ind)
                    sym = t.replace(".BK", "")
                    rows.append({"หุ้น": sym, "คะแนน": sc,
                                 "ราคา": round(float(ind["Close"].iloc[-1]), 2),
                                 "เข้าได้เลย": "✅" if entry_now(ind) else "—", **sig})
                except Exception:
                    pass
                del df
            del batch                            # คืนหน่วยความจำหลังจบแต่ละก้อน
            time.sleep(0.5)
        prog.empty()

        if only_buynow:
            rows = [r for r in rows if r["เข้าได้เลย"] == "✅"]
        ranked = sorted(rows, key=lambda r: r["คะแนน"], reverse=True)[:top_n]
        st.session_state.scan = {"ranked": ranked, "only_buynow": only_buynow}
        st.session_state.analyzed = set()   # ล้างรายการที่เคยกดวิเคราะห์ของรอบก่อน

    scan = st.session_state.get("scan")
    if scan and scan["ranked"]:
        st.dataframe(pd.DataFrame(scan["ranked"]),
                     use_container_width=True, hide_index=True,
                     column_config={"คะแนน": st.column_config.ProgressColumn(
                         "คะแนน", min_value=0, max_value=100, format="%.0f")})
        st.caption("คะแนนเต็ม 100 · ยิ่งสูง = โมเมนตัมขาขึ้นยิ่งแรง · ✅ ในคอลัมน์ 'เข้าได้เลย' = ราคาปัจจุบันเข้าซื้อได้โดยไม่ต้องรอย่อ")


        if API_KEY:
            st.markdown("##### บทวิเคราะห์ AI — คลิกเปิดดูได้ทุกตัว")
            st.caption("แต่ละตัวที่กดวิเคราะห์ใช้เครดิต API เล็กน้อย · เปิดซ้ำไม่เสียเพิ่ม")
            analyzed = st.session_state.setdefault("analyzed", set())
            for r in scan["ranked"]:
                sym = r["หุ้น"]
                with st.expander(f"{sym}  ·  คะแนน {r['คะแนน']}  ·  ราคา {r['ราคา']}"):
                    if st.button("วิเคราะห์ด้วย AI (3 TF)", key=f"ai_{sym}"):
                        analyzed.add(sym)
                    if sym in analyzed:
                        with st.spinner("AI กำลังดึง 3 timeframe และวิเคราะห์..."):
                            tk = sym + ".BK"
                            p = build_multi_payload(tk)
                            st.markdown(cached_analyze(tk, p["as_of"], p))
        else:
            st.info("ตั้ง API key ใน Secrets เพื่อเปิดบทวิเคราะห์ AI")
    elif scan is not None and not scan["ranked"]:
        if scan.get("only_buynow"):
            st.info("รอบนี้ไม่พบหุ้นที่ 'เข้าซื้อได้เลย' ตามเงื่อนไข — ลองเอาตัวกรองออก หรือสแกนใหม่ภายหลัง")
        else:
            st.error("ดึงข้อมูลไม่ได้ ลองใหม่อีกครั้ง")
