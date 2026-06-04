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
  .stApp { background: #0e1117; }
  h1, h2, h3 { letter-spacing: .3px; }
  .pill { display:inline-block; padding:4px 12px; border-radius:999px;
          font-size:13px; font-weight:600; }
  .ok  { background:#10341f; color:#41d77f; }
  .off { background:#3a2a10; color:#e0a341; }
</style>
""", unsafe_allow_html=True)

DEFAULT_UNIVERSE = ["PTT.BK", "AOT.BK", "KBANK.BK", "SCB.BK", "CPALL.BK",
                    "ADVANC.BK", "GULF.BK", "DELTA.BK", "BDMS.BK", "SCC.BK"]


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
#  ฟังก์ชันคำนวณ (เหมือนเวอร์ชันก่อนหน้า)
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"ไม่พบข้อมูลของ {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


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


def stochastic(df, k=14, d=3):
    lo = df["Low"].rolling(k).min(); hi = df["High"].rolling(k).max()
    pk = 100*(df["Close"]-lo)/(hi-lo); return pk, pk.rolling(d).mean()


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
    out["EMA20"] = ema(close, 20); out["EMA50"] = ema(close, 50); out["SMA200"] = sma(close, 200)
    out["RSI14"] = rsi(close, 14)
    ml, sl, hi = macd(close); out["MACD"] = ml; out["MACD_signal"] = sl; out["MACD_hist"] = hi
    sk, sd = stochastic(out); out["STOCH_K"] = sk; out["STOCH_D"] = sd
    out["VOL_AVG5"] = out["Volume"].rolling(5).mean(); out["VOL_AVG20"] = out["Volume"].rolling(20).mean()
    return out


def build_payload(df, ticker, tail=30):
    last = df.iloc[-1]; prev = df.iloc[-2]
    chg = last["Close"]-prev["Close"]; chgp = chg/prev["Close"]*100
    def f(x, n=2): return None if pd.isna(x) else round(float(x), n)
    recent = df.tail(tail).copy(); recent.index = recent.index.astype(str)
    series = recent[["Close", "RSI14", "MACD", "MACD_signal"]].round(2)
    levels = {"recent_high_week_5d": f(df.tail(5)["High"].max()),
              "recent_low_week_5d": f(df.tail(5)["Low"].min()),
              "recent_high_10d": f(df.tail(10)["High"].max()),
              "recent_low_10d": f(df.tail(10)["Low"].min()),
              "recent_high_30d": f(df.tail(tail)["High"].max()),
              "recent_low_30d": f(df.tail(tail)["Low"].min())}
    pats = detect_candlestick(df)
    return {"ticker": ticker, "as_of": str(df.index[-1]),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "snapshot": {"open": f(last["Open"]), "high": f(last["High"]), "low": f(last["Low"]),
                "close": f(last["Close"]), "change": f(chg), "change_pct": f(chgp),
                "volume": int(last["Volume"]),
                "avg_volume_5d": None if pd.isna(last["VOL_AVG5"]) else int(last["VOL_AVG5"]),
                "avg_volume_20d": None if pd.isna(last["VOL_AVG20"]) else int(last["VOL_AVG20"]),
                "volume_vs_avg20": ("สูงกว่าค่าเฉลี่ย" if not pd.isna(last["VOL_AVG20"]) and last["Volume"] > last["VOL_AVG20"] else "ต่ำกว่าค่าเฉลี่ย"),
                "rsi14": f(last["RSI14"]), "stoch_k": f(last["STOCH_K"]), "stoch_d": f(last["STOCH_D"]),
                "ema20": f(last["EMA20"]), "ema50": f(last["EMA50"]), "sma200": f(last["SMA200"]),
                "macd": f(last["MACD"], 4), "macd_signal": f(last["MACD_signal"], 4), "macd_hist": f(last["MACD_hist"], 4),
                "price_vs_ema20": "above" if last["Close"] > last["EMA20"] else "below",
                "price_vs_ema50": "above" if last["Close"] > last["EMA50"] else "below",
                "ema20_vs_ema50": "bullish_cross" if last["EMA20"] > last["EMA50"] else "bearish_cross"},
            "candlestick_pattern": pats if pats else ["ไม่พบรูปแบบเด่นชัด"],
            "support_resistance": levels,
            "recent_series": json.loads(series.to_json(orient="index"))}


def momentum_score(df):
    last = df.iloc[-1]; close = float(last["Close"]); score = 0.0; sig = {}
    if close > last["EMA20"]: score += 15
    if close > last["EMA50"]: score += 10
    if last["EMA20"] > last["EMA50"]: score += 10
    if last["MACD"] > last["MACD_signal"]: score += 10
    if last["MACD_hist"] > 0: score += 5
    r_ = last["RSI14"]
    if pd.notna(r_):
        if 50 <= r_ <= 70: score += 15
        elif 70 < r_ <= 80: score += 8
        elif r_ > 80: score += 2
        sig["RSI"] = round(float(r_), 1)
    if pd.notna(last["STOCH_K"]) and pd.notna(last["STOCH_D"]):
        if last["STOCH_K"] > last["STOCH_D"] and last["STOCH_K"] < 80: score += 10
        sig["Stoch %K"] = round(float(last["STOCH_K"]), 1)
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


ANALYSIS_PROMPT = """คุณเป็นผู้เชี่ยวชาญการลงทุนหุ้นไทย เน้นการเทรดทำกำไรระยะสั้น (swing trade ถือ 1-3 วัน เก็บ capital gain) วิเคราะห์จากค่าตัวเลข indicator ต่อไปนี้ (เป็นค่าจริง ไม่ใช่ภาพ) แล้วให้ข้อสรุป "สั้น กระชับ" เป็นภาษาไทย

ตอบตามรูปแบบนี้เท่านั้น ห้ามยืดยาว แต่ละหัวข้อไม่เกิน 2-3 บรรทัด:

【สัญญาณรวม】 เลือก 1 อย่าง: ✅ น่าสนใจ / ⏳ รอจังหวะ / ❌ หลีกเลี่ยง
【เหตุผล】 อ้างอิง RSI, Stochastic (%K/%D), MACD, ตำแหน่งราคาเทียบ EMA/SMA, ปริมาณซื้อขายเทียบค่าเฉลี่ย และรูปแบบแท่งเทียนที่ตรวจพบ (ถ้ามี)

【ถ้ายังไม่มีของ】
  - ควรเข้าซื้ออย่างไร: "ซื้อได้เลยที่ราคาปัจจุบัน" หรือ "รอซื้อที่โซน (ระบุช่วงราคา)" หรือ "ยังไม่ควรเข้า"
  - Stop loss: อิงแนวรับทางเทคนิคที่ใกล้สุด (ราคา + % ขาดทุนโดยประมาณ)
  - เป้าทำกำไร: แนวต้านถัดไป (ราคา + % กำไรโดยประมาณ)
  - Risk:Reward โดยประมาณ

【ถ้ามีของอยู่แล้ว】
  - ราคาขายทำกำไร: แนวต้านที่ควรขายล็อกกำไร (ราคา)
  - จุดตัดขาดทุน (stop loss): อิงแนวรับทางเทคนิคที่ใกล้สุด (ราคา)
  - คำแนะนำสั้น ๆ: "ถือต่อ" / "ทยอยขายลดพอร์ต" / "ขายออกเลย" พร้อมเหตุผล 1 บรรทัด

ปิดท้ายบรรทัดเดียว: เตือนว่าเป็นมุมมองเชิงเทคนิคเพื่อประกอบการตัดสินใจ ไม่ใช่คำแนะนำการลงทุน ความเสี่ยงเป็นของผู้ลงทุน

ข้อมูล:
__DATA__
"""


def analyze_with_claude(payload, model="claude-sonnet-4-6"):
    from anthropic import Anthropic
    client = Anthropic(api_key=API_KEY)
    msg = client.messages.create(model=model, max_tokens=1500,
        messages=[{"role": "user",
                   "content": ANALYSIS_PROMPT.replace("__DATA__", json.dumps(payload, ensure_ascii=False, indent=2))}])
    return "".join(b.text for b in msg.content if b.type == "text")


# ============================================================
#  ส่วนหน้าตา (UI)
# ============================================================
st.title("📊 วิเคราะห์หุ้นไทยด้วย AI")
st.caption("ข้อมูลดีเลย์ ~15 นาที · เพื่อประกอบการตัดสินใจ ไม่ใช่คำแนะนำการลงทุน")

with st.sidebar:
    st.header("สถานะ")
    if API_KEY:
        st.markdown('<span class="pill ok">✓ เชื่อมต่อ AI พร้อมใช้</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill off">โหมดดูตัวเลข (ยังไม่ได้ตั้ง API key)</span>', unsafe_allow_html=True)
        st.caption("ตั้ง ANTHROPIC_API_KEY ใน Settings → Secrets เพื่อเปิดใช้ AI")
    st.divider()
    st.caption("หุ้นในชุดสแกน:")
    st.caption(", ".join(t.replace(".BK", "") for t in DEFAULT_UNIVERSE))

tab1, tab2 = st.tabs(["📈 วิเคราะห์รายตัว", "🔍 สแกนโมเมนตัม"])

# ---------- แท็บ 1: วิเคราะห์รายตัว ----------
with tab1:
    c1, c2 = st.columns([3, 1])
    ticker = c1.text_input("สัญลักษณ์หุ้น (เช่น PTT.BK, AOT.BK, ^SET.BK)", value="PTT.BK")
    use_ai = c2.toggle("ให้ AI วิเคราะห์", value=bool(API_KEY), disabled=not API_KEY)

    if st.button("วิเคราะห์", type="primary", use_container_width=True):
        try:
            with st.spinner("กำลังดึงข้อมูลและคำนวณ..."):
                df = add_indicators(fetch_ohlcv(ticker.strip().upper()))
                payload = build_payload(df, ticker.strip().upper())
            snap = payload["snapshot"]

            m = st.columns(4)
            m[0].metric("ราคาล่าสุด", f"{snap['close']:,.2f}",
                        f"{snap['change_pct']:+.2f}%")
            m[1].metric("RSI (14)", snap["rsi14"])
            m[2].metric("Stochastic %K", snap["stoch_k"])
            m[3].metric("วอลุ่ม vs เฉลี่ย", snap["volume_vs_avg20"])

            st.line_chart(df.tail(60)[["Close", "EMA20", "EMA50"]])

            cc = st.columns(2)
            cc[0].markdown("**รูปแบบแท่งเทียน**\n\n" +
                           "\n".join(f"- {p}" for p in payload["candlestick_pattern"]))
            sr = payload["support_resistance"]
            cc[1].markdown(
                f"**แนวรับ/แนวต้าน**\n\n"
                f"- แนวต้านสัปดาห์: {sr['recent_high_week_5d']}\n"
                f"- แนวรับสัปดาห์: {sr['recent_low_week_5d']}\n"
                f"- สูงสุด 30 วัน: {sr['recent_high_30d']}\n"
                f"- ต่ำสุด 30 วัน: {sr['recent_low_30d']}")

            if use_ai and API_KEY:
                with st.spinner("AI กำลังวิเคราะห์..."):
                    st.divider()
                    st.subheader("ผลวิเคราะห์จาก Claude")
                    st.markdown(analyze_with_claude(payload))
            elif not API_KEY:
                st.info("อยู่ในโหมดดูตัวเลข — ก๊อปตัวเลขด้านบนไปถาม Claude.ai ได้ "
                        "หรือตั้ง API key เพื่อให้วิเคราะห์อัตโนมัติ")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")

# ---------- แท็บ 2: สแกนโมเมนตัม ----------
with tab2:
    s1, s2 = st.columns(2)
    top_n = s1.slider("แสดงกี่อันดับ", 1, 10, 10)
    ai_top = s2.slider("ให้ AI วิเคราะห์กี่อันดับแรก (0 = ไม่ใช้)", 0, 5, 0,
                       disabled=not API_KEY)

    if st.button("เริ่มสแกน", type="primary", use_container_width=True):
        rows = []
        prog = st.progress(0.0, text="กำลังสแกน...")
        data = {}
        for i, t in enumerate(DEFAULT_UNIVERSE, 1):
            prog.progress(i/len(DEFAULT_UNIVERSE),
                          text=f"กำลังสแกน {t.replace('.BK','')} ({i}/{len(DEFAULT_UNIVERSE)})")
            try:
                df = fetch_ohlcv(t)
                if len(df) >= 30:
                    data[t] = df
                    ind = add_indicators(df); sc, sig = momentum_score(ind)
                    rows.append({"หุ้น": t.replace(".BK", ""), "คะแนน": sc,
                                 "ราคา": round(float(ind["Close"].iloc[-1]), 2),
                                 **sig})
            except Exception:
                pass
            time.sleep(0.3)
        prog.empty()

        if rows:
            tbl = pd.DataFrame(sorted(rows, key=lambda r: r["คะแนน"], reverse=True)[:top_n])
            st.dataframe(tbl.style.background_gradient(subset=["คะแนน"], cmap="Greens"),
                         use_container_width=True, hide_index=True)
            st.caption("คะแนนเต็ม 100 · ยิ่งสูง = โมเมนตัมขาขึ้นยิ่งแรง")

            if ai_top > 0 and API_KEY:
                for r in sorted(rows, key=lambda r: r["คะแนน"], reverse=True)[:ai_top]:
                    t = r["หุ้น"] + ".BK"
                    with st.expander(f"📋 วิเคราะห์ {r['หุ้น']} (คะแนน {r['คะแนน']})"):
                        with st.spinner("AI กำลังวิเคราะห์..."):
                            st.markdown(analyze_with_claude(build_payload(add_indicators(data[t]), t)))
        else:
            st.error("ดึงข้อมูลไม่ได้ ลองใหม่อีกครั้ง")
