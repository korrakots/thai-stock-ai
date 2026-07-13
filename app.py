"""
app.py — สแกนหุ้นโมเมนตัม SET100 (แนวคิด swing trade "พร้อมวิ่ง")
Indicator: MACD, RSI, Volume, ATR, Bollinger Bands
Timeframe: D (รายวัน) / H (รายชั่วโมง) / 15m (15 นาที)

ออกแบบให้:
- ประหยัดหน่วยความจำ: สแกน SET100 ด้วยกราฟ "รายวัน" ก่อน (ทยอยทีละก้อน) แล้วค่อยยืนยัน H/15m รายตัวตอนกดดู
- ทนข้อมูลขาด: intraday ของหุ้นไทยบางตัวไม่มีใน yfinance -> ข้ามอย่างนุ่มนวล
"""
import os
import json
import time
import gc
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

st.set_page_config(page_title="สแกนโมเมนตัม SET100", page_icon="🚀", layout="wide")

# ============================================================
#  Universe: SET100 (รอบ H2 2026) — ตัด EA ออกตามที่เคยระบุ => 99 ตัว
# ============================================================
SET100 = [
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

# timeframe -> (period, interval) สำหรับ yfinance
TF = {"D": ("1y", "1d"), "H": ("3mo", "1h"), "15m": ("1mo", "15m")}


# ============================================================
#  API key (สำหรับบทวิเคราะห์ AI — ไม่บังคับ)
# ============================================================
def get_api_key():
    try:
        k = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        k = None
    return k or os.environ.get("ANTHROPIC_API_KEY")


API_KEY = get_api_key()


# ============================================================
#  ชั้นดึงข้อมูล
# ============================================================
@st.cache_data(ttl=120, max_entries=60, show_spinner=False)
def fetch(ticker, tf):
    """ดึง 1 timeframe ของ 1 ตัว; คืน None ถ้าไม่มีข้อมูล (เช่น 15m ของบางตัว)"""
    if yf is None:
        return None
    period, interval = TF[tf]
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=False, progress=False, threads=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    try:
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return None
    return df if len(df) >= 30 else None


def fetch_daily_batch(tickers):
    """ดึงกราฟรายวันหลายตัวพร้อมกัน (ลดจำนวนครั้งเรียก + เบา RAM ด้วย float32)"""
    out = {}
    if yf is None:
        return out
    try:
        data = yf.download(list(tickers), period="1y", interval="1d",
                           auto_adjust=False, progress=False,
                           group_by="ticker", threads=True)
    except Exception:
        return out
    if data is None or data.empty:
        return out
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for t in tickers:
        try:
            sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            sub = sub[cols].dropna().astype("float32")
            if len(sub) >= 30:
                out[t] = sub
        except Exception:
            pass
    del data
    return out


# ============================================================
#  Indicator (คำนวณเอง) — MACD, RSI, ATR, Bollinger, Volume
# ============================================================
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def macd(close, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def rsi(close, period=14):
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + ag / al))


def atr(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(close, period=20, mult=2.0):
    basis = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return basis, basis + mult * sd, basis - mult * sd


# ============================================================
#  หัวใจ: ให้คะแนน "โมเมนตัมพร้อมวิ่ง" ต่อ 1 timeframe
# ============================================================
def analyze_tf(df):
    """คืน dict: score(0-100), ready(bool), และรายละเอียดสัญญาณ — ใช้ MACD/RSI/Vol/ATR/BB"""
    close = df["Close"]
    if len(df) < 35:
        return None
    ml, sl, hist = macd(close)
    r = rsi(close, 14)
    a = atr(df, 14)
    basis, upper, lower = bollinger(close, 20, 2.0)
    vavg = df["Volume"].rolling(20).mean()

    def fx(s, i=-1):
        v = s.iloc[i]
        return None if pd.isna(v) else float(v)

    c = fx(close)
    macd_bull = fx(ml) > fx(sl)
    hist_now, hist_prev = fx(hist), fx(hist, -2)
    hist_rising = hist_now is not None and hist_prev is not None and hist_now > hist_prev
    rsi_v = fx(r)
    rsi_rising = rsi_v is not None and fx(r, -2) is not None and rsi_v > fx(r, -2)
    vr = (fx(df["Volume"]) / fx(vavg)) if fx(vavg) else None
    b_now = fx(basis)
    above_basis = b_now is not None and c > b_now
    up_now = fx(upper)
    breakout = up_now is not None and c >= up_now * 0.995
    # Bollinger squeeze: bandwidth แคบสุดในรอบ ~60 แท่ง = พลังงานสะสมพร้อมระเบิด
    bw = (upper - lower) / basis
    bw_now = fx(bw)
    bw_series = bw.tail(60).dropna()
    squeeze = bool(bw_now is not None and len(bw_series) >= 20 and bw_now <= bw_series.min() * 1.15)
    atr_now, atr_prev = fx(a), fx(a, -2)
    atr_rising = atr_now is not None and atr_prev is not None and atr_now > atr_prev

    score = 0
    if macd_bull:
        score += 20
    if hist_rising:
        score += 5
    if rsi_v is not None:
        if 50 <= rsi_v <= 68:
            score += 20
        elif 68 < rsi_v <= 75:
            score += 8
        elif rsi_v < 50 and rsi_rising:
            score += 8
    if vr is not None:
        score += 20 if vr >= 1.5 else 12 if vr >= 1.2 else 5 if vr >= 1.0 else 0
    if above_basis:
        score += 10
    if breakout:
        score += 10
    if squeeze:
        score += 5
    if squeeze and atr_rising:
        score += 5   # เริ่มขยายตัวหลังบีบ = จุดพร้อมวิ่ง
    score = min(score, 100)

    # "พร้อมวิ่ง" = โมเมนตัมหนุน + ยังไม่ overbought + ยืนเหนือเส้นกลาง BB + วอลุ่มไม่หด
    ready = bool(macd_bull and rsi_v is not None and 50 <= rsi_v <= 72
                 and above_basis and (vr is None or vr >= 1.0))

    return {
        "score": round(score),
        "ready": ready,
        "close": round(c, 2),
        "rsi": round(rsi_v, 1) if rsi_v is not None else None,
        "macd_bull": macd_bull,
        "vol_x": round(vr, 2) if vr is not None else None,
        "bb": ("breakout" if breakout else "เหนือกลาง" if above_basis else "ใต้กลาง"),
        "squeeze": squeeze,
        "atr": round(atr_now, 3) if atr_now is not None else None,
        "upper": round(up_now, 2) if up_now is not None else None,
        "basis": round(b_now, 2) if b_now is not None else None,
    }


def multi_tf(ticker):
    """วิเคราะห์ครบ D/H/15m; TF ไหนไม่มีข้อมูลเป็น None"""
    res = {}
    for tf in ("D", "H", "15m"):
        df = fetch(ticker, tf)
        res[tf] = analyze_tf(df) if df is not None else None
    aligned = sum(1 for v in res.values() if v and v["ready"])
    return res, aligned


# ============================================================
#  บทวิเคราะห์ AI (ไม่บังคับ)
# ============================================================
PROMPT = """คุณเป็นผู้เชี่ยวชาญเทรดหุ้นไทยแบบ swing (ถือ 1-3 วัน) วิเคราะห์จากค่า indicator จริง 3 timeframe
D = เทรนด์/setup, H = โมเมนตัมกำลังก่อตัว, 15m = จังหวะเข้า (ถ้า TF ไหนเป็น null ให้ข้ามและระบุว่าไม่มีข้อมูล)
indicator: MACD, RSI, Volume(เทียบเฉลี่ย), ATR, Bollinger Bands

ตอบสั้น กระชับ ภาษาไทย ครบทุกหัวข้อ:
【พร้อมวิ่งไหม】 ✅ พร้อม / ⏳ ใกล้ / ❌ ยัง — พร้อมเหตุผลจากการเรียงตัวของ 3 TF
【เหตุผล】 อ้าง MACD, RSI, Volume, Bollinger(squeeze/breakout), ATR
【ถ้าจะเข้า】 โซนเข้า, จุด stop loss (อิง ATR รายวัน ~1.5 เท่า หรือใต้เส้นกลาง BB), เป้าแรก (ขอบบน BB/แนวต้าน), Risk:Reward
【ถ้ามีของ】 จุดขายทำกำไร / stop / ถือต่อ-ทยอยขาย-ขายออก + เหตุผล 1 บรรทัด
ปิดท้าย: เป็นมุมมองเทคนิคเพื่อประกอบการตัดสินใจ ไม่ใช่คำแนะนำการลงทุน

ข้อมูล:
__DATA__
"""


@st.cache_data(ttl=900, max_entries=40, show_spinner=False)
def ai_analyze(ticker, as_of, payload):
    from anthropic import Anthropic
    client = Anthropic(api_key=API_KEY)
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1400,
        messages=[{"role": "user",
                   "content": PROMPT.replace("__DATA__", json.dumps(payload, ensure_ascii=False, indent=2))}])
    return "".join(b.text for b in msg.content if b.type == "text")


# ============================================================
#  UI
# ============================================================
st.title("🚀 สแกนหุ้นโมเมนตัม SET100 (Swing)")
st.caption("หา 'หุ้นพร้อมวิ่ง' ด้วย MACD · RSI · Volume · ATR · Bollinger Bands บนกราฟ D / H / 15m — "
           "เพื่อประกอบการตัดสินใจ ไม่ใช่คำแนะนำการลงทุน")

with st.sidebar:
    st.header("สถานะ")
    if API_KEY:
        st.success("✓ เชื่อม AI พร้อมใช้")
    else:
        st.warning("โหมดตัวเลข (ยังไม่ตั้ง ANTHROPIC_API_KEY)")
    st.divider()
    st.markdown(
        "**เกณฑ์ 'พร้อมวิ่ง'**\n\n"
        "- MACD เป็นบวก (line > signal)\n"
        "- RSI โซนโมเมนตัม 50–70\n"
        "- ราคายืนเหนือเส้นกลาง BB / ทะลุขอบบน\n"
        "- Volume ไม่หด (≥ ค่าเฉลี่ย)\n"
        "- BB squeeze + ATR เริ่มขยาย = พร้อมระเบิด")
    st.caption("แหล่งข้อมูล: yfinance (รายวันเชื่อถือได้ · intraday บางตัวอาจขาด)")

tab_scan, tab_one = st.tabs(["🔍 สแกน SET100", "🔎 ดูรายตัว (D/H/15m)"])

# ---------- แท็บสแกน ----------
with tab_scan:
    st.markdown("สแกนจัดอันดับด้วย **กราฟรายวัน** (เร็ว + เบา) จากนั้นกดดูรายตัวเพื่อยืนยัน **H / 15m**")
    c1, c2 = st.columns(2)
    only_ready = c1.checkbox("แสดงเฉพาะ 'พร้อมวิ่ง' (รายวัน)", value=True)
    top_choice = c2.selectbox("แสดงกี่อันดับ", ["10", "20", "30", "ทั้งหมด"], index=1)

    if st.button("เริ่มสแกน SET100", type="primary", use_container_width=True):
        st.session_state.run = {"idx": 0, "rows": [],
                                "only_ready": only_ready, "top": top_choice}
        st.session_state.pop("result", None)
        st.rerun()

    run = st.session_state.get("run")
    if run is not None:
        CHUNK = 10
        total = len(SET100)
        start = run["idx"]
        st.progress(min(start / total, 1.0), text=f"กำลังสแกน... ({start}/{total})")
        chunk = tuple(SET100[start:start + CHUNK])
        batch = fetch_daily_batch(chunk)
        for t in chunk:
            df = batch.get(t)
            if df is None:
                continue
            a = analyze_tf(df)
            if a is None:
                continue
            run["rows"].append({"หุ้น": t.replace(".BK", ""), "คะแนน": a["score"],
                                "ราคา": a["close"], "RSI": a["rsi"],
                                "Vol×": a["vol_x"], "BB": a["bb"],
                                "squeeze": "🔸" if a["squeeze"] else "",
                                "พร้อมวิ่ง": "✅" if a["ready"] else "—"})
            del df
        del batch
        gc.collect()
        run["idx"] = start + len(chunk)

        if run["idx"] >= total:
            rows = run["rows"]
            if run["only_ready"]:
                rows = [r for r in rows if r["พร้อมวิ่ง"] == "✅"]
            rows.sort(key=lambda r: r["คะแนน"], reverse=True)
            n = len(rows) if run["top"] == "ทั้งหมด" else int(run["top"])
            st.session_state.result = rows[:n]
            st.session_state.pop("run", None)
            st.rerun()
        else:
            time.sleep(0.15)
            st.rerun()

    result = st.session_state.get("result")
    if result is not None:
        if not result:
            st.info("รอบนี้ไม่พบหุ้น 'พร้อมวิ่ง' ตามเกณฑ์ — ลองเอาตัวกรองออก หรือสแกนใหม่ภายหลัง")
        else:
            st.dataframe(pd.DataFrame(result), use_container_width=True, hide_index=True,
                         column_config={"คะแนน": st.column_config.ProgressColumn(
                             "คะแนน", min_value=0, max_value=100, format="%d")})
            st.caption("🔸 = Bollinger squeeze (พลังงานสะสม) · กดดูรายตัวด้านล่างเพื่อยืนยัน H/15m")
            st.markdown("##### ยืนยันหลาย timeframe รายตัว")
            for r in result:
                sym = r["หุ้น"]
                with st.expander(f"{sym} · คะแนนรายวัน {r['คะแนน']} · {r['พร้อมวิ่ง']}"):
                    if st.button("ดู D / H / 15m", key=f"m_{sym}"):
                        with st.spinner("กำลังดึง 3 timeframe..."):
                            res, aligned = multi_tf(sym + ".BK")
                        st.markdown(f"**สอดคล้องกัน {aligned}/3 timeframe**")
                        cols = st.columns(3)
                        for col, tf in zip(cols, ("D", "H", "15m")):
                            v = res[tf]
                            if v is None:
                                col.markdown(f"**{tf}**\n\n_ไม่มีข้อมูล_")
                            else:
                                col.markdown(
                                    f"**{tf}** {'✅' if v['ready'] else '—'}\n\n"
                                    f"- คะแนน: {v['score']}\n"
                                    f"- RSI: {v['rsi']}\n"
                                    f"- MACD: {'บวก' if v['macd_bull'] else 'ลบ'}\n"
                                    f"- Vol×: {v['vol_x']}\n"
                                    f"- BB: {v['bb']}{' 🔸' if v['squeeze'] else ''}\n"
                                    f"- ATR: {v['atr']}")
                        if API_KEY:
                            with st.spinner("AI กำลังวิเคราะห์..."):
                                payload = {"ticker": sym, "timeframes": res, "aligned": aligned}
                                as_of = datetime.now(timezone.utc).strftime("%Y%m%d%H")
                                st.markdown("---")
                                st.markdown(ai_analyze(sym, as_of, payload))

# ---------- แท็บดูรายตัว ----------
with tab_one:
    sym = st.text_input("สัญลักษณ์ (เช่น PTT, AOT, CBG)", value="PTT").strip().upper().replace(".BK", "")
    if st.button("วิเคราะห์ D / H / 15m", type="primary"):
        with st.spinner("กำลังดึง 3 timeframe..."):
            res, aligned = multi_tf(sym + ".BK")
        if all(v is None for v in res.values()):
            st.error("ไม่พบข้อมูล — ตรวจสัญลักษณ์อีกครั้ง")
        else:
            st.markdown(f"### {sym} — สอดคล้องกัน {aligned}/3 timeframe")
            cols = st.columns(3)
            for col, tf in zip(cols, ("D", "H", "15m")):
                v = res[tf]
                if v is None:
                    col.markdown(f"**{tf}**\n\n_ไม่มีข้อมูล_")
                else:
                    col.metric(f"{tf} · คะแนน", v["score"], "พร้อมวิ่ง" if v["ready"] else "ยัง")
                    col.markdown(
                        f"- ราคา: {v['close']}\n"
                        f"- RSI: {v['rsi']} · MACD: {'บวก' if v['macd_bull'] else 'ลบ'}\n"
                        f"- Vol×: {v['vol_x']} · ATR: {v['atr']}\n"
                        f"- BB: {v['bb']}{' 🔸squeeze' if v['squeeze'] else ''}\n"
                        f"- ขอบบน BB: {v['upper']} · กลาง: {v['basis']}")
            if API_KEY:
                with st.spinner("AI กำลังวิเคราะห์..."):
                    payload = {"ticker": sym, "timeframes": res, "aligned": aligned}
                    as_of = datetime.now(timezone.utc).strftime("%Y%m%d%H")
                    st.markdown("---")
                    st.subheader("บทวิเคราะห์ AI")
                    st.markdown(ai_analyze(sym, as_of, payload))
            else:
                st.info("ตั้ง ANTHROPIC_API_KEY ใน Secrets เพื่อเปิดบทวิเคราะห์ AI")
