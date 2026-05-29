# =============================================================
# PHASE 6 — app.py  (Premium Website UI Redesign)
# Smart AI Real Estate Advisor — Mohali
# =============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="NestIQ — Mohali Real Estate",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "mohali_properties_final.csv"

def get_groq_key() -> str:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")

GROQ_KEY = get_groq_key()
if GROQ_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_KEY

FURNISH_MAP = {0: "Unfurnished", 1: "Semi-Furnished", 2: "Furnished"}

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

# =============================================================
# MASTER CSS — Full Website Redesign
# =============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* ═══════════════════════════════════════
   GLOBAL RESET
═══════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #0a0a12 !important;
    font-family: 'Inter', sans-serif;
    color: #e8e8f0;
}

/* Kill all Streamlit padding */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

[data-testid="stAppViewContainer"] > .main {
    padding: 0 !important;
}

/* Hide the default Streamlit header */
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }

/* ═══════════════════════════════════════
   NAVBAR
═══════════════════════════════════════ */
.navbar {
    position: sticky;
    top: 0;
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 60px;
    height: 66px;
    background: rgba(10,10,18,0.92);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.navbar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    text-decoration: none;
}
.navbar-logo-text {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: #f0f0ff;
    letter-spacing: -0.03em;
    -webkit-text-fill-color: #f0f0ff;
}
.navbar-logo-text span {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.navbar-links {
    display: flex;
    gap: 36px;
    list-style: none;
    align-items: center;
}
.navbar-links a {
    text-decoration: none;
    font-size: 0.88rem;
    font-weight: 500;
    color: rgba(200,200,230,0.5);
    letter-spacing: 0.01em;
    pointer-events: none;
    cursor: default;
}
.nav-cta {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    padding: 9px 22px !important;
    border-radius: 100px !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    pointer-events: auto !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.35) !important;
}
.nav-cta:hover { opacity: 0.85 !important; }

/* ═══════════════════════════════════════
   HERO SECTION — Premium Dark Redesign
═══════════════════════════════════════ */
.hero {
    min-height: auto;
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    align-items: center;
    padding: 72px 64px 80px 64px;
    gap: 56px;
    background:
        radial-gradient(ellipse 80% 60% at 70% 50%, rgba(99,102,241,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 10% 80%, rgba(139,92,246,0.14) 0%, transparent 55%),
        linear-gradient(145deg, #06060f 0%, #0d0d1a 40%, #0f0b1e 70%, #07050d 100%);
    position: relative;
    overflow: hidden;
}

/* Subtle grid texture */
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(99,102,241,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    mask-image: radial-gradient(ellipse 70% 90% at 50% 50%, black 30%, transparent 100%);
    -webkit-mask-image: radial-gradient(ellipse 70% 90% at 50% 50%, black 30%, transparent 100%);
}

/* Glow orb */
.hero::after {
    content: '';
    position: absolute;
    top: -120px; right: -80px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(139,92,246,0.15) 0%, transparent 65%);
    pointer-events: none;
    animation: orbPulse 6s ease-in-out infinite;
}

@keyframes orbPulse {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.08); }
}

.hero-left { position: relative; z-index: 2; }

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.73rem;
    font-weight: 700;
    color: #a5b4fc;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 22px;
    animation: fadeSlideUp 0.6s ease both;
}

.hero-eyebrow::before {
    content: '';
    width: 6px; height: 6px;
    background: #818cf8;
    border-radius: 50%;
    box-shadow: 0 0 8px #818cf8;
    animation: blinkDot 2s ease-in-out infinite;
}

@keyframes blinkDot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

.hero-h1 {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: clamp(2.6rem, 4.5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.06;
    letter-spacing: -0.035em;
    color: #f0f0ff;
    margin-bottom: 18px;
    animation: fadeSlideUp 0.7s 0.1s ease both;
}

.hero-h1 em {
    font-style: normal;
    background: linear-gradient(125deg, #a5b4fc 0%, #c084fc 50%, #f0abfc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-desc {
    font-size: 1.0rem;
    line-height: 1.65;
    color: rgba(200,200,230,0.65);
    max-width: 440px;
    margin-bottom: 28px;
    font-weight: 400;
    animation: fadeSlideUp 0.7s 0.2s ease both;
}

/* ── Embedded Hero Search Box ── */
.hero-search-wrap {
    position: relative;
    margin-bottom: 28px;
    animation: fadeSlideUp 0.7s 0.3s ease both;
}

.hero-search-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(165,180,252,0.6);
    margin-bottom: 10px;
    display: block;
}

.hero-search-box {
    display: flex;
    align-items: center;
    background: rgba(255,255,255,0.05);
    border: 1.5px solid rgba(99,102,241,0.4);
    border-radius: 16px;
    padding: 4px 4px 4px 18px;
    gap: 10px;
    backdrop-filter: blur(12px);
    box-shadow:
        0 0 0 1px rgba(99,102,241,0.1) inset,
        0 8px 32px rgba(0,0,0,0.4),
        0 0 60px rgba(99,102,241,0.08);
    transition: border-color 0.3s, box-shadow 0.3s;
}

.hero-search-box:focus-within {
    border-color: rgba(139,92,246,0.7);
    box-shadow:
        0 0 0 1px rgba(139,92,246,0.2) inset,
        0 8px 32px rgba(0,0,0,0.4),
        0 0 40px rgba(139,92,246,0.18);
}

.hero-search-icon {
    font-size: 1.1rem;
    opacity: 0.5;
    flex-shrink: 0;
}

.hero-search-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 12px 22px;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
    box-shadow: 0 4px 20px rgba(99,102,241,0.5);
    letter-spacing: 0.01em;
    transition: all 0.2s;
    text-decoration: none !important;
    display: inline-block;
}

.hero-search-btn:hover {
    box-shadow: 0 6px 28px rgba(99,102,241,0.7);
    transform: translateY(-1px);
}

.hero-search-suggestions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 12px;
}

.hero-suggestion {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 100px;
    padding: 5px 13px;
    font-size: 0.76rem;
    font-weight: 500;
    color: rgba(165,180,252,0.8) !important;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
    text-decoration: none !important;
}

.hero-suggestion:hover {
    background: rgba(99,102,241,0.18);
    border-color: rgba(99,102,241,0.5);
    color: #a5b4fc !important;
}

/* ── Stats Row ── */
.hero-stats {
    display: flex;
    gap: 28px;
    padding-top: 24px;
    border-top: 1px solid rgba(255,255,255,0.07);
    animation: fadeSlideUp 0.7s 0.4s ease both;
}

.stat-item {}

.stat-number {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e0e7ff, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    display: block;
}

.stat-label {
    font-size: 0.76rem;
    color: rgba(200,200,230,0.45);
    font-weight: 500;
    margin-top: 2px;
}

/* ══════════════════════════════════
   HERO RIGHT — Animated Dashboard
══════════════════════════════════ */
.hero-right {
    position: relative;
    z-index: 2;
}

.dashboard-panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 24px;
    padding: 20px;
    backdrop-filter: blur(16px);
    box-shadow:
        0 24px 80px rgba(0,0,0,0.5),
        0 0 0 1px rgba(255,255,255,0.04) inset;
    animation: fadeSlideLeft 0.8s 0.2s ease both;
}

@keyframes fadeSlideLeft {
    from { opacity: 0; transform: translateX(30px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}

.dash-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

.dash-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: rgba(200,210,255,0.7);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.dash-live-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 100px;
    padding: 3px 10px;
    font-size: 0.68rem;
    font-weight: 700;
    color: #6ee7b7;
    letter-spacing: 0.06em;
}

.dash-live-dot {
    width: 5px; height: 5px;
    background: #34d399;
    border-radius: 50%;
    box-shadow: 0 0 6px #34d399;
    animation: blinkDot 1.5s ease-in-out infinite;
}

/* Property mini-cards inside dashboard */
.dcard {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    animation: floatY 5s ease-in-out infinite;
    cursor: default;
}

.dcard:nth-child(2) { animation-delay: -1.8s; }
.dcard:nth-child(3) { animation-delay: -3.2s; }

.dcard:hover {
    background: rgba(99,102,241,0.1);
    border-color: rgba(99,102,241,0.3);
    transform: translateX(4px) !important;
}

.dcard::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 0 2px 2px 0;
}
.dcard.top::before { background: linear-gradient(180deg, #6366f1, #8b5cf6); }
.dcard.mid::before { background: linear-gradient(180deg, #10b981, #06b6d4); }
.dcard.low::before { background: linear-gradient(180deg, #f59e0b, #ef4444); }

@keyframes floatY {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
}

.dcard-top-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 8px;
}

.dcard-loc {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: rgba(165,180,252,0.7);
    margin-bottom: 3px;
}

.dcard-name {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: rgba(240,240,255,0.92);
    letter-spacing: -0.01em;
}

.dcard-price {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    color: #e0e7ff;
    letter-spacing: -0.02em;
    text-align: right;
}

.dcard-price-sub {
    font-size: 0.66rem;
    color: rgba(200,200,230,0.4);
    font-weight: 400;
    display: block;
    text-align: right;
}

.dcard-bottom {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 8px;
}

.dcard-tags {
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
}

.dcard-tag {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 100px;
    padding: 2px 9px;
    font-size: 0.65rem;
    font-weight: 600;
    color: rgba(165,180,252,0.8);
}

.dcard-score-wrap {
    display: flex;
    align-items: center;
    gap: 5px;
}

.dcard-score-num {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 0.78rem;
    font-weight: 800;
    color: #a5b4fc;
}

.dcard-score-bar {
    width: 48px;
    height: 3px;
    background: rgba(255,255,255,0.1);
    border-radius: 2px;
    overflow: hidden;
}

.dcard-score-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    border-radius: 2px;
    animation: growBar 1.5s ease both;
}

@keyframes growBar {
    from { width: 0 !important; }
}

/* Summary stats strip at bottom of dashboard */
.dash-stats-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.06);
}

.dash-stat {
    text-align: center;
    padding: 8px 6px;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.05);
}

.dash-stat-num {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
    letter-spacing: -0.02em;
}

.dash-stat-lbl {
    font-size: 0.64rem;
    color: rgba(200,200,230,0.4);
    font-weight: 500;
    margin-top: 1px;
}

/* AI badge float */
.ai-badge-float {
    position: absolute;
    top: -14px; right: 12px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 100px;
    box-shadow: 0 4px 16px rgba(99,102,241,0.5);
    letter-spacing: 0.04em;
    z-index: 5;
}

/* floating-card legacy compat (used in old hero, keep for safety) */
.floating-card { display: none; }
.fcard-location, .fcard-title, .fcard-price, .fcard-tags, .fcard-tag,
.fcard-score, .score-bar, .score-fill { display: none; }

/* ═══════════════════════════════════════
   SECTION WRAPPER
═══════════════════════════════════════ */
.section {
    padding: 80px 60px;
    background: #0d0d18;
}
.section-alt {
    padding: 80px 60px;
    background: #111120;
    border-top: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.section-label-top {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 12px;
    display: block;
}
.section-h2 {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: clamp(1.8rem, 3vw, 2.6rem);
    font-weight: 800;
    letter-spacing: -0.025em;
    color: #f0f0ff;
    margin-bottom: 14px;
    line-height: 1.15;
}
.section-sub {
    font-size: 1.0rem;
    color: rgba(200,200,230,0.55);
    max-width: 520px;
    line-height: 1.65;
    margin-bottom: 48px;
}

/* ═══════════════════════════════════════
   SEARCH PANEL
═══════════════════════════════════════ */
.search-panel {
    background: #fff;
    border-radius: 24px;
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 20px 80px rgba(0,0,0,0.10);
    padding: 40px 44px;
    position: relative;
    overflow: visible;
}
.search-panel::before {
    content: '';
    position: absolute;
    inset: -1px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(102,126,234,0.3), rgba(118,75,162,0.15), transparent);
    z-index: -1;
}
.pill-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 28px;
}
.pill-btn {
    background: #f5f5f5;
    border: 1.5px solid transparent;
    border-radius: 100px;
    padding: 9px 18px;
    font-size: 0.84rem;
    font-weight: 500;
    color: #333;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}
.pill-btn:hover {
    background: #eef0ff;
    border-color: #667eea;
    color: #667eea;
}

/* ═══════════════════════════════════════
   PROPERTY RESULT CARDS
═══════════════════════════════════════ */
.result-card {
    background: #fff;
    border-radius: 20px;
    border: 1px solid rgba(0,0,0,0.07);
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    margin-bottom: 20px;
}
.result-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.12);
}
.result-card-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 18px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.rcard-rank {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: rgba(255,255,255,0.25);
    line-height: 1;
}
.rcard-type-badge {
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 100px;
    letter-spacing: 0.04em;
}
.result-card-body {
    padding: 22px 24px;
}
.rcard-location {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #667eea;
    margin-bottom: 6px;
}
.rcard-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #0a0a0a;
    margin-bottom: 4px;
    letter-spacing: -0.02em;
}
.rcard-price {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #0a0a0a;
    letter-spacing: -0.02em;
    margin-bottom: 16px;
}
.rcard-price-sub {
    font-size: 0.82rem;
    color: #888;
    font-weight: 400;
    margin-left: 6px;
    font-family: 'Inter', sans-serif;
}
.rcard-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.rcard-stat {
    background: #f8f8f8;
    border-radius: 12px;
    padding: 12px 14px;
}
.rcard-stat-val {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #0a0a0a;
    display: block;
}
.rcard-stat-key {
    font-size: 0.73rem;
    color: #888;
    font-weight: 500;
    margin-top: 1px;
}
.rcard-tags { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 16px; }
.rcard-tag {
    background: #eef0ff;
    color: #4f46e5;
    border-radius: 100px;
    padding: 5px 13px;
    font-size: 0.76rem;
    font-weight: 600;
}
.rcard-tag-green { background: #ecfdf5; color: #059669; }
.rcard-tag-amber { background: #fffbeb; color: #d97706; }
.rcard-ai {
    background: linear-gradient(135deg, #f8f8ff, #f0f0ff);
    border-left: 3px solid #667eea;
    border-radius: 12px;
    padding: 14px 16px;
    font-size: 0.88rem;
    color: #333;
    line-height: 1.65;
    margin-top: 4px;
}
.rcard-detail-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}
.rcard-detail-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #f4f4f6;
    border: 1px solid #e0e0e8;
    border-radius: 100px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: #444;
    font-family: 'Inter', sans-serif;
}
.rcard-detail-chip b { font-weight: 700; color: inherit; }

/* ═══════════════════════════════════════
   FEATURES / HOW IT WORKS
═══════════════════════════════════════ */
.features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
    margin-top: 48px;
}
.feature-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 32px 28px;
    transition: border-color 0.25s, background 0.25s;
}
.feature-card:hover {
    background: rgba(99,102,241,0.08);
    border-color: rgba(99,102,241,0.3);
}
.feature-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    margin-bottom: 20px;
}
.feature-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f0f0ff;
    margin-bottom: 10px;
}
.feature-desc { font-size: 0.9rem; color: rgba(200,200,230,0.55); line-height: 1.65; }

/* ═══════════════════════════════════════
   INVESTMENT / RAG SECTION
═══════════════════════════════════════ */
.rag-layout {
    display: grid;
    grid-template-columns: 1fr 1.4fr;
    gap: 60px;
    align-items: start;
}
.question-chip-grid {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.q-chip {
    background: #fff;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 14px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #333;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 500;
}
.q-chip:hover {
    background: #eef0ff;
    border-color: #667eea;
    color: #4f46e5;
}
.q-chip-icon { font-size: 1rem; flex-shrink: 0; }

/* ═══════════════════════════════════════
   MARKET INSIGHTS SECTION
═══════════════════════════════════════ */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-bottom: 40px;
}
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 22px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: rgba(99,102,241,0.3); }
.kpi-value {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    display: block;
}
.kpi-label { font-size: 0.78rem; color: rgba(200,200,230,0.45); font-weight: 500; margin-top: 4px; }

/* ═══════════════════════════════════════
   SECTOR TABLE
═══════════════════════════════════════ */
.stbl { width:100%; border-collapse:collapse; border-radius:16px; overflow:hidden; }
.stbl th {
    background: rgba(99,102,241,0.2);
    color: #a5b4fc;
    padding: 14px 16px;
    text-align: left;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.stbl td {
    padding: 13px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.88rem;
    color: rgba(220,220,240,0.8);
    background: rgba(255,255,255,0.02);
}
.stbl tr:last-child td { border-bottom: none; }
.stbl tr:hover td { background: rgba(99,102,241,0.07); }
.rating-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 100px;
    font-weight: 700;
    font-size: 0.82rem;
}

/* ═══════════════════════════════════════
   ANSWER BOX
═══════════════════════════════════════ */
.answer-panel {
    background: rgba(255,255,255,0.04);
    border-radius: 20px;
    border: 1px solid rgba(99,102,241,0.2);
    box-shadow: 0 8px 40px rgba(0,0,0,0.3);
    padding: 30px 32px;
    font-size: 0.96rem;
    color: rgba(220,220,240,0.85);
    line-height: 1.75;
}
.answer-panel-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.0rem;
    font-weight: 700;
    color: #a5b4fc;
}

/* ═══════════════════════════════════════
   FOOTER
═══════════════════════════════════════ */
.site-footer {
    background: #0a0a0a;
    color: rgba(255,255,255,0.5);
    text-align: center;
    padding: 40px 60px;
    font-size: 0.82rem;
    line-height: 1.8;
}
.site-footer strong { color: rgba(255,255,255,0.85); }

/* ═══════════════════════════════════════
   STREAMLIT WIDGET OVERRIDES — DARK THEME
═══════════════════════════════════════ */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1.5px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    color: #f0f0ff !important;
    padding: 14px 18px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.96rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #6366f1 !important;
    background: rgba(99,102,241,0.08) !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.15) !important;
}
.stTextInput > div > div > input::placeholder {
    color: rgba(200,200,230,0.35) !important;
}
.stTextInput label, .stTextArea label {
    color: rgba(200,200,230,0.6) !important;
    font-size: 0.82rem !important;
}
.stButton > button {
    background: rgba(255,255,255,0.06) !important;
    color: rgba(200,200,230,0.8) !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 100px !important;
    padding: 8px 16px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: none !important;
    transition: background 0.2s, border-color 0.2s !important;
    cursor: pointer !important;
    white-space: normal !important;
    line-height: 1.3 !important;
}
.stButton > button:hover {
    background: rgba(99,102,241,0.15) !important;
    border-color: rgba(99,102,241,0.4) !important;
    color: #a5b4fc !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 13px 32px !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    letter-spacing: 0.01em !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 28px rgba(99,102,241,0.6) !important;
    transform: translateY(-2px) !important;
    opacity: 0.92 !important;
    color: #fff !important;
    border: none !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1.5px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #f0f0ff !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Bricolage Grotesque', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: #a5b4fc !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.76rem !important;
    color: rgba(200,200,230,0.45) !important;
    font-weight: 500 !important;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    padding: 18px 20px !important;
}
/* ═══════════════════════════════════════
   TABS SECTION WRAPPER — DARK
═══════════════════════════════════════ */
.stTabs {
    padding: 0 !important;
}
.stTabs > div:first-child {
    padding: 24px 60px 0 60px !important;
    background: #111120 !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-radius: 0 !important;
    padding: 0 !important;
    gap: 4px !important;
    border: none !important;
}
.stTabs button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: rgba(200,200,230,0.45) !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 12px 24px !important;
    transition: all 0.2s !important;
    border-bottom: 3px solid transparent !important;
    background: transparent !important;
}
.stTabs button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(99,102,241,0.1) !important;
    color: #a5b4fc !important;
    border-bottom: 3px solid #6366f1 !important;
    box-shadow: none !important;
}
[data-testid="stTabsContent"] {
    background: #0d0d18 !important;
    padding: 40px 60px !important;
}

/* ═══════════════════════════════════════
   SEARCH CARD (Tab 1) — DARK
═══════════════════════════════════════ */
.search-card {
    background: rgba(99,102,241,0.06);
    border: 1.5px solid rgba(99,102,241,0.2);
    border-radius: 20px;
    padding: 28px 32px 20px 32px;
    margin-bottom: 20px;
}

.sc-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 0;
}

.sc-eyebrow {
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 8px;
}

.sc-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: #f0f0ff;
    margin-bottom: 6px;
    line-height: 1.2;
}

.sc-sub {
    font-size: 0.88rem;
    color: rgba(200,200,230,0.55);
    margin: 0;
    line-height: 1.55;
    max-width: 500px;
}

.sc-badge-wrap {
    display: flex;
    gap: 8px;
    flex-shrink: 0;
    padding-top: 4px;
}

.sc-badge {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 100px;
    padding: 5px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #a5b4fc;
    white-space: nowrap;
}

.sc-divider {
    height: 1px;
    background: linear-gradient(90deg, #e2e0f5, transparent);
    margin: 18px 0 14px 0;
}

.sc-chips-label {
    font-size: 0.71rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #aaa;
    margin: 0 0 10px 0 !important;
}

.sc-input-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #555;
    margin: 16px 0 6px 0 !important;
}

.sc-note {
    display: flex;
    align-items: center;
    height: 100%;
    padding-top: 2px;
    font-size: 0.78rem;
    color: #aaa;
    font-style: italic;
    letter-spacing: 0.01em;
}
.streamlit-expanderHeader {
    background: #fff !important;
    border: 1px solid rgba(0,0,0,0.07) !important;
    border-radius: 14px !important;
    color: #0a0a0a !important;
    font-family: 'Bricolage Grotesque', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 16px 22px !important;
}
.streamlit-expanderContent {
    background: #fff !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-top: none !important;
    border-radius: 0 0 14px 14px !important;
    padding: 20px !important;
}
.stSuccess {
    background: #ecfdf5 !important;
    border: 1px solid rgba(16,185,129,0.25) !important;
    border-radius: 12px !important;
    color: #065f46 !important;
}
.stWarning {
    background: #fffbeb !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
    border-radius: 12px !important;
}
.stError {
    background: #fff1f2 !important;
    border: 1px solid rgba(244,63,94,0.25) !important;
    border-radius: 12px !important;
}
.stSpinner > div { border-top-color: #667eea !important; }
hr { border: none !important; border-top: 1px solid #f0f0f0 !important; margin: 32px 0 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #555; }
[data-testid="stSidebar"] { background: #0d0d1a !important; border-right: 1px solid rgba(255,255,255,0.07) !important; }

/* ── Responsive Hero ── */
@media (max-width: 900px) {
    .hero {
        grid-template-columns: 1fr !important;
        padding: 48px 24px 40px !important;
        min-height: auto !important;
        gap: 32px !important;
    }
    .hero-h1 { font-size: 2.2rem !important; }
    .dashboard-panel { margin-top: 0; }
    .hero-search-suggestions { display: none; }
    .dash-stats-strip { grid-template-columns: repeat(3, 1fr); }
}

@media (max-width: 600px) {
    .hero { padding: 36px 16px 32px !important; }
    .hero-h1 { font-size: 1.9rem !important; }
    .hero-stats { gap: 20px !important; }
    .stat-number { font-size: 1.4rem !important; }
    .dcard { padding: 12px 12px !important; }
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# NAVBAR
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<nav class="navbar">
<a class="navbar-logo" href="#">
<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect width="32" height="32" rx="9" fill="url(#logoGrad)"/>
<path d="M7 17L16 9L25 17V26H20V21H12V26H7V17Z" fill="white" opacity="0.95"/>
<circle cx="22" cy="12" r="3" fill="white" opacity="0.7"/>
<defs>
<linearGradient id="logoGrad" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
<stop stop-color="#6366f1"/>
<stop offset="1" stop-color="#8b5cf6"/>
</linearGradient>
</defs>
</svg>
<span class="navbar-logo-text">Nest<span>IQ</span></span>
</a>
<ul class="navbar-links">
<li><a href="#search-section">Search</a></li>
<li><a href="#invest-section">Invest</a></li>
<li><a href="#insights-section">Insights</a></li>
<li><a href="#search-section" class="nav-cta">Get Started →</a></li>
</ul>
</nav>
<div id="top"></div>
""", unsafe_allow_html=True)

# API key warning (minimal)
if not GROQ_KEY:
    st.markdown("""
    <div style="background:#fff3cd; border-left:4px solid #f59e0b; padding:12px 20px; font-size:0.87rem; color:#92400e; font-family:Inter,sans-serif;">
        ⚠️ <b>GROQ_API_KEY not set</b> — AI search & explanations are disabled. Set it in environment or Streamlit Secrets.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<section class="hero">
<div class="hero-left">
<div class="hero-eyebrow">AI-Powered Real Estate Intelligence</div>
<div class="hero-h1">Find your next home<br>in <em>Mohali</em> —<br>just describe it.</div>
<p class="hero-desc">Skip the filters. Tell our AI what you need in plain English and get ranked, scored properties that actually match your life.</p>
<div class="hero-search-wrap">
<span class="hero-search-label">Try a search</span>
<div class="hero-search-box">
<span class="hero-search-icon">🔍</span>
<span style="font-size:0.9rem;color:rgba(200,200,230,0.45);font-style:italic;flex:1;pointer-events:none;">e.g. "3BHK near hospital under 80L in sector 70"</span>
<a href="#search-section" class="hero-search-btn">Search with AI →</a>
</div>
<div class="hero-search-suggestions">
<a href="#search-section" class="hero-suggestion">Luxury villa sector 82</a>
<a href="#search-section" class="hero-suggestion">2BHK under 60L</a>
<a href="#search-section" class="hero-suggestion">Near metro furnished</a>
<a href="#search-section" class="hero-suggestion">Best investment</a>
</div>
</div>
<div class="hero-stats">
<div class="stat-item"><span class="stat-number">540+</span><span class="stat-label">Properties Listed</span></div>
<div class="stat-item"><span class="stat-number">18</span><span class="stat-label">Mohali Sectors</span></div>
<div class="stat-item"><span class="stat-number">80.9%</span><span class="stat-label">ML Accuracy</span></div>
</div>
</div>
<div class="hero-right">
<div class="dashboard-panel">
<div class="dash-header">
<span class="dash-title">🏙 NestIQ Matches</span>
<span class="dash-live-badge"><span class="dash-live-dot"></span> LIVE</span>
</div>
<div class="dcard top" style="position:relative;">
<span class="ai-badge-float">🤖 Top Pick</span>
<div class="dcard-top-row">
<div><div class="dcard-loc">📍 Sector 82, Mohali</div><div class="dcard-name">4 BHK Luxury Villa</div></div>
<div><div class="dcard-price">₹3.8 Cr</div><span class="dcard-price-sub">3,200 sqft</span></div>
</div>
<div class="dcard-bottom">
<div class="dcard-tags"><span class="dcard-tag">🏥 Hospital</span><span class="dcard-tag">🌳 Park</span><span class="dcard-tag">Furnished</span></div>
<div class="dcard-score-wrap"><span class="dcard-score-num">9.2</span><div class="dcard-score-bar"><div class="dcard-score-fill" style="width:92%"></div></div></div>
</div>
</div>
<div class="dcard mid">
<div class="dcard-top-row">
<div><div class="dcard-loc">📍 Sector 70, Mohali</div><div class="dcard-name">3 BHK Premium Flat</div></div>
<div><div class="dcard-price">₹1.9 Cr</div><span class="dcard-price-sub">1,850 sqft</span></div>
</div>
<div class="dcard-bottom">
<div class="dcard-tags"><span class="dcard-tag">🛍️ Mall</span><span class="dcard-tag">Semi-Furnished</span></div>
<div class="dcard-score-wrap"><span class="dcard-score-num">9.0</span><div class="dcard-score-bar"><div class="dcard-score-fill" style="width:90%"></div></div></div>
</div>
</div>
<div class="dcard low">
<div class="dcard-top-row">
<div><div class="dcard-loc">📍 Sector 68, Mohali</div><div class="dcard-name">2 BHK Family Flat</div></div>
<div><div class="dcard-price">₹85 L</div><span class="dcard-price-sub">1,100 sqft</span></div>
</div>
<div class="dcard-bottom">
<div class="dcard-tags"><span class="dcard-tag">🔥 Underpriced</span><span class="dcard-tag">New Build</span></div>
<div class="dcard-score-wrap"><span class="dcard-score-num">8.0</span><div class="dcard-score-bar"><div class="dcard-score-fill" style="width:80%"></div></div></div>
</div>
</div>
<div class="dash-stats-strip">
<div class="dash-stat"><span class="dash-stat-num">₹1.4Cr</span><div class="dash-stat-lbl">Avg Price</div></div>
<div class="dash-stat"><span class="dash-stat-num">8.7</span><div class="dash-stat-lbl">Avg IQ Score</div></div>
<div class="dash-stat"><span class="dash-stat-num">12%</span><div class="dash-stat-lbl">Appreciation</div></div>
</div>
</div>
</div>
</section>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HOW IT WORKS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<section class="section-alt">
  <span class="section-label-top">How It Works</span>
  <h2 class="section-h2">Three steps to your<br>perfect property</h2>
  <p class="section-sub">No filters to fiddle with. Just describe what you want, and our AI does the heavy lifting.</p>
  <div class="features-grid">
    <div class="feature-card">
      <div class="feature-icon">💬</div>
      <div class="feature-title">Describe in Plain English</div>
      <p class="feature-desc">Say things like "3BHK near hospital under 1 crore in sector 70." Our Llama 3.1 NLP understands context, budget ranges, and location nuance.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">🧠</div>
      <div class="feature-title">AI Ranks & Scores</div>
      <p class="feature-desc">XGBoost ML predicts fair market price. Every property gets a value score — so you instantly know if it's underpriced or overpriced.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">📊</div>
      <div class="feature-title">Get Investment Insight</div>
      <p class="feature-desc">Our RAG knowledge base grounds every answer in real Mohali sector data — appreciation rates, traffic levels, proximity scores.</p>
    </div>
  </div>
</section>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MAIN APP TABS
# ═══════════════════════════════════════════════════════════════
st.markdown('<div id="search-section"></div>', unsafe_allow_html=True)
st.markdown('<section class="section">', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍  Find Properties", "💡  Investment Advisor", "📊  Market Insights"])

# ╔═══════════════════════════════════════════════════════════╗
# ║  TAB 1 — PROPERTY SEARCH                                  ║
# ╚═══════════════════════════════════════════════════════════╝
with tab1:

    st.markdown("""
    <div class="search-card">

      <!-- Header row -->
      <div class="sc-header">
        <div>
          <div class="sc-eyebrow">🏠 AI Property Search</div>
          <h2 class="sc-title">Find properties in plain English</h2>
          <p class="sc-sub">Describe what you need — budget, BHK, sector, amenities. Our AI understands everything.</p>
        </div>
        <div class="sc-badge-wrap">
          <span class="sc-badge">🤖 Llama 3.1</span>
          <span class="sc-badge">📊 XGBoost</span>
        </div>
      </div>

      <!-- Divider -->
      <div class="sc-divider"></div>

      <!-- Quick examples label -->
      <p class="sc-chips-label">✦ Quick examples — click to try</p>

    </div>
    """, unsafe_allow_html=True)

    # Example chip buttons — compact, auto-width
    ex_cols = st.columns(4)
    examples = [
        "3BHK near hospital under 1 crore",
        "Luxury villa in sector 82",
        "2BHK furnished sector 70",
        "3BHK near park under 1.5 crore",
    ]
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex, key=f"ex_{i}"):
            st.session_state["search_query"] = ex

    # Search input
    st.markdown('<p class="sc-input-label">Your search query</p>', unsafe_allow_html=True)
    query = st.text_input(
        label="Your query",
        value=st.session_state.get("search_query", ""),
        placeholder="e.g.  3BHK near hospital under 80 lakhs in sector 70 — furnished",
        label_visibility="collapsed",
    )

    # CTA row
    col_btn, col_note = st.columns([1, 2])
    with col_btn:
        search_btn = st.button("Find My Properties →", type="primary", use_container_width=True)
    with col_note:
        st.markdown("""
        <div class="sc-note">
          <span>⚡ Powered by Groq · Llama 3.1 · XGBoost ML</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    if search_btn and query:
        if not GROQ_KEY:
            st.error("GROQ_API_KEY is not set. Cannot run search.")
        else:
            with st.spinner("🤖 AI is analysing your requirements..."):
                try:
                    from recommender import recommend_properties
                    from query_processor import explain_filters, generate_recommendation_text
                    results_df, filters, warning = recommend_properties(query, top_n=5, verbose=False)
                except Exception as e:
                    st.error(f"Error: {e}")
                    results_df = pd.DataFrame()
                    filters = {}
                    warning = None

            if filters:
                from query_processor import explain_filters
                explanation = explain_filters(filters, query)
                st.markdown(
                    f'<div style="background:rgba(99,102,241,0.12);border-left:3px solid #6366f1;border-radius:12px;padding:14px 18px;font-size:0.88rem;color:#a5b4fc;margin:20px 0;font-family:Inter,sans-serif;">🧠 <b style="color:#c7d2fe;">AI understood:</b> {explanation}</div>',
                    unsafe_allow_html=True
                )

            if warning:
                st.markdown(
                    f'<div style="background:rgba(245,158,11,0.1);border-left:3px solid #f59e0b;border-radius:12px;padding:12px 16px;font-size:0.87rem;color:#fcd34d;margin-bottom:16px;font-family:Inter,sans-serif;">⚠️ {warning}</div>',
                    unsafe_allow_html=True
                )

            if results_df.empty:
                st.warning("No properties found. Try a broader query.")
            else:
                st.markdown(
                    f'<div style="margin:28px 0 20px 0;">'
                    f'<div style="font-family:Bricolage Grotesque,sans-serif;font-size:1.6rem;font-weight:800;color:#f0f0ff;letter-spacing:-0.02em;margin-bottom:4px;">{len(results_df)} Properties Found</div>'
                    f'<p style="color:rgba(200,200,230,0.5);font-size:0.88rem;font-family:Inter,sans-serif;">Ranked by AI score — best match first</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                for rank, (_, prop) in enumerate(results_df.iterrows(), start=1):
                    prop_type  = prop.get("Property_Type", "Property")
                    sector     = prop.get("Sector", "?")
                    bhk        = prop.get("BHK", "?")
                    price      = prop.get("Price_Lakh", 0)
                    area       = prop.get("Area_sqft", 0)
                    pred_price = prop.get("predicted_price", price)
                    val_score  = prop.get("value_score", 0)
                    fin_score  = prop.get("final_score", 0)
                    loc_score  = prop.get("Location_Score", 0)
                    inv_score  = prop.get("Investment_Score", 0)
                    amenities  = prop.get("Amenities_Score", 0)
                    near_hosp  = prop.get("Near_Hospital", 0)
                    near_park  = prop.get("Near_Park", 0)
                    near_mall  = prop.get("Near_Mall", 0)
                    traffic    = prop.get("Traffic_Level_Label", "?")
                    raw_furn   = prop.get("Furnishing", "?")
                    furnishing = FURNISH_MAP.get(raw_furn, raw_furn)
                    age        = prop.get("Age", "?")
                    floor      = prop.get("Floor", "?")
                    parking    = prop.get("Parking", 0)

                    # ── Score normalization ──────────────────────────────
                    # Amenities_Score is 0–5 → display as X/5
                    amenities_display = f"{amenities}/5"

                    # Investment_Score: if raw value ≤ 10 assume it's already 0–10
                    # but values like 4.9 are correct on a 10-point scale — keep as is
                    inv_display = f"{inv_score:.1f}/10"

                    # Location_Score: already 0–10
                    loc_display = f"{loc_score:.1f}/10"

                    # final_score is an ML probability (0–1) → convert to % match score
                    match_pct = min(100, round(fin_score * 100))

                    # Traffic badge color
                    traffic_lower = str(traffic).lower()
                    if "high" in traffic_lower:
                        traffic_color = "#ef4444"
                        traffic_bg    = "#fff1f2"
                    elif "medium" in traffic_lower or "med" in traffic_lower:
                        traffic_color = "#f59e0b"
                        traffic_bg    = "#fffbeb"
                    else:
                        traffic_color = "#10b981"
                        traffic_bg    = "#ecfdf5"

                    parking_icon = "✅ Yes" if parking else "❌ No"

                    # Deal tag
                    deal_pct = val_score * 100
                    if deal_pct > 5:
                        deal_class = "rcard-tag rcard-tag-green"
                        deal_label = f"🔥 {deal_pct:.0f}% Underpriced"
                    elif deal_pct < -5:
                        deal_class = "rcard-tag rcard-tag-amber"
                        deal_label = f"⚠️ {abs(deal_pct):.0f}% Overpriced"
                    else:
                        deal_class = "rcard-tag"
                        deal_label = "✅ Fair Price"

                    tags_html = f'<span class="{deal_class}">{deal_label}</span>'
                    if near_hosp: tags_html += '<span class="rcard-tag rcard-tag-green">🏥 Hospital</span>'
                    if near_park: tags_html += '<span class="rcard-tag rcard-tag-green">🌳 Park</span>'
                    if near_mall: tags_html += '<span class="rcard-tag rcard-tag-green">🛍️ Mall</span>'

                    card_html = (
                        f'<div class="result-card">'
                        f'<div class="result-card-header">'
                        f'<div class="rcard-rank">#{rank:02d}</div>'
                        f'<div style="text-align:center;font-family:Bricolage Grotesque,sans-serif;font-size:1.1rem;font-weight:800;color:white;letter-spacing:-0.01em;">Sector {sector} · {bhk} BHK {prop_type}</div>'
                        f'<span class="rcard-type-badge">{prop_type.upper()}</span>'
                        f'</div>'
                        f'<div class="result-card-body">'
                        f'<div class="rcard-location">📍 Sector {sector}, Mohali</div>'
                        f'<div class="rcard-title">{bhk} BHK {prop_type}</div>'
                        f'<div class="rcard-price">₹{price:.0f}L <span class="rcard-price-sub">· {area:.0f} sqft · Floor {floor}</span></div>'
                        f'<div class="rcard-tags">{tags_html}</div>'
                        f'<div class="rcard-grid">'
                        f'<div class="rcard-stat"><span class="rcard-stat-val">{loc_display}</span><span class="rcard-stat-key">Location Score</span></div>'
                        f'<div class="rcard-stat"><span class="rcard-stat-val">{amenities_display}</span><span class="rcard-stat-key">Amenities Score</span></div>'
                        f'<div class="rcard-stat"><span class="rcard-stat-val">{inv_display}</span><span class="rcard-stat-key">Investment Score</span></div>'
                        f'<div class="rcard-stat"><span class="rcard-stat-val">₹{pred_price:.0f}L</span><span class="rcard-stat-key">ML Fair Price</span></div>'
                        f'<div class="rcard-stat"><span class="rcard-stat-val">{furnishing}</span><span class="rcard-stat-key">Furnishing</span></div>'
                        f'<div class="rcard-stat"><span class="rcard-stat-val">{age} yrs</span><span class="rcard-stat-key">Property Age</span></div>'
                        f'</div>'
                        f'<div class="rcard-detail-row">'
                        f'<span class="rcard-detail-chip" style="background:{traffic_bg};color:{traffic_color};border-color:{traffic_color}40;">🚦 Traffic: <b>{traffic}</b></span>'
                        f'<span class="rcard-detail-chip">🚗 Parking: <b>{parking_icon}</b></span>'
                        f'<span class="rcard-detail-chip" style="background:rgba(99,102,241,0.15);color:#a5b4fc;border-color:rgba(99,102,241,0.3);">🏆 AI Match: <b>{match_pct}%</b></span>'
                        f'</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

                    with st.spinner("Generating AI insight..."):
                        try:
                            ai_text = generate_recommendation_text(prop.to_dict(), rank)
                            st.markdown(f'<div class="rcard-ai">🤖 <b>AI Insight:</b> {ai_text}</div>', unsafe_allow_html=True)
                        except Exception:
                            st.markdown(f'<div class="rcard-ai">This {bhk} BHK in Sector {sector} offers great value at ₹{price:.0f}L with a location score of {loc_score:.1f}/10.</div>', unsafe_allow_html=True)

                    st.markdown('</div></div>', unsafe_allow_html=True)

    elif search_btn and not query:
        st.warning("Please enter a search query.")


# ╔═══════════════════════════════════════════════════════════╗
# ║  TAB 2 — INVESTMENT ADVISOR (RAG)                         ║
# ╚═══════════════════════════════════════════════════════════╝
with tab2:

    st.markdown("""
    <div style="margin: 32px 0 0 0;">
      <span class="section-label-top">Investment Advisor</span>
      <h2 class="section-h2" style="margin-bottom:8px;">Ask anything about<br>Mohali real estate</h2>
      <p class="section-sub">Answers grounded in local sector data — not generic AI guesses.</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.5], gap="large")

    with col_left:
        st.markdown('<p style="font-size:0.78rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#999; margin-bottom:14px; margin-top:28px;">Common Questions</p>', unsafe_allow_html=True)

        common_qs = [
            ("🏆", "Best sector for investment?"),
            ("🏙️", "Mohali vs Chandigarh?"),
            ("💰", "Budget of 1.2 crore — what to buy?"),
            ("📊", "Sector 70 vs Sector 82?"),
            ("👨‍👩‍👧", "Best sectors for families?"),
            ("📈", "Appreciation in Sector 70?"),
        ]
        full_qs = [
            "Which sector in Mohali is best for investment?",
            "Should I buy in Mohali or Chandigarh?",
            "I have a budget of 1.2 crore. What should I buy?",
            "Compare Sector 70 vs Sector 82 for investment",
            "Which sectors are best for families with children?",
            "What is the expected appreciation in Sector 70?",
        ]
        for i, (icon, label) in enumerate(common_qs):
            if st.button(f"{icon}  {label}", key=f"cq_{i}", use_container_width=True):
                st.session_state["rag_query"] = full_qs[i]

    with col_right:
        st.markdown('<div style="margin-top:28px;"></div>', unsafe_allow_html=True)

        rag_query = st.text_input(
            label="Your question",
            value=st.session_state.get("rag_query", ""),
            placeholder="e.g.  Which sector has the best appreciation rate?",
            label_visibility="collapsed",
        )
        ask_btn = st.button("💡  Get Investment Advice", type="primary", use_container_width=True)

        if ask_btn and rag_query:
            if not GROQ_KEY:
                st.error("GROQ_API_KEY is not set. Cannot answer.")
            else:
                with st.spinner("📚 Searching knowledge base..."):
                    try:
                        from rag_system import answer_investment_question
                        answer = answer_investment_question(rag_query)
                        st.markdown(
                            f'<div class="answer-panel"><div class="answer-panel-header"><span>💡</span> AI Investment Advice</div>{answer}</div>',
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"RAG error: {e}")
        elif ask_btn and not rag_query:
            st.warning("Please enter a question.")

    st.markdown("---")

    st.markdown(
        '<div style="font-family:Bricolage Grotesque,sans-serif;font-size:1.4rem;font-weight:800;color:#f0f0ff;letter-spacing:-0.02em;margin-bottom:20px;">📋 Sector Quick Reference</div>',
        unsafe_allow_html=True
    )

    sector_data = {
        "Sector": [66,67,68,69,70,71,74,75,76,77,78,79,80,81,82,83,84,85],
        "Avg Price (₹L)": ["120-180","100-160","150-220","130-190","180-280","160-240","90-140","100-150","110-170","130-200","150-230","120-180","170-260","140-210","250-500","110-170","90-130","80-120"],
        "Appreciation": ["8-10%","8-10%","10-12%","9-11%","13-15%","11-13%","7-9%","7-9%","8-10%","9-11%","10-12%","8-10%","12-14%","10-12%","15-18%","7-9%","6-8%","5-8%"],
        "Rating": [7.0,7.5,8.0,7.8,9.0,8.5,7.0,7.3,7.6,8.0,8.3,7.9,8.7,8.2,9.2,7.4,7.1,6.8],
        "Best For": ["Professionals","First buyers","Families","Families","IT/Investors","Professionals","Budget","Budget","Families","Families","Premium","Families","IT/Investors","Mid-range","Luxury/HNI","Budget","Very budget","Very budget"],
    }
    ref_df = pd.DataFrame(sector_data)

    def get_rating_color(rating):
        if rating >= 9.0: return "background:#dcfce7; color:#166534;"
        elif rating >= 8.5: return "background:#dbeafe; color:#1d4ed8;"
        elif rating >= 8.0: return "background:#ede9fe; color:#5b21b6;"
        elif rating < 7.2: return "background:#fef9c3; color:#854d0e;"
        return "background:#f5f5f5; color:#333;"

    html_table = '<div style="overflow-x:auto;"><table class="stbl"><thead><tr>'
    for col in ref_df.columns:
        html_table += f"<th>{col}</th>"
    html_table += "</tr></thead><tbody>"
    for _, row in ref_df.iterrows():
        html_table += "<tr>"
        for col in ref_df.columns:
            val = row[col]
            if col == "Rating":
                style = get_rating_color(val)
                html_table += f'<td><span class="rating-pill" style="{style}">{val}</span></td>'
            else:
                html_table += f"<td>{val}</td>"
        html_table += "</tr>"
    html_table += "</tbody></table></div>"
    st.markdown(html_table, unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════╗
# ║  TAB 3 — MARKET INSIGHTS                                  ║
# ╚═══════════════════════════════════════════════════════════╝
with tab3:

    st.markdown("""
    <div style="margin: 32px 0 0 0;">
      <span class="section-label-top">Market Insights</span>
      <h2 class="section-h2" style="margin-bottom:8px;">Mohali property market<br>at a glance</h2>
    </div>
    """, unsafe_allow_html=True)

    try:
        df = load_data()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Properties",   f"{len(df):,}")
        k2.metric("Avg Price",          f"₹{df['Price_Lakh'].mean():.0f}L")
        k3.metric("Avg Area",           f"{df['Area_sqft'].mean():.0f} sqft")
        k4.metric("Sectors Covered",    f"{df['Sector'].nunique()}")
        k5.metric("Avg Invest Score",   f"{df['Investment_Score'].mean():.1f}/10" if "Investment_Score" in df.columns else "N/A")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Average Price by Sector")
            sector_price = df.groupby("Sector")["Price_Lakh"].mean().round(1).reset_index().sort_values("Price_Lakh", ascending=False)
            sector_price.columns = ["Sector", "Avg Price (₹L)"]
            sector_price["Sector"] = "S" + sector_price["Sector"].astype(str)
            st.bar_chart(sector_price.set_index("Sector"), height=300)
        with c2:
            st.markdown("#### Investment Score by Sector")
            if "Investment_Score" in df.columns and pd.api.types.is_numeric_dtype(df["Investment_Score"]):
                inv_score = df.groupby("Sector")["Investment_Score"].mean().round(2).reset_index().sort_values("Investment_Score", ascending=False)
                inv_score.columns = ["Sector", "Investment Score"]
                inv_score["Sector"] = "S" + inv_score["Sector"].astype(str)
                st.bar_chart(inv_score.set_index("Sector"), height=300)
            else:
                st.info("Investment_Score column not available.")

        st.markdown("---")
        c3, c4, c5 = st.columns(3)
        with c3:
            st.markdown("#### Property Type Mix")
            type_dist = df["Property_Type"].value_counts().reset_index()
            type_dist.columns = ["Type", "Count"]
            st.bar_chart(type_dist.set_index("Type"), height=260)
        with c4:
            st.markdown("#### Price by BHK")
            bhk_price = df.groupby("BHK")["Price_Lakh"].mean().round(1).reset_index()
            bhk_price.columns = ["BHK", "Avg Price (₹L)"]
            bhk_price["BHK"] = bhk_price["BHK"].astype(str) + " BHK"
            st.bar_chart(bhk_price.set_index("BHK"), height=260)
        with c5:
            st.markdown("#### Furnishing Breakdown")
            furn_dist = df["Furnishing"].value_counts().reset_index()
            furn_dist.columns = ["Furnishing", "Count"]
            st.bar_chart(furn_dist.set_index("Furnishing"), height=260)

        st.markdown("---")
        st.markdown("#### Proximity Overview")
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Near Hospital",  f"{df['Near_Hospital'].sum()} ({df['Near_Hospital'].mean()*100:.0f}%)")
        p2.metric("Near Park",      f"{df['Near_Park'].sum()} ({df['Near_Park'].mean()*100:.0f}%)")
        p3.metric("Near Mall",      f"{df['Near_Mall'].sum()} ({df['Near_Mall'].mean()*100:.0f}%)")
        p4.metric("Metro Access",   f"{df['Nearby_Metro'].sum()} ({df['Nearby_Metro'].mean()*100:.0f}%)")
        p5.metric("Has Parking",    f"{df['Parking'].sum()} ({df['Parking'].mean()*100:.0f}%)")

        st.markdown("---")
        c6, c7 = st.columns(2)
        with c6:
            st.markdown("#### Price by Traffic Level")
            tp = df.groupby("Traffic_Level_Label")["Price_Lakh"].mean().round(1).reset_index()
            tp.columns = ["Traffic Level", "Avg Price (₹L)"]
            st.bar_chart(tp.set_index("Traffic Level"), height=260)
        with c7:
            st.markdown("#### Property Age Distribution")
            age_bins = pd.cut(df["Age"], bins=[0,2,5,10,15,30], labels=["0-2yr","3-5yr","6-10yr","11-15yr","15+yr"])
            age_dist = age_bins.value_counts().sort_index().reset_index()
            age_dist.columns = ["Age Group", "Count"]
            st.bar_chart(age_dist.set_index("Age Group"), height=260)

        st.markdown("---")
        with st.expander("🔎 Explore Raw Dataset"):
            cf1, cf2, cf3 = st.columns(3)
            sel_sector = cf1.selectbox("Sector",        ["All"] + sorted(df["Sector"].unique().tolist()))
            sel_type   = cf2.selectbox("Property Type", ["All"] + df["Property_Type"].unique().tolist())
            sel_bhk    = cf3.selectbox("BHK",           ["All"] + sorted(df["BHK"].unique().tolist()))
            vdf = df.copy()
            if sel_sector != "All": vdf = vdf[vdf["Sector"] == sel_sector]
            if sel_type   != "All": vdf = vdf[vdf["Property_Type"] == sel_type]
            if sel_bhk    != "All": vdf = vdf[vdf["BHK"] == sel_bhk]
            dcols = ["Sector","Property_Type","BHK","Area_sqft","Price_Lakh","Furnishing","Location_Score","Near_Hospital","Near_Park","Near_Mall","Traffic_Level_Label"]
            if "Investment_Score" in df.columns and pd.api.types.is_numeric_dtype(df["Investment_Score"]):
                dcols.insert(7, "Investment_Score")
            st.dataframe(vdf[dcols].reset_index(drop=True), use_container_width=True, height=400)
            st.caption(f"Showing {len(vdf)} of {len(df)} properties")

    except FileNotFoundError:
        st.error(f"Dataset not found at {DATA_PATH}")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown('</section>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<footer class="site-footer">
    <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:1.3rem; font-weight:800;
         color:white; margin-bottom:10px;">PropAI</div>
    <div>
        Smart AI Real Estate Advisor for Mohali &nbsp;·&nbsp;
        Built with <strong>XGBoost + Groq Llama 3.1 + RAG + Streamlit</strong>
    </div>
    <div style="margin-top:8px; font-size:0.78rem; color:rgba(255,255,255,0.3);">
        ⚠️ For informational purposes only. Always verify with a licensed real estate agent.
    </div>
</footer>
""", unsafe_allow_html=True)