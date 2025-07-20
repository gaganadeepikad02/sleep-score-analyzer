# ---------------- Imports ----------------
import streamlit as st
import mne
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tempfile
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import requests

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Sleep Score Analyzer", layout="wide")

# ---------------- Load Theme & CSS ----------------
st.markdown("""
    <style>
    /* Base: Clean light background */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f9f9fb !important;
        font-family: 'Segoe UI', sans-serif;
        color: #1e1e1e;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #eef2f7 !important;
        border-right: 1px solid #d6dbe2;
    }

    /* Titles */
    h1, h2, h3 {
        color: #243b55 !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* Paragraphs and general text */
    p, span, label, div, .css-18ni7ap, .css-1cpxqw2 {
        color: #1e1e1e !important;
        font-weight: 500 !important;
    }

    /* File uploader and labels */
    label, .stFileUploader {
        font-weight: 600 !important;
        color: #37474f !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1976d2;
        color: #ffffff;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.6em 1.2em;
        border: none;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #1565c0;
        transform: scale(1.01);
    }

    /* Metric Box */
    .metric-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 6px rgba(0, 0, 0, 0.05);
        transition: 0.2s ease-in-out;
    }

    .metric-container:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }

    /* Subheader styling */
    .stMarkdown h2 {
        color: #1a2b3c !important;
        font-weight: 600;
        margin-top: 1.5rem;
    }

    /* Success, info, warning messages */
    .stAlert {
        border-radius: 6px;
        padding: 1rem;
    }

    </style>
""", unsafe_allow_html=True)

# ---------------- Load Model and Scaler ----------------
@st.cache_resource
def load_model_and_scaler():
    model = load_model("sleep_score_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler
model, scaler = load_model_and_scaler()

# ---------------- Lottie Loader ----------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_sleep = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_tll0j4bb.json")

# ---------------- Prediction Function ----------------
def predict_sleep_stages(psg_path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, stim_channel=None)
    raw.pick_channels(['EEG Fpz-Cz', 'EOG horizontal', 'EMG submental'])
    raw.resample(100)
    raw.filter(0.3, 35)
    sfreq = raw.info['sfreq']
    epoch_length = int(30 * sfreq)
    n_epochs = int(raw.n_times // epoch_length)
    data = raw.get_data()[:, :n_epochs * epoch_length]
    data = data.reshape((3, n_epochs, epoch_length))
    data = np.transpose(data, (1, 2, 0))  # (epochs, time, channels)
    if data.shape[1] == 3000:
        data = np.pad(data, ((0, 0), (0, 1), (0, 0)), mode='constant')
    X_flat = data.reshape(-1, data.shape[-1])
    X_flat = scaler.transform(X_flat)
    X_scaled = X_flat.reshape(data.shape)
    y_pred_proba = model.predict(X_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)
    return y_pred

# ---------------- Compute Sleep Metrics ----------------
def compute_sleep_metrics(pred_labels, epoch_length_sec=30):
    stages = np.array(pred_labels)
    n_epochs = len(stages)
    time_in_bed_min = (n_epochs * epoch_length_sec) / 60
    W, N1, N2, N3, REM = 0, 1, 2, 3, 4
    try:
        sleep_onset_idx = np.where(stages != W)[0][0]
        sleep_latency_min = (sleep_onset_idx * epoch_length_sec) / 60
    except IndexError:
        sleep_latency_min = np.nan
        sleep_onset_idx = None
    sleep_epochs = np.sum(stages != W)
    TST_min = (sleep_epochs * epoch_length_sec) / 60
    try:
        rem_latency_idx = np.where((stages[sleep_onset_idx:] == REM))[0][0] + sleep_onset_idx
        rem_latency_min = (rem_latency_idx * epoch_length_sec) / 60 - sleep_latency_min
    except:
        rem_latency_min = np.nan
    if sleep_onset_idx is not None:
        waso_epochs = np.sum(stages[sleep_onset_idx:] == W)
    else:
        waso_epochs = 0
    WASO_min = (waso_epochs * epoch_length_sec) / 60
    SE = (TST_min / time_in_bed_min) * 100
    total_sleep_stages = stages[stages != W]
    if len(total_sleep_stages) > 0:
        stage_percent = {
            'N1%': np.sum(total_sleep_stages == N1) / len(total_sleep_stages) * 100,
            'N2%': np.sum(total_sleep_stages == N2) / len(total_sleep_stages) * 100,
            'N3%': np.sum(total_sleep_stages == N3) / len(total_sleep_stages) * 100,
            'REM%': np.sum(total_sleep_stages == REM) / len(total_sleep_stages) * 100,
        }
    else:
        stage_percent = {'N1%': 0, 'N2%': 0, 'N3%': 0, 'REM%': 0}
    awakenings = np.sum((stages[1:] == W) & (stages[:-1] != W))
    metrics = {
        'Total Sleep Time (min)': round(TST_min, 1),
        'Sleep Efficiency (%)': round(SE, 1),
        'Sleep Latency (min)': round(sleep_latency_min, 1),
        'Rapid Eye Movement Latency (min)': round(rem_latency_min, 1),
        'Wake After Sleep Onset (min)': round(WASO_min, 1),
        'Awakenings': awakenings,
        **{k: round(v, 1) for k, v in stage_percent.items()}
    }
    return metrics

# ---------------- Sleep Score ----------------
def sleep_score(metrics):
    score = 0
    # Sleep Efficiency
    if metrics['Sleep Efficiency (%)'] >= 90:
        score += 25
    elif metrics['Sleep Efficiency (%)'] >= 85:
        score += 20
    elif metrics['Sleep Efficiency (%)'] >= 80:
        score += 15
    else:
        score += 10

    # Total Sleep Time
    if metrics['Total Sleep Time (min)'] >= 420:
        score += 25
    elif metrics['Total Sleep Time (min)'] >= 360:
        score += 20
    elif metrics['Total Sleep Time (min)'] >= 300:
        score += 15
    else:
        score += 10

    # WASO
    if metrics['Wake After Sleep Onset (min)'] <= 20:
        score += 20
    elif metrics['Wake After Sleep Onset (min)'] <= 40:
        score += 15
    else:
        score += 10

    # Sleep Latency
    if metrics['Sleep Latency (min)'] <= 15:
        score += 15
    elif metrics['Sleep Latency (min)'] <= 30:
        score += 10
    else:
        score += 5

    # REM%
    if metrics['REM%'] >= 20:
        score += 15
    elif metrics['REM%'] >= 15:
        score += 10
    else:
        score += 5
    return min(score, 100)

# ---------------- Hypnogram Plot ----------------
def plot_hypnogram(pred):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(pred, drawstyle='steps-post', color='blue')
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['W', 'N1', 'N2', 'N3', 'REM'])
    ax.set_xlabel("Epoch (30s)")
    ax.set_ylabel("Stage")
    ax.grid(True)
    st.pyplot(fig)

# ---------------- Enhanced UI ----------------

colA, colB = st.columns([1, 3])
with colA:
    st_lottie(lottie_sleep, height=120)
with colB:
    st.title("🌙 Sleep Score Analyzer")
st.markdown("Upload your **PSG `.edf` file** to get a detailed sleep analysis and score.")
uploaded_file = st.file_uploader("Upload PSG File (.edf)", type=["edf"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    st.info("Analyzing your sleep data...")
    y_pred = predict_sleep_stages(tmp_path)
    metrics = compute_sleep_metrics(y_pred)
    score = sleep_score(metrics)
    st.success(f" Final Sleep Score: **{score}/100**")
    if score >= 85:
        st.markdown("✅ **Excellent Sleep!** Your sleep quality is very good.")
    elif score >= 70:
        st.markdown("😌 **Good Sleep** — some areas can improve.")
    else:
        st.markdown("⚠️ **Sleep Needs Improvement** — consider reviewing your sleep habits.")
    # Metrics comparison
    st.subheader("Sleep Metrics vs Ideal")
    ideal = {
        'Total Sleep Time (min)': '≥ 420',
        'Sleep Efficiency (%)': '≥ 85%',
        'Sleep Latency (min)': '≤ 15',
        'Rapid Eye Movement Latency (min)': '70–120',
        'Wake After Sleep Onset (min)': '≤ 20',
        'Awakenings': '≤ 5',
        'N1%': '5–10%',
        'N2%': '45–55%',
        'N3%': '13–23%',
        'REM%': '20–25%'
    }
    for k, v in metrics.items():
        col1, col2, col3 = st.columns([3, 2, 2])
        col1.markdown(f"**{k}**")
        col2.markdown(f"`{v}`")
        col3.markdown(f"`{ideal.get(k, 'N/A')}`")
    st.markdown("Predicted Hypnogram")
    plot_hypnogram(y_pred)
    st.markdown("---")
    st.caption("Made by your sleep science assistant")
