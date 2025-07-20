##  Project Title
**SleepSense: Automated Sleep Score Analyzer usimg CNN & Transformer**

---

##  Description
Understanding sleep patterns is crucial for diagnosing and improving sleep health. This project provides an interactive web application that automatically analyzes overnight sleep patterns using EEG data from polysomnography (PSG) .edf files. Built with a hybrid CNN + Transformer deep learning model, the system classifies sleep stages, computes critical sleep metrics, and generates an overall sleep quality score.

The app is designed for clinicians, researchers, and individuals interested in understanding sleep health. Using signal preprocessing, stage-wise classification, and custom metrics, it mimics clinical grade scoring - all accessible through an elegant and responsive Streamlit interface.

---

##  Key Features

- Upload `.edf` PSG file with EEG, EOG, and EMG signals
- Automated prediction of sleep stages (W, N1, N2, N3, REM)
- Calculation of clinical sleep metrics:
  - Total Sleep Time
  - Sleep Efficiency
  - Sleep Latency
  - REM Latency
  - WASO (Wake After Sleep Onset)
  - Sleep Stage Percentages
  - Awakenings
- Final Sleep Score out of 100 with interpretation
- Hypnogram visualization

---

##  Dataset Used

- **Sleep-EDF Expanded Dataset**
  - Source: PhysioNet [https://physionet.org/content/sleep-edfx/1.0.0/](https://physionet.org/content/sleep-edfx/1.0.0/)
  - Channels used: `EEG Fpz-Cz`, `EOG horizontal`, `EMG submental`

---

##  Model Details

- **Architecture:** 1D Convolutional Neural Network (CNN) + Transformer Encoder  
- **Input Shape:** `(Epochs, 3000 samples, 3 channels)` The 3 channels represent EEG, EOG, and EMG signals  
- **Output:** Softmax probability distribution across 5 sleep stages: `W` – Wake  `N1`, `N2`, `N3` – Non-REM stages  `REM` – Rapid Eye Movement  
- **Training Summary:** Trained on cleaned and preprocessed **Sleep-EDF** dataset  where preprocessing steps include filtering, standardization  and 30-second epoch segmentation 

**Model Training Notebook:**  
[Open in Google Colab](https://colab.research.google.com/drive/14rxhuFzm_2SQaor5CTH9BbLVLkvuOw5y?usp=sharing)

---

##  Tools & Technologies

| Tool             | Purpose                           |
|------------------|-----------------------------------|
| Python           | Core programming language         |
| TensorFlow/Keras | Model building and training       |
| MNE              | Signal loading and preprocessing  |
| Scikit-learn     | StandardScaler and utilities      |
| Matplotlib       | Visualization (Hypnogram)         |
| Streamlit        | UI and Web App                    |
| Lottie           | UI animation                      |

---

##  Files Included

| File/Folder              | Description                                     |
|--------------------------|-------------------------------------------------|
| `app.py`                 | Main Streamlit application                      |
| `sleep_score_model.keras`| Trained CNN+Transformer model                   |
| `scaler.pkl`             | StandardScaler used to normalize input data     |
| `config.toml`            | Streamlit theme customization                   |
| `requirements.txt`       | All dependencies to run the app                 |
| `sample_psg.edf`         | Same PSG file in .edf format                    |

---

##  How to Use

###  Option 1: Use the Live App  
> _Hosted on Streamlit Cloud_  
🔗 [Launch Live Demo](https://your-streamlit-app-link)

### 🖥️ Option 2: Run Locally

1. **Clone the repository**  
   Go to GitHub → `Code` → `Download ZIP` or clone via GitHub Desktop

2. **Install dependencies**  
   *(Use a virtual environment for best results)*
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
    streamlit run app.py

4. **Upload your .edf file to begin the analysis**

--

## Contact
Gagana Deepika D
gaganadeepikad2004@gmail.com
