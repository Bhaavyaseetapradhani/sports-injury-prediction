#  Cricket Player Injury Risk Predictor

> ML-powered injury risk prediction for cricket players based on workload, fatigue, and physical profile — built with Random Forest and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple?style=flat-square)

---

##  What It Does

Input a cricket player's workload and physical stats → Get:

- **Injury Risk Level** — Low / Medium / High
- **Risk probabilities** with confidence breakdown
- **Plain-English key risk factors** explaining *why* the risk is high/low
- **Feature importance chart** showing which factors matter most
- **Multi-player comparison** with stacked bar visualization

---

##  ML Model

| Detail | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Trees | 200 estimators |
| Training samples | 3,000 synthetic cricket workload records |
| Features | 14 (workload, fatigue, physical, career) |
| Classes | Low / Medium / High risk |
| Accuracy | ~82% on held-out test set |

### Features Used

| Feature | Description |
|---|---|
| Age | Player age in years |
| Matches (30d) | Matches played in last 30 days |
| Balls bowled (30d) | Bowling workload — key for fast bowlers |
| Batting innings (30d) | Batting workload |
| Days since rest | Fatigue accumulation proxy |
| Career matches | Experience / cumulative wear |
| Previous injuries | Re-injury risk multiplier |
| Travel hours (14d) | Sleep/recovery disruption |
| Training hours/week | Training load on top of matches |
| Fatigue score | Self-reported 1–10 scale |
| Pitch hardness | Impact stress (especially for bowlers) |
| Player role | Batsman / Bowler / All-Rounder / Keeper |
| Match format | T20 / ODI / Test |
| BMI | Body composition proxy |

---

##  Quick Start

```bash
git clone https://github.com/Bhaavyaseetapradhani/sports-injury-prediction.git
cd sports-injury-prediction
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`

---

##  Built-In Sample Players

| Player | Scenario |
|---|---|
| Virat Kohli (Peak 2018) | Well-managed star batsman |
| Jasprit Bumrah (Heavy Workload) | Overloaded fast bowler |
| MS Dhoni (Late Career) | Aging keeper-batsman |
| Rohit Sharma (IPL Season) | Busy T20 captain |

---

##  Risk Factors (Sports Science Basis)

- **Bowling workload > 300 balls/month** — exceeds recommended threshold for fast bowlers
- **No rest day in 20+ days** — fatigue accumulation increases soft tissue injury risk
- **3+ previous injuries** — re-injury probability doubles
- **Fatigue score ≥ 7/10** — self-reported fatigue correlates with injury onset
- **Age > 33** — recovery time increases, injury susceptibility rises
- **Travel > 40 hours/14 days** — disrupts sleep, impairs muscle recovery

---

## ️ Tech Stack

- **Python 3.9+**
- **scikit-learn** — Random Forest, preprocessing
- **Streamlit** — web interface
- **Plotly** — interactive charts
- **pandas / numpy** — data handling

---

##  Author

**Bhaavya Seeta Pradhani** — ML Engineer | Data Science  
[LinkedIn](https://linkedin.com/in/bhaavya-seeta-pradhani) · [GitHub](https://github.com/Bhaavyaseetapradhani)

---

*Disclaimer: For educational and research purposes only. Not a substitute for professional medical or physiotherapy assessment.*
