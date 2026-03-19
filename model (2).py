"""
Cricket Player Injury Risk Prediction Model
Uses Random Forest classifier trained on synthetic cricket workload data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")


# ── Feature definitions ───────────────────────────────────────────────────────
FEATURES = [
    "age",
    "matches_last_30_days",
    "balls_bowled_last_30_days",
    "batting_innings_last_30_days",
    "days_since_last_rest",
    "career_matches",
    "previous_injuries",
    "travel_hours_last_14_days",
    "training_hours_per_week",
    "fatigue_score",        # 1-10 self-reported
    "pitch_hardness",       # 1-10 scale
    "player_role",          # encoded: 0=batsman, 1=bowler, 2=all-rounder, 3=keeper
    "format_intensity",     # encoded: 0=T20, 1=ODI, 2=Test
    "bmi",
]

FEATURE_LABELS = {
    "age": "Age (years)",
    "matches_last_30_days": "Matches in last 30 days",
    "balls_bowled_last_30_days": "Balls bowled (last 30 days)",
    "batting_innings_last_30_days": "Batting innings (last 30 days)",
    "days_since_last_rest": "Days since last rest day",
    "career_matches": "Career matches played",
    "previous_injuries": "Previous injuries count",
    "travel_hours_last_14_days": "Travel hours (last 14 days)",
    "training_hours_per_week": "Training hours per week",
    "fatigue_score": "Fatigue score (1-10)",
    "pitch_hardness": "Pitch hardness (1-10)",
    "player_role": "Player role",
    "format_intensity": "Match format",
    "bmi": "BMI",
}

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS = {"Low": "#00ff88", "Medium": "#ffcc00", "High": "#ff4444"}


def generate_training_data(n_samples: int = 3000) -> pd.DataFrame:
    """
    Generate realistic synthetic cricket player workload and injury data.
    Based on published sports science research on cricket workload management.
    """
    np.random.seed(42)

    data = []
    for _ in range(n_samples):
        age = np.random.randint(18, 40)
        role = np.random.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.20, 0.10])
        fmt = np.random.choice([0, 1, 2])

        # Workload varies by role
        if role == 1:  # bowler — more balls bowled
            balls = np.random.randint(0, 600)
            bat_inn = np.random.randint(0, 4)
        elif role == 0:  # batsman
            balls = np.random.randint(0, 60)
            bat_inn = np.random.randint(0, 12)
        elif role == 2:  # all-rounder
            balls = np.random.randint(0, 360)
            bat_inn = np.random.randint(0, 8)
        else:  # keeper
            balls = 0
            bat_inn = np.random.randint(0, 10)

        matches = np.random.randint(0, 18)
        rest = np.random.randint(0, 45)
        career = np.random.randint(10, 350)
        prev_inj = np.random.randint(0, 8)
        travel = np.random.randint(0, 80)
        training = np.random.randint(8, 35)
        fatigue = np.random.randint(1, 11)
        pitch = np.random.randint(1, 11)
        bmi = round(np.random.uniform(20, 32), 1)

        # Injury risk calculation (domain-based heuristic for labels)
        risk_score = 0

        # Age factor
        if age > 33: risk_score += 2
        elif age > 28: risk_score += 1

        # Workload
        if matches > 12: risk_score += 2
        elif matches > 8: risk_score += 1

        if balls > 400: risk_score += 3
        elif balls > 250: risk_score += 2
        elif balls > 150: risk_score += 1

        # Rest
        if rest > 25: risk_score += 2
        elif rest > 15: risk_score += 1

        # Previous injuries
        if prev_inj >= 4: risk_score += 3
        elif prev_inj >= 2: risk_score += 1

        # Fatigue
        if fatigue >= 8: risk_score += 2
        elif fatigue >= 6: risk_score += 1

        # Travel
        if travel > 50: risk_score += 1

        # Training overload
        if training > 28: risk_score += 1

        # BMI
        if bmi > 28 or bmi < 21: risk_score += 1

        # Format (Test = more physical demand)
        if fmt == 2: risk_score += 1

        # Add noise
        risk_score += np.random.randint(-2, 3)
        risk_score = max(0, risk_score)

        # Assign risk class
        if risk_score <= 3:
            risk_class = 0  # Low
        elif risk_score <= 7:
            risk_class = 1  # Medium
        else:
            risk_class = 2  # High

        data.append({
            "age": age,
            "matches_last_30_days": matches,
            "balls_bowled_last_30_days": balls,
            "batting_innings_last_30_days": bat_inn,
            "days_since_last_rest": rest,
            "career_matches": career,
            "previous_injuries": prev_inj,
            "travel_hours_last_14_days": travel,
            "training_hours_per_week": training,
            "fatigue_score": fatigue,
            "pitch_hardness": pitch,
            "player_role": role,
            "format_intensity": fmt,
            "bmi": bmi,
            "injury_risk": risk_class,
        })

    return pd.DataFrame(data)


def train_model():
    """Train Random Forest model and return model + scaler + metrics."""
    df = generate_training_data()

    X = df[FEATURES]
    y = df["injury_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"], output_dict=True)

    return model, scaler, accuracy, report


def predict_injury_risk(model, scaler, player_data: dict) -> dict:
    """
    Predict injury risk for a single player.
    player_data: dict with keys matching FEATURES
    Returns: dict with risk_class, risk_label, probabilities, feature_importance
    """
    X = pd.DataFrame([player_data])[FEATURES]
    X_scaled = scaler.transform(X)

    risk_class = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    # Feature importance for this prediction (global importances)
    importances = model.feature_importances_
    feature_importance = sorted(
        zip(FEATURES, importances),
        key=lambda x: x[1],
        reverse=True
    )[:6]

    # Generate plain-English insights
    insights = generate_insights(player_data, risk_class)

    return {
        "risk_class": int(risk_class),
        "risk_label": RISK_LABELS[risk_class],
        "risk_color": RISK_COLORS[RISK_LABELS[risk_class]],
        "probabilities": {
            "Low": round(float(proba[0]) * 100, 1),
            "Medium": round(float(proba[1]) * 100, 1),
            "High": round(float(proba[2]) * 100, 1),
        },
        "top_risk_factors": feature_importance,
        "insights": insights,
    }


def generate_insights(data: dict, risk_class: int) -> list:
    """Generate plain-English risk factor explanations."""
    insights = []

    if data["previous_injuries"] >= 3:
        insights.append(f"️ History of {data['previous_injuries']} previous injuries significantly increases re-injury risk.")

    if data["balls_bowled_last_30_days"] > 300:
        insights.append(f"️ {data['balls_bowled_last_30_days']} balls bowled this month exceeds recommended workload threshold (300).")

    if data["days_since_last_rest"] > 20:
        insights.append(f"️ {data['days_since_last_rest']} days without a rest day — fatigue accumulation is a major concern.")

    if data["fatigue_score"] >= 7:
        insights.append(f"️ Self-reported fatigue of {data['fatigue_score']}/10 is high. Physical breakdown risk elevated.")

    if data["matches_last_30_days"] > 10:
        insights.append(f"️ {data['matches_last_30_days']} matches in 30 days is an aggressive schedule.")

    if data["age"] >= 33:
        insights.append(f"️ Age {data['age']} — recovery time increases and injury susceptibility rises after 32.")

    if data["travel_hours_last_14_days"] > 40:
        insights.append(f"️ {data['travel_hours_last_14_days']} hours of travel in 2 weeks disrupts sleep and recovery.")

    if data["training_hours_per_week"] > 28:
        insights.append(f"️ {data['training_hours_per_week']} training hours/week on top of match duties risks overtraining syndrome.")

    if not insights:
        if risk_class == 0:
            insights.append(" Player workload and physical profile are within safe limits.")
            insights.append(" Adequate rest and manageable match schedule detected.")
        else:
            insights.append("️ Multiple moderate risk factors present — monitor closely.")

    return insights


# Famous cricketers for comparison (approximate career stats patterns)
SAMPLE_PLAYERS = {
    "Virat Kohli (Peak 2018)": {
        "age": 29, "matches_last_30_days": 8, "balls_bowled_last_30_days": 30,
        "batting_innings_last_30_days": 8, "days_since_last_rest": 12,
        "career_matches": 230, "previous_injuries": 2, "travel_hours_last_14_days": 35,
        "training_hours_per_week": 25, "fatigue_score": 5, "pitch_hardness": 7,
        "player_role": 0, "format_intensity": 1, "bmi": 24.0,
    },
    "Jasprit Bumrah (Heavy Workload)": {
        "age": 28, "matches_last_30_days": 12, "balls_bowled_last_30_days": 480,
        "batting_innings_last_30_days": 3, "days_since_last_rest": 28,
        "career_matches": 145, "previous_injuries": 3, "travel_hours_last_14_days": 60,
        "training_hours_per_week": 30, "fatigue_score": 8, "pitch_hardness": 8,
        "player_role": 1, "format_intensity": 2, "bmi": 23.5,
    },
    "MS Dhoni (Late Career)": {
        "age": 37, "matches_last_30_days": 6, "balls_bowled_last_30_days": 0,
        "batting_innings_last_30_days": 6, "days_since_last_rest": 8,
        "career_matches": 350, "previous_injuries": 4, "travel_hours_last_14_days": 30,
        "training_hours_per_week": 20, "fatigue_score": 4, "pitch_hardness": 6,
        "player_role": 3, "format_intensity": 0, "bmi": 26.0,
    },
    "Rohit Sharma (IPL Season)": {
        "age": 35, "matches_last_30_days": 14, "balls_bowled_last_30_days": 0,
        "batting_innings_last_30_days": 14, "days_since_last_rest": 22,
        "career_matches": 280, "previous_injuries": 5, "travel_hours_last_14_days": 55,
        "training_hours_per_week": 22, "fatigue_score": 7, "pitch_hardness": 7,
        "player_role": 0, "format_intensity": 0, "bmi": 27.5,
    },
}
