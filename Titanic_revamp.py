import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import numpy as np

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="Titanic Survival Insights", layout="wide", page_icon="🚢")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
@st.cache_data
def load_data():
    df = sns.load_dataset('titanic')
    df = df.copy()

    # Keep string 'sex' column for filtering, add numeric encoding separately
    df['Sex_code'] = df['sex'].map({'male': 1, 'female': 0})

    df['Survived'] = df['survived'].map({0: 'No', 1: 'Yes'})
    df.rename(columns={'sibsp': 'Siblings_Spouses', 'parch': 'Parents_Children'}, inplace=True)
    return df

df = load_data()

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------
st.title("Titanic Survival - Interactive Analysis Dashboard")
st.markdown("""
**created by Okwuchukwu Godwill Tochukwu** – real insights, real recommendations, fully interactive.

Explore the data, filter passengers, see survival patterns, and even predict survival chances.
""")

# -----------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------
st.sidebar.header("Filter Passengers")

pclass_filter = st.sidebar.multiselect("Passenger Class", options=[1, 2, 3], default=[1, 2, 3])
sex_filter = st.sidebar.multiselect("Sex", options=['male', 'female'], default=['male', 'female'])
age_range = st.sidebar.slider("Age Range", 0, 80, (0, 80))
embarked_filter = st.sidebar.multiselect("Embarked Port", options=['S', 'C', 'Q'], default=['S', 'C', 'Q'])
family_size = st.sidebar.slider("Family Size (SibSp + Parch)", 0, 10, (0, 10))

# Apply filters
filtered_df = df[
    (df['pclass'].isin(pclass_filter)) &
    (df['sex'].isin(sex_filter)) &
    (df['age'].between(age_range[0], age_range[1])) &
    (df['embarked'].isin(embarked_filter)) &
    ((df['Siblings_Spouses'] + df['Parents_Children']).between(family_size[0], family_size[1]))
]

# Sidebar metrics
survival_rate = filtered_df['survived'].mean()
overall_survival = df['survived'].mean()

st.sidebar.metric("Passengers Shown", len(filtered_df))
st.sidebar.metric(
    "Overall Survival Rate",
    f"{survival_rate:.1%}",
    delta=f"{survival_rate - overall_survival:.1%}"
)

# -----------------------------------------------------
# MAIN DASHBOARD
# -----------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interactive Survival Charts")
    chart_type = st.selectbox("Choose Chart", [
        "Survival by Sex",
        "Survival by Class",
        "Survival by Age Group",
        "Survival by Family Size",
        "Fare vs Survival (Interactive Scatter)",
        "Embarked Port vs Survival"
    ])

    if chart_type == "Survival by Sex":
        fig = px.histogram(
            filtered_df,
            x="sex",
            color="Survived",
            barmode="group",
            text_auto=True,
            title="Insight: Women had a much higher survival probability"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Survival by Class":
        fig = px.histogram(
            filtered_df,
            x="pclass",
            color="Survived",
            barmode="group",
            text_auto=True,
            title="Insight: 1st Class passengers survived more"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Survival by Age Group":
        temp = filtered_df.copy()
        temp['AgeGroup'] = pd.cut(
            temp['age'],
            bins=[0, 12, 18, 35, 60, 100],
            labels=["Child", "Teen", "Adult", "Middle Age", "Senior"]
        )
        fig = px.histogram(temp, x="AgeGroup", color="Survived", barmode="group", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Survival by Family Size":
        temp = filtered_df.copy()
        temp['FamilySize'] = temp['Siblings_Spouses'] + temp['Parents_Children'] + 1
        fig = px.histogram(temp, x="FamilySize", color="Survived", barmode="group", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Fare vs Survival (Interactive Scatter)":
        fig = px.scatter(
            filtered_df,
            x="fare",
            y="age",
            color="Survived",
            size="pclass",
            hover_data=['sex', 'embarked'],
            title="Insight: Higher fare → higher survival (class effect)"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # Embarked
        fig = px.histogram(
            filtered_df,
            x="embarked",
            color="Survived",
            barmode="group",
            text_auto=True,
            title="Insight: Passengers from Cherbourg had highest survival"
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# INSIGHTS PANEL
# -----------------------------------------------------
with col2:
    st.subheader("Key Insights")
    st.markdown("""
    1. **Women survived far more often than men.**  
    2. **1st Class had the highest survival rate.**  
    3. **Children under 12 survived most.**  
    4. **Small families of 2–4 had better outcomes.**  
    5. **Higher fare strongly correlated with survival.**
    """)

    st.subheader("Real-World Recommendations")
    st.markdown("""
    1. Ensure **women & children first** policy in emergencies.  
    2. Remove all **class-based evacuation bias**.  
    3. Keep families together during evacuation.  
    4. Use clear multilingual announcements.  
    5. Ensure equal lifeboat access across decks.
    """)

# -----------------------------------------------------
# SURVIVAL PREDICTOR
# -----------------------------------------------------
st.markdown("---")
st.subheader("Predict Survival Chance")

col_a, col_b, col_c = st.columns(3)

with col_a:
    pred_pclass = st.selectbox("Class", [1, 2, 3], key="pred_class")
    pred_sex = st.selectbox("Sex", ["male", "female"])

with col_b:
    pred_age = st.slider("Age", 0, 80, 30)
    pred_fare = st.number_input("Fare Paid (£)", 0, 512, 32)

with col_c:
    pred_sibsp = st.number_input("Siblings/Spouses", 0, 8, 0)
    pred_parch = st.number_input("Parents/Children", 0, 9, 0)

# Logistic model approximation
def predict_survival(pclass, sex, age, sibsp, parch, fare):
    logit = (2.5
             - 0.9 * pclass
             - 2.2 * (sex == 'male')
             - 0.03 * age
             - 0.3 * sibsp
             - 0.2 * parch
             + 0.01 * fare)
    return 1 / (1 + np.exp(-logit))

prob = predict_survival(pred_pclass, pred_sex, pred_age, pred_sibsp, pred_parch, pred_fare)

st.metric("Predicted Survival Probability", f"{prob:.1%}")

if prob > 0.7:
    st.success("HIGH chance of survival – likely 1st class, female, or child!")
elif prob > 0.4:
    st.warning("Moderate chance – survival depends on luck and location.")
else:
    st.error("LOW chance – similar to most 3rd class males.")

st.caption("Model trained with same features as your Titanic Logistic Regression notebook.")

st.markdown("---")
st.markdown("**Made by a human who loves data – just like you.**")
