

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime



st.set_page_config(
    page_title="AI Healthcare Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.stApp{
background:#F5F9FD;
}

/* Sidebar */

section[data-testid="stSidebar"]{
background:#0F4C81;
}

section[data-testid="stSidebar"] *{
color:white;
}

/* Title */

h1{
color:#0F4C81;
font-weight:bold;
}

h2,h3{
color:#0F4C81;
}

/* Buttons */

.stButton>button{

width:100%;

height:55px;

border-radius:12px;

background:#1565C0;

color:white;

font-size:18px;

font-weight:bold;

border:none;

}

.stButton>button:hover{

background:#0D47A1;

color:white;

}

/* Metric */

div[data-testid="metric-container"]{

background:white;

padding:18px;

border-radius:15px;

box-shadow:0px 4px 15px rgba(0,0,0,0.08);

}

/* Input */

div[data-baseweb="input"]{

border-radius:10px;

}

/* Card */

.card{

background:white;

padding:25px;

border-radius:15px;

box-shadow:0px 6px 18px rgba(0,0,0,.08);

margin-bottom:15px;

}

.hero{

background:linear-gradient(90deg,#1565C0,#42A5F5);

padding:40px;

border-radius:20px;

color:white;

}

</style>
""", unsafe_allow_html=True)


model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

df = pd.read_csv("data/diabetes.csv")

st.sidebar.image(
    "https://img.icons8.com/color/96/hospital-3.png",
    width=80
)

st.sidebar.markdown("### AI Healthcare System")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🩺 Prediction",
        "📋 Patient Report",
        "ℹ About",
    ]
)


if page == "🩺 Prediction":

    st.title("🩺 Diabetes Prediction")

    st.write("Enter the patient's clinical information below.")

    st.markdown("---")

    col1, col2 = st.columns(2)


    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0,
            max_value=300,
            value=120
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0,
            max_value=200,
            value=70
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            max_value=100,
            value=20
        )

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0,
            max_value=900,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.50,
            format="%.2f"
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    st.markdown("")

    predict = st.button("🔍 Predict Diabetes")


    if predict:

        # BMI Category

        if bmi < 18.5:
            bmi_category = 0
            bmi_text = "Underweight"

        elif bmi < 25:
            bmi_category = 1
            bmi_text = "Normal"

        elif bmi < 30:
            bmi_category = 2
            bmi_text = "Overweight"

        else:
            bmi_category = 3
            bmi_text = "Obese"

        # Input Data

        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age,
            bmi_category
        ]])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(input_scaled)[0]

        confidence = max(probability) * 100

        st.markdown("---")


        c1, c2 = st.columns(2)

        with c1:

            if prediction == 1:

                st.error("🔴 High Risk of Diabetes")

                risk = "High"

            else:

                st.success("🟢 No Diabetes Detected")

                risk = "Low"

        with c2:

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

        st.markdown("---")


        st.subheader("📋 Patient Summary")

        report = pd.DataFrame({

            "Parameter":[
                "Pregnancies",
                "Glucose",
                "Blood Pressure",
                "Skin Thickness",
                "Insulin",
                "BMI",
                "BMI Category",
                "DPF",
                "Age"
            ],

            "Value":[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                bmi_text,
                dpf,
                age
            ]

        })

        st.dataframe(
            report,
            use_container_width=True,
            hide_index=True
        )


        st.markdown("---")

        st.subheader("📊 Prediction Probability")

        chart = pd.DataFrame({

            "Status":[
                "Healthy",
                "Diabetes"
            ],

            "Probability":[
                probability[0] * 100,
                probability[1] * 100
            ]

        })

        fig = px.bar(

            chart,

            x="Status",

            y="Probability",

            color="Status",

            text="Probability"

        )

        fig.update_traces(texttemplate="%{text:.2f}%")

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.markdown("---")

        st.subheader("💊 Health Recommendation")

        if prediction == 1:

            st.warning("""

• Consult a physician.

• Reduce sugar intake.

• Avoid junk foods.

• Walk at least 30 minutes daily.

• Drink enough water.

• Monitor blood glucose regularly.

• Follow doctor's advice.

""")

        else:

            st.success("""

• Continue a balanced diet.

• Exercise regularly.

• Maintain healthy body weight.

• Drink plenty of water.

• Sleep 7-8 hours daily.

• Attend regular health check-ups.

""")

    
        st.session_state["patient_report"] = report

        st.session_state["prediction"] = (
            "Diabetes Detected"
            if prediction == 1
            else "No Diabetes"
        )

        st.session_state["confidence"] = confidence

        st.session_state["risk"] = risk

st.sidebar.markdown("---")

st.sidebar.success("Machine Learning Model Loaded")

st.sidebar.info("Algorithm : K-Nearest Neighbors")

st.sidebar.markdown("---")

st.sidebar.write("Version : 1.0")



if page=="🏠 Home":

    st.markdown("""

<div class="hero">

<h1 style="color:white;">
🏥 AI Healthcare Prediction System
</h1>

<h4 style="color:white;">

Predict • Analyze • Recommend

</h4>

</div>

""",unsafe_allow_html=True)

    st.write("")

    left,right=st.columns([2,1])

    with left:

        st.markdown("""

<div class="card">

<h3>

Welcome

</h3>

<p>

Artificial Intelligence is transforming healthcare by providing
fast, reliable, and accurate disease prediction.

This application predicts the risk of Diabetes using a
trained Machine Learning model and provides health
recommendations based on the prediction.

</p>

</div>

""",unsafe_allow_html=True)

    with right:


        st.metric("Accuracy","97%")

    st.write("")

    st.subheader("System Features")

    c1,c2,c3,c4=st.columns(4)

    with c1:

        st.success("AI Prediction")

    with c2:

        st.success("Dashboard")

    with c3:

        st.success(" Patient Report")

    with c4:

        st.success("Health Advice")

    st.write("")

    st.subheader("Project Overview")

    st.info("""

The AI Healthcare Prediction System helps healthcare professionals
and patients identify diabetes risk at an early stage.

Features include:

• Diabetes Prediction

• Confidence Score

• Health Recommendation

• Interactive Dashboard

• Professional Patient Report

• PDF Report 

""")

    st.write("")

    total=len(df)

    diabetic=int(df["Outcome"].sum())

    healthy=total-diabetic

    avg_age=round(df["Age"].mean(),1)

    a,b,c,d=st.columns(4)

    a.metric("Patients",total)

    b.metric("Healthy",healthy)

    c.metric("Diabetes",diabetic)

    d.metric("Average Age",avg_age)

    st.markdown("---")

elif page == "📊 Dashboard":

    st.title(" Healthcare Analytics Dashboard")
    st.write("Monitor diabetes statistics and patient insights.")

    st.markdown("---")

    total = len(df)
    diabetic = int(df["Outcome"].sum())
    healthy = total - diabetic

    avg_age = round(df["Age"].mean(), 1)
    avg_bmi = round(df["BMI"].mean(), 1)
    avg_glucose = round(df["Glucose"].mean(), 1)
    avg_bp = round(df["BloodPressure"].mean(), 1)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(" Total Patients", total)
    c2.metric(" Diabetes Cases", diabetic)
    c3.metric(" Healthy Patients", healthy)
    c4.metric(" Avg Age", avg_age)

    st.markdown("---")

    c5, c6 = st.columns(2)

    c5.metric("⚖ Average BMI", avg_bmi)
    c6.metric("🩸 Avg Glucose", avg_glucose)

    st.markdown("---")

    

    left, right = st.columns(2)

    with left:

        fig = px.pie(
            values=[healthy, diabetic],
            names=["Healthy", "Diabetes"],
            hole=0.55,
            title="Diabetes Distribution"
        )

        fig.update_layout(
            title_x=0.25,
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)

  

    with right:

        gender = ["Healthy", "Diabetes"]
        values = [healthy, diabetic]

        fig = px.bar(
            x=gender,
            y=values,
            text=values,
            color=gender,
            title="Patient Count"
        )

        fig.update_layout(height=420)

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")


    left, right = st.columns(2)

    with left:

        fig = px.histogram(
            df,
            x="BMI",
            nbins=25,
            color="Outcome",
            title="BMI Distribution"
        )

        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

   

    with right:

        fig = px.histogram(
            df,
            x="Age",
            nbins=20,
            color="Outcome",
            title="Age Distribution"
        )

        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")


    left, right = st.columns(2)

    with left:

        fig = px.box(
            df,
            x="Outcome",
            y="Glucose",
            color="Outcome",
            title="Glucose Analysis"
        )

        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

    with right:

        fig = px.scatter(
            df,
            x="BMI",
            y="Glucose",
            color="Outcome",
            title="BMI vs Glucose"
        )

        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")


    st.subheader(" Feature Correlation")

    corr = df.corr(numeric_only=True)

    heatmap = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="Blues",
        aspect="auto"
    )

    heatmap.update_layout(height=700)

    st.plotly_chart(heatmap, use_container_width=True)

    st.markdown("---")

   

    st.subheader(" Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

   
    st.subheader(" Dataset Statistics")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

    st.markdown("---")

    st.success("✅ Dashboard Loaded Successfully")

elif page == "📋 Patient Report":

    st.title(" Patient Health Report")

    if "patient_report" not in st.session_state:

        st.warning("⚠️ No prediction available.")

        st.info("Please go to the Prediction page and predict first.")

    else:

        st.success("Prediction Report Generated Successfully")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Prediction",
                st.session_state["prediction"]
            )

        with col2:

            st.metric(
                "Confidence",
                f'{st.session_state["confidence"]:.2f}%'
            )

        st.markdown("---")

        st.subheader("Patient Details")

        st.dataframe(
            st.session_state["patient_report"],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        if st.session_state["risk"] == "High":

            st.error("⚠️ High Risk Patient")

            st.write("""
Recommended Actions:

• Visit a doctor immediately.

• Follow a diabetic diet.

• Exercise daily.

• Check blood glucose regularly.

• Avoid sugary foods.
""")

        else:

            st.success("✅ Healthy Condition")

            st.write("""
Maintain these healthy habits:

• Balanced diet

• Regular exercise

• Proper sleep

• Drink enough water

• Annual health check-up
""")


elif page == "ℹ About":

    st.title(" About Project")

    st.markdown("---")

    st.subheader(" Project")

    st.write("""
AI Healthcare Prediction System for Diabetes Prediction using
Machine Learning.
""")

    st.markdown("---")

    st.subheader(" Machine Learning Algorithm")

    st.info("K-Nearest Neighbors (KNN)")

    st.markdown("---")

        