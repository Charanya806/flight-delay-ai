import streamlit as st
import pandas as pd
import joblib
import os
# Prediction History
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Flight Delay AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(
        "models/flight_delay_model.pkl"
    )

    frequency_mappings = joblib.load(
        "models/frequency_mappings.pkl"
    )

    return model, frequency_mappings


model, frequency_mappings = load_model()

# Prediction threshold
THRESHOLD = 0.20


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b
    );
    color: white;
    margin-bottom: 28px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    color: #cbd5e1;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 18px;
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    background: white;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 18px rgba(0,0,0,0.06);
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 45px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.markdown("## ✈️ Flight AI")

    st.markdown("---")

    st.markdown("### About")

    st.write(
        "AI-powered flight delay prediction "
        "using pre-flight information."
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.write("HistGradientBoosting Classifier")

    st.markdown("### 📊 Dataset")

    st.write("500,000 flight records")

    st.markdown("### 🎯 ROC-AUC")

    st.write("0.7028")

    st.markdown("### ⚙️ Decision Threshold")

    st.write("20%")

    st.markdown("---")

    st.caption(
        "AI & Data Science Project"
    )


# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown("""
<div class="hero">

<h1>✈️ Flight Delay AI</h1>

<p>
Predict the probability of flight arrival delays
before departure using Machine Learning.
</p>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
st.markdown(
    '<div class="section-title">'
    '🛫 Enter Flight Details'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


# --------------------------------------------------
# SCHEDULE
# --------------------------------------------------
with col1:

    st.markdown("#### 📅 Schedule")

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=7
    )

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=15
    )

    day_of_week = st.number_input(
        "Day of Week",
        min_value=1,
        max_value=7,
        value=3,
        help="1 = Monday, 7 = Sunday"
    )


# --------------------------------------------------
# ROUTE
# --------------------------------------------------
with col2:

    st.markdown("#### 🗺️ Route")

    airline = st.text_input(
        "Airline Code",
        value="AA"
    )

    origin = st.text_input(
        "Origin Airport",
        value="ATL"
    )

    destination = st.text_input(
        "Destination Airport",
        value="LAX"
    )


# --------------------------------------------------
# FLIGHT INFORMATION
# --------------------------------------------------
with col3:

    st.markdown("#### 🕐 Flight Information")

    scheduled_departure = st.number_input(
        "Scheduled Departure (HHMM)",
        min_value=0,
        max_value=2359,
        value=1430
    )

    distance = st.number_input(
        "Distance (miles)",
        min_value=1.0,
        value=1946.0
    )

    scheduled_time = st.number_input(
        "Scheduled Flight Time (minutes)",
        min_value=1.0,
        value=270.0
    )


st.divider()


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------
predict_button = st.button(
    "🔮  Predict Flight Delay",
    use_container_width=True,
    type="primary"
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if predict_button:
    

    # Clean input
    airline = airline.strip().upper()
    origin = origin.strip().upper()
    destination = destination.strip().upper()

    if not airline or not origin or not destination:

        st.warning(
            "⚠️ Please enter Airline, Origin Airport "
            "and Destination Airport."
        )

    else:

        # ------------------------------------------
        # FEATURE ENGINEERING
        # ------------------------------------------

        dep_hour = (
            scheduled_departure // 100
        )

        dep_minute = (
            scheduled_departure % 100
        )

        is_weekend = (
            1 if day_of_week >= 6 else 0
        )

        route = (
            origin + "_" + destination
        )

        airline_route = (
            airline + "_" + route
        )

        # Distance category
        if distance < 500:
            distance_category = "Short"

        elif distance < 1500:
            distance_category = "Medium"

        elif distance < 2500:
            distance_category = "Long"

        else:
            distance_category = "Very_Long"


        # ------------------------------------------
        # CREATE INPUT DATAFRAME
        # ------------------------------------------

        flight = pd.DataFrame([{

            "MONTH": month,

            "DAY": day,

            "DAY_OF_WEEK": day_of_week,

            "AIRLINE": airline,

            "ORIGIN_AIRPORT": origin,

            "DESTINATION_AIRPORT": destination,

            "SCHEDULED_DEPARTURE":
                scheduled_departure,

            "DEP_HOUR":
                dep_hour,

            "DEP_MINUTE":
                dep_minute,

            "DISTANCE":
                distance,

            "SCHEDULED_TIME":
                scheduled_time,

            "IS_WEEKEND":
                is_weekend,

            "ROUTE":
                route,

            "AIRLINE_ROUTE":
                airline_route,

            "DISTANCE_CATEGORY":
                distance_category

        }])


        # ------------------------------------------
        # FREQUENCY ENCODING
        # ------------------------------------------

        categorical_columns = [

            "AIRLINE",

            "ORIGIN_AIRPORT",

            "DESTINATION_AIRPORT",

            "ROUTE",

            "AIRLINE_ROUTE",

            "DISTANCE_CATEGORY"

        ]

        for col in categorical_columns:

            frequency = (
                frequency_mappings[col]
            )

            flight[
                col + "_FREQ"
            ] = (

                flight[col]
                .map(frequency)
                .fillna(0)

            )


        # ------------------------------------------
        # FINAL FEATURES
        # ------------------------------------------

        numerical_columns = [

            "MONTH",

            "DAY",

            "DAY_OF_WEEK",

            "SCHEDULED_DEPARTURE",

            "DEP_HOUR",

            "DEP_MINUTE",

            "DISTANCE",

            "SCHEDULED_TIME",

            "IS_WEEKEND"

        ]

        frequency_columns = [

            col + "_FREQ"

            for col in categorical_columns

        ]

        final_features = (
            numerical_columns
            + frequency_columns
        )


        X = flight[
            final_features
        ].astype(float)


        # ------------------------------------------
        # MODEL PREDICTION
        # ------------------------------------------

        probability = (
            model
            .predict_proba(X)[0][1]
        )

        probability_percent = (
            probability * 100
        )

        prediction = (
            1
            if probability >= THRESHOLD
            else 0
        )


        st.divider()


        # ------------------------------------------
        # RESULT
        # ------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '📊 Prediction Result'
            '</div>',
            unsafe_allow_html=True
        )


        if prediction == 1:

            st.error(
                "🔴 FLIGHT IS LIKELY TO BE DELAYED"
            )

            status = "Delayed"

        else:

            st.success(
                "🟢 FLIGHT IS LIKELY TO BE ON TIME"
            )

            status = "On Time"
            # ------------------------------------------
# SAVE PREDICTION TO HISTORY
# ------------------------------------------

        st.session_state.prediction_history.append({
                "Airline": airline,
                "Route": f"{origin} → {destination}",
                "Scheduled Departure": f"{scheduled_departure:04d}",
                "Distance": f"{distance:.0f} mi",
                "Delay Probability": f"{probability_percent:.2f}%",
                "Prediction": status
    })


        # ------------------------------------------
        # METRICS
        # ------------------------------------------

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Delay Probability",
                f"{probability_percent:.2f}%"
            )

        with m2:

            st.metric(
                "Prediction",
                status
            )

        with m3:

            st.metric(
                "Decision Threshold",
                f"{THRESHOLD * 100:.0f}%"
            )


        # ------------------------------------------
        # PROBABILITY
        # ------------------------------------------

        st.markdown(
            "#### 📈 Delay Probability"
        )

        st.progress(
            min(
                max(
                    int(probability_percent),
                    0
                ),
                100
            )
        )


        # ------------------------------------------
        # FLIGHT SUMMARY
        # ------------------------------------------

        st.markdown(
            "#### 🧾 Flight Summary"
        )

        s1, s2, s3, s4 = st.columns(4)

        with s1:

            st.metric(
                "Airline",
                airline
            )

        with s2:

            st.metric(
                "Route",
                f"{origin} → {destination}"
            )

        with s3:

            st.metric(
                "Distance",
                f"{distance:.0f} mi"
            )

        with s4:

            st.metric(
                "Scheduled",
                f"{scheduled_departure:04d}"
            )


        # ------------------------------------------
        # RISK EXPLANATION
        # ------------------------------------------

        st.markdown(
            "#### 💡 Risk Assessment"
        )

        if probability_percent >= 60:

            st.warning(
                f"⚠️ High delay risk. "
                f"The model estimates a "
                f"{probability_percent:.2f}% "
                f"probability of arrival delay."
            )

        elif probability_percent >= 30:

            st.info(
                f"ℹ️ Moderate delay risk. "
                f"The estimated probability "
                f"is {probability_percent:.2f}%."
            )

        else:

            st.success(
                f"✅ Lower delay risk. "
                f"The estimated probability "
                f"is {probability_percent:.2f}%."
            )
            
# --------------------------------------------------
# MODEL INSIGHTS
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">'
    '🧠 Model Insights'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "The model uses pre-flight information to estimate "
    "the probability of an arrival delay."
)

insight_col1, insight_col2 = st.columns(2)

with insight_col1:

    st.markdown("### 📊 Top Influencing Features")

    feature_data = pd.DataFrame({
        "Feature": [
            "Scheduled Departure",
            "Month",
            "Day",
            "Airline Pattern",
            "Day of Week"
        ],
        "Importance": [
            0.088554,
            0.065164,
            0.050290,
            0.030011,
            0.017647
        ]
    })

    st.dataframe(
        feature_data,
        use_container_width=True,
        hide_index=True
    )

with insight_col2:

    st.markdown("### 🎯 Model Performance")

    p1, p2 = st.columns(2)

    with p1:
        st.metric(
            "ROC-AUC",
            "0.7028"
        )

    with p2:
        st.metric(
            "Threshold",
            "20%"
        )

    st.info(
        "The 20% decision threshold was selected to "
        "improve detection of delayed flights. "
        "Accuracy is interpreted together with "
        "precision, recall and F1-score."
    )
    
# --------------------------------------------------
# EVALUATION DASHBOARD
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">'
    '📈 Evaluation Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "The model evaluation results are based on the "
    "test dataset using the selected 20% decision threshold."
)

# Performance Metrics
e1, e2, e3, e4 = st.columns(4)

with e1:
    st.metric("Accuracy", "69.14%")

with e2:
    st.metric("Delayed Precision", "31%")

with e3:
    st.metric("Delayed Recall", "57%")

with e4:
    st.metric("Delayed F1-Score", "40%")

st.markdown("### 📊 Model Evaluation Charts")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    if os.path.exists("outputs/confusion_matrix.png"):

        st.image(
            "outputs/confusion_matrix.png",
            caption="Confusion Matrix",
            use_container_width=True
        )

    else:
        st.warning(
            "Confusion Matrix image not found."
        )

with chart_col2:

    if os.path.exists("outputs/roc_curve.png"):

        st.image(
            "outputs/roc_curve.png",
            caption="ROC Curve",
            use_container_width=True
        )

    else:
        st.warning(
            "ROC Curve image not found."
        )

# Precision-Recall Curve

st.markdown("### 🎯 Precision–Recall Curve")

if os.path.exists(
    "outputs/precision_recall_curve.png"
):

    st.image(
        "outputs/precision_recall_curve.png",
        caption="Precision–Recall Curve",
        use_container_width=True
    )

else:

    st.warning(
        "Precision–Recall Curve image not found."
    )
    # --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">'
    '🕘 Prediction History'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

else:
    st.info(
        "No predictions yet. Make a flight prediction to see "
        "your prediction history here."
    )
# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">

✈️ <b>Flight Delay Prediction System</b><br>

Machine Learning • Python • Streamlit •
HistGradientBoosting

</div>
""", unsafe_allow_html=True)
