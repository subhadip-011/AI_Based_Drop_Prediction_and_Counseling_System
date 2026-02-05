import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from alerts import send_email_alert  
# Import your email function


# Load model artifacts

try:
    model_type = joblib.load("model_type.pkl")
    scaler = joblib.load("scaler.pkl")
    label_enc = joblib.load("label_encoder.pkl")
    model_features = joblib.load("features.pkl")

    if model_type == "sklearn":
        clf = joblib.load("model.pkl")
    else:
        from tensorflow.keras.models import load_model
        clf = load_model("dl_model.h5")

    st.title("AVSAR (AI-based Virtual System for Academic Retention)")
    st.info("Upload a CSV file to begin.")

except FileNotFoundError:
    st.error("Model artifacts not found. Please run train.py first.")
    st.stop()


# File Upload

uploaded = st.file_uploader("Upload student data CSV", type="csv")

if uploaded:
    # Read CSV
    try:
        # Auto-detect delimiter for CSV (; or ,)
        df = pd.read_csv(uploaded, sep=None, engine="python")
        st.success("File uploaded successfully.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Ensure student_id exists
    if "student_id" not in df.columns:
        df["student_id"] = range(1, len(df) + 1)

    
    # Feature Processing
    X = df.drop(columns=["Target"], errors="ignore")
    X = pd.get_dummies(X)

    # Ensure all training-time features exist
    for col in model_features:
        if col not in X.columns:
            X[col] = 0
    # Reorder columns to match training
    X = X[model_features]

    # Scale features
    X_scaled = scaler.transform(X)

    # Predictions
    
    if model_type == "sklearn":
        preds = clf.predict(X_scaled)
        probs = clf.predict_proba(X_scaled)
    else:
        probs = clf.predict(X_scaled)
        preds = np.argmax(probs, axis=1)

    # Decode label indices → class names
    try:
        df["Prediction"] = label_enc.inverse_transform(preds)
    except Exception as e:
        st.error(f"Error decoding predictions with label encoder: {e}")
        st.stop()

    # Compute dropout probability using correct class index
    classes_list = list(label_enc.classes_)
    try:
        dropout_index = classes_list.index("Dropout")
        df["Dropout_Probability"] = probs[:, dropout_index]
    except ValueError:
        st.warning(f"'Dropout' not found in label classes {classes_list}. Using first probability column as fallback.")
        df["Dropout_Probability"] = probs[:, 0]

 
    # Risk Classification
    RED_ENROLLED_THRESHOLD = 75
    AMBER_LOWER_THRESHOLD = 40
    RED_DROPOUT_THRESHOLD = 90

    def risk_level(pred, p_dropout):
        p_dropout_percent = p_dropout * 100
        if pred == "Graduate":
            return "Green"
        elif pred == "Enrolled":
            if p_dropout_percent > RED_ENROLLED_THRESHOLD:
                return "Red"
            elif AMBER_LOWER_THRESHOLD <= p_dropout_percent <= RED_ENROLLED_THRESHOLD:
                return "Amber"
            else:
                return "Green"
        elif pred == "Dropout":
            if p_dropout_percent >= RED_DROPOUT_THRESHOLD:
                return "Red"
            elif AMBER_LOWER_THRESHOLD <= p_dropout_percent < RED_DROPOUT_THRESHOLD:
                return "Amber"
            else:
                return "Green"
        else:
            return "Green"

    df["Risk_Level"] = [
        risk_level(pred, p) for pred, p in zip(df["Prediction"], df["Dropout_Probability"])
    ]

    # Risky subset (Red + Amber)
    risky = df[df["Risk_Level"].isin(["Red", "Amber"])]

    # Sidebar – Rewards & Schemes
    st.sidebar.title("Rewards & Schemes Summary")

    reward_low_risk = df[df["Dropout_Probability"] < 0.25]
    reward_post_counsel = df[
        df["Risk_Level"].isin(["Red", "Amber"]) & (df["Dropout_Probability"] < 0.75)
    ]
    reward_eligible = pd.concat([reward_low_risk, reward_post_counsel]).drop_duplicates()

    st.sidebar.metric("Reward Eligible Students", len(reward_eligible))

    st.sidebar.markdown("### Rewards")
    st.sidebar.markdown(
        """
        - Free Stationery  
        - Books  
        - Digital Vouchers  
        - Recognition on Dashboard
        """
    )

    high_risk_count = len(risky)
    with st.sidebar.expander(f" High Risk Students & Schemes ({high_risk_count})", expanded=True):
        st.markdown("**Suggested Schemes:**")
        st.markdown(
            """
            - General Counseling  
            - Mentorship Program  
            - Financial Aid Guidance  
            - Girl-Specific Schemes (Kanyashree, Beti Bachao, Scholarships)  
            """
        )

    
    # Tabs for Dashboard & Risk Groups
    amber_count = (df["Risk_Level"] == "Amber").sum()
    red_count = (df["Risk_Level"] == "Red").sum()

    tab1, tab2, tab3 = st.tabs([
        " Dashboard",
        f"Amber Zone ({amber_count})",
        f"Red Zone ({red_count})"
    ])

    # Dashboard Tab 
    with tab1:
        st.subheader("Predictions & Risk Summary")

        # FIX: make dataframe Arrow-compatible before displaying
        df_display = df.copy()
        # Convert dtypes so that object columns become proper string/nullable types
        df_display = df_display.convert_dtypes()

        st.dataframe(
            df_display.style.apply(
                lambda row: [
                    "background-color: #ff4d4d" if row.Risk_Level == "Red"
                    else "background-color: #ffa500" if row.Risk_Level == "Amber"
                    else "background-color: #90ee90"
                    for _ in row
                ],
                axis=1
            )
        )

        st.download_button(
            "Download Predictions",
            df.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv"
        )

        st.subheader(" Risk Summary")
        st.markdown(f" **Red:** {(df['Risk_Level'] == 'Red').sum()} students")
        st.markdown(f" **Amber:** {(df['Risk_Level'] == 'Amber').sum()} students")
        st.markdown(f" **Green:** {(df['Risk_Level'] == 'Green').sum()} students")

        st.subheader("Dropout Probability Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["Dropout_Probability"], bins=20, edgecolor="black")
        ax.set_title("Distribution of Dropout Probability")
        ax.set_xlabel("Dropout Probability")
        ax.set_ylabel("Number of Students")
        st.pyplot(fig)

    # Amber Zone Tab 
    with tab2:
        st.subheader("Amber Zone Students - Detailed View")
        amber_students = df[df["Risk_Level"] == "Amber"]

        if amber_students.empty:
            st.info("No students in Amber Zone.")
        else:
            selected_id = st.selectbox(
                "Select an Amber Zone Student:", amber_students["student_id"].tolist()
            )
            student = amber_students[amber_students["student_id"] == selected_id].iloc[0]

            st.write("### Student Details")
            st.write(student)

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=student["Dropout_Probability"] * 100,
                    title={'text': "Dropout Probability (%)"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "orange"}}
                )
            )
            st.plotly_chart(fig_gauge)

    # Red Zone Tab 
    with tab3:
        st.subheader("Red Zone Students - Detailed View")
        red_students = df[df["Risk_Level"] == "Red"]

        if red_students.empty:
            st.info("No students in Red Zone.")
        else:
            selected_id = st.selectbox(
                "Select a Red Zone Student:", red_students["student_id"].tolist()
            )
            student = red_students[red_students["student_id"] == selected_id].iloc[0]

            st.write("### Student Details")
            st.write(student)

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=student["Dropout_Probability"] * 100,
                    title={'text': "Dropout Probability (%)"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red"}}
                )
            )
            st.plotly_chart(fig_gauge)

    # Email Alerts Section
    st.subheader("Automatic Email Alerts")
    st.info("Use a Gmail App Password (recommended).")

    sender_email = st.text_input("Sender Email (Gmail)")
    password = st.text_input("Email Password / App Password", type="password")

    if st.button("Send Counseling Alerts"):
        if not sender_email or not password:
            st.error("Please enter sender email and password.")
        else:
            count_sent = 0
            for _, row in risky.iterrows():
                recipients = []
                if row["Risk_Level"] == "Red":
                    recipients = [row.get("guardian_mail", ""), row.get("mentor_mail", "")]
                elif row["Risk_Level"] == "Amber":
                    recipients = [row.get("student_mail", ""), row.get("mentor_mail", "")]
                recipients = [r for r in recipients if r]  # remove empty

                if recipients:
                    log_entry = send_email_alert(
                        row,
                        recipients,
                        risk=row["Risk_Level"],
                        sender_email=sender_email,
                        sender_password=password
                    )
                    if log_entry.get("status") == "Sent":
                        count_sent += 1

            st.success(f"Emails sent successfully to {count_sent} recipients!")
