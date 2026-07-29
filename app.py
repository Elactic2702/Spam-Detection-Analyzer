import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Spam Detection Analyzer",
    page_icon="📩",
    layout="wide"
)

# -----------------------------
# Load Model and Vectorizer
# -----------------------------
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# -----------------------------
# Header
# -----------------------------
st.title("📩 Spam Detection Analyzer")
st.write(
    "Analyze SMS or email messages using Machine Learning "
    "and Natural Language Processing."
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ About the Project")

st.sidebar.info(
    """
    This application uses:

    • TF-IDF Vectorization
    • Multinomial Naive Bayes
    • Natural Language Processing
    • Machine Learning

    The model classifies messages as:

    🟢 Not Spam
    🔴 Spam
    """
)

# -----------------------------
# Single Message Detection
# -----------------------------
st.subheader("🔍 Analyze a Message")

message = st.text_area(
    "Enter your SMS or Email message:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize. Click here to claim..."
)

if st.button("🔎 Analyze Message", type="primary"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:
        # Convert message into TF-IDF features
        message_vector = vectorizer.transform([message])

        # Make prediction
        prediction = model.predict(message_vector)[0]

        # Get probability
        probability = model.predict_proba(message_vector)[0]

        # Spam probability
        spam_probability = probability[1] * 100

        # Not Spam probability
        not_spam_probability = probability[0] * 100

        st.divider()

        # -----------------------------
        # Display Result
        # -----------------------------
        if prediction == 1:

            st.error("🚨 SPAM MESSAGE DETECTED!")

            st.metric(
                "Spam Probability",
                f"{spam_probability:.2f}%"
            )

            st.progress(
                int(spam_probability)
            )

            st.write(
                "⚠️ This message appears to be potentially "
                "unwanted or suspicious."
            )

        else:

            st.success("✅ NOT SPAM")

            st.metric(
                "Not Spam Probability",
                f"{not_spam_probability:.2f}%"
            )

            st.progress(
                int(not_spam_probability)
            )

            st.write(
                "✅ This message appears to be legitimate."
            )

        # -----------------------------
        # Probability Details
        # -----------------------------
        st.subheader("📊 Prediction Details")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Not Spam",
                f"{not_spam_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Spam",
                f"{spam_probability:.2f}%"
            )


# -----------------------------
# Batch CSV Analysis
# -----------------------------
st.divider()

st.subheader("📁 Bulk Message Analysis")

st.write(
    "Upload a CSV file containing a column named `message` "
    "to analyze multiple messages."
)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "message" not in df.columns:

        st.error(
            "❌ CSV must contain a column named 'message'."
        )

    else:

        # Transform messages
        messages_vector = vectorizer.transform(
            df["message"].astype(str)
        )

        # Predict
        predictions = model.predict(messages_vector)

        # Add predictions
        df["prediction"] = predictions

        df["prediction"] = df["prediction"].map({
            0: "Not Spam",
            1: "Spam"
        })

        # Display results
        st.success(
            f"✅ Successfully analyzed {len(df)} messages!"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        # -----------------------------
        # Statistics
        # -----------------------------
        st.subheader("📊 Analysis Summary")

        spam_count = (df["prediction"] == "Spam").sum()
        not_spam_count = (
            df["prediction"] == "Not Spam"
        ).sum()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🚨 Spam Messages",
                spam_count
            )

        with col2:
            st.metric(
                "✅ Not Spam Messages",
                not_spam_count
            )