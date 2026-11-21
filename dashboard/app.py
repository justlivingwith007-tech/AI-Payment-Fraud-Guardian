
"""Streamlit demonstration UI; accepts CSV rows with anonymised features."""

import pandas as pd
import requests
import streamlit as st

from src.preprocess import FEATURES


# 1. Page configuration
st.set_page_config(
    page_title="AI Payment Fraud Guardian",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Payment Fraud Guardian")

st.caption(
    "Educational demonstration. "
    "Not suitable for approving or blocking real payments."
)

st.info(
    "This dataset uses anonymised V1–V28 features. "
    "Upload a CSV row with Time, V1–V28 and Amount; "
    "the Class label is not sent to the API."
)


# 2. API configuration
api_url = st.text_input(
    "FastAPI base URL",
    "http://127.0.0.1:8000"
)


# 3. CSV file upload
file = st.file_uploader(
    "Upload CSV with at least one transaction",
    type=["csv"]
)


# 4. Process uploaded CSV
if file is not None:

    try:
        data = pd.read_csv(file)

        missing = sorted(set(FEATURES) - set(data.columns))

        if missing:
            st.error(f"Missing columns: {missing}")

        elif data.empty:
            st.error("The uploaded file has no rows")

        else:
            # Display uploaded transactions
            st.dataframe(
                data.head(5),
                use_container_width=True
            )

            # Select transaction row
            index = st.number_input(
                "Row index to score",
                min_value=0,
                max_value=len(data) - 1,
                value=0
            )

            # 5. Predict fraud risk
            if st.button("Predict fraud risk", type="primary"):

                payload = data.iloc[int(index)][FEATURES].to_dict()

                response = requests.post(
                    f"{api_url.rstrip('/')}/predict",
                    json=payload,
                    timeout=20
                )

                # 6. Display prediction result
                if response.ok:

                    result = response.json()

                    st.metric(
                        "Fraud probability",
                        f"{result['fraud_probability']:.2%}"
                    )

                    st.metric(
                        "Model threshold",
                        f"{result['threshold']:.2%}"
                    )

                    # Display fraud warning
                    if result["is_flagged"]:
                        st.warning("⚠️ Flag for review")
                    else:
                        st.success("✅ Not flagged")

                    # Display complete JSON response
                    st.json(result)

                    # Display historical label if available
                    if "Class" in data.columns:

                        actual_class = int(
                            data.iloc[int(index)]["Class"]
                        )

                        st.caption(
                            f"Historical label for this row: "
                            f"{actual_class} "
                            "(0=legitimate, 1=fraud)"
                        )

                else:
                    st.error(
                        f"API returned {response.status_code}: "
                        f"{response.text}"
                    )

    except (
        pd.errors.ParserError,
        ValueError,
        requests.RequestException
    ) as exc:

        st.error(f"Unable to process request: {exc}")
