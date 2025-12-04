import os
import re
from io import BytesIO

import joblib
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ==========================
# Config
# ==========================
MODEL_PATH = "campaign_model_merchant_only.joblib"
DEFAULT_CONF_THRESHOLD = 0.8  # adjust default in UI if needed


# ==========================
# Text Preprocessor
# ==========================
class SimpleTextCleaner(BaseEstimator, TransformerMixin):
    """
    Basic text cleaning:
    - lowercasing
    - remove extra spaces
    - keep alphanumeric, underscore, spaces, & some separators
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = []
        for text in X:
            if not isinstance(text, str):
                text = str(text)
            t = text.lower()
            # keep letters, numbers, underscore, space, dash, slash
            t = re.sub(r"[^0-9a-zA-Z_\-\s/]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            cleaned.append(t)
        return cleaned


# ==========================
# Helper Functions
# ==========================
def load_excel_with_sheet_selection(uploaded_file, key_prefix):
    """
    Load Excel file and allow user to select sheet if multiple sheets exist.
    Returns a pandas DataFrame or None.
    """
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheets = xls.sheet_names

        if len(sheets) > 1:
            sheet = st.selectbox(
                f"Select sheet for {key_prefix}:",
                sheets,
                key=f"{key_prefix}_sheet_select",
            )
        else:
            sheet = sheets[0]

        df = pd.read_excel(uploaded_file, sheet_name=sheet)
        return df

    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None


def train_model(df: pd.DataFrame):
    """
    Train a classifier:
    Input  : Category Name (campaign name)
    Output : Merchant Category
    Strategy & Campaign Type are rule-based, not learned.
    """
    required_cols = ["Category Name", "Merchant Category"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training file: {missing}")

    X = df["Category Name"].astype(str)
    y = df["Merchant Category"].astype(str)

    pipeline = Pipeline(
        [
            ("cleaner", SimpleTextCleaner()),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),  # unigrams + bigrams
                    min_df=2,            # ignore very rare terms
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",  # handle imbalanced Merchant Category
                    multi_class="auto",
                ),
            ),
        ]
    )

    pipeline.fit(X, y)
    return pipeline


def derive_strategy_campaign(campaign_name: str):
    """
    Derive Strategy and Campaign Type purely from naming rules.

    Rules (case-insensitive, after handling 'PMax: '):
      - brand-, brand_, 'brand '   -> Strategy = BRAND      ; Campaign Type = BRAND
      - dsa                        -> Strategy = DSA        ; Campaign Type = GENERIC
      - lia_                       -> Strategy = LIA        ; Campaign Type = SHOPPING
      - nb_                        -> Strategy = NON BRAND  ; Campaign Type = GENERIC
      - pla_                       -> Strategy = PLA        ; Campaign Type = SHOPPING
      - pmax-local_                -> Strategy = PMAX LOCAL ; Campaign Type = SEARCH (default)
      - shopping_                  -> Strategy = SHOPPING   ; Campaign Type = SHOPPING
      - sb_                        -> Strategy = SUB BRAND  ; Campaign Type = GENERIC

    Special:
      - If original starts with 'PMax: ', we strip that and then apply rules to the rest.
      - For 'PMax: ' cases:
          * startswith 'pla'      -> PLA / SHOPPING
          * startswith 'lia'      -> LIA / SHOPPING
          * startswith 'shopping' -> SHOPPING / SHOPPING
    """
    if not isinstance(campaign_name, str):
        campaign_name = str(campaign_name)

    original = campaign_name.strip()
    lower_orig = original.lower()

    # Handle "PMax: " specially (don't confuse with 'pmax-')
    # We strip the prefix ONLY if it literally starts with "pmax: " (case-insensitive)
    core = original
    if lower_orig.startswith("pmax: "):
        core = original[len("PMax: "):].lstrip()

    lower_core = core.lower()

    # BRAND rules
    if lower_core.startswith("brand-") or lower_core.startswith("brand_") or lower_core.startswith("brand "):
        return "BRAND", "BRAND"

    # DSA
    if lower_core.startswith("dsa"):
        return "DSA", "GENERIC"

    # LIA
    if lower_core.startswith("lia_"):
        return "LIA", "SHOPPING"

    # NON BRAND
    if lower_core.startswith("nb_"):
        return "NON BRAND", "GENERIC"

    # PLA
    if lower_core.startswith("pla_"):
        return "PLA", "SHOPPING"

    # PMAX LOCAL (not from PMax:, but 'pmax-local_')
    if lower_core.startswith("pmax-local_"):
        # In your data: can be SEARCH or GENERIC; default to SEARCH
        return "PMAX LOCAL", "SEARCH"

    # SHOPPING
    if lower_core.startswith("shopping_"):
        return "SHOPPING", "SHOPPING"

    # SUB BRAND
    if lower_core.startswith("sb_"):
        return "SUB BRAND", "GENERIC"

    # --- PMax: derived rules (after stripping "PMax: ") ---
    # These kick in only for campaigns that *originally* started with PMax:
    if lower_orig.startswith("pmax: "):
        # after PMax: we already stripped; now look at core
        if lower_core.startswith("pla"):
            return "PLA", "SHOPPING"
        if lower_core.startswith("lia"):
            return "LIA", "SHOPPING"
        if lower_core.startswith("shopping"):
            return "SHOPPING", "SHOPPING"
        # fallback for PMax: unknown subtype – treat as SHOPPING generic
        return "SHOPPING", "SHOPPING"

    # If nothing matched, we don't enforce a strategy/type here.
    # You could add more heuristic rules later.
    return None, None


def predict_with_confidences(model, campaigns: pd.Series, conf_threshold: float):
    """
    Predict Merchant Category and then derive Strategy & Campaign Type with rules.

    Returns DataFrame with:
      - Category Name
      - Merchant Category
      - Merchant Category Confidence
      - Strategy
      - Campaign Type
      - Overall Confidence  (== Merchant Category Confidence)
      - Needs Review        (based on Overall Confidence vs threshold)
    """
    X_new = campaigns.astype(str)

    cleaner = model.named_steps["cleaner"]
    tfidf = model.named_steps["tfidf"]
    clf: LogisticRegression = model.named_steps["clf"]

    X_clean = cleaner.transform(X_new)
    X_features = tfidf.transform(X_clean)

    probs = clf.predict_proba(X_features)
    classes = clf.classes_
    max_indices = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)
    preds = classes[max_indices]

    result = pd.DataFrame({"Category Name": X_new}).reset_index(drop=True)
    result["Merchant Category"] = preds
    result["Merchant Category Confidence"] = max_probs

    # Derive Strategy & Campaign Type from rules
    strategies = []
    campaign_types = []
    for name in X_new:
        strat, ctype = derive_strategy_campaign(name)
        strategies.append(strat)
        campaign_types.append(ctype)

    result["Strategy"] = strategies
    result["Campaign Type"] = campaign_types

    # Overall Confidence driven only by Merchant Category Confidence
    result["Overall Confidence"] = result["Merchant Category Confidence"]
    result["Needs Review"] = result["Overall Confidence"] < conf_threshold

    return result


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    return output.getvalue()


def save_model(model, path: str = MODEL_PATH):
    joblib.dump(model, path)


def load_model(path: str = MODEL_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None


# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Campaign Strategy Tagger", layout="wide")

st.title("📊 Campaign Strategy Tagger")
st.caption(
    "Model predicts Merchant Category; Strategy & Campaign Type come from hard media rules."
)

# Sidebar controls
st.sidebar.header("⚙️ Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold for 'Needs Review' (Merchant Category only)",
    min_value=0.5,
    max_value=0.99,
    value=DEFAULT_CONF_THRESHOLD,
    step=0.01,
)
st.sidebar.write(
    f"`Overall Confidence` = **Merchant Category Confidence**.\n"
    f"Rows with confidence < {conf_threshold:.2f} will be **Needs Review = True**."
)

# Load existing model if present
if "model" not in st.session_state:
    st.session_state.model = load_model()

model_loaded = st.session_state.model is not None
st.sidebar.write(
    f"Model status: {'✅ Loaded from disk' if model_loaded else '❌ No saved model yet'}"
)

if model_loaded:
    if st.sidebar.button("🗑️ Delete Saved Model"):
        try:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            st.session_state.model = None
            st.sidebar.success("Saved model deleted.")
        except Exception as e:
            st.sidebar.error(f"Failed to delete model: {e}")

# Layout: 2 columns
col_train, col_predict = st.columns(2)

# ======================================================
# 1️⃣ TRAINING COLUMN
# ======================================================
with col_train:
    st.markdown("### 1️⃣ Upload & Train Merchant Category Model")

    train_file = st.file_uploader(
        "Training Excel (.xlsx / .xls) with historical tagging",
        type=["xlsx", "xls"],
        key="train_file",
    )

    train_df = None

    if train_file is not None:
        train_df = load_excel_with_sheet_selection(train_file, key_prefix="train")

        if train_df is not None:
            st.write("Preview of training data:")
            st.dataframe(train_df.head())

            # Optional analytics
            with st.expander("📈 Training Data Analytics", expanded=False):
                if "Merchant Category" in train_df.columns:
                    st.write("Distribution of **Merchant Category**:")
                    st.bar_chart(train_df["Merchant Category"].value_counts())

                if "Strategy" in train_df.columns:
                    st.write("Distribution of **Strategy**:")
                    st.bar_chart(train_df["Strategy"].value_counts())

                if "Campaign Type" in train_df.columns:
                    st.write("Distribution of **Campaign Type**:")
                    st.bar_chart(train_df["Campaign Type"].value_counts())

            if st.button("✅ Train / Retrain Model"):
                try:
                    with st.spinner("Training Merchant Category model..."):
                        model = train_model(train_df)
                        st.session_state.model = model
                        save_model(model)
                    st.success("Model trained and saved successfully!")
                except Exception as e:
                    st.error(f"Training error: {e}")
        else:
            st.info("Upload a valid Excel file to proceed.")
    else:
        st.info("Upload your historical tagging Excel to train the model.")

# ======================================================
# 2️⃣ PREDICTION COLUMN
# ======================================================
with col_predict:
    st.markdown("### 2️⃣ Tag New Campaigns")

    if st.session_state.model is None:
        st.warning("No trained model available. Please train one in step 1.")
    else:
        tab_text, tab_file = st.tabs(["✏️ Enter Manually", "📁 Upload File"])

        new_campaigns_df = None

        # --------------------------
        # A) Manual text input
        # --------------------------
        with tab_text:
            text_input = st.text_area(
                "Enter campaign names (one per line):",
                height=250,
                placeholder=(
                    "Example:\n"
                    "brand_bts_LEGO_unid_cons_non-match_all\n"
                    "PMax: nb_LEGO_summer_sale\n"
                    "pmax-local_nb_generic_test\n"
                ),
            )

            if st.button("🔍 Predict from Text Input"):
                lines = [l.strip() for l in text_input.split("\n") if l.strip()]
                if len(lines) == 0:
                    st.error("Please enter at least one campaign name.")
                else:
                    with st.spinner("Predicting..."):
                        new_campaigns_df = predict_with_confidences(
                            st.session_state.model,
                            pd.Series(lines),
                            conf_threshold=conf_threshold,
                        )

        # --------------------------
        # B) File upload input
        # --------------------------
        with tab_file:
            pred_file = st.file_uploader(
                "Upload CSV/Excel with `Category Name` column",
                type=["csv", "xlsx", "xls"],
                key="predict_file",
            )

            if st.button("🔍 Predict from File"):
                if pred_file is None:
                    st.error("Upload a file first.")
                else:
                    try:
                        if pred_file.name.lower().endswith(".csv"):
                            pred_df = pd.read_csv(pred_file)
                        else:
                            pred_df = load_excel_with_sheet_selection(
                                pred_file, key_prefix="predict"
                            )

                        if pred_df is not None:
                            if "Category Name" not in pred_df.columns:
                                st.error(
                                    "Uploaded file must contain a `Category Name` column."
                                )
                            else:
                                with st.spinner("Predicting..."):
                                    new_campaigns_df = predict_with_confidences(
                                        st.session_state.model,
                                        pred_df["Category Name"],
                                        conf_threshold=conf_threshold,
                                    )
                    except Exception as e:
                        st.error(f"Error reading prediction file: {e}")

        # --------------------------
        # RESULTS + DOWNLOAD
        # --------------------------
        if new_campaigns_df is not None:
            st.markdown("---")
            st.markdown("### 3️⃣ Results")

            st.success(
                f"Predictions generated for {len(new_campaigns_df)} campaign(s)."
            )

            cols_to_show = [
                "Category Name",
                "Merchant Category",
                "Strategy",
                "Campaign Type",
                "Overall Confidence",
                "Needs Review",
            ]
            cols_to_show = [c for c in cols_to_show if c in new_campaigns_df.columns]

            st.dataframe(new_campaigns_df[cols_to_show])

            with st.expander("🔬 Full Prediction Data (with confidence)"):
                st.dataframe(new_campaigns_df)

            excel_output = to_excel_bytes(new_campaigns_df)

            st.download_button(
                "📥 Download Predictions as Excel",
                data=excel_output,
                file_name="campaign_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# Footer
st.markdown("---")
st.caption("AI-driven Merchant Category prediction • Strategy & Campaign Type from media rules.")
