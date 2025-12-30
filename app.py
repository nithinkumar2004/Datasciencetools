import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
import duckdb  # Added for efficient whole-dataset SQL querying
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from config import OPENROUTER_API_KEY


# --- 1. SETTINGS & CONFIG ---
#OPENROUTER_API_KEY = "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"
FREE_MODEL_LIST = [
    "tngtech/deepseek-r1t2-chimera:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2-7b-instruct:free"
]


def call_openrouter(prompt, json_mode=False):
    for model_id in FREE_MODEL_LIST:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "DSBot Advanced",
                },
                data=json.dumps({
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1 if json_mode else 0.7
                }),
                timeout=15
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
        except:
            continue
    return None


# --- UI SETUP ---
st.set_page_config(page_title="DSBot: Smart Agentic Dashboard", layout="wide")
st.markdown(
    """<style>.stMetric { background: #16213e; padding: 15px; border-radius: 10px; border: 1px solid #4ecca3; }</style>""",
    unsafe_allow_html=True)


# --- 2. ML & UTILS ---
def detect_model_type(df, target_col):
    """Smart Detection: Identifies if a task is Classification or Regression."""
    unique_vals = df[target_col].nunique()
    if df[target_col].dtype == 'O' or unique_vals < 15:
        return "auto_classification"
    return "auto_regression"


def run_ml_pipeline(df, target, pipeline_tokens):
    # Dropping high-cardinality IDs
    cols_to_drop = [col for col in df.columns if df[col].nunique() > 100 and df[col].dtype == 'O']
    if target in cols_to_drop: cols_to_drop.remove(target)

    X = df.drop(columns=[target] + cols_to_drop)
    y = df[target]
    X = X.fillna(X.mean(numeric_only=True))
    X_processed = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    is_clf = "auto_classification" in pipeline_tokens
    model = RandomForestClassifier(n_estimators=100) if is_clf else RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    m1, m2 = st.columns(2)
    with m1:
        score = accuracy_score(y_test, preds) if is_clf else r2_score(y_test, preds)
        st.metric("Model Performance", f"{score:.2%}" if is_clf else f"{score:.4f} (R²)")
    with m2:
        fig, ax = plt.subplots(figsize=(6, 4));
        plt.style.use('dark_background')
        if is_clf:
            sns.heatmap(pd.crosstab(y_test, preds), annot=True, fmt='d', cmap="rocket", ax=ax)
        else:
            sns.regplot(x=y_test, y=preds, scatter_kws={'color': '#4ecca3'}, line_kws={'color': '#ff2e63'}, ax=ax)
        st.pyplot(fig)

    st.session_state.trained_brain = model
    st.session_state.model_columns = X_processed.columns.tolist()
    st.session_state.raw_features = X.columns.tolist()
    st.session_state.target_name = target


# --- 3. MAIN APP ---
st.title("🚀 DSBot: Smart Agentic Data Science")
tab_data, tab_chat, tab_viz, tab_ml, tab_infer = st.tabs(
    ["📁 Data", "💬 Smart Q&A", "📊 Agentic EDA", "🤖 Smart AutoML", "🔮 Inference"])

if uploaded_file := tab_data.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx']):
    st.session_state.df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(
        uploaded_file)
    tab_data.dataframe(st.session_state.df.head(10), use_container_width=True)

# SMART Q&A (DuckDB Text-to-SQL)
with tab_chat:
    if 'df' in st.session_state:
        if prompt := st.chat_input("Ask a question about the whole dataset..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                # Agentic SQL Generation
                schema = f"Table name: 'df_table'. Columns: {list(st.session_state.df.columns)}"
                sql_prompt = f"System: Use DuckDB SQL. {schema}. Query for: {prompt}. Return ONLY SQL code in a code block."
                sql_query = call_openrouter(sql_prompt)

                try:
                    clean_sql = sql_query.split("sql")[1].split("")[
                        0].strip() if "```sql" in sql_query else sql_query.strip()
                    df_table = st.session_state.df  # DuckDB reads local variables
                    result = duckdb.query(clean_sql).to_df()
                    st.write("📊 Query Result:")
                    st.dataframe(result)
                    # Natural Language Summary
                    summary = call_openrouter(f"Data: {result.to_dict()}. User Question: {prompt}. Summarize findings.")
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Agentic SQL failed. Retrying... Error: {e}")
    else:
        st.info("Upload data first.")

# AGENTIC EDA
with tab_viz:
    if 'df' in st.session_state:
        viz_query = st.text_input("What patterns should I visualize?")
        if st.button("Generate AI Visualization Suite"):
            # Only send numeric/categorical column groups to prevent errors
            num_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = st.session_state.df.select_dtypes(exclude=np.number).columns.tolist()
            prompt = f"Numeric: {num_cols}. Categorical: {cat_cols}. Request: {viz_query}. Return ONLY JSON list of 6 dicts with title, type, x, y."
            res = call_openrouter(prompt, json_mode=True)
            try:
                st.session_state.viz_plan = json.loads(res[res.find("["):res.rfind("]") + 1])
            except:
                st.error("AI Visualization Agent Busy.")

        if 'viz_plan' in st.session_state:
            cols = st.columns(3)
            for i, p in enumerate(st.session_state.viz_plan):
                with cols[i % 3]:
                    st.markdown(f"*{p['title']}*")
                    fig, ax = plt.subplots(figsize=(5, 4));
                    plt.style.use('dark_background')
                    try:
                        if p['type'] == 'bar':
                            sns.barplot(data=st.session_state.df.head(15), x=p['x'], y=p['y'], ax=ax)
                        elif p['type'] == 'line':
                            sns.lineplot(data=st.session_state.df, x=p['x'], y=p['y'], ax=ax)
                        else:
                            sns.scatterplot(data=st.session_state.df, x=p['x'], y=p['y'], ax=ax)
                        st.pyplot(fig)
                    except:
                        st.caption("Debug Agent: Column mismatch. Retrying next graph...")

# SMART AUTOML
with tab_ml:
    if 'df' in st.session_state:
        target = st.selectbox("Select Prediction Target", st.session_state.df.columns)
        ml_query = st.text_input("Model Goal (Optional)", placeholder="e.g., Predict prices")
        if st.button("Train Smart Model"):
            with st.spinner("🤖 Detecting optimal algorithm..."):
                # AUTOMATED TYPE DETECTION
                detected_type = detect_model_type(st.session_state.df, target)

                # Check LLM for user preference, otherwise use detected type
                prompt = f"User Goal: {ml_query}. Target: {target}. Available: [auto_classification, auto_regression]. Return ONLY JSON: {{'pipeline': ['token']}}"
                res = call_openrouter(prompt, json_mode=True)

                try:
                    pipe_data = json.loads(res[res.find("{"):res.rfind("}") + 1])
                    tokens = pipe_data.get('pipeline', [detected_type])
                except:
                    tokens = [detected_type]

                st.info(f"🚀 *Auto-Detected Strategy:* {tokens[0].replace('_', ' ').title()}")
                run_ml_pipeline(st.session_state.df, target, tokens)
                st.success("✅ Training Complete!")

# INFERENCE
with tab_infer:
    if 'trained_brain' in st.session_state:
        with st.form("inference_form"):
            user_inputs = {}
            cols = st.columns(2)
            for i, feat in enumerate(st.session_state.raw_features):
                with cols[i % 2]:
                    if st.session_state.df[feat].dtype == 'O':
                        user_inputs[feat] = st.selectbox(feat, list(st.session_state.df[feat].unique()))
                    else:
                        user_inputs[feat] = st.number_input(feat, value=float(st.session_state.df[feat].mean()))
            if st.form_submit_button("Generate Prediction"):
                input_df = pd.DataFrame(columns=st.session_state.model_columns).fillna(0)
                input_df.loc[0] = 0
                for col, val in user_inputs.items():
                    if col in input_df.columns:
                        input_df.at[0, col] = val
                    elif f"{col}{val}" in input_df.columns:
                        input_df.at[0, f"{col}{val}"] = 1
                pred = st.session_state.trained_brain.predict(input_df)[0]
                st.balloons();
                st.metric(f"Prediction for {st.session_state.target_name}", f"{pred}")