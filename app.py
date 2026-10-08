import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import ollama
import io

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Data Analysis & Report Generator",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

if "ai_error" not in st.session_state:
    st.session_state.ai_error = None

if "report_text" not in st.session_state:
    st.session_state.report_text = None

# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(99,102,241,0.12), transparent 30%),
        radial-gradient(circle at 85% 15%, rgba(168,85,247,0.10), transparent 30%),
        radial-gradient(circle at 50% 90%, rgba(59,130,246,0.08), transparent 35%),
        #080b14;
    color: #f5f7ff;
}

/* Main container */

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #0b0e18 0%, #090c14 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}

section[data-testid="stSidebar"] * {
    color: #dce1ee;
}

section[data-testid="stSidebar"] .stRadio label {
    color: #dce1ee !important;
    font-weight: 500;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #c8cede;
}

.sidebar-brand {
    padding: 10px 5px 25px 5px;
}

.sidebar-brand-title {
    font-size: 22px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
}

.sidebar-brand-subtitle {
    font-size: 11px;
    color: #8d96aa;
    margin-top: 4px;
    letter-spacing: 1px;
}

/* Hero */

.hero {
    padding: 38px 42px;
    border-radius: 28px;
    background:
        linear-gradient(135deg,
        rgba(20,25,45,0.94),
        rgba(11,15,29,0.94));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow:
        0 25px 80px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.04);
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.hero:after {
    content: "";
    position: absolute;
    width: 380px;
    height: 380px;
    right: -160px;
    top: -180px;
    border-radius: 50%;
    background: rgba(99,102,241,0.16);
    filter: blur(50px);
}

.hero-kicker {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #9ca8ff;
    margin-bottom: 13px;
}

.hero-title {
    font-size: clamp(38px, 5vw, 68px);
    line-height: 1.02;
    font-weight: 800;
    letter-spacing: -3px;
    color: #ffffff;
}

.hero-title span {
    background: linear-gradient(90deg, #a5b4fc, #c084fc, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    max-width: 850px;
    color: #aeb7ca;
    font-size: 16px;
    line-height: 1.7;
    margin-top: 18px;
}

/* Cards */

.metric-card {
    padding: 22px;
    min-height: 120px;
    border-radius: 20px;
    background: rgba(18,23,38,0.82);
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 15px 40px rgba(0,0,0,0.18);
}

.metric-label {
    color: #8993aa;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.metric-value {
    margin-top: 8px;
    font-size: 31px;
    font-weight: 800;
    color: #ffffff;
}

.metric-small {
    color: #7e89a0;
    font-size: 12px;
    margin-top: 4px;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #ffffff;
    margin-top: 30px;
    margin-bottom: 6px;
}

.section-subtitle {
    color: #8993aa;
    font-size: 13px;
    margin-bottom: 18px;
}

.feature-card {
    padding: 18px;
    border-radius: 17px;
    background: rgba(17,22,37,0.72);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 10px;
}

.feature-icon {
    font-size: 21px;
    margin-right: 8px;
}

.feature-title {
    color: #f4f6ff;
    font-weight: 650;
}

.feature-text {
    color: #8791a8;
    font-size: 13px;
    margin-top: 4px;
}

.ai-card {
    padding: 28px;
    border-radius: 23px;
    background:
        linear-gradient(135deg,
        rgba(25,22,50,0.94),
        rgba(13,20,39,0.94));
    border: 1px solid rgba(129,140,248,0.20);
    box-shadow:
        0 20px 60px rgba(0,0,0,0.25),
        inset 0 1px 0 rgba(255,255,255,0.04);
}

.ai-title {
    font-size: 24px;
    font-weight: 750;
    color: #ffffff;
}

.ai-subtitle {
    color: #929bb0;
    font-size: 13px;
    margin-top: 5px;
}

.info-box {
    padding: 18px 20px;
    border-radius: 16px;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(129,140,248,0.16);
    color: #cbd2ff;
}

.footer {
    margin-top: 70px;
    padding-top: 25px;
    border-top: 1px solid rgba(255,255,255,0.07);
    text-align: center;
    color: #68738a;
    font-size: 12px;
}

</style>
""")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">✦ AI Analytics</div>
        <div class="sidebar-brand-subtitle">
            INTELLIGENT DATA PLATFORM
        </div>
    </div>
    """)

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Data Upload",
            "Statistical Analysis",
            "Visual Analytics",
            "AI Insights",
            "Report Generator"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.html("""
    <div style="
        padding:14px;
        border-radius:14px;
        background:rgba(99,102,241,0.08);
        border:1px solid rgba(129,140,248,0.12);
        margin-top:10px;
    ">
        <div style="
            color:#9ca8ff;
            font-size:11px;
            font-weight:700;
            letter-spacing:1px;
        ">
            LOCAL AI
        </div>
        <div style="
            color:#ffffff;
            font-size:14px;
            font-weight:650;
            margin-top:5px;
        ">
            Qwen 2.5 • 7B
        </div>
        <div style="
            color:#7f8aa2;
            font-size:11px;
            margin-top:3px;
        ">
            Powered by Ollama
        </div>
    </div>
    """)

# ============================================================
# FUNCTIONS
# ============================================================

def clean_value(value):
    if pd.isna(value):
        return "N/A"

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return round(float(value), 4)

    return value


def detect_anomalies(df):
    """
    Detect numerical outliers using the IQR method.
    """

    numeric_df = df.select_dtypes(include=np.number)

    anomaly_results = {}

    for column in numeric_df.columns:

        series = numeric_df[column].dropna()

        if len(series) < 5:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            anomaly_results[column] = 0
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = ((series < lower) | (series > upper)).sum()

        anomaly_results[column] = int(count)

    return anomaly_results


def detect_trends(df):
    """
    Simple directional trend detection based on
    first and last valid observations.
    """

    numeric_df = df.select_dtypes(include=np.number)

    results = []

    for column in numeric_df.columns:

        series = numeric_df[column].dropna()

        if len(series) < 5:
            continue

        first = float(series.iloc[0])
        last = float(series.iloc[-1])

        if first == 0:
            change_pct = None
        else:
            change_pct = ((last - first) / abs(first)) * 100

        if change_pct is None:
            direction = "Stable / cannot determine percentage change"
        elif change_pct > 5:
            direction = "Increasing"
        elif change_pct < -5:
            direction = "Decreasing"
        else:
            direction = "Relatively stable"

        results.append({
            "column": column,
            "first_value": round(first, 4),
            "last_value": round(last, 4),
            "change_percent": (
                round(change_pct, 2)
                if change_pct is not None
                else None
            ),
            "direction": direction
        })

    return results


def create_dataset_profile(df):
    """
    Creates a compact dataset profile to send to the local AI.
    This avoids sending the entire CSV to Qwen.
    """

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    profile = []

    profile.append("DATASET OVERVIEW")
    profile.append(f"Rows: {len(df)}")
    profile.append(f"Columns: {len(df.columns)}")
    profile.append(
        f"Numerical columns: {', '.join(numeric_columns) if numeric_columns else 'None'}"
    )
    profile.append(
        f"Categorical columns: {', '.join(categorical_columns) if categorical_columns else 'None'}"
    )

    profile.append("\nDATA TYPES")

    for column in df.columns:
        profile.append(
            f"- {column}: {str(df[column].dtype)}"
        )

    profile.append("\nMISSING VALUES")

    if len(missing) == 0:
        profile.append("No missing values.")
    else:
        for column, count in missing.items():
            percentage = (count / len(df)) * 100
            profile.append(
                f"- {column}: {count} ({percentage:.2f}%)"
            )

    profile.append("\nNUMERICAL STATISTICS")

    if numeric_columns:

        desc = df[numeric_columns].describe().T

        for column in numeric_columns:

            row = desc.loc[column]

            profile.append(
                f"- {column}: "
                f"mean={clean_value(row['mean'])}, "
                f"median={clean_value(df[column].median())}, "
                f"std={clean_value(row['std'])}, "
                f"min={clean_value(row['min'])}, "
                f"max={clean_value(row['max'])}"
            )

    profile.append("\nCATEGORICAL SUMMARY")

    for column in categorical_columns[:15]:

        values = df[column].value_counts(dropna=True).head(8)

        profile.append(
            f"- {column}: "
            + ", ".join(
                [
                    f"{str(index)}={count}"
                    for index, count in values.items()
                ]
            )
        )

    # Correlation summary

    if len(numeric_columns) >= 2:

        corr = df[numeric_columns].corr()

        profile.append("\nSTRONG CORRELATIONS")

        correlations = []

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):

                value = corr.iloc[i, j]

                if not pd.isna(value):

                    correlations.append(
                        (
                            abs(value),
                            corr.columns[i],
                            corr.columns[j],
                            value
                        )
                    )

        correlations.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        for _, a, b, value in correlations[:8]:

            profile.append(
                f"- {a} vs {b}: correlation={value:.3f}"
            )

    return "\n".join(profile)


def build_ai_prompt(df):

    profile = create_dataset_profile(df)

    anomalies = detect_anomalies(df)

    trends = detect_trends(df)

    anomaly_text = "\n".join(
        [
            f"- {column}: {count} potential outliers"
            for column, count in anomalies.items()
        ]
    )

    if not anomaly_text:
        anomaly_text = "No numerical anomaly results available."

    trend_text = "\n".join(
        [
            f"- {item['column']}: "
            f"{item['direction']}, "
            f"change={item['change_percent']}%"
            for item in trends
        ]
    )

    if not trend_text:
        trend_text = "No sufficient numerical data for trend analysis."

    # Sample data only.
    sample = df.head(10).copy()

    sample_text = sample.to_string(
        index=False,
        max_cols=15
    )

    prompt = f"""
You are an expert senior data analyst.

Analyze the dataset information below.

IMPORTANT RULES:
1. Use ONLY the information provided below.
2. Do not invent values, business facts, causes, or conclusions.
3. Clearly distinguish calculated facts from reasonable interpretations.
4. Keep the analysis professional and useful for a report.
5. If something cannot be determined from the dataset, say so.
6. Do not claim causation from correlation.
7. Do not claim a trend is statistically significant unless statistical significance was actually calculated.

DATASET PROFILE
----------------
{profile}

LOCAL ANOMALY DETECTION
-----------------------
{anomaly_text}

LOCAL TREND DETECTION
---------------------
{trend_text}

SAMPLE RECORDS
--------------
{sample_text}

Prepare the following sections:

## Executive Summary
Give a concise overview of the dataset and the most important findings.

## Key Findings
List the most important numerical and categorical observations.

## Trends
Discuss meaningful increasing, decreasing, or stable patterns based only on the supplied trend information.

## Anomalies
Discuss potential outliers detected by the IQR-based analysis.
Do not automatically assume an outlier is an error.

## Relationships
Discuss strong correlations if they are provided.
Explicitly state that correlation does not prove causation.

## Data Quality
Discuss missing values, duplicates, data types, and other visible quality issues.

## Recommendations
Give practical recommendations based strictly on the available evidence.

## Limitations
State what cannot be concluded from this dataset.

Write in professional business-report language.
"""

    return prompt


def ask_qwen(prompt):

    try:

        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional data analyst. "
                        "Be accurate, concise, evidence-based, "
                        "and never fabricate information."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.2
            }
        )

        return response["message"]["content"]

    except Exception as error:

        error_text = str(error)

        if "connection" in error_text.lower():
            return (
                "ERROR: Could not connect to Ollama. "
                "Please make sure the Ollama application is running."
            )

        if "not found" in error_text.lower():
            return (
                "ERROR: The Qwen 2.5 7B model was not found. "
                "Run `ollama list` and verify qwen2.5:7b is installed."
            )

        return f"ERROR: {error_text}"


def generate_basic_report(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    anomalies = detect_anomalies(df)

    trends = detect_trends(df)

    lines = []

    lines.append("=" * 70)
    lines.append("AI DATA ANALYSIS & REPORT GENERATOR")
    lines.append("=" * 70)
    lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append("")

    lines.append("DATASET OVERVIEW")
    lines.append("-" * 70)
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Columns: {len(df.columns)}")
    lines.append(
        f"Numerical columns: {len(numeric_columns)}"
    )
    lines.append(
        f"Categorical columns: {len(categorical_columns)}"
    )
    lines.append(
        f"Missing cells: {int(df.isnull().sum().sum())}"
    )
    lines.append(
        f"Duplicate rows: {int(df.duplicated().sum())}"
    )
    lines.append("")

    if numeric_columns:

        lines.append("NUMERICAL SUMMARY")
        lines.append("-" * 70)

        for column in numeric_columns:

            lines.append(
                f"{column}: "
                f"mean={df[column].mean():.4f}, "
                f"median={df[column].median():.4f}, "
                f"min={df[column].min():.4f}, "
                f"max={df[column].max():.4f}"
            )

        lines.append("")

    lines.append("TREND DETECTION")
    lines.append("-" * 70)

    for item in trends:

        lines.append(
            f"{item['column']}: "
            f"{item['direction']} "
            f"({item['change_percent']}%)"
        )

    lines.append("")

    lines.append("ANOMALY DETECTION")
    lines.append("-" * 70)

    for column, count in anomalies.items():

        lines.append(
            f"{column}: {count} potential outlier(s)"
        )

    lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">

    <div class="hero-kicker">
        ✦ AI-POWERED DATA ANALYTICS
    </div>

    <div class="hero-title">
        AI Data Analysis<br>
        <span>& Report Generator</span>
    </div>

    <div class="hero-description">
        Transform raw CSV data into meaningful statistics,
        visual analytics, trends, anomaly detection and
        intelligent AI-generated reports using local
        Qwen 2.5 7B intelligence.
    </div>

</div>
""")

# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    if st.session_state.df is None:

        st.html("""
        <div class="section-title">
            Welcome to your AI Data Platform
        </div>

        <div class="section-subtitle">
            Upload a CSV dataset to begin intelligent analysis.
        </div>
        """)

        uploaded_file = st.file_uploader(
            "Upload CSV Dataset",
            type=["csv"],
            help="Upload a CSV file for analysis."
        )

        if uploaded_file is not None:

            try:

                df = pd.read_csv(uploaded_file)

                st.session_state.df = df
                st.session_state.ai_result = None
                st.session_state.ai_error = None

                st.success(
                    f"Dataset loaded successfully: "
                    f"{uploaded_file.name}"
                )

                st.rerun()

            except Exception as error:

                st.error(
                    f"Could not read the CSV file: {error}"
                )

        st.html("""
        <div style="margin-top:28px;">

            <div class="feature-card">
                <span class="feature-icon">✓</span>
                <span class="feature-title">
                    Validate dataset
                </span>
                <div class="feature-text">
                    Automatically inspect structure, data types and quality.
                </div>
            </div>

            <div class="feature-card">
                <span class="feature-icon">▦</span>
                <span class="feature-title">
                    Statistical analysis
                </span>
                <div class="feature-text">
                    Calculate averages, medians, ranges and distributions.
                </div>
            </div>

            <div class="feature-card">
                <span class="feature-icon">⌁</span>
                <span class="feature-title">
                    Visual analytics
                </span>
                <div class="feature-text">
                    Explore numerical trends and correlations.
                </div>
            </div>

            <div class="feature-card">
                <span class="feature-icon">◈</span>
                <span class="feature-title">
                    AI-powered insights
                </span>
                <div class="feature-text">
                    Ask local Qwen 2.5 7B to interpret the dataset.
                </div>
            </div>

            <div class="feature-card">
                <span class="feature-icon">⚡</span>
                <span class="feature-title">
                    Automated report
                </span>
                <div class="feature-text">
                    Generate a professional downloadable analysis report.
                </div>
            </div>

        </div>
        """)

    else:

        df = st.session_state.df

        st.html("""
        <div class="section-title">
            Dataset Dashboard
        </div>

        <div class="section-subtitle">
            Overview of the uploaded dataset.
        </div>
        """)

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-label">Rows</div>
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-small">records</div>
            </div>
            """)

        with c2:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-label">Columns</div>
                <div class="metric-value">{len(df.columns):,}</div>
                <div class="metric-small">features</div>
            </div>
            """)

        with c3:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-label">Missing</div>
                <div class="metric-value">{int(df.isnull().sum().sum()):,}</div>
                <div class="metric-small">cells</div>
            </div>
            """)

        with c4:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-label">Duplicates</div>
                <div class="metric-value">{int(df.duplicated().sum()):,}</div>
                <div class="metric-small">rows</div>
            </div>
            """)

        with c5:
            st.html(f"""
            <div class="metric-card">
                <div class="metric-label">Numerical</div>
                <div class="metric-value">
                    {len(df.select_dtypes(include=np.number).columns)}
                </div>
                <div class="metric-small">columns</div>
            </div>
            """)

        st.html("""
        <div class="section-title">
            Data Preview
        </div>
        """)

        st.dataframe(
            df.head(100),
            use_container_width=True,
            height=360
        )

# ============================================================
# DATA UPLOAD
# ============================================================

elif page == "Data Upload":

    st.html("""
    <div class="section-title">
        Data Upload
    </div>

    <div class="section-subtitle">
        Upload or replace your CSV dataset.
    </div>
    """)

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.session_state.df = df
            st.session_state.ai_result = None
            st.session_state.report_text = None

            st.success(
                f"Successfully loaded {uploaded_file.name}"
            )

        except Exception as error:

            st.error(
                f"Could not read the CSV: {error}"
            )

    if st.session_state.df is not None:

        df = st.session_state.df

        st.html("""
        <div class="section-title">
            Current Dataset
        </div>
        """)

        st.write(
            f"**{len(df):,} rows × {len(df.columns):,} columns**"
        )

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

elif page == "Statistical Analysis":

    if st.session_state.df is None:

        st.warning(
            "Please upload a CSV dataset first."
        )

    else:

        df = st.session_state.df

        st.html("""
        <div class="section-title">
            Statistical Analysis
        </div>

        <div class="section-subtitle">
            Descriptive statistics for numerical variables.
        </div>
        """)

        numeric_df = df.select_dtypes(
            include=np.number
        )

        if numeric_df.empty:

            st.info(
                "No numerical columns were detected."
            )

        else:

            summary = numeric_df.describe().T

            summary["median"] = numeric_df.median()

            summary = summary[
                [
                    "count",
                    "mean",
                    "median",
                    "std",
                    "min",
                    "25%",
                    "50%",
                    "75%",
                    "max"
                ]
            ]

            st.dataframe(
                summary,
                use_container_width=True
            )

            st.html("""
            <div class="section-title">
                Column Statistics
            </div>
            """)

            selected_column = st.selectbox(
                "Select numerical column",
                numeric_df.columns.tolist()
            )

            col_data = numeric_df[selected_column]

            a, b, c, d = st.columns(4)

            with a:
                st.metric(
                    "Mean",
                    f"{col_data.mean():,.3f}"
                )

            with b:
                st.metric(
                    "Median",
                    f"{col_data.median():,.3f}"
                )

            with c:
                st.metric(
                    "Minimum",
                    f"{col_data.min():,.3f}"
                )

            with d:
                st.metric(
                    "Maximum",
                    f"{col_data.max():,.3f}"
                )

# ============================================================
# VISUAL ANALYTICS
# ============================================================

elif page == "Visual Analytics":

    if st.session_state.df is None:

        st.warning(
            "Please upload a CSV dataset first."
        )

    else:

        df = st.session_state.df

        st.html("""
        <div class="section-title">
            Visual Analytics
        </div>

        <div class="section-subtitle">
            Explore trends and relationships using interactive charts.
        </div>
        """)

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if not numeric_columns:

            st.info(
                "No numerical columns available for visualization."
            )

        else:

            st.html("""
            <div class="section-title">
                Average Values
            </div>
            """)

            averages = df[numeric_columns].mean()

            st.bar_chart(
                averages,
                use_container_width=True
            )

            st.html("""
            <div class="section-title">
                Numerical Trend
            </div>
            """)

            selected_column = st.selectbox(
                "Select a column for trend analysis",
                numeric_columns,
                key="visual_column"
            )

            st.line_chart(
                df[selected_column],
                use_container_width=True
            )

            if len(numeric_columns) >= 2:

                st.html("""
                <div class="section-title">
                    Correlation Matrix
                </div>
                """)

                correlation = df[numeric_columns].corr()

                st.dataframe(
                    correlation.round(3),
                    use_container_width=True
                )

# ============================================================
# AI INSIGHTS
# ============================================================

elif page == "AI Insights":

    if st.session_state.df is None:

        st.warning(
            "Please upload a CSV dataset first."
        )

    else:

        df = st.session_state.df

        st.html("""
        <div class="section-title">
            Local AI Insights
        </div>

        <div class="section-subtitle">
            Analyze your dataset locally using Qwen 2.5 7B through Ollama.
        </div>
        """)

        st.html("""
        <div class="ai-card">

            <div class="ai-title">
                ✦ Qwen 2.5 • Local Intelligence
            </div>

            <div class="ai-subtitle">
                Your data is analyzed by the local AI model.
                No external AI API is required.
            </div>

        </div>
        """)

        st.write("")

        anomalies = detect_anomalies(df)
        trends = detect_trends(df)

        a, b = st.columns(2)

        with a:

            st.html("""
            <div class="metric-card">
                <div class="metric-label">
                    Potential Outliers
                </div>
            """)

            total_anomalies = sum(
                anomalies.values()
            )

            st.html(f"""
                <div class="metric-value">
                    {total_anomalies:,}
                </div>
                <div class="metric-small">
                    detected using IQR
                </div>
            </div>
            """)

        with b:

            increasing = len(
                [
                    x for x in trends
                    if x["direction"] == "Increasing"
                ]
            )

            decreasing = len(
                [
                    x for x in trends
                    if x["direction"] == "Decreasing"
                ]
            )

            st.html(f"""
            <div class="metric-card">
                <div class="metric-label">
                    Trend Signals
                </div>

                <div class="metric-value">
                    {len(trends):,}
                </div>

                <div class="metric-small">
                    ↑ {increasing} increasing
                    &nbsp; • &nbsp;
                    ↓ {decreasing} decreasing
                </div>
            </div>
            """)

        st.write("")

        if st.button(
            "🤖 Generate AI Analysis",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Qwen 2.5 7B is analyzing your dataset..."
            ):

                prompt = build_ai_prompt(df)

                result = ask_qwen(prompt)

                st.session_state.ai_result = result

                if result.startswith("ERROR:"):

                    st.session_state.ai_error = result

                else:

                    st.session_state.ai_error = None

        if st.session_state.ai_result:

            st.divider()

            if st.session_state.ai_error:

                st.error(
                    st.session_state.ai_result
                )

            else:

                st.markdown(
                    st.session_state.ai_result
                )

                st.success(
                    "AI analysis generated successfully using local Qwen 2.5 7B."
                )

        else:

            st.html("""
            <div class="info-box" style="margin-top:20px;">
                Click <b>Generate AI Analysis</b> to let
                Qwen 2.5 7B analyze the uploaded dataset.
            </div>
            """)

# ============================================================
# REPORT GENERATOR
# ============================================================

elif page == "Report Generator":

    if st.session_state.df is None:

        st.warning(
            "Please upload a CSV dataset first."
        )

    else:

        df = st.session_state.df

        st.html("""
        <div class="section-title">
            Report Generator
        </div>

        <div class="section-subtitle">
            Generate a professional data analysis report.
        </div>
        """)

        if st.button(
            "📄 Generate Complete Report",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Generating AI-powered report..."
            ):

                basic_report = generate_basic_report(df)

                prompt = build_ai_prompt(df)

                ai_result = ask_qwen(prompt)

                complete_report = (
                    basic_report
                    + "\n\n"
                    + "=" * 70
                    + "\n"
                    + "AI-GENERATED ANALYSIS"
                    + "\n"
                    + "=" * 70
                    + "\n\n"
                    + ai_result
                )

                st.session_state.report_text = complete_report

        if st.session_state.report_text:

            st.success(
                "Complete report generated successfully."
            )

            st.text_area(
                "Report Preview",
                st.session_state.report_text,
                height=650
            )

            report_bytes = (
                st.session_state.report_text
                .encode("utf-8")
            )

            st.download_button(
                label="⬇️ Download Complete Report",
                data=report_bytes,
                file_name=(
                    "AI_Data_Analysis_Report_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S")
                    + ".txt"
                ),
                mime="text/plain",
                use_container_width=True
            )

        else:

            st.html("""
            <div class="info-box">
                Generate a report to combine statistical analysis,
                anomaly detection, trend detection and local AI insights.
            </div>
            """)

# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">
    <div>
        ✦ AI Data Analysis & Report Generator
    </div>

    <div style="margin-top:6px;">
        Python • Pandas • Streamlit • Ollama • Qwen 2.5 7B
    </div>
</div>
""")