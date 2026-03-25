from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


st.set_page_config(
    page_title="Notebook Model Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_PATH = "task1_dataset.csv"
NUMERIC_IMPUTE_COLUMNS = ["income", "loan_amount", "credit_score", "annual_spend"]
OUTLIER_COLUMNS = ["income", "loan_amount", "credit_score", "annual_spend", "num_transactions"]
CATEGORICAL_COLUMNS = ["city", "employment_type", "loan_type"]
SCALED_COLUMNS = ["income", "loan_amount", "credit_score", "annual_spend", "num_transactions"]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(196, 224, 255, 0.9), transparent 28%),
                radial-gradient(circle at top right, rgba(255, 220, 188, 0.75), transparent 24%),
                linear-gradient(180deg, #f7f6f2 0%, #f2efe7 100%);
            color: #18212f;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(16, 27, 46, 0.96), rgba(24, 39, 61, 0.96));
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #f5f1e8;
        }

        .hero-card, .glass-card, .prediction-card {
            background: rgba(255, 255, 255, 0.72);
            backdrop-filter: blur(18px);
            border: 1px solid rgba(255, 255, 255, 0.7);
            border-radius: 24px;
            box-shadow: 0 24px 60px rgba(31, 41, 55, 0.12);
            animation: riseUp 0.7s ease both;
        }

        .hero-card {
            padding: 2rem 2.2rem;
            margin-bottom: 1.2rem;
            background:
                linear-gradient(135deg, rgba(10, 32, 56, 0.95), rgba(43, 76, 110, 0.88)),
                linear-gradient(120deg, rgba(255, 255, 255, 0.1), transparent);
            color: #f8f4ea;
            overflow: hidden;
            position: relative;
        }

        .hero-title {
            font-size: 3rem;
            line-height: 1;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.6rem;
        }

        .hero-copy {
            font-size: 1.02rem;
            max-width: 760px;
            color: rgba(248, 244, 234, 0.88);
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin: 1rem 0 0.6rem;
        }

        .metric-tile {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .metric-label {
            font-size: 0.82rem;
            color: rgba(248, 244, 234, 0.7);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-value {
            font-size: 1.6rem;
            font-weight: 800;
            margin-top: 0.3rem;
        }

        .glass-card {
            padding: 1.25rem;
            margin-bottom: 1rem;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            color: #152033;
        }

        .section-copy {
            color: #465264;
            margin-bottom: 1rem;
        }

        .prediction-card {
            padding: 1.4rem;
            background: linear-gradient(160deg, rgba(19, 41, 69, 0.96), rgba(34, 82, 94, 0.9));
            color: #f9f3e9;
        }

        .prediction-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.78rem;
            opacity: 0.72;
        }

        .prediction-value {
            font-size: 2.4rem;
            font-weight: 800;
            margin-top: 0.35rem;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.65rem;
            background: rgba(255, 255, 255, 0.45);
            border: 1px solid rgba(183, 132, 85, 0.18);
            padding: 0.45rem;
            border-radius: 999px;
            width: fit-content;
            box-shadow: 0 16px 34px rgba(21, 32, 51, 0.08);
        }

        div[data-testid="stTabs"] [data-baseweb="tab"] {
            height: auto;
            padding: 0.7rem 1.15rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.35);
            color: #7d5a3a;
            font-weight: 700;
            border: 1px solid transparent;
            transition: all 0.2s ease;
        }

        div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
            background: rgba(215, 184, 148, 0.28);
            color: #5d4028;
        }

        div[data-testid="stTabs"] [aria-selected="true"] {
            background: linear-gradient(135deg, #b78455, #d7b894);
            color: #152033;
            border-color: rgba(183, 132, 85, 0.4);
            box-shadow: 0 10px 24px rgba(183, 132, 85, 0.22);
        }

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background: transparent;
        }

        div[data-testid="stButton"] > button {
            border-radius: 999px;
            border: none;
            padding: 0.8rem 1.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #b78455, #d7b894);
            color: #152033;
            box-shadow: 0 14px 30px rgba(183, 132, 85, 0.28);
        }

        @keyframes riseUp {
            from {
                opacity: 0;
                transform: translateY(18px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def cap_series_with_bounds(series: pd.Series, lower_bound: float, upper_bound: float) -> pd.Series:
    return pd.Series(np.clip(series, lower_bound, upper_bound), index=series.index)


def prepare_training_data(df: pd.DataFrame):
    cleaned_df = df.copy()

    medians: dict[str, float] = {}
    for column in NUMERIC_IMPUTE_COLUMNS:
        median_value = float(cleaned_df[column].median())
        cleaned_df[column] = cleaned_df[column].fillna(median_value)
        medians[column] = median_value

    outlier_bounds: dict[str, tuple[float, float]] = {}
    for column in OUTLIER_COLUMNS:
        q1 = cleaned_df[column].quantile(0.25)
        q3 = cleaned_df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = float(q1 - 1.5 * iqr)
        upper_bound = float(q3 + 1.5 * iqr)
        cleaned_df[column] = cap_series_with_bounds(cleaned_df[column], lower_bound, upper_bound)
        outlier_bounds[column] = (lower_bound, upper_bound)

    y = cleaned_df["target"].copy()
    X = cleaned_df.drop(columns=["target"]).copy()

    X["date"] = pd.to_datetime(X["date"], errors="coerce")
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["dayofweek"] = X["date"].dt.dayofweek
    X = X.drop(columns=["date"])

    for column in CATEGORICAL_COLUMNS:
        X[column] = X[column].astype(str)

    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=int)

    fill_values = X.median(numeric_only=True).to_dict()
    X = X.fillna(fill_values)

    scaler = StandardScaler()
    X[SCALED_COLUMNS] = scaler.fit_transform(X[SCALED_COLUMNS])

    metadata = {
        "medians": medians,
        "outlier_bounds": outlier_bounds,
        "fill_values": fill_values,
        "model_columns": X.columns.tolist(),
        "scaler": scaler,
        "categorical_options": {
            column: sorted(cleaned_df[column].dropna().astype(str).unique().tolist())
            for column in CATEGORICAL_COLUMNS
        },
        "numeric_defaults": {
            column: float(cleaned_df[column].median())
            for column in ["age", "income", "loan_amount", "credit_score", "num_transactions", "annual_spend"]
        },
        "date_default": str(pd.to_datetime(df["date"], errors="coerce").dropna().median().date()),
    }
    return X, y, metadata


@st.cache_resource(show_spinner=False)
def train_notebook_model(file_path: str):
    df = load_data(file_path)
    X, y, metadata = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_dist = {
        "n_estimators": [200, 300, 500, 700],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    rf_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="r2",
        random_state=42,
        n_jobs=1,
    )
    rf_search.fit(X_train, y_train)

    model = rf_search.best_estimator_
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    comparison_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.001, random_state=42),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(
            random_state=42,
            n_estimators=300,
            min_samples_split=5,
            min_samples_leaf=4,
            max_features=None,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "ExtraTrees": ExtraTreesRegressor(random_state=42, n_estimators=300),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(),
    }

    comparison_results = []
    for name, estimator in comparison_models.items():
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        comparison_results.append(
            {
                "Model": name,
                "MSE": mean_squared_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
            }
        )

    comparison_df = pd.DataFrame(comparison_results).sort_values(
        by="R2", ascending=False
    ).reset_index(drop=True)

    return {
        "raw_df": df,
        "features": X,
        "model": model,
        "best_params": rf_search.best_params_,
        "mse": mse,
        "r2": r2,
        "comparison_df": comparison_df,
        "metadata": metadata,
    }


def build_prediction_frame(inputs: dict[str, object], metadata: dict[str, object]) -> pd.DataFrame:
    row = pd.DataFrame([inputs]).copy()

    for column, median_value in metadata["medians"].items():
        row[column] = row[column].fillna(median_value)

    for column, bounds in metadata["outlier_bounds"].items():
        lower_bound, upper_bound = bounds
        row[column] = cap_series_with_bounds(row[column].astype(float), lower_bound, upper_bound)

    row["date"] = pd.to_datetime(row["date"], errors="coerce")
    row["year"] = row["date"].dt.year
    row["month"] = row["date"].dt.month
    row["day"] = row["date"].dt.day
    row["dayofweek"] = row["date"].dt.dayofweek
    row = row.drop(columns=["date"])

    for column in CATEGORICAL_COLUMNS:
        row[column] = row[column].astype(str)

    row = pd.get_dummies(row, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=int)
    row = row.reindex(columns=metadata["model_columns"], fill_value=0)
    row = row.fillna(metadata["fill_values"])

    scaler: StandardScaler = metadata["scaler"]
    row[SCALED_COLUMNS] = scaler.transform(row[SCALED_COLUMNS])
    return row


def render_hero(results: dict[str, object]) -> None:
    st.markdown(
        f"""
        <section class="hero-card">
            <div class="hero-title">Notebook Model, now live in Streamlit</div>
            <div class="hero-copy">
                This app uses the modeling flow from <strong>task-1.ipynb</strong> with preprocessing,
                tuned Random Forest training, and a polished premium-style dashboard.
            </div>
            <div class="metric-grid">
                <div class="metric-tile">
                    <div class="metric-label">Rows In Dataset</div>
                    <div class="metric-value">{len(results["raw_df"]):,}</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-label">Notebook Model R²</div>
                    <div class="metric-value">{results["r2"]:.4f}</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-label">Notebook Model MSE</div>
                    <div class="metric-value">{results["mse"]:,.0f}</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def build_sidebar_inputs(metadata: dict[str, object]) -> dict[str, object]:
    st.sidebar.markdown("## Prediction Controls")
    st.sidebar.caption("These inputs use the same preprocessing as the notebook model.")

    numeric_defaults = metadata["numeric_defaults"]
    categorical_options = metadata["categorical_options"]
    raw_df = load_data(DATASET_PATH)

    return {
        "date": st.sidebar.date_input(
            "Date",
            value=pd.to_datetime(metadata["date_default"]).date(),
        ),
        "age": st.sidebar.slider(
            "Age",
            min_value=int(raw_df["age"].min()),
            max_value=int(raw_df["age"].max()),
            value=int(numeric_defaults["age"]),
        ),
        "income": st.sidebar.number_input("Income", min_value=0.0, value=float(numeric_defaults["income"]), step=1000.0),
        "loan_amount": st.sidebar.number_input("Loan Amount", min_value=0.0, value=float(numeric_defaults["loan_amount"]), step=5000.0),
        "credit_score": st.sidebar.number_input("Credit Score", min_value=0.0, value=float(numeric_defaults["credit_score"]), step=1.0),
        "num_transactions": st.sidebar.slider(
            "Transactions",
            min_value=int(raw_df["num_transactions"].min()),
            max_value=int(raw_df["num_transactions"].max()),
            value=int(numeric_defaults["num_transactions"]),
        ),
        "annual_spend": st.sidebar.number_input("Annual Spend", min_value=0.0, value=float(numeric_defaults["annual_spend"]), step=5000.0),
        "city": st.sidebar.selectbox("City", options=categorical_options["city"]),
        "employment_type": st.sidebar.selectbox("Employment Type", options=categorical_options["employment_type"]),
        "loan_type": st.sidebar.selectbox("Loan Type", options=categorical_options["loan_type"]),
    }


def render_prediction(results: dict[str, object], inputs: dict[str, object]) -> None:
    prediction_frame = build_prediction_frame(inputs, results["metadata"])
    prediction = float(results["model"].predict(prediction_frame)[0])

    st.markdown(
        f"""
        <section class="prediction-card">
            <div class="prediction-label">Predicted Target</div>
            <div class="prediction-value">{prediction:,.2f}</div>
            <div>The live prediction uses the tuned Random Forest model selected from the notebook workflow.</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Prediction payload"):
        st.dataframe(pd.DataFrame([inputs]), use_container_width=True)


def main() -> None:
    inject_styles()

    try:
        results = train_notebook_model(DATASET_PATH)
    except Exception as exc:
        st.error(f"Unable to train the notebook model: {exc}")
        st.stop()

    render_hero(results)
    inputs = build_sidebar_inputs(results["metadata"])

    tab_overview, tab_predict, tab_model = st.tabs(
        ["Overview", "Prediction Studio", "Model Details"]
    )

    with tab_overview:
        st.dataframe(results["raw_df"].head(12), use_container_width=True, height=320)
        st.dataframe(results["comparison_df"].head(8), use_container_width=True, height=320)
        st.line_chart(results["features"][SCALED_COLUMNS].head(40), height=260)

    with tab_predict:
        if st.button("Generate Prediction", type="primary"):
            render_prediction(results, inputs)
        else:
            st.info("Set your inputs in the sidebar, then click Generate Prediction.")

    with tab_model:
        st.json(results["best_params"])
        st.dataframe(results["comparison_df"], use_container_width=True, height=420)


if __name__ == "__main__":
    main()
