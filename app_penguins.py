import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Penguin Species Prediction",
    page_icon="🐧",
    layout="wide"
)

st.title("🐧 Penguin Species Prediction Dashboard")
st.caption(
    "Dashboard untuk EDA, pemeriksaan data, dan pengujian model "
    "machine learning penguin species."
)

# =========================================================
# SIDEBAR - UPLOAD
# =========================================================
st.sidebar.header("📂 Upload File")

uploaded_model = st.sidebar.file_uploader(
    "Upload model (.joblib)",
    type=["joblib"]
)

uploaded_data = st.sidebar.file_uploader(
    "Upload dataset penguins.csv",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Model yang digunakan sebaiknya merupakan pipeline lengkap "
    "yang sudah mencakup preprocessing."
)

# =========================================================
# LOAD MODEL
# =========================================================
model = None

if uploaded_model is not None:
    try:
        model = joblib.load(uploaded_model)
        st.sidebar.success("Model berhasil dimuat.")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model: {e}")

# =========================================================
# LOAD DATA
# =========================================================
df = None

if uploaded_data is not None:
    try:
        df = pd.read_csv(uploaded_data)
        st.sidebar.success(f"Dataset berhasil dimuat: {len(df)} baris")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca CSV: {e}")

# =========================================================
# CHECK DATA
# =========================================================
if df is None:
    st.warning(
        "Silakan upload `penguins.csv` melalui sidebar untuk melihat EDA."
    )
    st.stop()

# =========================================================
# DETECT COLUMNS
# =========================================================
target_col = "species"

required_features = [
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex"
]

missing_features = [
    col for col in required_features
    if col not in df.columns
]

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "🔍 Data Inspection",
    "📈 EDA",
    "⚖️ Balance",
    "🧩 Relationships",
    "🤖 Prediction",
    "📋 Model Evaluation"
])

# =========================================================
# TAB 1 - OVERVIEW
# =========================================================
with tab1:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Jumlah Baris", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])

    if target_col in df.columns:
        col3.metric(
            "Jumlah Species",
            df[target_col].nunique()
        )
    else:
        col3.metric("Jumlah Species", "N/A")

    col4.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.subheader("Preview Dataset")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Informasi Dataset")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Missing": df.isnull().sum().values,
        "Unique": df.nunique().values
    })

    st.dataframe(info_df, use_container_width=True)

# =========================================================
# TAB 2 - DATA INSPECTION
# =========================================================
with tab2:
    st.header("🔍 Data Inspection")

    st.subheader("Head")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Tail")
    st.dataframe(df.tail(), use_container_width=True)

    st.subheader("Statistical Description")

    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

    st.subheader("Duplicate Data")

    duplicate_count = df.duplicated().sum()

    if duplicate_count == 0:
        st.success("Tidak ditemukan duplicate row.")
    else:
        st.warning(
            f"Ditemukan {duplicate_count} duplicate row."
        )
        st.dataframe(
            df[df.duplicated(keep=False)],
            use_container_width=True
        )

    st.subheader("Missing Values")

    missing_summary = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing Percentage": (
            df.isnull().mean() * 100
        ).round(2)
    })

    st.dataframe(
        missing_summary,
        use_container_width=True
    )

    st.subheader("Full Rows Containing Missing Values")

    rows_with_missing = df[
        df.isnull().any(axis=1)
    ].copy()

    if len(rows_with_missing) == 0:
        st.success("Tidak ada baris yang mengandung missing value.")
    else:
        st.warning(
            f"Ditemukan {len(rows_with_missing)} baris "
            "yang memiliki missing value."
        )

        st.dataframe(
            rows_with_missing,
            use_container_width=True
        )

        missing_detail = rows_with_missing.copy()

        missing_detail["Missing Columns"] = (
            missing_detail.isnull()
            .apply(
                lambda row: list(row[row].index),
                axis=1
            )
        )

        st.subheader("Missing Columns per Row")

        st.dataframe(
            missing_detail,
            use_container_width=True
        )

    st.subheader("Completely Empty Rows")

    empty_rows = df[
        df.isnull().all(axis=1)
    ]

    if len(empty_rows) == 0:
        st.success("Tidak ada baris yang seluruh kolomnya kosong.")
    else:
        st.dataframe(
            empty_rows,
            use_container_width=True
        )

# =========================================================
# TAB 3 - EDA
# =========================================================
with tab3:
    st.header("📈 Exploratory Data Analysis")

    numeric_cols = [
        col for col in [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g"
        ]
        if col in df.columns
    ]

    categorical_cols = [
        col for col in ["island", "sex", "species"]
        if col in df.columns
    ]

    # Numerical distributions
    st.subheader("Numerical Feature Distributions")

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(
            data=df,
            x=col,
            kde=True,
            ax=ax
        )
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
        plt.close(fig)

    # Boxplots
    st.subheader("Boxplots")

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.boxplot(
            data=df,
            x=col,
            ax=ax
        )
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)
        plt.close(fig)

    # Categorical distributions
    st.subheader("Categorical Feature Distributions")

    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(
            data=df,
            x=col,
            ax=ax
        )
        ax.set_title(f"Distribution of {col}")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
        plt.close(fig)

    # Species vs sex
    if "species" in df.columns and "sex" in df.columns:
        st.subheader("Species vs Sex")

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(
            data=df,
            x="species",
            hue="sex",
            ax=ax
        )
        ax.set_title("Species vs Sex")
        st.pyplot(fig)
        plt.close(fig)

    # Species vs island
    if "species" in df.columns and "island" in df.columns:
        st.subheader("Species vs Island")

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(
            data=df,
            x="species",
            hue="island",
            ax=ax
        )
        ax.set_title("Species vs Island")
        st.pyplot(fig)
        plt.close(fig)

# =========================================================
# TAB 4 - BALANCE
# =========================================================
with tab4:
    st.header("⚖️ Class / Data Balance")

    if target_col not in df.columns:
        st.warning(
            "Kolom `species` tidak ditemukan."
        )
    else:
        class_counts = df[target_col].value_counts()
        class_percent = (
            df[target_col]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        balance_df = pd.DataFrame({
            "Count": class_counts,
            "Percentage": class_percent
        })

        st.dataframe(
            balance_df,
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.countplot(
            data=df,
            x=target_col,
            order=class_counts.index,
            ax=ax
        )

        ax.set_title("Species Class Balance")
        ax.set_xlabel("Species")
        ax.set_ylabel("Count")

        st.pyplot(fig)
        plt.close(fig)

# =========================================================
# TAB 5 - RELATIONSHIPS
# =========================================================
with tab5:
    st.header("🧩 Relationships and Patterns")

    numeric_cols = [
        col for col in [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g"
        ]
        if col in df.columns
    ]

    if "species" in df.columns and len(numeric_cols) > 0:

        # Numerical feature vs species
        st.subheader("Numerical Features vs Species")

        selected_feature = st.selectbox(
            "Select numerical feature",
            numeric_cols
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        sns.boxplot(
            data=df,
            x="species",
            y=selected_feature,
            ax=ax
        )

        ax.set_title(
            f"{selected_feature} vs Species"
        )

        st.pyplot(fig)
        plt.close(fig)

        # Pairplot
        st.subheader("Pairplot")

        pairplot_cols = numeric_cols + ["species"]

        pair_df = df[pairplot_cols].dropna()

        if len(pair_df) > 0:
            pair_fig = sns.pairplot(
                pair_df,
                hue="species"
            )

            st.pyplot(pair_fig.figure)
            plt.close(pair_fig.figure)

    # Correlation
    if len(numeric_cols) >= 2:
        st.subheader("Correlation Matrix")

        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )

        ax.set_title("Correlation Matrix")

        st.pyplot(fig)
        plt.close(fig)

# =========================================================
# TAB 6 - PREDICTION
# =========================================================
with tab6:
    st.header("🤖 Test Model")

    if model is None:
        st.warning(
            "Upload `best_penguin_model.joblib` "
            "melalui sidebar terlebih dahulu."
        )
    elif missing_features:
        st.error(
            "Kolom berikut tidak ditemukan pada dataset: "
            + ", ".join(missing_features)
        )
    else:
        st.subheader("Single Penguin Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            island = st.selectbox(
                "Island",
                sorted(
                    df["island"].dropna().unique()
                )
            )

            bill_length = st.number_input(
                "Bill Length (mm)",
                min_value=0.0,
                value=50.2,
                step=0.1
            )

        with col2:
            bill_depth = st.number_input(
                "Bill Depth (mm)",
                min_value=0.0,
                value=18.7,
                step=0.1
            )

            flipper_length = st.number_input(
                "Flipper Length (mm)",
                min_value=0,
                value=198,
                step=1
            )

        with col3:
            body_mass = st.number_input(
                "Body Mass (g)",
                min_value=0,
                value=3775,
                step=25
            )

            sex_options = sorted(
                df["sex"].dropna().unique()
            )

            sex = st.selectbox(
                "Sex",
                sex_options
            )

        input_data = pd.DataFrame({
            "island": [island],
            "bill_length_mm": [bill_length],
            "bill_depth_mm": [bill_depth],
            "flipper_length_mm": [flipper_length],
            "body_mass_g": [body_mass],
            "sex": [sex]
        })

        st.subheader("Input Data")
        st.dataframe(
            input_data,
            use_container_width=True
        )

        if st.button(
            "🔮 Predict Species",
            type="primary"
        ):
            try:
                prediction = model.predict(
                    input_data
                )[0]

                st.success(
                    f"Predicted Species: **{prediction}**"
                )

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(
                        input_data
                    )[0]

                    classes = model.classes_

                    probability_df = pd.DataFrame({
                        "Species": classes,
                        "Probability": probabilities
                    })

                    probability_df[
                        "Probability"
                    ] = (
                        probability_df["Probability"] * 100
                    ).round(2)

                    st.subheader("Prediction Probability")

                    st.dataframe(
                        probability_df,
                        use_container_width=True
                    )

                    fig, ax = plt.subplots(
                        figsize=(8, 4)
                    )

                    sns.barplot(
                        data=probability_df,
                        x="Species",
                        y="Probability",
                        ax=ax
                    )

                    ax.set_ylabel(
                        "Probability (%)"
                    )
                    ax.set_title(
                        "Prediction Probability"
                    )

                    st.pyplot(fig)
                    plt.close(fig)

            except Exception as e:
                st.error(
                    f"Prediction gagal: {e}"
                )

        # Batch prediction
        st.markdown("---")
        st.subheader("Batch Prediction dari CSV")

        batch_file = st.file_uploader(
            "Upload CSV untuk prediction",
            type=["csv"],
            key="batch_prediction"
        )

        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)

            st.write("Data yang di-upload:")
            st.dataframe(
                batch_df,
                use_container_width=True
            )

            missing_batch = [
                col for col in required_features
                if col not in batch_df.columns
            ]

            if missing_batch:
                st.error(
                    "Kolom yang dibutuhkan tidak ditemukan: "
                    + ", ".join(missing_batch)
                )
            else:
                batch_X = batch_df[
                    required_features
                ]

                batch_predictions = model.predict(
                    batch_X
                )

                result_df = batch_df.copy()

                result_df[
                    "Predicted Species"
                ] = batch_predictions

                st.subheader("Prediction Results")

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                csv_result = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Prediction Result",
                    data=csv_result,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )

# =========================================================
# TAB 7 - MODEL EVALUATION
# =========================================================
with tab7:
    st.header("📋 Model Evaluation")

    if model is None:
        st.info(
            "Upload model `.joblib` untuk menggunakan "
            "fitur evaluasi."
        )
    elif target_col not in df.columns:
        st.warning(
            "Dataset tidak memiliki kolom `species`, "
            "sehingga evaluasi tidak dapat dilakukan."
        )
    else:
        eval_X = df[
            required_features
        ].copy()

        eval_y = df[
            target_col
        ].copy()

        if eval_X.isnull().any().any():
            st.warning(
                "Dataset memiliki missing values. "
                "Pipeline model akan menangani missing values "
                "jika preprocessing-nya memang sudah disimpan "
                "di dalam pipeline."
            )

        try:
            eval_pred = model.predict(
                eval_X
            )

            accuracy = accuracy_score(
                eval_y,
                eval_pred
            )

            precision = precision_score(
                eval_y,
                eval_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                eval_y,
                eval_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                eval_y,
                eval_pred,
                average="weighted",
                zero_division=0
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Accuracy",
                f"{accuracy:.2%}"
            )

            c2.metric(
                "Precision",
                f"{precision:.2%}"
            )

            c3.metric(
                "Recall",
                f"{recall:.2%}"
            )

            c4.metric(
                "F1 Score",
                f"{f1:.2%}"
            )

            st.subheader("Confusion Matrix")

            labels = sorted(
                eval_y.dropna().unique()
            )

            cm = confusion_matrix(
                eval_y,
                eval_pred,
                labels=labels
            )

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax
            )

            ax.set_xlabel(
                "Predicted Species"
            )
            ax.set_ylabel(
                "Actual Species"
            )
            ax.set_title(
                "Confusion Matrix"
            )

            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Classification Report")

            report = classification_report(
                eval_y,
                eval_pred,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(
                report_df,
                use_container_width=True
            )

            st.subheader(
                "Prediction Results on Dataset"
            )

            evaluation_df = df.copy()

            evaluation_df[
                "Predicted Species"
            ] = eval_pred

            evaluation_df[
                "Correct"
            ] = (
                evaluation_df[target_col]
                == evaluation_df[
                    "Predicted Species"
                ]
            )

            st.dataframe(
                evaluation_df,
                use_container_width=True
            )

        except Exception as e:
            st.error(
                f"Evaluasi gagal: {e}"
            )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Penguin Species Prediction Dashboard | "
    "Predictive Analytics"
)
