"""
Module 9: Machine Learning Model Performance & Multi-Model Evaluation Dashboard
Location: ./pages/9_Model_Performance.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.ml_engine import load_trained_artifacts

st.set_page_config(page_title="Model Performance | GTI-ARP", page_icon="📊", layout="wide")

# Custom Dark Command-Center Theme Styling
st.markdown("""
<style>
    .model-header {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #58A6FF;
        margin-bottom: 2px;
    }
    .hud-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(13, 17, 23, 0.98) 100%);
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }
    .hud-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8B949E;
    }
    .hud-value {
        font-size: 22px;
        font-weight: 800;
        color: #F0F6FC;
        margin-top: 2px;
    }
    .hud-sub {
        font-size: 11px;
        font-weight: 600;
        margin-top: 2px;
    }
    .summary-card {
        background: linear-gradient(135deg, #161B22 0%, #0D1117 100%);
        border-left: 4px solid #58A6FF;
        border-radius: 6px;
        padding: 14px 18px;
        border-top: 1px solid #30363D;
        border-right: 1px solid #30363D;
        border-bottom: 1px solid #30363D;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

model, preprocessor, label_encoder, metadata = load_trained_artifacts()

# 1. Page Title & Subtitle
st.markdown("<div class='model-header'>📊 Machine Learning Model Performance & Multi-Model Suite</div>", unsafe_allow_html=True)
st.caption("Empirical multi-model benchmarking, temporal out-of-time holdout validation, and confusion matrix analytics.")

if metadata is None:
    st.error("⚠️ Model metadata artifact not detected. Run `python train_pipeline.py` first.")
    st.stop()

benchmark_list = metadata.get("benchmark_comparison", [])
df_bench = pd.DataFrame(benchmark_list)

if df_bench.empty:
    st.error("Benchmark comparison data unavailable in metadata.")
    st.stop()

# Identify Best Model by Macro F1
best_model_name = metadata.get("best_model_name", df_bench.iloc[0]["Model"])
available_models = df_bench["Model"].tolist()
best_idx = available_models.index(best_model_name) if best_model_name in available_models else 0

# 2. Model Selector
sel_col1, _ = st.columns([1.8, 3.2])
with sel_col1:
    selected_model_name = st.selectbox(
        "Select Model Architecture for Inspection",
        available_models,
        index=best_idx,
        help="Dynamically updates the performance KPIs, confusion matrix, and generalization diagnostics."
    )

# Retrieve metrics for the selected model
models_eval_dict = metadata.get("models_evaluation", {})
if selected_model_name in models_eval_dict:
    selected_model_data = models_eval_dict[selected_model_name]
    val_acc = selected_model_data.get("validation_accuracy", 0.0)
    test_mets = selected_model_data.get("test_metrics", {})
    cm_raw = np.array(selected_model_data.get("confusion_matrix", []))
else:
    # Fallback to general metadata if individual model dict is not yet populated
    bench_row = df_bench[df_bench["Model"] == selected_model_name].iloc[0]
    val_acc = float(bench_row["Accuracy (%)"])
    test_mets = metadata.get("best_model_metrics", {
        "accuracy": val_acc,
        "balanced_accuracy": float(bench_row["Balanced Accuracy (%)"]),
        "macro_f1": float(bench_row["Macro F1 (%)"]),
        "weighted_f1": float(bench_row["Weighted F1 (%)"])
    })
    cm_raw = np.array(metadata.get("confusion_matrix", []))

test_acc = float(test_mets.get("accuracy", 0.0))
bal_acc = float(test_mets.get("balanced_accuracy", 0.0))
macro_f1 = float(test_mets.get("macro_f1", 0.0))
weighted_f1 = float(test_mets.get("weighted_f1", 0.0))

# 3. Dynamic KPI Cards
st.markdown("<br>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Test Accuracy</div>
        <div class="hud-value" style="color:#58A6FF;">{test_acc:.2f}%</div>
        <div class="hud-sub" style="color:#8B949E;">Out-of-Time Holdout</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Balanced Accuracy</div>
        <div class="hud-value" style="color:#39D353;">{bal_acc:.2f}%</div>
        <div class="hud-sub" style="color:#39D353;">Macro Class Average</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Macro F1-Score</div>
        <div class="hud-value" style="color:#FFA657;">{macro_f1:.2f}%</div>
        <div class="hud-sub" style="color:#FFA657;">Unweighted Class Mean</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="hud-card">
        <div class="hud-title">Weighted F1-Score</div>
        <div class="hud-value" style="color:#D29922;">{weighted_f1:.2f}%</div>
        <div class="hud-sub" style="color:#8B949E;">Support-Adjusted Mean</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 4. Multi-Model Benchmark Comparison (Table + Chart)
st.subheader("1. Multi-Model Benchmark Comparison")

# Format benchmark table with visual highlight for best model
df_bench_display = df_bench.copy()
df_bench_display["Status"] = [
    "★ Optimal (Highest Macro F1)" if m == best_model_name else "Baseline / Alternative"
    for m in df_bench_display["Model"]
]
df_bench_display = df_bench_display[["Status", "Model", "Accuracy (%)", "Balanced Accuracy (%)", "Macro F1 (%)", "Weighted F1 (%)"]]

st.dataframe(
    df_bench_display,
    column_config={
        "Status": "Selection Status",
        "Model": "Model Architecture",
        "Accuracy (%)": st.column_config.NumberColumn("Accuracy", format="%.2f%%"),
        "Balanced Accuracy (%)": st.column_config.NumberColumn("Balanced Acc", format="%.2f%%"),
        "Macro F1 (%)": st.column_config.NumberColumn("Macro F1", format="%.2f%%"),
        "Weighted F1 (%)": st.column_config.NumberColumn("Weighted F1", format="%.2f%%")
    },
    hide_index=True,
    use_container_width=True
)

# Benchmark Horizontal Comparison Chart
df_melted = df_bench.melt(id_vars=["Model"], var_name="Evaluation Metric", value_name="Score (%)")

fig_bench = px.bar(
    df_melted,
    x="Score (%)",
    y="Model",
    color="Evaluation Metric",
    barmode="group",
    orientation="h",
    template="plotly_dark",
    color_discrete_sequence=["#58A6FF", "#39D353", "#FFA657", "#D29922"],
    title="Comparative Multi-Model Performance Across Evaluation Metrics"
)
fig_bench.update_layout(
    yaxis=dict(autorange="reversed"),
    height=320,
    margin=dict(l=10, r=10, t=35, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=10))
)
st.plotly_chart(fig_bench, use_container_width=True)

# 5. Model Selection Insight
gen_gap = round(abs(val_acc - test_acc), 2)
is_selected_best = (selected_model_name == best_model_name)

st.markdown(f"""
<div class="summary-card">
    <div style="font-size:14px;font-weight:800;color:#58A6FF;text-transform:uppercase;">💡 Model Selection & Generalization Insight</div>
    <div style="font-size:13px;color:#E6EDF3;margin-top:6px;line-height:1.5;">
        • <b>Active Inspection:</b> <code>{selected_model_name}</code> {'(★ Recommended Top Model)' if is_selected_best else ''}<br>
        • <b>Generalization Profile:</b> Validation Accuracy is <b>{val_acc:.2f}%</b> vs. Out-of-Time Test Accuracy of <b>{test_acc:.2f}%</b> (Generalization Gap: <b>{gen_gap} percentage points</b>).<br>
        • <b>Methodological Evaluation:</b> <b>{best_model_name}</b> was selected as the platform default because it maximizes <b>Macro F1 ({df_bench[df_bench['Model']==best_model_name]['Macro F1 (%)'].values[0]:.2f}%)</b>, ensuring high tactical discrimination across severely imbalanced minority attack categories.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Multiclass Confusion Matrix
st.subheader("2. Multiclass Confusion Matrix")

classes = metadata.get("target_classes", [])

if cm_raw.size > 0 and len(classes) == len(cm_raw):
    cm_col1, _ = st.columns([1.5, 3.5])
    with cm_col1:
        cm_mode = st.radio("Matrix Metric Format", ["Raw Counts", "Normalized Recall (%)"], horizontal=True)

    if "Normalized" in cm_mode:
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm_raw.astype(float), row_sums, where=row_sums != 0) * 100.0
        cm_display = np.round(cm_display, 1)
        color_label = "Class Recall (%)"
    else:
        cm_display = cm_raw
        color_label = "Incident Count"

    fig_cm = px.imshow(
        cm_display,
        x=classes,
        y=classes,
        text_auto=True,
        template="plotly_dark",
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Tactical Class", y="Actual Ground Truth Class", color=color_label),
        title=f"Confusion Matrix: {selected_model_name} ({cm_mode})"
    )
    fig_cm.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_tickangle=-30
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    st.caption("ℹ️ *Rows represent actual ground-truth classes; columns represent model predicted classes. Diagonal cells reflect correct classifications.*")
else:
    st.info("Confusion matrix data unavailable for the active model selection.")

st.markdown("<br>", unsafe_allow_html=True)

# 7. Validation Strategy & Sample Sizes
st.subheader("3. Validation Strategy & Sample Sizes")

train_n = metadata.get("train_samples", 0)
val_n = metadata.get("val_samples", 0)
test_n = metadata.get("test_samples", 0)
total_samples = train_n + val_n + test_n

st.markdown(f"""
- **Validation Protocol:** `{metadata.get('validation_strategy', 'Temporal Out-of-Time Holdout')}`
- **Chronological Partitions:** `{metadata.get('split_periods', {}).get('train', 'Base')} (Train)` → `{metadata.get('split_periods', {}).get('val', 'Middle')} (Val)` → `{metadata.get('split_periods', {}).get('test', 'Recent Holdout')} (Test)`
- **Sample Distribution:** Total Corpus: **{total_samples:,} samples** (Train: `{train_n:,}`, Validation: `{val_n:,}`, Test Holdout: `{test_n:,}`).
- **Academic Justification:** Temporal partitioning ensures model evaluation is strictly out-of-time, preventing lookahead data leakage in longitudinal incident streams.
""")