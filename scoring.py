from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf

# ─────────────────────────  ANSWER KEYS  ─────────────────────────
PRE_CONFIRM_SETS = [
    {"Card showing 4", "Card showing E"},
    {"Person drinking a beer", "Person is 25 years old"},
    {"Card showing A", "Card showing B", "Card showing 2"},
    {"Person is 16", "Person has no voter I.D"},
    {"Card showing 6", "Card showing Z", "Card showing K"},
    {"Card showing red", "Card showing 8"},
    {"Wearing glasses", "Reading a book"},
    {"Car has a D sticker", "Car is from Germany"},
]
POST_CONFIRM_SETS = [
    {"Bird", "Can fly"},
    {"Studies hard", "Passed the exam"},
    {"Teacher", "Works at a school"},
    {"Apple", "Red"},
    {"Motorcycle", "Two wheels"},
    {"Married", "Wears ring"},
    {"Wearing blue shirt", "Wearing brown shoes"},
    {"Under 30", "Green shirt"},
]
N_ITEMS = 8

# ─────────────────────  SCORING UTILITIES  ─────────────────────

def _is_conf(resp: str, key: set[str]) -> int:
    return int(str(resp).strip() in key)

def score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("timestamp")], errors="ignore")
    pre_cols, post_cols = df.columns[4:12], df.columns[12:20]
    if len(pre_cols) != N_ITEMS:
        raise ValueError("Need 8 PRE columns at positions E–L.")

    for c, k in zip(pre_cols, PRE_CONFIRM_SETS):
        df[f"score_{c}"] = df[c].apply(_is_conf, key=k)
    for c, k in zip(post_cols, POST_CONFIRM_SETS):
        df[f"score_{c}"] = df[c].apply(_is_conf, key=k)

    df["pre"]  = df[[f"score_{c}" for c in pre_cols]].mean(axis=1)
    df["post"] = df[[f"score_{c}" for c in post_cols]].mean(axis=1)
    df["delta"] = df["post"] - df["pre"]
    return df

# ─────────────────────────  STREAMLIT UI  ───────────────────────

st.set_page_config(page_title="Confirmation Bias Analyzer", layout="wide")

st.title("📊 Confirmation Bias Analyzer")

file = st.file_uploader("Upload Google‑Form CSV", type="csv")
if file is None:
    st.stop()

try:
    df = score(pd.read_csv(file))
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

if "Game/Video" not in df.columns:
    st.error("Column 'Game/Video' is required to separate conditions.")
    st.stop()

# Sidebar – choose grouping variable and whether to show regression
with st.sidebar:
    grouping = st.selectbox("Group by", options=[c for c in ["Gender", "Field"] if c in df.columns], index=0)
    show_reg = st.checkbox("Show Age regression section", value=True)

# ─────────────────────  BAR CHARTS: Pre and Post  ───────────────────

for metric in ("pre", "post"):
    st.subheader(f"{metric.capitalize()} confirmation bias by {grouping} and Game/Video")
    pivot = df.pivot_table(values=metric, index=grouping, columns="Game/Video", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Confirmation‑bias score")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# ――― Distribution histogram ―――

st.subheader("Distribution of Pre vs Post (all participants)")
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.hist(df["pre"], bins=8, alpha=0.6, label="Pre")
ax2.hist(df["post"], bins=8, alpha=0.6, label="Post")
ax2.legend()
ax2.set_xlabel("Score")
ax2.set_ylabel("Participants")
st.pyplot(fig2)

# ――― Age regression (both Pre and Post automatically) ―――
if show_reg and "Age" in df.columns and pd.api.types.is_numeric_dtype(df["Age"]):
    st.header("Age regression by Game/Video (Pre and Post)")
    for dep in ("pre", "post"):
        st.subheader(f"{dep.capitalize()} vs Age")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        for cond, sub in df.groupby("Game/Video"):
            if sub["Age"].nunique() < 2:
                continue
            model = smf.ols(f"{dep} ~ Age", data=sub).fit()
            xs = np.linspace(sub["Age"].min(), sub["Age"].max(), 100)
            ax3.scatter(sub["Age"], sub[dep], alpha=0.5, label=f"{cond} data")
            ax3.plot(xs, model.params[0] + model.params[1]*xs, label=f"{cond} fit")
        ax3.set_xlabel("Age")
        ax3.set_ylabel(dep.capitalize())
        ax3.legend()
        st.pyplot(fig3)
else:
    st.info("Numeric 'Age' column not found or regression disabled.")

# Collapsible About Section
with st.expander("ℹ️ About this app"):
    st.markdown("""
    Created by **Omar Kamel** for a **bachelor thesis**.

    This tool calculates a confirmation-bias score based on participant answers before and after an intervention (e.g. video or game). It visualizes the average scores and optionally performs regression by age.
    """)

st.markdown("---")
st.caption("Each section shows both Pre and Post data grouped by your selected variable.")
