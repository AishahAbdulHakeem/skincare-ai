import streamlit as st
import pandas as pd
import numpy as np
import pickle
import itertools

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skincare Compatibility Checker",
    page_icon="🧴",
    layout="wide"
)

# ── Load all model artifacts ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("models/pair_scores.pkl", "rb") as f:
        pair_scores = pickle.load(f)
    with open("models/product_ingredients.pkl", "rb") as f:
        product_ingredients = pickle.load(f)
    with open("models/product_ratings.pkl", "rb") as f:
        product_ratings = pickle.load(f)
    with open("models/product_review_counts.pkl", "rb") as f:
        product_review_counts = pickle.load(f)
    with open("models/product_names.pkl", "rb") as f:
        product_names = pickle.load(f)
    with open("models/brand_names.pkl", "rb") as f:
        brand_names = pickle.load(f)
    with open("models/review_summaries.pkl", "rb") as f:
        review_summaries = pickle.load(f)
    with open("data/processed/top_ingredients.pkl", "rb") as f:
        top_ingredients = pickle.load(f)

    return (pair_scores, product_ingredients, product_ratings,
            product_review_counts, product_names, brand_names,
            review_summaries, top_ingredients)

(pair_scores, product_ingredients, product_ratings,
 product_review_counts, product_names, brand_names,
 review_summaries, top_ingredients) = load_artifacts()

# ── Helper: build product search list ──────────────────────────────────────────
@st.cache_data
def get_product_options():
    options = {
        pid: f"{brand_names.get(pid, '')} — {product_names.get(pid, pid)}"
        for pid in product_names
    }
    # Sort alphabetically by display name
    return dict(sorted(options.items(), key=lambda x: x[1]))

product_options = get_product_options()
# Reverse lookup: display name → product_id
name_to_id = {v: k for k, v in product_options.items()}

# ── Helper: compatibility check ─────────────────────────────────────────────────
def check_compatibility(routine_ingredients, product_id):
    if product_id not in product_ingredients:
        return None, "Product not found", []

    new_ings = product_ingredients[product_id]

    routine_top = [i for i in routine_ingredients if i in top_ingredients]
    new_top     = [i for i in new_ings if i in top_ingredients]

    if not routine_top or not new_top:
        return None, "Not enough recognized ingredients", []

    cross_pairs = []
    for r_ing in routine_top:
        for n_ing in new_top:
            pair = tuple(sorted([r_ing, n_ing]))
            cross_pairs.append(pair)

    scores = [(pair, pair_scores.get(pair, 1.0)) for pair in cross_pairs]
    avg_score = np.mean([s for _, s in scores])

    flagged = sorted(
        [(p, s) for p, s in scores if s < 0.85],
        key=lambda x: x[1]
    )[:5]

    if avg_score >= 1.2:
        label = "Low Risk"
    elif avg_score >= 0.9:
        label = "Moderate Risk"
    else:
        label = "High Risk"

    return round(avg_score, 4), label, flagged

# ── Helper: render a product card ──────────────────────────────────────────────
def render_product_card(product_id, routine_ingredients):
    name    = product_names.get(product_id, product_id)
    brand   = brand_names.get(product_id, "")
    rating  = product_ratings.get(product_id, None)
    reviews = product_review_counts.get(product_id, 0)
    summary = review_summaries.get(product_id, "No reviews available.")

    score, label, flagged = check_compatibility(routine_ingredients, product_id)

    # Risk color
    color_map = {
        "Low Risk":      "#2ecc71",
        "Moderate Risk": "#f39c12",
        "High Risk":     "#e74c3c"
    }
    emoji_map = {
        "Low Risk":      "✅",
        "Moderate Risk": "⚠️",
        "High Risk":     "❌"
    }
    color = color_map.get(label, "#888")
    emoji = emoji_map.get(label, "")

    # Card
    st.markdown(f"### 🧴 {brand}")
    st.markdown(f"**{name}**")
    st.divider()

    # Rating
    if rating:
        st.markdown(f"⭐ **Rating:** {rating:.2f} / 5 &nbsp;&nbsp; "
                    f"*({int(reviews):,} reviews)*")
    else:
        st.markdown("⭐ **Rating:** Not available")

    # Compatibility badge
    st.markdown(
        f"<div style='background-color:{color};padding:10px;border-radius:8px;"
        f"color:white;font-weight:bold;text-align:center;margin:10px 0'>"
        f"{emoji} {label} &nbsp; (score: {score})</div>",
        unsafe_allow_html=True
    )

    # Flagged pairs
    if flagged:
        with st.expander("⚠️ Flagged ingredient combinations"):
            for pair, s in flagged:
                st.markdown(f"- `{pair[0]}` + `{pair[1]}`")
    else:
        st.success("No problematic ingredient combinations found.")

    # Review summary
    st.markdown("**📝 What people say:**")
    st.info(summary)

# ── App layout ──────────────────────────────────────────────────────────────────
st.title("🧴 Skincare Compatibility Checker")
st.markdown(
    "Select your current routine ingredients, then search for up to "
    "3 Sephora products to compare their compatibility side by side."
)
st.divider()

# ── Sidebar: routine ingredients ───────────────────────────────────────────────
with st.sidebar:
    st.header("Your Current Routine")
    st.markdown("Select all ingredients already in your skincare routine:")

    routine_ingredients = st.multiselect(
        label="Routine ingredients",
        options=sorted(list(top_ingredients)),
        placeholder="Search and select ingredients...",
        label_visibility="collapsed"
    )

    if routine_ingredients:
        st.success(f"{len(routine_ingredients)} ingredient(s) selected")
    else:
        st.warning("Select at least one ingredient to check compatibility.")

# ── Main: product search ────────────────────────────────────────────────────────
st.subheader("Search Products to Compare")

col1, col2, col3 = st.columns(3)

with col1:
    p1_name = st.selectbox(
        "Product 1",
        options=[""] + list(name_to_id.keys()),
        index=0,
        key="p1"
    )

with col2:
    p2_name = st.selectbox(
        "Product 2",
        options=[""] + list(name_to_id.keys()),
        index=0,
        key="p2"
    )

with col3:
    p3_name = st.selectbox(
        "Product 3",
        options=[""] + list(name_to_id.keys()),
        index=0,
        key="p3"
    )

st.divider()

# ── Results ─────────────────────────────────────────────────────────────────────
selected = [p for p in [p1_name, p2_name, p3_name] if p != ""]

if not routine_ingredients:
    st.info("👈 Start by selecting your routine ingredients in the sidebar.")

elif not selected:
    st.info("Search for at least one product above to see the compatibility report.")

else:
    st.subheader("Product Reports")
    cols = st.columns(len(selected))

    for col, product_name_str in zip(cols, selected):
        product_id = name_to_id[product_name_str]
        with col:
            render_product_card(product_id, routine_ingredients)