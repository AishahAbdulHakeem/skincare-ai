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

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .stApp {
            background-color: #f9f7f4;
        }
        [data-testid="stSidebar"] {
            background-color: #f0ece6;
        }
        .stButton > button {
            background-color: #c4a882;
            color: white;
            border: none;
            border-radius: 6px;
        }
        hr {
            border-color: #e0d8cf;
        }
        h1, h2, h3 {
            color: #4a4a4a;
            font-weight: 500;
        }
        .stInfo {
            background-color: #f0ece6;
            color: #4a4a4a;
        }
    </style>
""", unsafe_allow_html=True)

# ── Load all artifacts ─────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("results/models/pair_scores.pkl", "rb") as f:
        pair_scores = pickle.load(f)
    with open("results/models/product_ingredients.pkl", "rb") as f:
        product_ingredients = pickle.load(f)
    with open("results/models/product_ratings.pkl", "rb") as f:
        product_ratings = pickle.load(f)
    with open("results/models/product_review_counts.pkl", "rb") as f:
        product_review_counts = pickle.load(f)
    with open("results/models/product_names.pkl", "rb") as f:
        product_names = pickle.load(f)
    with open("results/models/brand_names.pkl", "rb") as f:
        brand_names = pickle.load(f)
    with open("results/models/review_summaries.pkl", "rb") as f:
        review_summaries = pickle.load(f)
    with open("data/processed/top_ingredients.pkl", "rb") as f:
        top_ingredients = pickle.load(f)
    with open("results/models/best_model_final.pkl", "rb") as f:
        model = pickle.load(f)
    with open("results/models/scaler_final.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("results/models/top_ingredients_list.pkl", "rb") as f:
        top_ingredients_list = pickle.load(f)

    return (pair_scores, product_ingredients, product_ratings,
            product_review_counts, product_names, brand_names,
            review_summaries, top_ingredients, model, scaler,
            top_ingredients_list)

(pair_scores, product_ingredients, product_ratings,
 product_review_counts, product_names, brand_names,
 review_summaries, top_ingredients, model, scaler,
 top_ingredients_list) = load_artifacts()

# ── Load sentiment lookup for accurate predictions ─────────────────────────────
@st.cache_data
def load_sentiment_lookup():
    skincare = pd.read_pickle("data/processed/skincare_final.pkl")
    return skincare.set_index('product_id')[
        ['mean_sentiment', 'pct_positive', 'review_count']
    ].to_dict(orient='index')

sentiment_lookup = load_sentiment_lookup()

# ── Product search options ─────────────────────────────────────────────────────
@st.cache_data
def get_product_options():
    options = {
        pid: f"{brand_names.get(pid, '')} — {product_names.get(pid, pid)}"
        for pid in product_names
    }
    return dict(sorted(options.items(), key=lambda x: x[1]))

product_options  = get_product_options()
name_to_id       = {v: k for k, v in product_options.items()}

# ── Compatibility function ─────────────────────────────────────────────────────
def check_compatibility(routine_ingredients, product_id):
    if product_id not in product_ingredients:
        return None, "Product not found", []

    new_ings    = product_ingredients[product_id]
    routine_top = [i for i in routine_ingredients if i in top_ingredients]
    new_top     = [i for i in new_ings if i in top_ingredients]

    if not routine_top or not new_top:
        return None, "Not enough recognized ingredients", []

    cross_pairs = []
    for r_ing in routine_top:
        for n_ing in new_top:
            pair = tuple(sorted([r_ing, n_ing]))
            cross_pairs.append(pair)

    scores    = [(pair, pair_scores.get(pair, 1.0)) for pair in cross_pairs]
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

# ── ML prediction function ─────────────────────────────────────────────────────
def predict_rating(product_id):
    if product_id not in product_ingredients:
        return None

    ing_list   = product_ingredients[product_id]

    # 224 ingredient binary features
    ing_vector = [1 if ing in ing_list else 0 for ing in top_ingredients_list]

    # 3 sentiment features — use actual stored values, fall back to dataset means
    sent = sentiment_lookup.get(product_id, {
        'mean_sentiment': 0.65,
        'pct_positive':   0.86,
        'review_count':   0
    })
    sentiment_scaled = scaler.transform([[
        sent['mean_sentiment'],
        sent['pct_positive'],
        sent['review_count']
    ]])

    # Full 227-feature vector — same as Notebook 03
    features  = np.hstack([ing_vector, sentiment_scaled[0]])
    predicted = model.predict([features])[0]

    return round(float(np.clip(predicted, 1.0, 5.0)), 2)

# ── Product card renderer ──────────────────────────────────────────────────────
def render_product_card(product_id, routine_ingredients):
    name    = product_names.get(product_id, product_id)
    brand   = brand_names.get(product_id, "")
    summary = review_summaries.get(product_id, "No reviews available.")

    # Ratings
    actual_rating = product_ratings.get(product_id, None)
    review_count  = product_review_counts.get(product_id, 0)
    predicted     = predict_rating(product_id)

    # Compatibility
    score, label, flagged = check_compatibility(routine_ingredients, product_id)

    # Risk colors
    color_map = {
        "Low Risk":      "#6b8f71",
        "Moderate Risk": "#c4a882",
        "High Risk":     "#a67c7c"
    }
    emoji_map = {
        "Low Risk":      "✅",
        "Moderate Risk": "⚠️",
        "High Risk":     "❌"
    }
    color = color_map.get(label, "#888")
    emoji = emoji_map.get(label, "")

    # ── Card ──
    st.markdown(f"### 🧴 {brand}")
    st.markdown(f"**{name}**")
    st.divider()

    # Actual rating
    if actual_rating:
        st.markdown(
            f"⭐ **Actual Rating:** {actual_rating:.2f} / 5 "
            f"&nbsp;&nbsp; *({int(review_count):,} reviews)*"
        )
    else:
        st.markdown("⭐ **Actual Rating:** Not available")

    # Predicted rating from ML model
    if predicted and actual_rating:
        delta = predicted - actual_rating
        if delta > 0.2:
            note = " *(ingredient profile suggests higher potential)*"
        elif delta < -0.2:
            note = " *(ingredient profile suggests lower potential)*"
        else:
            note = " *(aligns with actual rating)*"
        st.markdown(f"🤖 **Predicted Rating:** {predicted:.2f} / 5{note}")
    elif predicted:
        st.markdown(f"🤖 **Predicted Rating:** {predicted:.2f} / 5")

    st.divider()

    # Compatibility badge
    st.markdown(
        f"<div style='background-color:{color};padding:10px;border-radius:8px;"
        f"color:white;font-weight:bold;text-align:center;margin:10px 0;"
        f"letter-spacing:0.5px;font-size:0.95em'>"
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

    st.divider()

    # Review summary
    st.markdown("**📝 What people say:**")
    st.info(summary)

# ── App layout ─────────────────────────────────────────────────────────────────
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

# ── Results ────────────────────────────────────────────────────────────────────
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
