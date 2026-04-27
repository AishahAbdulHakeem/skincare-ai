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

# ── Load sentiment lookup ──────────────────────────────────────────────────────
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

product_options = get_product_options()
name_to_id      = {v: k for k, v in product_options.items()}

# ── ML predicted rating ────────────────────────────────────────────────────────
def predict_rating(product_id):
    """
    Runs the trained Gradient Boosting model on the product's
    227-feature vector (224 ingredient features + 3 sentiment features)
    to produce a predicted rating.
    """
    if product_id not in product_ingredients:
        return None

    ing_list   = product_ingredients[product_id]

    # 224 binary ingredient features
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

# ── Compatibility check ────────────────────────────────────────────────────────
def check_compatibility(routine_ingredients, product_id):
    """
    Scores cross-pairs between routine ingredients and new product ingredients
    using the co-occurrence lookup table built in Notebook 04.
    Returns raw ratio score, risk label, and flagged pairs.
    """
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

# ── Personalized routine score ─────────────────────────────────────────────────
def get_routine_score(product_id, routine_ingredients):
    """
    Combines ML predicted rating (60%) + normalized ingredient
    compatibility ratio (40%) into a single personalized 1-5 score.

    This score changes based on the user's specific routine ingredients,
    making it a personalized output the user cannot get from reviews alone.
    """
    predicted = predict_rating(product_id)
    if predicted is None:
        return None, None, []

    compat_score, label, flagged = check_compatibility(
        routine_ingredients, product_id
    )
    if compat_score is None:
        return None, label, flagged

    # Normalize compatibility ratio → 1-5 scale
    # ratio = 1.0 (neutral)       → 3.0
    # ratio >= 2.0 (compatible)   → 5.0
    # ratio <= 0.5 (incompatible) → 1.0
    compat_normalized = np.clip(1 + (compat_score - 0.5) * (4 / 1.5), 1.0, 5.0)

    # Weighted combination
    final_score = (predicted * 0.6) + (compat_normalized * 0.4)
    final_score = round(float(np.clip(final_score, 1.0, 5.0)), 2)

    return final_score, label, flagged

# ── App explanation generator ──────────────────────────────────────────────────
def generate_app_explanation(routine_score, label, flagged, actual_rating):
    """
    Generates a plain English explanation of the routine score,
    combining risk label, flagged pairs, and score context.
    """
    if routine_score is None:
        return "Not enough ingredient data to generate an explanation."

    lines = []

    # Score context
    if routine_score >= 4.0:
        lines.append(
            f"This product scores {routine_score} / 5 for your routine — "
            f"a strong match based on its ingredient profile and overall quality."
        )
    elif routine_score >= 3.0:
        lines.append(
            f"This product scores {routine_score} / 5 for your routine — "
            f"a reasonable match, though some caution is advised."
        )
    else:
        lines.append(
            f"This product scores {routine_score} / 5 for your routine — "
            f"its ingredient profile may not complement your current routine well."
        )

    # Risk label context
    if label == "Low Risk":
        lines.append(
            "Its ingredients tend to co-occur with high-rated products "
            "alongside your routine — generally a safe addition."
        )
    elif label == "Moderate Risk":
        lines.append(
            "Some ingredient combinations show mixed signals with your routine. "
            "A patch test is recommended before full use."
        )
    elif label == "High Risk":
        lines.append(
            "Several ingredient combinations tend to appear in lower-rated products "
            "alongside your routine. Use with caution."
        )

    # Flagged pairs context
    if flagged:
        pair_strs = " and ".join(
            [f"{p[0]} + {p[1]}" for p, _ in flagged[:2]]
        )
        lines.append(
            f"{len(flagged)} potentially problematic combination(s) detected "
            f"({pair_strs}). These pairs frequently appear in lower-rated products."
        )

    # Actual vs routine score gap
    if actual_rating and routine_score:
        gap = routine_score - actual_rating
        if gap > 0.3:
            lines.append(
                "Notably, this product scores higher for your specific routine "
                "than its general rating suggests — your ingredients may bring "
                "out its best qualities."
            )
        elif gap < -0.3:
            lines.append(
                "This product's general rating is higher than its score for your "
                "routine — it may perform better for different ingredient combinations."
            )

    return " ".join(lines)

# ── Product card ───────────────────────────────────────────────────────────────
def render_product_card(product_id, routine_ingredients):
    name    = product_names.get(product_id, product_id)
    brand   = brand_names.get(product_id, "")
    summary = review_summaries.get(product_id, "No reviews available.")

    actual_rating = product_ratings.get(product_id, None)
    review_count  = product_review_counts.get(product_id, 0)

    # Get personalized routine score + risk label + flagged pairs
    routine_score, label, flagged = get_routine_score(
        product_id, routine_ingredients
    )

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

    # ── Card header ──
    st.markdown(f"### 🧴 {brand}")
    st.markdown(f"**{name}**")
    st.divider()

    # ── Actual rating ──
    if actual_rating:
        st.markdown(
            f"⭐ **Actual Rating:** {actual_rating:.2f} / 5 "
            f"&nbsp;&nbsp; *({int(review_count):,} reviews)*"
        )
    else:
        st.markdown("⭐ **Actual Rating:** Not available")

    # ── Personalized routine score ──
    if routine_score:
        st.markdown(
            f"<div style='background-color:#f0ece6;padding:12px;"
            f"border-radius:8px;border-left:4px solid #c4a882;margin:10px 0'>"
            f"<span style='color:#4a4a4a;font-weight:bold;font-size:1.05em'>"
            f"🔬 Your Routine Score: {routine_score} / 5</span><br>"
            f"<span style='color:#888;font-size:0.85em'>"
            f"Personalized based on your ingredients + product quality"
            f"</span></div>",
            unsafe_allow_html=True
        )

    st.divider()

    # ── Risk badge ──
    st.markdown(
        f"<div style='background-color:{color};padding:10px;border-radius:8px;"
        f"color:white;font-weight:bold;text-align:center;margin:10px 0;"
        f"letter-spacing:0.5px;font-size:0.95em'>"
        f"{emoji} {label}</div>",
        unsafe_allow_html=True
    )

    # ── Flagged pairs ──
    if flagged:
        with st.expander("⚠️ Flagged ingredient combinations"):
            for pair, s in flagged:
                st.markdown(f"- `{pair[0]}` + `{pair[1]}`")
    else:
        st.success("No problematic ingredient combinations found.")

    st.divider()

    # ── What people say ──
    st.markdown("**📝 What people say:**")
    st.info(summary)

    # ── What our app says ──
    explanation = generate_app_explanation(
        routine_score, label, flagged, actual_rating
    )
    st.markdown("**🤖 What our app says:**")
    st.markdown(
        f"<div style='background-color:#f0ece6;padding:12px;"
        f"border-radius:8px;border-left:4px solid #6b8f71;margin:10px 0;"
        f"color:#4a4a4a;font-size:0.95em;line-height:1.6'>"
        f"{explanation}"
        f"</div>",
        unsafe_allow_html=True
    )

# ── App layout ─────────────────────────────────────────────────────────────────
st.title("🧴 Skincare Compatibility Checker")
st.markdown(
    "Select your current routine ingredients, then search for up to "
    "3 Sephora products to compare their compatibility side by side."
)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
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

# ── Product search ─────────────────────────────────────────────────────────────
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