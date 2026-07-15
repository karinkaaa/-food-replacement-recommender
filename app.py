"""
Запуск:
    pip install streamlit
    python -m streamlit run app.py
"""

import pickle
 
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
 
st.set_page_config(page_title="Замена блюд", page_icon="", layout="centered")
 
TARGETS = ["target_weightloss", "target_gainmass", "target_balance", "target_sugar"]
LABELS = ["Похудение", "Набор массы", "Баланс/ЗОЖ", "Контроль сахара"]
LABEL_TO_TARGET = dict(zip(LABELS, TARGETS))
 
NUTRITION_FEATURES = [
    "Calories (kcal)", "Protein (g)", "Carbohydrates (g)",
    "Fat (g)", "Fiber (g)", "Sugars (g)", "Sodium (mg)",
]
NUTRIENT_LABELS = {
    "Calories (kcal)": "Калории, ккал",
    "Protein (g)": "Белки, г",
    "Carbohydrates (g)": "Углеводы, г",
    "Fat (g)": "Жиры, г",
    "Fiber (g)": "Клетчатка, г",
    "Sugars (g)": "Сахар, г",
    "Sodium (mg)": "Натрий, мг",
}
 
 
 
@st.cache_data
def load_data():
    return pd.read_csv("data/russian_food_labeled.csv")
 
 
@st.cache_resource
def load_best_models():
    with open("data/best_models.pkl", "rb") as f:
        return pickle.load(f)
 
 
df = load_data()
best_models = load_best_models()
 
 
def find_replacement_knn(dish, category, target_col):
    category_mask = (df["Category"] == category) & (df["Food_Item"] != dish["Food_Item"])
    df_cat = df[category_mask].reset_index(drop=True)
    if len(df_cat) < 2:
        return None
 
    scaler = StandardScaler()
    X_cat = scaler.fit_transform(df_cat[NUTRITION_FEATURES].astype(float))
    X_dish = scaler.transform(
        pd.DataFrame([dish[NUTRITION_FEATURES].astype(float)])[NUTRITION_FEATURES]
    )
 
    knn = NearestNeighbors(n_neighbors=len(df_cat), metric="euclidean")
    knn.fit(X_cat)
    _, indices = knn.kneighbors(X_dish)
 
    for idx in indices[0]:
        neighbor = df_cat.iloc[idx]
        if int(neighbor[target_col]) == 0:
            return neighbor
    return None
 
 
def recommend(food_item, goal):
    target_col = LABEL_TO_TARGET[goal]
    dish_rows = df[df["Food_Item"] == food_item]
    if len(dish_rows) == 0:
        return {"status": "error"}
 
    dish = dish_rows.iloc[0]
    category = dish["Category"]
    needs_replacement = int(dish[target_col]) == 1
 
    if not needs_replacement:
        return {"status": "ok", "food_item": food_item, "category": category, "dish": dish}
 
    replacement = find_replacement_knn(dish, category, target_col)
    if replacement is None:
        return {"status": "no_replacement", "food_item": food_item, "category": category}
 
    return {
        "status": "replaced", "food_item": food_item, "category": category,
        "dish": dish, "replacement": replacement,
    }
 
 
# Интерфейс 
 
st.title("Рекомендации по замене блюд")
st.caption(
    f"ML-проект: классификация + kNN · {len(df)} блюд, {df['Category'].nunique()} категорий · значения на 100г"
)
 

 
food_item = st.selectbox("Блюдо", options=sorted(df["Food_Item"].unique()))
goal = st.selectbox("Цель", options=LABELS)
 
check = st.button("Проверить", type="primary", use_container_width=True)
 
def fmt(v):
    """Форматирует число без лишних нулей: 245.0 -> '245', 11.5 -> '11.5'."""
    v = float(v)
    return f"{v:g}"
 
 
if check:
    result = recommend(food_item, goal)
    st.divider()
 
    if result["status"] == "error":
        st.error("Блюдо не найдено")
 
    elif result["status"] == "ok":
        st.success(f" «{result['food_item']}» подходит для цели «{goal}» — замена не нужна")
        table = pd.DataFrame({
            NUTRIENT_LABELS[f]: [fmt(result["dish"][f])] for f in NUTRITION_FEATURES
        }, index=["на 100г"]).T
        st.table(table)
 
    elif result["status"] == "no_replacement":
        st.warning(
            f" «{result['food_item']}» не подходит для цели «{goal}», но замены "
            f"в категории «{result['category']}» не нашлось"
        )
 
    elif result["status"] == "replaced":
        st.error(f" «{result['food_item']}» не подходит для цели «{goal}»")
        st.markdown(f"**Рекомендуем заменить на:** {result['replacement']['Food_Item']}")
 
        table = pd.DataFrame({
            NUTRIENT_LABELS[f]: [fmt(result["dish"][f]), fmt(result["replacement"][f])]
            for f in NUTRITION_FEATURES
        }, index=[result["food_item"], result["replacement"]["Food_Item"]]).T
        st.table(table)