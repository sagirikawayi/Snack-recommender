import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import time
import os

# 页面配置
st.set_page_config(page_title="AI Smart Food Recommender", page_icon="🧠", layout="centered")
st.title("🧠 AI Smart Food Recommender")
st.subheader("(Powered by KNN Algorithm)")
st.markdown(
    "This system connects to a real food database and uses **K-Nearest Neighbors (KNN)** data mining to find the perfect meals or food items for you.")
st.markdown("---")


# ==========================================
# 1. 真实数据集加载与数据清洗 (Data Mining 核心步骤)
# ==========================================
@st.cache_data  # 使用缓存加速网页刷新
def load_and_mine_data():
    try:
        # 【数据挖掘步骤 1：读取真实数据】
        df_nutri = pd.read_csv('healthy_foods_database.csv')
        df_allergen = pd.read_csv('foods_allergens.csv')

        # 【数据挖掘步骤 2：特征清洗与提取】
        df_nutri = df_nutri[['food_name', 'calories', 'protein_g']].dropna()

        # 数据过滤 (Data Filtering)
        df_main = df_nutri.sample(20, random_state=42).reset_index(drop=True)

        # 模拟数据融合 (Data Integration)
        np.random.seed(42)
        df_main['is_nut_free'] = np.random.choice([0, 1], size=len(df_main), p=[0.2, 0.8])
        df_main['is_dairy_free'] = np.random.choice([0, 1], size=len(df_main), p=[0.3, 0.7])

        # 【数据挖掘步骤 3：人为注入“异常值 (Outlier / Dirty Data)”用于教学】
        bug_food = pd.DataFrame({
            'food_name': ['⚠️ Chocolate Peanut Cookie (Data Error Case)'],
            'calories': [450.0],
            'protein_g': [12.0],
            'is_nut_free': [1],  # 致命漏洞：明明叫花生，却标记为无坚果(1)
            'is_dairy_free': [0]
        })
        df_final = pd.concat([df_main, bug_food], ignore_index=True)

        return df_final

    except FileNotFoundError:
        st.error("🚨 Dataset Not Found! Please ensure `healthy_foods_database.csv` is in the correct folder.")
        return pd.DataFrame()


# 加载数据
df = load_and_mine_data()

if df.empty:
    st.stop()

# ==========================================
# 2. 特征工程与 KNN 算法
# ==========================================
features = ['calories', 'protein_g', 'is_nut_free', 'is_dairy_free']
X = df[features].copy()

# 数据归一化 (Min-Max Scaling)
scaler = MinMaxScaler()
X[['calories', 'protein_g']] = scaler.fit_transform(X[['calories', 'protein_g']])

# ==========================================
# 3. 前端界面与用户输入 (Fixed Synchronization)
# ==========================================
st.subheader("📋 Step 1: Set Your Dietary Goals")


# --- 1. 定义同步回调函数 ---
def update_slider_cal():
    st.session_state.cal_slider = st.session_state.cal_input


def update_input_cal():
    st.session_state.cal_input = st.session_state.cal_slider


def update_slider_pro():
    st.session_state.pro_slider = st.session_state.pro_input


def update_input_pro():
    st.session_state.pro_input = st.session_state.pro_slider


# --- 2. 初始化 Session State ---
if 'cal_slider' not in st.session_state:
    st.session_state.cal_slider = 150
if 'pro_slider' not in st.session_state:
    st.session_state.pro_slider = 10

col1, col2 = st.columns(2)

with col1:
    require_nut_free = st.checkbox("🥜 Must be Nut-Free")
    require_dairy_free = st.checkbox("🥛 Must be Dairy-Free")
    st.markdown("<br>", unsafe_allow_html=True)
    show_math = st.toggle("🔍 Activate 'Algorithm X-Ray Vision'")

with col2:
    # --- 能量控制 (Calories) ---
    st.write("**Ideal Calories (kcal)**")
    c1, c2 = st.columns([3, 1.2])
    with c1:
        # 当滑动条改变时，触发 update_input_cal
        target_calories = st.slider(
            "Cal Slider", int(df['calories'].min()), int(df['calories'].max()),
            key="cal_slider", on_change=update_input_cal, label_visibility="collapsed"
        )
    with c2:
        # 当输入框改变时，触发 update_slider_cal
        st.number_input(
            "Cal Input", int(df['calories'].min()), int(df['calories'].max()),
            key="cal_input", on_change=update_slider_cal, label_visibility="collapsed",
            value=st.session_state.cal_slider  # 初始值取自 slider 的状态
        )

    # 欧盟标准卡片
    current_cal = st.session_state.cal_slider
    if current_cal <= 120:
        st.success("🍏 **Low Energy** (EU Light Standard)")
    elif current_cal <= 250:
        st.info("🥪 **Standard Energy** (Balanced)")
    else:
        st.warning("⚡ **High Energy** (Energy-Dense)")

    st.markdown("---")

    # --- 蛋白质控制 (Protein) ---
    st.write("**Ideal Protein (g)**")
    p1, p2 = st.columns([3, 1.2])
    with p1:
        target_protein = st.slider(
            "Pro Slider", int(df['protein_g'].min()), int(df['protein_g'].max()),
            key="pro_slider", on_change=update_input_pro, label_visibility="collapsed"
        )
    with p2:
        st.number_input(
            "Pro Input", int(df['protein_g'].min()), int(df['protein_g'].max()),
            key="pro_input", on_change=update_slider_pro, label_visibility="collapsed",
            value=st.session_state.pro_slider
        )

    # 欧盟标准卡片
    current_pro = st.session_state.pro_slider
    if current_pro < 5:
        st.warning("📉 **Low Protein**")
    elif current_pro <= 12:
        st.info("🥚 **Source of Protein** (EU Standard)")
    else:
        st.success("💪 **High Protein** (EU High-Pro Standard)")

# 注意：最后算法部分的代码，请统一使用 st.session_state.cal_slider 和 st.session_state.pro_slider

# ==========================================
# 4. 执行推荐
# ==========================================
if st.button("🚀 Run KNN Algorithm", type="primary"):
    with st.spinner('⏳ Mining database and calculating Euclidean distances...'):
        time.sleep(1)

    search_df = X.copy()
    if require_nut_free:
        search_df = search_df[search_df['is_nut_free'] == 1]
    if require_dairy_free:
        search_df = search_df[search_df['is_dairy_free'] == 1]

    if len(search_df) == 0:
        st.error("❌ No food items match your specific requirements in this dataset.")
    else:
        valid_indices = search_df.index
        valid_X = search_df[['calories', 'protein_g']].values

        user_num_target = scaler.transform([[st.session_state.cal_slider, st.session_state.pro_slider]])

        k = min(3, len(valid_X))
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(valid_X)
        distances, indices = knn.kneighbors(user_num_target)

        st.success(f"✅ Based on real data, here are your Top {k} Matches:")

        for i in range(k):
            original_idx = valid_indices[indices[0][i]]
            row = df.iloc[original_idx]
            dist = distances[0][i]
            # 计算匹配度得分
            match_score = max(0, (1 - dist) * 100)

            with st.container():
                st.markdown(f"### 🏆 Rank {i + 1}: {row['food_name']}")
                st.write(f"**Energy:** {row['calories']} kcal | **Protein:** {row['protein_g']} g")

                if show_math:
                    st.info(f"**[Algorithm Backend Data]**\n\n"
                            f"- Dataset Row Index: `{original_idx}`\n"
                            f"- Euclidean Distance: `{dist:.4f}`\n"
                            f"- Confidence Score: `{match_score:.1f}%`")
                else:
                    st.progress(int(match_score))
                st.markdown("---")
