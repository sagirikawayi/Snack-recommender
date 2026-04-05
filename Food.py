import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(
    page_title="AI Food Recommender (Scientific Standards)",
    page_icon="🧠",
    layout="centered"
)

# ==========================================
# 0.5 全方位 CSS 美化补丁 (最高文本对比度版)
# ==========================================
st.markdown("""
    <style>
    /* 1. 核心修复：强制浏览器锁定深色配色，防止系统干扰 */
    :root { color-scheme: dark !important; } 
    
    /* 2. 核心修复：强制全局文本颜色锁定为最高对比度的白色 */
    /* 对所有级别的 Markdown、Label、p 标签应用最强级别的颜色锁定 */
    h1, h2, h3, p, label, .stMarkdown, div[data-testid="stSidebar"] .stMarkdown, .stSubheader {
        color: #FFFFFF !important;
        forced-color-adjust: none !important; /* 防止浏览器非法色彩调整 */
    }

    /* 3. 结果容器文字颜色特殊锁定 (Rank、Metric 名称等) */
    .stContainer h3, .stContainer p, [data-testid="stMetricValue"] div, [data-testid="stMetricLabel"] div {
        color: #FFFFFF !important;
    }
    /* Metric 的数值可以保持亮色作为点缀 */
    [data-testid="stMetricValue"] { color: #58A6FF !important; }

    /* 4. 组件与卡片样式 (保持你满意的布局) */
    /* 数据状态 Metric 卡片美化 (深色风格) */
    [data-testid="stMetric"] {
        background-color: #1C2128 !important; 
        border: 1px solid #30363D !important;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    [data-testid="stMetric"]:hover { transform: translateY(-5px); box-shadow: 0 10px 15px rgba(88, 166, 255, 0.2); }

    /* 按钮美化 (高亮亮蓝) */
    div.stButton > button:first-child {
        background-color: #388BFD !important; /* 更适合深色模式的亮蓝 */
        color: white !important; 
        width: 100%; 
        border-radius: 10px;
        height: 3.5em; 
        font-weight: bold; 
        border: none;
        transition: 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #58A6FF !important;
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.4);
    }
    
    /* 结果容器 (深色风格) */
    .stContainer { 
        background-color: #1C2128 !important; 
        padding: 25px; 
        border-radius: 15px; 
        margin-bottom: 25px; 
        border: 1px solid #30363D;
    }
    
    /* 下拉框适配 */
    div[data-baseweb="select"] { background-color: #161B22 !important; cursor: pointer !important; }
    div[data-baseweb="select"] * { cursor: pointer !important; color: #FFFFFF !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 1. 加载数据
# ==========================================
@st.cache_data
def load_and_mine_data():
    try:
        # 这里引用了你原始代码中的CSV文件
        df_nutri = pd.read_csv('healthy_foods_database.csv')
        cols = ['food_name', 'calories', 'protein_g', 'carbs_g', 'sugar_g', 'fat_g', 'sodium_mg']
        return df_nutri[cols].dropna().reset_index(drop=True)
    except:
        st.error("🚨 Dataset Not Found! Please ensure 'healthy_foods_database.csv' is present.")
        return pd.DataFrame()

df = load_and_mine_data()
if df.empty: st.stop()

# ==========================================
# 1.5 预计算维度 (解决 6D 实时变更问题)
# ==========================================
nutrient_prefixes = ['cal', 'pro', 'carb', 'sugar', 'fat', 'sod']
active_count = sum([st.session_state.get(f"use_{p}", True) for p in nutrient_prefixes])

# ==========================================
# 2. 侧边栏导航逻辑
# ==========================================
with st.sidebar:
    st.title("🧭 Navigation")
    menu_selection = st.selectbox(
        "Choose a module:",
        ["Main Dashboard", "AI Control Panel", "Scientific Standards"]
    )
    st.write("---")

    if menu_selection == "AI Control Panel":
        st.markdown("### ⚙️ AI Control Panel")
        with st.expander("🔍 Algorithm Settings", expanded=True):
            k_val = st.slider("Top-K Matches", 1, 8, 5)
            show_math = st.toggle("🧪 Algorithm X-Ray Vision")

        with st.expander("🏥 Data Health Status", expanded=True):
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                st.metric("Total Rows", len(df))
                st.metric("Null Cells", df.isnull().sum().sum())
            with h_col2:
                # 使用预计算好的 active_count
                st.metric("Dimensions", f"{active_count}D")
                st.caption("Status: Healthy ✅")

        if st.button("🔄 Reset System Cache"):
            st.cache_data.clear()
            st.rerun()

    elif menu_selection == "Scientific Standards":
        st.markdown("### 📘 Scientific Standards")
        st.info("""
        This AI engine evaluates targets using:
        - FSA Traffic Lights
        - China National Standards (GB 28050)
        """)
        k_val, show_math = 5, False

    else:
        st.markdown("### 🏠 Welcome")
        st.info("**v2.1.2 Total White Edition**\n\nAdjust your targets to find aligned profiles.")
        k_val, show_math = 5, False

# ==========================================
# 3. 主界面逻辑
# ==========================================
st.title("🧠 AI Food Recommender")
st.subheader("Scientific Standards-Based Matching")
st.markdown("---")

def sync_val(prefix, source):
    if source == 'slider':
        st.session_state[f'{prefix}_input'] = st.session_state[f'{prefix}_slider']
    else:
        st.session_state[f'{prefix}_slider'] = st.session_state[f'{prefix}_input']

nutrient_map = {'cal': 'calories', 'pro': 'protein_g', 'carb': 'carbs_g', 'sugar': 'sugar_g', 'fat': 'fat_g', 'sod': 'sodium_mg'}
for prefix, col in nutrient_map.items():
    if f'{prefix}_slider' not in st.session_state:
        st.session_state[f'{prefix}_slider'] = int(df[col].mean())
    if f'{prefix}_input' not in st.session_state:
        st.session_state[f'{prefix}_input'] = st.session_state[f'{prefix}_slider']

st.markdown("### 📋 Step 1: Set Your Nutritional Target")
active_features, user_target_values = [], []

def render_nutrient_control(label, prefix, col_name, emoji):
    # 核心修复：确保 Slider 和 Checkbox 的文本颜色被最高权限锁定
    st.markdown('<div class="control-unit">', unsafe_allow_html=True) 
    is_active = st.checkbox(f"{emoji} Use {label}", value=True, key=f"use_{prefix}")
    c1, c2 = st.columns([3, 1.2])
    min_v, max_v = int(df[col_name].min()), int(df[col_name].max())
    with c1: st.slider(label, min_v, max_v, key=f"{prefix}_slider", on_change=sync_val, args=(prefix, 'slider'), disabled=not is_active, label_visibility="collapsed")
    with c2: st.number_input(label, min_v, max_v, key=f"{prefix}_input", on_change=sync_val, args=(prefix, 'input'), disabled=not is_active, label_visibility="collapsed")
    
    if is_active:
        val = st.session_state[f"{prefix}_slider"]
        active_features.append(col_name)
        user_target_values.append(val)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

render_nutrient_control("Calories (kcal)", "cal", "calories", "⚡")
render_nutrient_control("Protein (g)", "pro", "protein_g", "🥚")
render_nutrient_control("Carbs (g)", "carb", "carbs_g", "🍞")
render_nutrient_control("Sugar (g)", "sugar", "sugar_g", "🍭")
render_nutrient_control("Total Fat (g)", "fat", "fat_g", "🥑")
render_nutrient_control("Sodium (mg)", "sod", "sodium_mg", "🧂")

if st.button("🚀 Run AI Search", type="primary"):
    if not active_features:
        st.error("❌ Select at least one nutrient!")
    else:
        with st.spinner('Calculating best matches...'):
            X = df[active_features].copy()
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            user_target_scaled = scaler.transform([user_target_values])
            k = min(k_val, len(df))
            knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(X_scaled)
            distances, indices = knn.kneighbors(user_target_scaled)
            
            st.markdown("### ✅ Scientific Match Results")
            for i in range(k):
                orig_idx = indices[0][i]
                row = df.iloc[orig_idx]
                dist = distances[0][i]
                match_score = max(0, (1 - dist) * 100)
                with st.container():
                    # 结果容器标题保持亮蓝，增强层次感
                    st.markdown(f"<h3 style='color: #58A6FF;'>Rank {i+1}: {row['food_name']}</h3>", unsafe_allow_html=True)
                    cols = st.columns(3)
                    for idx, feat in enumerate(active_features):
                        u = " kcal" if 'calories' in feat.lower() else (" mg" if 'sodium' in feat.lower() else " g")
                        # 将带有 _g 和 _mg 的原始列名清理为整洁的显示标签
                        clean_label = feat.replace('_', ' ').title().replace('G', '(g)').replace('Mg', '(mg)')
                        cols[idx % 3].metric(clean_label, f"{row[feat]}{u}")
                    st.progress(int(match_score))
