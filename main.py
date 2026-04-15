import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. LOAD ASSETS
df = pd.read_pickle('models/phone_data.pkl')
knn = joblib.load('models/knn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(layout='wide', page_title="Phone Finder")

# 2. CUSTOM CSS (Cleaned up emojis and refined card design)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        background-color: #ff4b4b; 
        color: white; 
        font-weight: bold;
        border: none;
        padding: 10px;
    }
     .phone-card {
            padding: 20px; 
            border-radius: 12px; 
            background-color: rgba(72, 66, 82, 0.200);
            box-shadow: 0px 4px 6px rgba(0,0,0,0.05); 
            margin-bottom: 20px;
            border-left: 3px solid #ff4b4b;
            # border : 0.5px solid #997977;

    }
    </style>
    """, unsafe_allow_html=True)

st.title("Smartphone Recommender System")
st.write("Set your preferences to find the best match.")

# 3. SIDE-BY-SIDE LAYOUT
# On desktop: col1 is left, col2 is right. On mobile: col1 is top, col2 is bottom.
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Preferences")
    
    st.write("**Budget Range (₹)**")
    c1, c2 = st.columns(2)
    with c1:
        price_from = st.number_input("From", min_value=0, value=10000, step=1000)
    with c2:
        price_to = st.number_input("To", min_value=0, value=50000, step=1000)

    if price_to < price_from:
        st.error("Error: 'To' price must be greater than 'From'")
        valid_budget = False
    else:
        valid_budget = True

    num_rec = st.number_input("Number of Matches", 1, 100, 5)
    
    st.divider()
    st.write("**Priority Scores**")
    gaming = st.slider("Gaming", 0.0, 1.0, 0.5)
    camera = st.slider("Camera", 0.0, 1.0, 0.5)
    battery = st.slider("Battery", 0.0, 1.0, 0.5)
    
    search_clicked = st.button("Find My Match")

with col2:
    st.subheader("Results")
    
    if search_clicked:
        if not valid_budget:
            st.warning("Please correct the budget range on the left.")
        else:
            total = gaming + camera + battery

            if total == 0:
                st.error("Please move at least one slider.")
            else:
                # Prediction Logic
                user_vector = np.array([[gaming, camera, battery]])
                user_scaled = scaler.transform(user_vector)
                distances, indices = knn.kneighbors(user_scaled, n_neighbors=len(df))

                # Filtering
                recommendations = df.iloc[indices[0]].copy()
                final_list = recommendations[
                    (recommendations['price'] >= price_from) & 
                    (recommendations['price'] <= price_to)
                ].head(num_rec)

                if final_list.empty:
                    st.warning(f"No phones found between ₹{price_from:,} and ₹{price_to:,}.")
                else:
                    for _, phone in final_list.iterrows():
                        st.markdown(f"""
                            <div class="phone-card">
                                <div class="card-title">{phone['model']}</div>
                                <div class="card-price">₹{phone['price']:,}</div>
                                <div class="card-text"><b>RAM:</b> {int(phone['ram_capacity'])}GB | <b>Processor:</b> {phone['processor_brand']}</div>
                                <div class="card-text"><b>Battery:</b> {int(phone['battery_capacity'])} mAh</div>
                            </div>
                        """, unsafe_allow_html=True)