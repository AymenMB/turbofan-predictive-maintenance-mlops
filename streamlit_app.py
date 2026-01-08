"""
AeroGuard AI - Turbofan Engine RUL Prediction
Professional Streamlit frontend for predictive maintenance
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ==================== Configuration ====================
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AeroGuard AI | RUL Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Professional Theme CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary: #0891b2;
        --primary-light: #22d3ee;
        --primary-dark: #0e7490;
        --secondary: #4f46e5;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-primary: #ffffff;
        --bg-secondary: #f0f9ff;
        --text-primary: #0f172a;
        --text-secondary: #475569;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .stApp {
        background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 100%);
        font-family: 'Inter', sans-serif !important;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #0891b2 0%, #0e7490 50%, #164e63 100%);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(8, 145, 178, 0.25);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #ffffff;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
    
    .hero-badge-dot {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0f172a;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }
    
    .status-safe {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981 !important;
        border-radius: 16px;
        padding: 1.5rem;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 2px solid #f59e0b !important;
        border-radius: 16px;
        padding: 1.5rem;
    }
    
    .status-critical {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444 !important;
        border-radius: 16px;
        padding: 1.5rem;
    }
    
    .status-online {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #ecfdf5;
        border: 1px solid #10b981;
        color: #059669;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-offline {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #fef2f2;
        border: 1px solid #ef4444;
        color: #dc2626;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.875rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(8, 145, 178, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(8, 145, 178, 0.5) !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Helper Functions ====================

def check_api_health():
    """Check API health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def make_prediction(features: dict):
    """Make a single RUL prediction"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=features, timeout=30)
        if response.status_code == 200:
            return True, response.json()
        return False, response.text
    except Exception as e:
        return False, str(e)

def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def get_status_class(rul: float):
    """Get status based on RUL"""
    if rul < 30:
        return "Critical", "status-critical", "#ef4444"
    elif rul < 80:
        return "Warning", "status-warning", "#f59e0b"
    else:
        return "Safe", "status-safe", "#10b981"

def create_rul_gauge(rul: float, status: str):
    """Create RUL gauge chart"""
    _, _, color = get_status_class(rul)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rul,
        number={'suffix': ' cycles', 'font': {'size': 36, 'color': '#0f172a'}},
        gauge={
            'axis': {'range': [0, 125], 'tickcolor': '#cbd5e1'},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': '#f1f5f9',
            'steps': [
                {'range': [0, 30], 'color': '#fee2e2'},
                {'range': [30, 80], 'color': '#fef3c7'},
                {'range': [80, 125], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.85,
                'value': rul
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
        margin=dict(l=30, r=30, t=40, b=20),
        font={'family': 'Inter'}
    )
    
    return fig

def create_status_chart(predictions_df):
    """Create status distribution chart"""
    status_counts = predictions_df['Status'].value_counts()
    colors = {'Safe': '#10b981', 'Warning': '#f59e0b', 'Critical': '#ef4444'}
    
    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.6,
        marker=dict(colors=[colors.get(s, '#4f46e5') for s in status_counts.index]),
        textposition='outside',
        textinfo='label+percent'
    )])
    
    fig.add_annotation(
        text=f"<b>{len(predictions_df)}</b><br><span style='font-size:11px'>Engines</span>",
        x=0.5, y=0.5, font=dict(size=20), showarrow=False
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False
    )
    
    return fig

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%); border-radius: 16px; display: flex; align-items: center; justify-content: center; margin: 0 auto; box-shadow: 0 8px 20px rgba(8, 145, 178, 0.3);">
            <span style="font-size: 1.8rem;">✈️</span>
        </div>
        <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-top: 1rem;">AeroGuard AI</div>
        <div style="font-size: 0.8rem; color: #64748b;">Predictive Maintenance</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🎯 Predict RUL", "📊 Batch Analysis", "📡 Dashboard"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # API Status
    is_healthy, _ = check_api_health()
    if is_healthy:
        st.markdown('<div class="status-online"><span style="color: #10b981;">●</span> API Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-offline"><span>●</span> API Offline</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("##### 📊 Model Info")
    st.markdown("**RMSE**: 18.64 cycles")
    st.markdown("**R²**: 0.79")
    st.markdown("**Dataset**: NASA FD001")
    
    st.markdown("---")
    st.caption("Powered by XGBoost • FastAPI")

# ==================== Hero Header ====================
st.markdown("""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="hero-title">✈️ AeroGuard AI</h1>
            <p class="hero-subtitle">Turbofan Engine Remaining Useful Life Prediction</p>
            <div class="hero-badge">
                <div class="hero-badge-dot"></div>
                Predictive Maintenance Active
            </div>
        </div>
        <div style="text-align: right; color: rgba(255,255,255,0.9);">
            <div style="font-weight: 700;">NASA C-MAPSS Dataset</div>
            <div style="opacity: 0.8;">XGBoost • RMSE 18.64 cycles</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== Single Prediction Page ====================
if page == "🎯 Predict RUL":
    
    col_left, col_right = st.columns([1.3, 1])
    
    with col_left:
        st.markdown("### 🔧 Engine Sensor Readings")
        st.markdown("Enter sensor measurements from the turbofan engine.")
        
        # Operational Settings
        st.markdown("**Operational Settings**")
        set_col1, set_col2, set_col3 = st.columns(3)
        with set_col1:
            setting_1 = st.number_input("Setting 1", value=0.0, format="%.4f")
        with set_col2:
            setting_2 = st.number_input("Setting 2", value=0.0, format="%.4f")
        with set_col3:
            setting_3 = st.number_input("Setting 3", value=100.0, format="%.2f")
        
        # Sensors
        st.markdown("**Sensor Measurements**")
        
        # Row 1
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            s_1 = st.number_input("S1 (const)", value=518.67, format="%.2f")
        with s_col2:
            s_2 = st.number_input("S2 (T24)", value=642.0, format="%.2f")
        with s_col3:
            s_3 = st.number_input("S3 (T30)", value=1590.0, format="%.2f")
        with s_col4:
            s_4 = st.number_input("S4 (T50)", value=1408.0, format="%.2f")
        
        # Row 2
        s_col5, s_col6, s_col7, s_col8 = st.columns(4)
        with s_col5:
            s_5 = st.number_input("S5 (const)", value=14.62, format="%.2f")
        with s_col6:
            s_6 = st.number_input("S6 (T2)", value=21.6, format="%.2f")
        with s_col7:
            s_7 = st.number_input("S7 (Ps30)", value=554.0, format="%.2f")
        with s_col8:
            s_8 = st.number_input("S8 (Nf)", value=2388.0, format="%.2f")
        
        # Row 3
        s_col9, s_col10, s_col11, s_col12 = st.columns(4)
        with s_col9:
            s_9 = st.number_input("S9 (Nc)", value=9050.0, format="%.2f")
        with s_col10:
            s_10 = st.number_input("S10 (const)", value=1.3, format="%.2f")
        with s_col11:
            s_11 = st.number_input("S11 (Ps30) ⭐", value=47.5, format="%.2f")
        with s_col12:
            s_12 = st.number_input("S12 (phi)", value=522.0, format="%.2f")
        
        # Row 4
        s_col13, s_col14, s_col15, s_col16 = st.columns(4)
        with s_col13:
            s_13 = st.number_input("S13 (NRf)", value=2388.0, format="%.2f")
        with s_col14:
            s_14 = st.number_input("S14 (NRc)", value=8140.0, format="%.2f")
        with s_col15:
            s_15 = st.number_input("S15 (BPR)", value=8.44, format="%.4f")
        with s_col16:
            s_16 = st.number_input("S16 (const)", value=0.03, format="%.2f")
        
        # Row 5
        s_col17, s_col18, s_col19, s_col20 = st.columns(4)
        with s_col17:
            s_17 = st.number_input("S17 (htBleed)", value=391.0, format="%.2f")
        with s_col18:
            s_18 = st.number_input("S18 (const)", value=2388.0, format="%.2f")
        with s_col19:
            s_19 = st.number_input("S19 (const)", value=100.0, format="%.2f")
        with s_col20:
            s_20 = st.number_input("S20 (W31)", value=39.0, format="%.2f")
        
        # Row 6
        s_col21, _, _, _ = st.columns(4)
        with s_col21:
            s_21 = st.number_input("S21 (W32)", value=23.4, format="%.4f")
        
        predict_btn = st.button("🚀 Predict RUL", use_container_width=True)
    
    with col_right:
        st.markdown("### 📊 Prediction Results")
        
        if predict_btn:
            features = {
                "setting_1": setting_1, "setting_2": setting_2, "setting_3": setting_3,
                "s_1": s_1, "s_2": s_2, "s_3": s_3, "s_4": s_4, "s_5": s_5,
                "s_6": s_6, "s_7": s_7, "s_8": s_8, "s_9": s_9, "s_10": s_10,
                "s_11": s_11, "s_12": s_12, "s_13": s_13, "s_14": s_14, "s_15": s_15,
                "s_16": s_16, "s_17": s_17, "s_18": s_18, "s_19": s_19, "s_20": s_20,
                "s_21": s_21
            }
            
            with st.spinner("Analyzing engine data..."):
                success, result = make_prediction(features)
            
            if success:
                rul = result['RUL']
                status = result['status']
                status_text, status_class, status_color = get_status_class(rul)
                
                # RUL Gauge
                st.plotly_chart(create_rul_gauge(rul, status), use_container_width=True)
                
                # Status Card
                st.markdown(f"""
                <div class="{status_class}" style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 800; color: {status_color};">
                        {status_text.upper()}
                    </div>
                    <div style="color: #475569; margin-top: 0.5rem;">
                        Engine Status: {status}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Metrics Row
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Predicted RUL", f"{rul:.1f} cycles")
                with m2:
                    st.metric("Confidence", result.get('confidence', 'High'))
                
                # Recommendation
                if status_text == "Critical":
                    st.error("⚠️ **Immediate Action Required**: Schedule maintenance within 30 cycles.")
                elif status_text == "Warning":
                    st.warning("🔔 **Plan Maintenance**: Monitor closely, maintenance needed soon.")
                else:
                    st.success("✅ **Engine Healthy**: Normal operation, continue monitoring.")
            else:
                st.error(f"Prediction failed: {result}")
        else:
            st.info("👈 Enter sensor readings and click **Predict RUL** to analyze engine health.")

# ==================== Batch Analysis Page ====================
elif page == "📊 Batch Analysis":
    
    st.markdown("### 📊 Batch Prediction")
    st.markdown("Upload a CSV file with sensor readings to analyze multiple engines.")
    
    # Sample CSV format
    with st.expander("📄 CSV Format Requirements"):
        st.markdown("""
        Your CSV should contain these columns:
        - `setting_1`, `setting_2`, `setting_3`
        - `s_1` through `s_21`
        
        **Example row:**
        ```
        setting_1,setting_2,setting_3,s_1,s_2,...,s_21
        0.0,0.0,100.0,518.67,642.0,...,23.4
        ```
        """)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded {len(df)} records**")
        
        if st.button("🚀 Analyze All Engines", use_container_width=True):
            results = []
            progress = st.progress(0)
            status_text = st.empty()
            
            for i, row in df.iterrows():
                status_text.text(f"Analyzing engine {i+1}/{len(df)}...")
                progress.progress((i + 1) / len(df))
                
                features = row.to_dict()
                success, result = make_prediction(features)
                
                if success:
                    status_text_val, _, _ = get_status_class(result['RUL'])
                    results.append({
                        'Engine': i + 1,
                        'RUL': result['RUL'],
                        'Status': status_text_val,
                        'Confidence': result.get('confidence', 'High')
                    })
            
            status_text.text("Analysis complete!")
            
            if results:
                results_df = pd.DataFrame(results)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Engines", len(results_df))
                with col2:
                    critical = len(results_df[results_df['Status'] == 'Critical'])
                    st.metric("Critical", critical, delta_color="inverse")
                with col3:
                    warning = len(results_df[results_df['Status'] == 'Warning'])
                    st.metric("Warning", warning)
                with col4:
                    avg_rul = results_df['RUL'].mean()
                    st.metric("Avg RUL", f"{avg_rul:.1f}")
                
                # Charts
                ch1, ch2 = st.columns(2)
                with ch1:
                    st.plotly_chart(create_status_chart(results_df), use_container_width=True)
                with ch2:
                    fig = px.histogram(results_df, x='RUL', nbins=20, 
                                       color_discrete_sequence=['#0891b2'])
                    fig.update_layout(title="RUL Distribution", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.markdown("### 📋 Detailed Results")
                st.dataframe(results_df, use_container_width=True, height=400)
                
                # Download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results CSV",
                    csv,
                    "rul_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )

# ==================== Dashboard Page ====================
elif page == "📡 Dashboard":
    
    st.markdown("### 📡 System Dashboard")
    
    # API Status
    is_healthy, health_data = check_api_health()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #10b981;">●</div>
            <div class="metric-label">API Status</div>
        </div>
        """ if is_healthy else """
        <div class="metric-card">
            <div class="metric-value" style="color: #ef4444;">●</div>
            <div class="metric-label">API Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">18.64</div>
            <div class="metric-label">RMSE (cycles)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0.79</div>
            <div class="metric-label">R² Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Info
    st.markdown("### 📊 Model Information")
    
    success, model_info = get_model_info()
    
    if success:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Model Details**")
            st.json({
                "Type": model_info.get('model_type', 'XGBoost'),
                "Version": model_info.get('version', '2.0.0'),
                "RMSE": model_info.get('performance', {}).get('test_rmse', 18.64),
                "Improvement": model_info.get('performance', {}).get('improvement_pct', '62.8%')
            })
        
        with col_b:
            st.markdown("**Top Features**")
            features = model_info.get('top_feature_importance', [
                "s_4_mean (35.2%)",
                "s_11_mean (16.8%)",
                "s_15_mean (14.0%)",
                "s_9_mean (5.6%)",
                "s_21_mean (4.1%)"
            ])
            for f in features:
                st.markdown(f"• {f}")
    else:
        st.info("Start the API to view model details: `python -m uvicorn api.main:app --port 8000`")
    
    st.markdown("---")
    
    # Quick Links
    st.markdown("### 🔗 Quick Links")
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.markdown(f"[📚 API Docs]({API_BASE_URL}/docs)")
    with col_l2:
        st.markdown("[💻 GitHub](https://github.com/AymenMB/turbofan-predictive-maintenance-mlops)")
    with col_l3:
        st.markdown(f"[❤️ Health Check]({API_BASE_URL}/health)")

if __name__ == "__main__":
    pass
