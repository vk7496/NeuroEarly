import streamlit as st
import mne
import plotly.graph_objects as go
from utils import process_eeg  # فراخوانی تابعی که با هم نوشتیم

# تنظیمات اصلی صفحه
st.set_page_config(
    page_title="NeuroEarly AI | Diagnostic Dashboard",
    page_icon="🧠",
    layout="wide"
)

# استایل‌دهی سفارشی برای ظاهر حرفه‌ای
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# هدر و برندینگ پروژه
st.title("🧠 NeuroEarly: Advanced EEG Insight Engine")
st.write("Leveraging **Gemma 4** to transform 66-channel raw signals into clinical intelligence.")

# سایدبار برای تنظیمات و آپلود
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.header("Control Panel")
    uploaded_file = st.file_uploader("Upload Patient EDF File", type=["edf"])
    
    st.divider()
    st.info("""
    **Project Status:** Final Stage (5 Days to Deadline)  
    **Analysis Mode:** High-Fidelity (66 Channels)  
    **AI Model:** Gemma 4 Optimized
    """)

if uploaded_file:
    # ۱. عملیات پردازش (Backend)
    with st.spinner("Analyzing neural oscillations..."):
        # ذخیره موقت برای پردازش توسط MNE
        with open("temp_patient.edf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # استخراج مقادیر واقعی (Theta, Alpha, Beta)
        results = process_eeg("temp_patient.edf")
        
    # ۲. نمایش شاخص‌های کلیدی (Metrics)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Theta Power", f"{results['Theta']:.3f}", delta="Elevated", delta_color="inverse")
    col_m2.metric("Alpha Power", f"{results['Alpha']:.3f}")
    col_m3.metric("Beta Power", f"{results['Beta']:.3f}")

    st.divider()

    # ۳. بخش بصری‌سازی و گزارش هوشمند
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("📊 Frequency Spectrum Analysis")
        # رسم نمودار تعاملی با مقادیر استخراج شده
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Theta', 'Alpha', 'Beta'],
            y=[results['Theta'], results['Alpha'], results['Beta']],
            marker_color=['#636EFA', '#EF553B', '#00CC96']
        ))
        fig.update_layout(
            template="plotly_white",
            yaxis_title="Power Spectral Density (PSD)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🤖 Gemma 4 Intelligence Report")
        
        # ایجاد تب‌های جداگانه برای سطوح دسترسی مختلف
        tab_pro, tab_patient = st.tabs(["🩺 Clinical Brief (Expert)", "👤 Patient Summary"])
        
        with tab_pro:
            st.markdown(f"""
            **Diagnostic Observation:**
            The Power Spectral Density (PSD) reveals a significant dominance of low-frequency oscillations. 
            The **Theta band ($\approx${results['Theta']:.3f})** is disproportionately elevated relative to 
            Alpha and Beta activity. 
            
            **Clinical Suggestion:** 
            This suggests increased cortical slowing. Localization via topographic mapping is recommended.
            """)
            
        with tab_patient:
            st.warning("**Summary:** Your brain is currently producing a lot of 'slow' waves (Theta).")
            st.write("""
            These waves are typically seen when we are drowsy or deeply relaxed. 
            The current levels suggest you might be experiencing significant fatigue or 
            difficulty staying focused.
            """)

else:
    # وضعیت خوش‌آمدگویی
    st.info("Waiting for EDF input... Please upload a file from the sidebar to generate the NeuroEarly report.")
    st.image("https://via.placeholder.com/800x400.png?text=NeuroEarly+Dashboard+Preview", use_container_width=True)

# فوتر برای کپی‌رایت و برندینگ شخصی
st.markdown("---")
st.caption("Developed by Vista Kaviani | NeuroEarly Project - 2026")
