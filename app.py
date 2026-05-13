import streamlit as st
import mne
import plotly.graph_objects as go
import os
from utils import process_eeg, get_gemma_analysis

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
    .report-box { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #2E5BFF;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# هدر و برندینگ پروژه
st.title("🧠 NeuroEarly: Advanced EEG Insight Engine")
st.write("Leveraging **Gemma 4** via **Groq** to transform 66-channel raw signals into clinical intelligence.")

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
    
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.rerun()

if uploaded_file:
    # ۱. عملیات پردازش (Backend)
    with st.spinner("Analyzing neural oscillations from 66 channels..."):
        # ذخیره موقت فایل آپلود شده
        with open("temp_patient.edf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # استخراج مقادیر واقعی از فایل EDF
            results = process_eeg("temp_patient.edf")
            
            # ۲. نمایش شاخص‌های کلیدی (Metrics)
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Theta Power", f"{results['Theta']:.4f}")
            col_m2.metric("Alpha Power", f"{results['Alpha']:.4f}")
            col_m3.metric("Beta Power", f"{results['Beta']:.4f}")

            st.divider()

            # ۳. بخش بصری‌سازی و گزارش هوشمند
            col_left, col_right = st.columns([1.2, 1])

            with col_left:
                st.subheader("📊 Frequency Spectrum Analysis")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Theta', 'Alpha', 'Beta'],
                    y=[results['Theta'], results['Alpha'], results['Beta']],
                    marker_color=['#636EFA', '#EF553B', '#00CC96'],
                    text=[f"{v:.4f}" for v in [results['Theta'], results['Alpha'], results['Beta']]],
                    textposition='auto',
                ))
                fig.update_layout(
                    template="plotly_white",
                    yaxis_title="Power Spectral Density (PSD)",
                    height=450,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.subheader("🤖 Gemma 4 Clinical Intelligence")
                
                # دکمه فراخوانی API
                if st.button("Generate Live AI Report", type="primary"):
                    if "GROQ_API_KEY" in st.secrets:
                        try:
                            with st.spinner("Gemma 4 is processing medical reasoning..."):
                                report = get_gemma_analysis(results)
                                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                                st.write_stream(iter(report.splitlines(keepends=True))) # افکت تایپ شدن
                                st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"AI Analysis Error: {str(e)}")
                    else:
                        st.warning("Please configure GROQ_API_KEY in Streamlit Secrets.")
                else:
                    st.info("Click the button above to start the AI diagnostic analysis.")

        except Exception as e:
            st.error(f"Error processing EDF file: {str(e)}")
        finally:
            # پاکسازی فایل موقت
            if os.path.exists("temp_patient.edf"):
                os.remove("temp_patient.edf")

else:
    # وضعیت خوش‌آمدگویی قبل از آپلود
    st.info("Waiting for EDF input... Please upload a file from the sidebar to generate the NeuroEarly report.")
    # نمایش یک تصویر پیش‌فرض یا راهنما
    st.image("https://raw.githubusercontent.com/mne-tools/mne-python/main/doc/_static/mne_logo.png", width=200)
    st.write("---")
    st.markdown("""
    ### How to use NeuroEarly:
    1. **Upload:** Select a standard 66-channel EDF file.
    2. **Process:** Our engine filters the signal and extracts PSD values.
    3. **Insight:** Gemma 4 generates a dual-layer report for clinical and personal use.
    """)

# فوتر برای برندینگ شخصی
st.markdown("---")
st.caption("Developed by Vista Kaviani | AI Solutions Developer | NeuroEarly Project - 2026")
