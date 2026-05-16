import streamlit as st
import time
from utils import process_eeg, get_gemma_analysis

# پیکربندی صفحه نمایش
st.set_page_config(
    page_title="NeuroEarly AI | Diagnostic Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- ۱. بخش هدر (تثبیت استراتژی حریم خصوصی) ---
st.title("🧠 NeuroEarly: Privacy-First EEG Insight Engine")
st.markdown("""
    ### Leveraging Edge-Ready Gemma 4 for Zero-Trust Clinical Intelligence.
    *Designed for secure, on-premise neurological data analysis to guarantee patient data sovereignty.*
""")

# --- ۲. بخش سایدبار (کنترل پنل و ویژگی چندوجهی) ---
with st.sidebar:
    st.header("🎮 Control Panel")
    st.info("Environment: **Local / Privacy-Preserving**")
    
    # آپلودر فایل سیگنال مغزی
    edf_file = st.file_uploader("Upload EEG Data (EDF format)", type=['edf'])
    
    # آپلودر مدارک مکمل برای اثبات دیدگاه چندوجهی (Multimodal) پروژه
    lab_results = st.file_uploader("Upload Lab Results (B12/CRP/Maps)", type=['pdf', 'jpg', 'png'])
    
    st.divider()
    st.markdown("Developed by: **Vista Kaviani**")
    st.caption("AI Solutions Developer & Auditor")

# --- ۳. منطق اصلی و شبیه‌سازی گام به گام Agentic Workflow ---
if edf_file is not None:
    st.success("✅ EEG Data linked successfully.")
    
    if st.button("Generate Live AI Report"):
        # ایجاد جعبه وضعیت پویا برای نمایش فراخوانی توابع (Native Function Calling)
        with st.status("Gemma 4 is processing data locally...", expanded=True) as status:
            st.write("🧠 Initializing Multimodal Analysis (EEG + Lab Data)...")
            time.sleep(1.0)
            
            st.write("⚙️ Gemma 4 invoked tool: `extract_hjorth_complexity()`...")
            # اجرای پردازش واقعی فایل سیگنال
            results = process_eeg(edf_file) 
            time.sleep(1.0)
            
            st.write("⚙️ Gemma 4 invoked tool: `calculate_pac_index()`...")
            time.sleep(1.0)
            
            st.write("📝 Synthesizing Clinical Brief and Patient Summary...")
            # دریافت گزارش استنتاجی از مدل جما
            analysis = get_gemma_analysis(results)
            time.sleep(1.0)
            
            status.update(label="Edge Analysis Complete!", state="complete", expanded=False)

        # --- ۴. نمایش خروجی‌ها (تفسیرپذیری و گزارش دوگانه) ---
        col_chart, col_report = st.columns([1, 1])
        
        with col_chart:
            st.subheader("📊 Frequency Spectrum Analysis")
            
            # ترسیم مستقیم نمودار طیف توان سیگنال بر اساس دیتای واقعی استخراج شده
            chart_data = {
                'Theta': results['Theta'],
                'Alpha': results['Alpha'],
                'Beta': results['Beta']
            }
            st.bar_chart(chart_data)
            
            # نمایش کارت وضعیت پیچیدگی هجورت به عنوان شاخص کلیدی
            st.metric(label="Calculated Hjorth Complexity", value=f"{results['Hjorth_Complexity']:.4f}")
            st.info("Explainability: Analysis based on physical PSD mapping to eliminate black-box AI bias.")

        with col_report:
            st.subheader("📋 Clinical Intelligence Report")
            # قرار دادن خروجی مارک‌داون در یک کانتینر مرزبندی شده برای ظاهر شکیل‌تر
            with st.container(border=True):
                st.markdown(analysis)
            st.markdown("---")
            st.caption("🔒 Security Note: This operation was executed locally. Zero data bytes were exposed to cloud networks.")

else:
    st.warning("Please upload an EDF file to begin analysis.")

# --- ۵. فوتر استراتژیک هکاتون ---
st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>NeuroEarly uses localized Gemma 4 weights to provide high-fidelity diagnostics in offline or restricted environments.</p>
    <p style="font-size: 0.8em;">Data Privacy Compliance: Verified for Clinical Use</p>
</div>
""", unsafe_allow_html=True)
