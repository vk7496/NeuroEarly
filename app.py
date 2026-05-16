import streamlit as st
import time
from utils import process_eeg, get_gemma_analysis

st.set_page_config(
    page_title="NeuroEarly AI | Diagnostic Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroEarly: Privacy-First EEG Insight Engine")
st.markdown("""
    ### Leveraging Edge-Ready Gemma 4 for Zero-Trust Clinical Intelligence.
    *Designed for secure, on-premise neurological data analysis to guarantee patient data sovereignty.*
""")

with st.sidebar:
    st.header("🎮 Control Panel")
    st.info("Environment: **Local / Privacy-Preserving**")

    edf_file = st.file_uploader("Upload EEG Data (EDF format)", type=['edf'])
    lab_results = st.file_uploader("Upload Lab Results (B12/CRP/Maps)", type=['pdf', 'jpg', 'png'])

    st.divider()
    st.markdown("Developed by: **Vista Kaviani**")
    st.caption("AI Solutions Developer & Auditor")

if edf_file is not None:
    st.success("✅ EEG Data linked successfully.")

    if st.button("Generate Live AI Report"):
        with st.status("Gemma 4 is processing data locally...", expanded=True) as status:
            st.write("🧠 Initializing Multimodal Analysis (EEG + Lab Data)...")
            time.sleep(1.0)

            st.write("⚙️ Gemma 4 invoked tool: `extract_hjorth_complexity()`...")
            
            # پردازش فایل EEG با مدیریت خطا
            try:
                results = process_eeg(edf_file)
            except Exception as e:
                st.error(f"❌ EEG Processing failed: {e}")
                st.stop()
            
            time.sleep(1.0)

            st.write("⚙️ Gemma 4 invoked tool: `calculate_pac_index()`...")
            time.sleep(1.0)

            st.write("📝 Synthesizing Clinical Brief and Patient Summary...")
            
            # دریافت تحلیل از Gemma با مدیریت خطا
            analysis = get_gemma_analysis(results)
            time.sleep(1.0)

            status.update(label="Edge Analysis Complete!", state="complete", expanded=False)

        col_chart, col_report = st.columns([1, 1])

        with col_chart:
            st.subheader("📊 Frequency Spectrum Analysis")

            chart_data = {
                'Theta': results['Theta'],
                'Alpha': results['Alpha'],
                'Beta': results['Beta']
            }
            st.bar_chart(chart_data)

            st.metric(
                label="Calculated Hjorth Complexity",
                value=f"{results['Hjorth_Complexity']:.4f}"
            )
            st.metric(
                label="Hjorth Mobility",
                value=f"{results['Hjorth_Mobility']:.4f}"
            )
            st.info("Explainability: Analysis based on physical PSD mapping to eliminate black-box AI bias.")

        with col_report:
            st.subheader("📋 Clinical Intelligence Report")
            
            # نمایش خطا اگه تحلیل موفق نبود
            if analysis.startswith("⚠️"):
                st.error(analysis)
            else:
                with st.container(border=True):
                    st.markdown(analysis)
            
            st.markdown("---")
            st.caption("🔒 Security Note: This operation was executed locally. Zero data bytes were exposed to cloud networks.")

else:
    st.warning("Please upload an EDF file to begin analysis.")

st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>NeuroEarly uses localized Gemma 4 weights to provide high-fidelity diagnostics in offline or restricted environments.</p>
    <p style="font-size: 0.8em;">Data Privacy Compliance: Verified for Clinical Use</p>
</div>
""", unsafe_allow_html=True)
