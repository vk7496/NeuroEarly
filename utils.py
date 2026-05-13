import mne
import numpy as np
import streamlit as st
from groq import Groq

def process_eeg(file_path):
    """
    پردازش فایل EDF و استخراج توان باندهای فرکانسی برای ۶۶ کانال
    """
    # بارگذاری فایل بدون لود کردن تمام دیتا در رم (برای سرعت بیشتر)
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    # فیلتر کردن سیگنال بین ۱ تا ۴۰ هرتز
    raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
    
    # محاسبه Power Spectral Density (PSD)
    # استفاده از روش Welch برای استخراج میانگین توان فرکانسی
    spectrum = raw.compute_psd(method='welch', fmin=1.0, fmax=40.0, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    # تعریف باندهای فرکانسی
    bands = {
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30)
    }
    
    results = {}
    for band, (fmin, fmax) in bands.items():
        # پیدا کردن ایندکس‌های مربوط به هر باند
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        # محاسبه میانگین توان در آن باند برای تمام ۶۶ کانال
        band_psd = psds[:, idx_band].mean()
        results[band] = band_psd
        
    return results

def get_gemma_analysis(results):
    """
    ارسال نتایج عددی به Gemma 4 (Groq) و دریافت تحلیل پزشکی
    """
    # فراخوانی کلاینت Groq با استفاده از کلید ذخیره شده در Secrets
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # طراحی پرامپت مهندسی شده
    prompt = f"""
    You are an expert Neurologist AI system named NeuroEarly. 
    Analyze these EEG Power Spectral Density (PSD) values extracted from a professional 66-channel recording:
    
    - Theta Band Power: {results['Theta']:.4f}
    - Alpha Band Power: {results['Alpha']:.4f}
    - Beta Band Power: {results['Beta']:.4f}
    
    Please provide:
    1. **Clinical Brief**: Technical interpretation of the Theta/Beta ratio and potential cortical slowing for medical professionals.
    2. **Patient Summary**: An empathetic, non-technical explanation of what these brainwave patterns mean for the patient's cognitive state.
    
    Language: English
    """
    
    # درخواست از مدل Gemma 2 (نسخه جدید و جایگزین)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a specialized medical AI assistant for neuro-diagnostic analysis."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gemma2-9b-it",
        temperature=0.7,
        max_tokens=1000
    )
    
    return chat_completion.choices[0].message.content
