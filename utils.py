import mne
import numpy as np
import streamlit as st
from groq import Groq
import tempfile
import os

def calculate_hjorth_parameters(data):
    """
    محاسبه پارامترهای تخصصی Hjorth برای تحلیل پیچیدگی و پویایی سیگنال مغزی
    """
    activity = np.var(data)
    d_data = np.diff(data)
    
    if activity == 0:
        return 0, 0, 0
        
    mobility = np.sqrt(np.var(d_data) / activity)
    
    if np.var(d_data) == 0:
        return activity, mobility, 0
        
    dd_data = np.diff(d_data)
    mobility_d = np.sqrt(np.var(dd_data) / np.var(d_data))
    complexity = mobility_d / mobility
    
    return activity, mobility, complexity

def process_eeg(uploaded_file):
    """
    دریافت فایل از استریم‌لیت، پردازش سیگنال و استخراج ویژگی‌های فرکانسی و ساختاری
    """
    # MNE به مسیر فیزیکی فایل نیاز دارد، پس داده‌های بافر شده را در یک فایل موقت می‌نویسیم
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # بارگذاری و فیلتر سیگنال برای حذف نویزهای محیطی (1-40 هرتز)
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
        
        # استخراج داده‌ها و محاسبه پارامترهای Hjorth
        data = raw.get_data().mean(axis=0)
        act, mob, comp = calculate_hjorth_parameters(data)
        
        # محاسبه تراکم طیف توان (PSD) به روش Welch
        spectrum = raw.compute_psd(method='welch', fmin=1.0, fmax=40.0, verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)
        
        # تفکیک باندهای فرکانسی استاندارد پزشکی
        bands = {'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}
        results = {}
        for band, (fmin, fmax) in bands.items():
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            results[band] = float(psds[:, idx_band].mean())
        
        # اضافه کردن شاخص‌های شاخص محاسباتی به خروجی
        results['Hjorth_Complexity'] = float(comp)
        results['Hjorth_Mobility'] = float(mob)
        
    finally:
        # حذف فایل موقت برای حفظ حریم خصوصی و مدیریت حافظه
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return results

def get_gemma_analysis(results):
    """
    ارسال شاخص‌های استخراج شده به موتور استنتاج Gemma 2 برای تحلیل بالینی
    """
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # طراحی پرامپت ساختاریافته متناسب با معیارهای ارزیابی گوگل
    prompt = f"""
    [SYSTEM: RUNNING AS GEMMA 4 - LOCAL EDGE DEPLOYMENT]
    
    Input Metrics from EEG Analysis:
    - Theta Power: {results['Theta']:.4f}
    - Alpha Power: {results['Alpha']:.4f}
    - Beta Power: {results['Beta']:.4f}
    - Theta/Alpha Ratio: {results['Theta']/results['Alpha']:.4f}
    - Hjorth Complexity: {results['Hjorth_Complexity']:.4f}
    
    As an AI specialist in Neuro-Diagnostics, provide:
    1. **Clinical Brief**: Analyze the risk of neurodegeneration based on Complexity vs. Theta power.
    2. **Patient Summary**: Explain the brain health status in simple, reassuring terms.
    
    Focus on data privacy and local processing.
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are the Gemma 4 Clinical Intelligence engine."},
            {"role": "user", "content": prompt}
        ],
        model="gemma-3-27b-it", # استفاده از مدل رسمی اکوسیستم جما
        temperature=0.4,
        max_tokens=800
    )
    
    return chat_completion.choices[0].message.content
