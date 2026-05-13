# utils.py
import mne
import numpy as np

def process_eeg(file_path):
    """
    پردازش فایل EDF و استخراج توان باندهای فرکانسی
    """
    raw = mne.io.read_raw_edf(file_path, preload=True)
    # فیلتر کردن سیگنال (همان کاری که در کگل کردیم)
    raw.filter(1., 40., fir_design='firwin')
    
    # محاسبه PSD (Power Spectral Density)
    psd = raw.compute_psd(method='welch', fmin=1, fmax=40)
    psds, freqs = psd.get_data(return_freqs=True)
    
    # میانگین‌گیری روی تمام ۶۶ کانال
    avg_psd = np.mean(psds, axis=0)
    
    # استخراج مقادیر برای باندها (نمونه)
    theta = avg_psd[(freqs >= 4) & (freqs <= 8)].mean()
    alpha = avg_psd[(freqs >= 8) & (freqs <= 12)].mean()
    beta = avg_psd[(freqs >= 12) & (freqs <= 30)].mean()
    
    return {"Theta": theta, "Alpha": alpha, "Beta": beta}
import streamlit as st
from groq import Groq

def get_gemma_analysis(results):
    # فراخوانی کلید از Secrets استریم‌لایت
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # طراحی پرامپت مهندسی‌شده برای Gemma 4
    prompt = f"""
    You are an expert Neurologist AI. Analyze these EEG PSD values from a professional 66-channel recording:
    - Theta Power: {results['Theta']:.4f}
    - Alpha Power: {results['Alpha']:.4f}
    - Beta Power: {results['Beta']:.4f}
    
    Task:
    1. Provide a technical 'Clinical Brief' for doctors.
    2. Provide a simple, empathetic 'Patient Summary'.
    Focus on Theta/Beta ratios and cortical slowing.
    """
    
    # در فایل utils.py این بخش را پیدا و اصلاح کن:
chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="gemma2-9b-it", # مطمئن شو که دقیقاً همین نام باشد
)
    
    return chat_completion.choices[0].message.content
