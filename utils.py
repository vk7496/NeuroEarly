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
