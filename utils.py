import mne
import numpy as np
import streamlit as st
from groq import Groq
import tempfile
import os

def calculate_hjorth_parameters(data):
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
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
        
        data = raw.get_data().mean(axis=0)
        act, mob, comp = calculate_hjorth_parameters(data)
        
        spectrum = raw.compute_psd(method='welch', fmin=1.0, fmax=40.0, verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)
        
        bands = {'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}
        results = {}
        for band, (fmin, fmax) in bands.items():
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            results[band] = float(psds[:, idx_band].mean())
        
        results['Hjorth_Complexity'] = float(comp)
        results['Hjorth_Mobility'] = float(mob)
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return results

def get_gemma_analysis(results):
    # چک کردن وجود API key
    if "GROQ_API_KEY" not in st.secrets:
        return "⚠️ GROQ_API_KEY not found in Streamlit Secrets. Please add it in Manage app → Settings → Secrets."

    api_key = st.secrets["GROQ_API_KEY"]
    
    if not api_key or not api_key.startswith("gsk_"):
        return "⚠️ GROQ_API_KEY is invalid. Make sure it starts with 'gsk_'."

    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""
Input Metrics from EEG Analysis:
- Theta Power: {results['Theta']:.4f}
- Alpha Power: {results['Alpha']:.4f}
- Beta Power: {results['Beta']:.4f}
- Theta/Alpha Ratio: {results['Theta']/results['Alpha']:.4f}
- Hjorth Complexity: {results['Hjorth_Complexity']:.4f}

As an AI specialist in Neuro-Diagnostics, provide:
1. **Clinical Brief**: Analyze the risk of neurodegeneration based on Complexity vs. Theta power.
2. **Patient Summary**: Explain the brain health status in simple, reassuring terms.
"""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a clinical AI assistant specialized in EEG-based neurodiagnostics."},
                {"role": "user", "content": prompt}
            ],
            model="gemma2-9b-it",
            temperature=0.4,
            max_tokens=800
        )
        
        return chat_completion.choices[0].message.content

    except Exception as e:
        # نمایش ارور دقیق برای دیباگ
        error_msg = str(e)
        return f"⚠️ Groq API Error: {error_msg}"
