"""
Interfața Streamlit pentru monitorizarea video cu modelul de detecție de anomalii.

Flux simplificat:
- utilizatorul selectează un fișier `.npy` cu secvențe video preprocesate,
- aplicația încarcă modelul `ConvLSTMAutoencoder` din `models/trained_model.pt`,
- pentru fiecare fereastră temporală calculează un scor de anomalie (MSE pe top 5% pixeli),
- decide „NORMAL” / „ANORMAL” în funcție de prag (`threshold`) și persistență (`min_frames`),
- afișează frame-urile, graficul scorului și un log cu alertele apărute.
"""

import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from src.neural_network.model import ConvLSTMAutoencoder

ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = ROOT / "models" / "trained_model.pt"
DATA_DIR = ROOT / "data"
DEVICE = torch.device("cpu")

st.set_page_config(page_title="Surveillance Monitor", layout="wide")

@st.cache_resource
def load_model() -> ConvLSTMAutoencoder | None:
    """
    Încarcă modelul antrenat de pe disc și îl pune în modul `eval`.

    Folosim `cache_resource` pentru a evita reîncărcarea modelului la fiecare
    interacțiune cu UI-ul.
    """
    model = ConvLSTMAutoencoder().to(DEVICE)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    st.error(f"Modelul nu a fost găsit la calea: {MODEL_PATH}")
    return None

def main() -> None:
    """
    Construcția layout-ului Streamlit și rularea buclei de inferență.
    """
    st.title("🛡️ Monitorizare Video")
    model = load_model()
    
    with st.sidebar:
        st.header("Control & Log")
        # Enumerăm toate fișierele .npy disponibile pentru testare.
        files = [str(p.relative_to(DATA_DIR)) for p in DATA_DIR.rglob("*.npy")]
        selected = st.selectbox("Fisier Video:", files)
        fps = st.slider("Viteza (FPS)", 1, 60, 25)
        threshold = st.slider("Prag Detecție", 0.001, 0.100, 0.025, step=0.001, format="%.3f")
        min_frames = st.slider("Persistență", 1, 10, 3)
        run = st.button("▶️ START")
        st.divider()
        log_container = st.container()

    if run:
        full_path = DATA_DIR / selected
        data = np.load(full_path, mmap_mode="r")
        
        # Layout: Video în stânga, Grafic în dreapta
        col_left, col_right = st.columns([1, 1.5])
        
        with col_left:
            video_placeholder = st.empty()
            # Metricile de bază sub video
            m_col1, m_col2, m_col3 = st.columns(3)
            m_frame = m_col1.empty()
            m_time = m_col2.empty()
            m_score = m_col3.empty()
        
        with col_right:
            st.write("### Evoluție Scor (Ultimile 50 cadre)")
            chart_placeholder = st.empty()

        # Istoricul scorurilor de anomalie și numărul de cadre consecutive
        # peste prag (folosit pentru a stabiliza decizia de anomalie).
        history = []
        consecutive = 0
        anomalies_logged = set()

        for i in range(data.shape[0]):
            # Luăm o singură fereastră temporală (shape: 1, T, H, W, C).
            sample = np.array(data[i : i + 1]).copy()
            input_tensor = torch.FloatTensor(sample).permute(0, 2, 1, 3, 4).to(DEVICE)

            # Reconstruim secvența și calculăm scorul de anomalie.
            # Folosim media celor mai mari 5% erori de reconstrucție pentru
            # a ne concentra pe regiuni critice din volum.
            with torch.no_grad():
                output = model(input_tensor)
                diff = (output - input_tensor) ** 2
                top_k = int(diff.numel() * 0.05)
                top_v, _ = torch.topk(diff.view(-1), top_k)
                mse = top_v.mean().item()

            if mse > threshold:
                consecutive += 1
            else:
                consecutive = 0
            
            is_anomaly = consecutive >= min_frames
            history.append(mse)

            # Update Metrici
            m_frame.metric("Cadru", i)
            m_time.metric("Timp", time.strftime('%M:%S', time.gmtime(i/fps)))
            m_score.metric("Scor Actual", f"{mse:.4f}")

            # Update Grafic (Dreapta) – evoluția scorului în timp.
            fig_chart, ax_chart = plt.subplots(figsize=(6, 3.2))
            ax_chart.plot(history[-50:], color='#1f77b4', linewidth=2)
            ax_chart.axhline(y=threshold, color='red', linestyle='--', label='Prag')
            ax_chart.set_ylim(0, max(max(history[-50:] if history else [0.1]), threshold) * 1.2)
            ax_chart.set_facecolor('#0e1117') # Match Streamlit dark theme if needed
            fig_chart.patch.set_facecolor('#0e1117')
            ax_chart.tick_params(colors='white')
            ax_chart.grid(alpha=0.2)
            chart_placeholder.pyplot(fig_chart)
            plt.close(fig_chart)

            # Update Video (Stânga) – afișăm primul cadru din secvență.
            fig_vid, ax_vid = plt.subplots(figsize=(4, 3))
            ax_vid.imshow(sample[0, 0, 0, :, :], cmap='gray')
            if is_anomaly:
                ax_vid.set_title("ANORMAL", color='red', fontsize=12, fontweight='bold')
                for spine in ax_vid.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(4)
                
                # Log în sidebar
                t_str = time.strftime('%M:%S', time.gmtime(i/fps))
                if i not in anomalies_logged:
                    log_container.error(f"🚨 {t_str} - Anomalie!")
                    anomalies_logged.add(i)
            else:
                ax_vid.set_title("NORMAL", color='green', fontsize=12)
            
            ax_vid.axis('off')
            plt.tight_layout(pad=0)
            video_placeholder.pyplot(fig_vid)
            plt.close(fig_vid)

            time.sleep(1/fps)
            del input_tensor, output, sample
            if i % 20 == 0: gc.collect()

if __name__ == "__main__":
    main()