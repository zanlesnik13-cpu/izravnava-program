import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st
import plotly.graph_objects as go


def render_deformacije_module(current_user: dict):
    st.markdown("""
    <div style="background-color: #9f1239; color: white; padding: 18px 24px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0; font-size: 1.5rem;">📊 DEFORMACIJSKA ANALIZA IN PREMIKI GEODETSKIH TOČK</h2>
        <p style="color: #fecdd3; margin: 4px 0 0 0; font-size: 0.95rem;">
            Dvofazna analiza epohe 0 in epohe 1 • Kongruenčni testi • Pelzerjeva metoda • Vektorji premikov
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_analiza, tab_grafik, tab_teorija = st.tabs([
        "🔍 1. Dvofazna analiza premikov (E0 in E1)",
        "🗺️ 2. Vizualizacija vektorjev premikov (Plotly)",
        "📖 3. Metoda kongruence in S-transformacija"
    ])

    if "df_deform_tocke" not in st.session_state:
        st.session_state.df_deform_tocke = pd.DataFrame({
            "Točka": ["P1 (Ref)", "P2 (Ref)", "P3 (Objekt)", "P4 (Objekt)"],
            "Tip točke": ["Referenčna", "Referenčna", "Objektna", "Objektna"],
            "X0 [m]": [100.000, 250.000, 180.000, 210.000],
            "Y0 [m]": [100.000, 120.000, 220.000, 260.000],
            "X1 [m]": [100.001, 250.002, 180.015, 210.024],
            "Y1 [m]": [099.999, 120.001, 219.982, 259.976],
            "σ_X [mm]": [1.5, 1.5, 1.8, 1.8],
            "σ_Y [mm]": [1.5, 1.5, 1.8, 1.8]
        })

    with tab_analiza:
        st.subheader("Primerjava koordinat točk med dvema epohama")
        st.caption("Vnesite izravnane koordinate točk v začetni epohi (E0) in kontrolni epohi (E1):")
        st.session_state.df_deform_tocke = st.data_editor(st.session_state.df_deform_tocke, use_container_width=True, key="ed_deform")

        alpha_def = st.slider("Stopnja značilnosti testa α:", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

        df_t = st.session_state.df_deform_tocke
        # Izračun premikov
        dx = (df_t["X1 [m]"] - df_t["X0 [m]"]) * 1000.0  # mm
        dy = (df_t["Y1 [m]"] - df_t["Y0 [m]"]) * 1000.0  # mm
        d_disp = np.sqrt(dx**2 + dy**2)                  # mm

        # Natančnost premika: sigma_d = sqrt(sigma_0^2 + sigma_1^2)
        sig_dx = np.sqrt(df_t["σ_X [mm]"]**2 + df_t["σ_X [mm]"]**2)
        sig_dy = np.sqrt(df_t["σ_Y [mm]"]**2 + df_t["σ_Y [mm]"]**2)
        sig_d = np.sqrt((dx**2 * sig_dx**2 + dy**2 * sig_dy**2) / np.maximum(1e-6, d_disp**2))

        # Testna statistika T = d / sig_d
        z_crit = stats.norm.ppf(1.0 - alpha_def / 2.0)
        t_stat = d_disp / sig_d

        status_premika = ["POMIK STATISTIČNO ZNAČILEN!" if t > z_crit else "Stabilno (ni premika)" for t in t_stat]

        df_rez_def = pd.DataFrame({
            "Točka": df_t["Točka"],
            "Tip": df_t["Tip točke"],
            "ΔX [mm]": dx,
            "ΔY [mm]": dy,
            "Premik d [mm]": d_disp,
            "σ_d [mm]": sig_d,
            "Test T": t_stat,
            "Z_krit": z_crit,
            "Sklep": status_premika
        })

        st.subheader("Statistična ocena premikov točk")
        st.dataframe(df_rez_def.style.format({
            "ΔX [mm]": "{:.1f}", "ΔY [mm]": "{:.1f}",
            "Premik d [mm]": "{:.1f}", "σ_d [mm]": "{:.1f}",
            "Test T": "{:.2f}", "Z_krit": "{:.2f}"
        }), use_container_width=True)

    with tab_grafik:
        st.subheader("Vektorji premikov z merilom povečave")
        povezava_merilo = st.slider("Povečava vektorjev premika:", min_value=100, max_value=5000, value=1500, step=100)

        fig_def = go.Figure()
        # Točke E0
        fig_def.add_trace(go.Scatter(
            x=df_t["X0 [m]"], y=df_t["Y0 [m]"],
            mode="markers+text", marker=dict(size=10, color="#1e3a8a"),
            text=df_t["Točka"], textposition="bottom center",
            name="Epoha 0 (Začetna)"
        ))

        # Vektorji premikov
        for i, r in df_t.iterrows():
            x0, y0 = r["X0 [m]"], r["Y0 [m]"]
            vx_mm, vy_mm = dx[i], dy[i]
            x_end = x0 + (vx_mm / 1000.0) * povezava_merilo
            y_end = y0 + (vy_mm / 1000.0) * povezava_merilo

            barva = "#ef4444" if "ZNAČILEN" in status_premika[i] else "#10b981"
            fig_def.add_trace(go.Scatter(
                x=[x0, x_end], y=[y0, y_end],
                mode="lines", line=dict(color=barva, width=3),
                hoverinfo="text",
                text=f"{r['Točka']}: d = {d_disp[i]:.1f} mm ({status_premika[i]})",
                name=f"Vektor {r['Točka']}"
            ))

        fig_def.update_layout(
            title=f"Vektorji premikov točk (Povečava: {povezava_merilo}×)",
            xaxis_title="X [m]", yaxis_title="Y [m]",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            height=500
        )
        st.plotly_chart(fig_def, use_container_width=True)

    with tab_teorija:
        st.markdown("""
        ### Geodetska deformacijska analiza
        - **Model Pelzerja za referenčne točke:**
          Najprej preverimo globalno stabilnost referenčnih točk. Če je test $T > F_{krit}$, sistematično izločujemo nestabilne točke (ena po ena z največjim odstopanjem).
        - **S-transformacija:**
          Omogoča prehod med različnimi geodetskimi datumi z minimalno normo koordinat na izbrani podmnožici stabilnih točk.
        """)
