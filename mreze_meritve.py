import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules import mnk_izravnava


def render_mreze_module(current_user: dict):
    st.markdown("""
    <div style="background-color: #0f766e; color: white; padding: 18px 24px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0; font-size: 1.5rem;">🗺️ IZRAVNAVA 1D/2D GEODETSKIH MREŽ IZ TERENSKIH MERITEV</h2>
        <p style="color: #ccfbf1; margin: 4px 0 0 0; font-size: 0.95rem;">
            Nivelmanske višinske mreže • Poligonske in girusne smeri • Dolžine • Samodejna linearizacija opazovanj
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_1d, tab_2d, tab_info = st.tabs([
        "📏 1. Geometrični nivelman (1D Višinska mreža)",
        "📐 2. Tahimetrična 2D mreža (Smeri in dolžine)",
        "ℹ️ O modulu in matematična osnova"
    ])

    with tab_1d:
        st.subheader("Izravnava nivelmanske mreže po MNK")
        st.markdown("""
        Vnesite višinske razlike $\Delta h_{ij}$ med točkami in dolžine nivelmanskih potez $d_{ij}$ v kilometrih.
        Uteži opazovanj se določijo po pravilu $p_k = \frac{1}{d_k \text{ [km]}}$ oz. $p_k = \frac{1}{n_{stojišč}}$.
        """)

        # Prednastavljeni primer nivelmanske mreže
        if "nivelman_df" not in st.session_state:
            st.session_state.nivelman_reperji = pd.DataFrame({
                "Točka": ["R1", "R2", "R3"],
                "Tip": ["Dani reper (fiksen)", "Dani reper (fiksen)", "Dani reper (fiksen)"],
                "Višina H [m]": [284.1520, 286.9140, 281.4050]
            })
            st.session_state.nivelman_neznanke = pd.DataFrame({
                "Točka": ["T1", "T2"],
                "Približna višina H0 [m]": [285.2000, 283.7000]
            })
            st.session_state.nivelman_opazovanja = pd.DataFrame({
                "Od točke": ["R1", "T1", "T1", "T2", "R3"],
                "Do točke": ["T1", "T2", "R2", "R3", "T1"],
                "Izmerjena Δh [m]": [1.0530, -1.4920, 1.7080, -2.2980, 3.7910],
                "Dolžina d [km]": [0.65, 0.80, 1.20, 0.95, 1.40]
            })

        c1, c2 = st.columns([1, 1.3], gap="medium")
        with c1:
            st.markdown("##### 📌 Dani reperji (Fiksne točke)")
            st.session_state.nivelman_reperji = st.data_editor(st.session_state.nivelman_reperji, use_container_width=True, key="ed_rep")
            st.markdown("##### 📍 Nove točke (Neznane višine)")
            st.session_state.nivelman_neznanke = st.data_editor(st.session_state.nivelman_neznanke, use_container_width=True, key="ed_nez")

        with c2:
            st.markdown("##### 🔭 Merjene nivelmanske poteze (Δh)")
            st.session_state.nivelman_opazovanja = st.data_editor(st.session_state.nivelman_opazovanja, use_container_width=True, key="ed_opaz")

        if st.button("🚀 Samodejno sestavi matriko B in prenesi v MNK modul", type="primary", use_container_width=True):
            rep_dict = dict(zip(st.session_state.nivelman_reperji["Točka"], st.session_state.nivelman_reperji["Višina H [m]"]))
            nez_tocke = list(st.session_state.nivelman_neznanke["Točka"])
            nez_h0 = list(st.session_state.nivelman_neznanke["Približna višina H0 [m]"])
            nez_dict = dict(zip(nez_tocke, nez_h0))

            n_op = len(st.session_state.nivelman_opazovanja)
            u_nez = len(nez_tocke)

            B_mat = np.zeros((n_op, u_nez))
            f_vec = np.zeros(n_op)
            l_vec = np.zeros(n_op)
            stds = np.zeros(n_op)

            sigma0_km = 0.002  # 2 mm / sqrt(km)

            for i, row in st.session_state.nivelman_opazovanja.iterrows():
                od_t = str(row["Od točke"]).strip()
                do_t = str(row["Do točke"]).strip()
                dh_merjeno = float(row["Izmerjena Δh [m]"])
                d_km = max(0.01, float(row["Dolžina d [km]"]))

                h_od = rep_dict[od_t] if od_t in rep_dict else nez_dict[od_t]
                h_do = rep_dict[do_t] if do_t in rep_dict else nez_dict[do_t]
                dh_pribl = h_do - h_od

                # f = dh_merjeno - dh_pribl (oz. B*Delta = f)
                # v + B*Delta = f ==> l + v = h_do - h_od ==> v - Delta_do + Delta_od = (h_do0 - h_od0) - dh
                if do_t in nez_tocke:
                    B_mat[i, nez_tocke.index(do_t)] = -1.0
                if od_t in nez_tocke:
                    B_mat[i, nez_tocke.index(od_t)] = 1.0

                f_vec[i] = dh_pribl - dh_merjeno
                l_vec[i] = dh_merjeno
                stds[i] = sigma0_km * np.sqrt(d_km)

            # Prenos v session state MNK modula
            st.session_state.metoda = "POSREDNA"
            st.session_state.n = n_op
            st.session_state.u = u_nez
            st.session_state.c = n_op
            st.session_state.sigma0_sq = 1.0
            st.session_state.tip_utezi = "STD_ODKLONI"

            mnk_izravnava.init_default_matrices("POSREDNA", n_op, n_op, u_nez, "STD_ODKLONI")
            st.session_state.df_B.iloc[:, :] = B_mat
            st.session_state.df_f.iloc[:, 0] = f_vec
            st.session_state.df_l["Merjeno l_i"] = l_vec
            st.session_state.df_l["σ_i (odklon)"] = stds
            st.session_state.df_x0["Ime neznanke"] = [f"H_{t}" for t in nez_tocke]
            st.session_state.df_x0["Približna vrednost x₀"] = nez_h0

            mnk_izravnava.izvedi_izravnavo_jedro()
            st.success(f"Nivelmanska mreža uspešno sestavljena (n={n_op}, u={u_nez}) in prenesena v MNK modul! Preklopite na modul 1 za podroben elaborat.")

    with tab_2d:
        st.subheader("2D Geodetska mreža s koti, smermi in dolžinami")
        st.info("💡 **Funkcionalnost v pripravi:** Tukaj boste lahko uvažali tahimetrične datoteke (GSI, Leica XML, Trimble RAW) z girusnimi smermi in elektrooptično merjenimi dolžinami z avtomatskim izračunom Taylorjevih odvodov.")

        fig_prev = go.Figure()
        # Primer mreže
        pts_x = [100, 250, 400, 220]
        pts_y = [150, 320, 180, 100]
        names = ["T1 (Dani)", "T2 (Dani)", "T3 (Novi)", "T4 (Novi)"]
        fig_prev.add_trace(go.Scatter(
            x=pts_x, y=pts_y, mode="markers+text",
            marker=dict(size=12, color=["#1e3a8a", "#1e3a8a", "#ef4444", "#ef4444"]),
            text=names, textposition="top center"
        ))
        # Vizure
        edges = [(0, 1), (0, 2), (1, 2), (0, 3), (2, 3), (1, 3)]
        for e in edges:
            fig_prev.add_trace(go.Scatter(
                x=[pts_x[e[0]], pts_x[e[1]]], y=[pts_y[e[0]], pts_y[e[1]]],
                mode="lines", line=dict(color="#94a3b8", dash="dash"),
                showlegend=False
            ))
        fig_prev.update_layout(
            title="Predogled 2D mreže z opazovanji",
            xaxis_title="Y [m]", yaxis_title="X [m]",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            height=420
        )
        st.plotly_chart(fig_prev, use_container_width=True)

    with tab_info:
        st.markdown("""
        ### Matematična osnova za izravnavo geodetskih mrež
        - **Linearizacija nivelmanskih enačb:**
          $$h_j - h_i - \Delta h_{ij} = v_{ij}$$
          kjer je $h_i = h_i^{(0)} + \Delta_i$.
        - **Linearizacija dolžin:**
          $$d_{ij} = \sqrt{(X_j - X_i)^2 + (Y_j - Y_i)^2}$$
          $$\\frac{\partial d}{\partial X_j} = \frac{X_j - X_i}{d_{ij}^{(0)}}, \quad \\frac{\partial d}{\partial Y_j} = \frac{Y_j - Y_i}{d_{ij}^{(0)}}$$
        - **Linearizacija smernih kotov $\\nu_{ij}$:**
          $$\\nu_{ij} = \\arctan\\left(\\frac{Y_j - Y_i}{X_j - X_i}\\right)$$
        """)
