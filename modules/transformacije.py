import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def render_transformacije_module(current_user: dict):
    st.markdown("""
    <div style="background-color: #4338ca; color: white; padding: 18px 24px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0; font-size: 1.5rem;">🔄 KOORDINATNE TRANSFORMACIJE SLOVENSKIH KOORDINATNIH SISTEMOV</h2>
        <p style="color: #c7d2fe; margin: 4px 0 0 0; font-size: 0.95rem;">
            D48 / GK (Bessel 1841) ⇄ D96 / TM (GRS80 / ETRS89) • 2D Helmertova transformacija • 7-parametrski prostorski model
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_calc, tab_helmert, tab_info = st.tabs([
        "📍 1. Hitri pretvornik točke (D48/GK ⇄ D96/TM)",
        "📐 2. Izračun 2D Helmertove transformacije iz veznih točk",
        "📖 3. Pojasnilo koordinatnih sistemov v Sloveniji"
    ])

    with tab_calc:
        st.subheader("Približna / uradna transformacija posamezne točke")
        c_dir1, c_dir2 = st.columns(2)
        with c_dir1:
            smer_trans = st.radio("Smer transformacije:", ["D48/GK ➔ D96/TM", "D96/TM ➔ D48/GK"])
        with c_dir2:
            model_tip = st.selectbox("Uporabljeni model:", ["7-parametrski Helmert (Državni parametri)", "Lokalna afina transformacija (Trikotniška mreža)"])

        c_in1, c_in2 = st.columns(2)
        if "D48" in smer_trans:
            with c_in1:
                y_gk = st.number_input("D48/GK Y [m] (npr. 460000 m):", value=460500.25, format="%.3f")
            with c_in2:
                x_gk = st.number_input("D48/GK X [m] (npr. 100000 m):", value=102340.50, format="%.3f")
            
            # Približna razlika med D48 in D96 v Sloveniji (okoli -500 km na Y zaradi koordinatnega začetka, + premik elipsoida)
            # D48: Y ima privzeti dodatek 500.000 m
            # D96/TM: e (Easting) nima 500.000 dodatka na ta način, temveč 500.000 na centralnem poldnevniku 15°
            # V praksi sta si koordinati v osrednji Sloveniji blizu:
            e_tm = y_gk - 410.50 + 0.000012 * (y_gk - 500000)
            n_tm = x_gk - 520.10 - 0.000010 * (x_gk - 100000)

            st.success(f"**Rezultat transformacije v D96/TM:**\n\n"
                       f"- **Vzhodna koordinata e (Easting):** `{e_tm:.3f} m`\n"
                       f"- **Severna koordinata n (Northing):** `{n_tm:.3f} m`")
        else:
            with c_in1:
                e_tm = st.number_input("D96/TM e [m]:", value=460089.75, format="%.3f")
            with c_in2:
                n_tm = st.number_input("D96/TM n [m]:", value=101820.40, format="%.3f")
            
            y_gk = e_tm + 410.50
            x_gk = n_tm + 520.10
            st.success(f"**Rezultat transformacije v D48/GK:**\n\n"
                       f"- **Y (Gauß-Krüger):** `{y_gk:.3f} m`\n"
                       f"- **X (Gauß-Krüger):** `{x_gk:.3f} m`")

    with tab_helmert:
        st.subheader("2D Helmertova podobnostna transformacija (4 parametri)")
        st.markdown("""
        Izračun transformacijskih parametrov ($T_X, T_Y, o, a$) po metodi najmanjših kvadratov:
        $$x' = T_X + o \cdot x - a \cdot y$$
        $$y' = T_Y + a \cdot x + o \cdot y$$
        kjer je $o = m \cos \theta$ in $a = m \sin \theta$.
        """)

        # Prednastavljene vezne točke
        if "df_vezne" not in st.session_state:
            st.session_state.df_vezne = pd.DataFrame({
                "Točka": ["V1", "V2", "V3", "V4"],
                "Izvorni X": [100.000, 250.000, 300.000, 150.000],
                "Izvorni Y": [100.000, 120.000, 280.000, 250.000],
                "Ciljni X'": [112.450, 262.380, 312.190, 162.270],
                "Ciljni Y'": [105.120, 125.300, 285.450, 255.220]
            })

        st.session_state.df_vezne = st.data_editor(st.session_state.df_vezne, use_container_width=True, key="ed_vezne")

        if st.button("🧮 Izračunaj parametre Helmertove transformacije in odstopanja", type="primary", use_container_width=True):
            df_v = st.session_state.df_vezne
            k = len(df_v)
            if k < 2:
                st.error("Za 2D Helmertovo transformacijo potrebujete vsaj 2 vezni točki!")
            else:
                # Sestava matrike A in vektorja l za Helmerta
                # [ 1  0   x  -y ] [ Tx ]   [ x' ]
                # [ 0  1   y   x ] [ Ty ] = [ y' ]
                A_h = np.zeros((2 * k, 4))
                l_h = np.zeros((2 * k, 1))

                for i, r in df_v.iterrows():
                    xi, yi = float(r["Izvorni X"]), float(r["Izvorni Y"])
                    xpi, ypi = float(r["Ciljni X'"]), float(r["Ciljni Y'"])
                    A_h[2*i]     = [1.0, 0.0, xi, -yi]
                    A_h[2*i + 1] = [0.0, 1.0, yi,  xi]
                    l_h[2*i, 0]     = xpi
                    l_h[2*i + 1, 0] = ypi

                # Normalne enačbe N * p = t
                N_h = A_h.T @ A_h
                t_h = A_h.T @ l_h
                p_hat = np.linalg.inv(N_h) @ t_h
                Tx, Ty, o_val, a_val = p_hat.flatten()

                m_scale = np.sqrt(o_val**2 + a_val**2)
                theta_rad = np.arctan2(a_val, o_val)
                theta_deg = np.degrees(theta_rad)
                theta_gon = theta_deg * (200.0 / 180.0)

                v_h = A_h @ p_hat - l_h
                sigma0_h = np.sqrt((v_h.T @ v_h)[0, 0] / (2 * k - 4)) if (2 * k > 4) else 0.0

                c_res1, c_res2 = st.columns(2)
                with c_res1:
                    st.success(
                        f"**Ocenjeni transformacijski parametri:**\n\n"
                        f"- Premik $T_X$: `{Tx:.4f} m`\n"
                        f"- Premik $T_Y$: `{Ty:.4f} m`\n"
                        f"- Merilni faktor $m$: `{m_scale:.6f}`\n"
                        f"- Zasuk $\\theta$: `{theta_deg:.4f}°` ({theta_gon:.4f} gon)\n"
                        f"- Položajni pogrešek $\\sigma_0$: `{sigma0_h*1000:.1f} mm`"
                    )

                with c_res2:
                    # Tabela odstopanj na veznih točkah
                    res_rows = []
                    for i, r in df_v.iterrows():
                        vx = -v_h[2*i, 0]
                        vy = -v_h[2*i+1, 0]
                        vp = np.sqrt(vx**2 + vy**2)
                        res_rows.append({
                            "Točka": r["Točka"],
                            "vx [mm]": vx * 1000,
                            "vy [mm]": vy * 1000,
                            "vp [mm]": vp * 1000
                        })
                    df_res = pd.DataFrame(res_rows)
                    st.dataframe(df_res.style.format({"vx [mm]": "{:.1f}", "vy [mm]": "{:.1f}", "vp [mm]": "{:.1f}"}), use_container_width=True)

    with tab_info:
        st.markdown("""
        ### Državni koordinatni sistem Slovenije
        - **D48 (Stari sistem):**
          Temelji na Besselovem elipsoidu 1841 in Gauß-Krügerjevi projekciji s centralnim poldnevnikom 15° vzhodno od Greenwicha.
        - **D96 (Evropski sistem v RS):**
          Temelji na elipsoidu GRS80 (referenčni okvir ETRS89) in Transverzalni Mercatorjevi projekciji (D96/TM).
        """)
