import io
import os
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import plotly.graph_objects as go
import streamlit as st

# Zagotovitev, da je korenski imenik aplikacije vedno v sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import auth


# =============================================================================
# INICIALIZACIJA IN PRIVZETE MATRIKE
# =============================================================================
def init_default_matrices(metoda, c, n, u, tip_utezi):
    # Matrika A
    if metoda == "POSREDNA" and c == n:
        A_init = np.eye(n)
    else:
        A_init = np.zeros((c, n))
        for i in range(min(c, n)):
            A_init[i, i] = 1.0
    st.session_state.df_A = pd.DataFrame(
        A_init,
        index=[f"Enačba {i+1}" for i in range(c)],
        columns=[f"v_{j+1}" for j in range(n)]
    )

    # Matrika B
    if u > 0:
        st.session_state.df_B = pd.DataFrame(
            np.zeros((c, u)),
            index=[f"Enačba {i+1}" for i in range(c)],
            columns=[f"Δ_{k+1}" for k in range(u)]
        )
    else:
        st.session_state.df_B = pd.DataFrame()

    # Vektor prostih členov f
    st.session_state.df_f = pd.DataFrame(
        {"f (prosti člen)": np.zeros(c)},
        index=[f"Enačba {i+1}" for i in range(c)]
    )

    # Opazovanja l in diagonalne uteži / odkloni
    oznaka_u = "σ_i (odklon)" if tip_utezi == "STD_ODKLONI" else "p_i (utež)"
    st.session_state.df_l = pd.DataFrame({
        "Merjeno l_i": np.zeros(n),
        oznaka_u: np.ones(n)
    }, index=[f"l_{j+1}" for j in range(n)])

    # Polna kovariančna matrika Sigma
    st.session_state.df_Sigma = pd.DataFrame(
        np.eye(n),
        index=[f"l_{i+1}" for i in range(n)],
        columns=[f"l_{j+1}" for j in range(n)]
    )

    # Neznanke x0 in imena
    if u > 0:
        privzeta_imena = ["a"] if u == 1 else [f"X{(k//2)+1}" if k % 2 == 0 else f"Y{(k//2)+1}" for k in range(u)]
        st.session_state.df_x0 = pd.DataFrame({
            "Ime neznanke": privzeta_imena,
            "Približna vrednost x₀": np.zeros(u)
        }, index=[f"Δ_{k+1}" for k in range(u)])
    else:
        st.session_state.df_x0 = pd.DataFrame()


def init_mnk_session_state(user_profile=None):
    if "mnk_initialized" not in st.session_state:
        st.session_state.metoda = "POSREDNA"
        st.session_state.n = 6
        st.session_state.u = 1
        st.session_state.c = 6
        st.session_state.sigma0_sq = float(user_profile.get("default_sigma0_sq", 1.0) if user_profile else 1.0)
        st.session_state.alpha = float(user_profile.get("default_alpha", 0.05) if user_profile else 0.05)
        st.session_state.alpha0 = float(user_profile.get("default_alpha0", 0.001) if user_profile else 0.001)
        st.session_state.beta0 = float(user_profile.get("default_beta0", 0.80) if user_profile else 0.80)
        st.session_state.tip_utezi = "STD_ODKLONI"
        st.session_state.zadnji_rezultati = None
        init_default_matrices("POSREDNA", 6, 6, 1, "STD_ODKLONI")
        st.session_state.mnk_initialized = True


# =============================================================================
# PREDPRIPRAVLJENI ŠOLSKI PRIMERI IZ SKRIPTE
# =============================================================================
def nalozi_primer_1():
    st.session_state.metoda = "POSREDNA"
    st.session_state.c, st.session_state.n, st.session_state.u = 6, 6, 1
    st.session_state.sigma0_sq = 1.0
    st.session_state.tip_utezi = "STD_ODKLONI"
    init_default_matrices("POSREDNA", 6, 6, 1, "STD_ODKLONI")

    B_vals = [-1.0, -np.sqrt(2), -np.sqrt(3), -4.0, -20.0, -300.0]
    f_vals = [0.0, -0.2579, 0.0205, -0.1000, -1.0, 0.0]
    l_vals = [10.0, 14.4, 17.3, 40.1, 101.0, 1000.0]
    stds = [0.2, 0.24, 0.27, 0.5, 1.0, 5.0]

    st.session_state.df_B.iloc[:, 0] = B_vals
    st.session_state.df_f.iloc[:, 0] = f_vals
    st.session_state.df_l["Merjeno l_i"] = l_vals
    st.session_state.df_l["σ_i (odklon)"] = stds
    st.session_state.df_x0["Ime neznanke"] = ["stranica_a"]
    st.session_state.df_x0["Približna vrednost x₀"] = [10.0]
    st.toast("📦 Naložen Primer 1: Kocka — Posredna izravnava (str. 71–73)")


def nalozi_primer_2():
    st.session_state.metoda = "POGOJNA"
    st.session_state.c, st.session_state.n, st.session_state.u = 5, 6, 0
    st.session_state.sigma0_sq = 1.0
    st.session_state.tip_utezi = "STD_ODKLONI"
    init_default_matrices("POGOJNA", 5, 6, 0, "STD_ODKLONI")

    A_mat = [
        [1.0, -1.0/np.sqrt(2), 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, -1.0/np.sqrt(3), 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, -0.25, 0.0, 0.0],
        [20.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [300.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    ]
    l_vals = [10.0, 14.4, 17.3, 40.1, 101.0, 1000.0]
    f_vals = [
        -(10.0 - 14.4/np.sqrt(2)),
        -(10.0 - 17.3/np.sqrt(3)),
        -(10.0 - 40.1/4.0),
        -(100.0 - 101.0),
        -(1000.0 - 1000.0)
    ]
    stds = [0.2, 0.24, 0.27, 0.5, 1.0, 5.0]

    st.session_state.df_A.iloc[:, :] = A_mat
    st.session_state.df_f.iloc[:, 0] = f_vals
    st.session_state.df_l["Merjeno l_i"] = l_vals
    st.session_state.df_l["σ_i (odklon)"] = stds
    st.toast("📐 Naložen Primer 2: Kocka — Pogojna izravnava (str. 74–75)")


def nalozi_primer_3():
    st.session_state.metoda = "SPLOSNA"
    st.session_state.c, st.session_state.n, st.session_state.u = 4, 8, 2
    st.session_state.sigma0_sq = 0.0100
    st.session_state.tip_utezi = "STD_ODKLONI"
    init_default_matrices("SPLOSNA", 4, 8, 2, "STD_ODKLONI")

    a0 = 1.9863
    A_mat = [
        [-a0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -a0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -a0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -a0, 1.0]
    ]
    B_mat = [[-5.0, -1.0], [-10.0, -1.0], [-15.0, -1.0], [-20.0, -1.0]]
    f_vals = [-0.0400, 0.0914, 0.0229, -0.0457]
    l_vals = [5.0, 10.2, 10.0, 20.0, 15.0, 30.0, 20.0, 40.0]
    variances = [0.01, 0.10, 0.04, 0.08, 0.04, 0.08, 0.04, 0.08]

    st.session_state.df_A.iloc[:, :] = A_mat
    st.session_state.df_B.iloc[:, :] = B_mat
    st.session_state.df_f.iloc[:, 0] = f_vals
    st.session_state.df_l["Merjeno l_i"] = l_vals
    st.session_state.df_l["σ_i (odklon)"] = [np.sqrt(v) for v in variances]
    st.session_state.df_x0["Ime neznanke"] = ["naklon_a", "odsek_b"]
    st.session_state.df_x0["Približna vrednost x₀"] = [1.9863, 0.2286]
    st.toast("📏 Naložen Primer 3: Premica skozi 4 točke — Splošni model")


def nalozi_primer_4():
    st.session_state.metoda = "POSREDNA"
    st.session_state.c, st.session_state.n, st.session_state.u = 7, 7, 4
    st.session_state.sigma0_sq = 1.0
    st.session_state.tip_utezi = "STD_ODKLONI"
    init_default_matrices("POSREDNA", 7, 7, 4, "STD_ODKLONI")

    B_mat = [
        [-1.0,  0.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0,  0.0],
        [-0.7, -0.7,  0.0,  0.0],
        [ 0.0,  0.0, -1.0,  0.0],
        [ 0.0,  0.0,  0.0, -1.0],
        [ 1.0,  0.0, -1.0,  0.0],
        [ 0.5, -0.8, -0.5,  0.8]
    ]
    f_vals = [0.012, -0.018, 0.005, -0.009, 0.015, 0.021, -0.014]
    l_vals = [100.012, 200.018, 223.605, 150.009, 250.015, 50.021, 70.714]
    stds = [0.015, 0.020, 0.012, 0.018, 0.015, 0.010, 0.014]

    st.session_state.df_B.iloc[:, :] = B_mat
    st.session_state.df_f.iloc[:, 0] = f_vals
    st.session_state.df_l["Merjeno l_i"] = l_vals
    st.session_state.df_l["σ_i (odklon)"] = stds
    st.session_state.df_x0["Ime neznanke"] = ["X1", "Y1", "X2", "Y2"]
    st.session_state.df_x0["Približna vrednost x₀"] = [100.0, 200.0, 150.0, 250.0]
    st.toast("🗺️ Naložen Primer 4: 2D Geodetska mreža 2 točk")


# =============================================================================
# POMOŽNE FUNKCIJE ZA FORMATIRANJE IN LATEX
# =============================================================================
def format_matrika(M, naziv, predpona="  "):
    if M is None:
        return f"{naziv}: Ni definirana v tem modelu.\n"
    vrstice = [f"{naziv} ({M.shape[0]}×{M.shape[1]}):"]
    for row in M:
        vrstice.append(predpona + " ".join(f"{val:11.5f}" for val in row))
    return "\n".join(vrstice) + "\n"


def matrix_to_latex(M: np.ndarray, decimals: int = 4) -> str:
    """Pretvori numpy matriko v LaTeX bmatrix obliko."""
    if M is None:
        return "% Matrika ni definirana v tem modelu"
    lines = []
    lines.append(r"\begin{bmatrix}")
    fmt = f"{{:.{decimals}f}}"
    for row in M:
        row_str = " & ".join(fmt.format(val) for val in row)
        lines.append(f"  {row_str} \\\\")
    lines.append(r"\end{bmatrix}")
    return "\n".join(lines)


def dataframe_to_latex_table(df: pd.DataFrame, caption: str = "Tabela rezultatov") -> str:
    """Pretvori DataFrame v lepo oblikovano LaTeX tabelo."""
    if df is None:
        return ""
    col_def = "l" + "r" * (len(df.columns) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{col_def}}}",
        r"\hline",
        " & ".join(f"\\textbf{{{c}}}" for c in df.columns) + r" \\",
        r"\hline"
    ]
    for _, row in df.iterrows():
        row_vals = []
        for val in row:
            if isinstance(val, (int, np.integer)):
                row_vals.append(str(val))
            elif isinstance(val, (float, np.floating)):
                row_vals.append(f"{val:.4f}")
            else:
                row_vals.append(str(val).replace("_", r"\_"))
        lines.append(" & ".join(row_vals) + r" \\")
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}"
    ])
    return "\n".join(lines)


# =============================================================================
# MATEMATIČNO JEDRO IZRAVNAVE IN STATISTIČNA ANALIZA (Z BAARDA ZANESLJIVOSTJO)
# =============================================================================
def izvedi_izravnavo_jedro():
    metoda = st.session_state.metoda
    c = int(st.session_state.c)
    n = int(st.session_state.n)
    u = int(st.session_state.u)
    sigma0_sq = float(st.session_state.sigma0_sq)
    alpha = float(st.session_state.alpha)
    alpha0 = float(st.session_state.alpha0)
    beta0 = float(st.session_state.get("beta0", 0.80))
    tip_u = st.session_state.tip_utezi

    if sigma0_sq <= 0:
        st.error("Referenčna varianca a-priori (σ₀²) mora biti pozitivno število (> 0).")
        return False

    A = st.session_state.df_A.to_numpy(dtype=float)
    B = st.session_state.df_B.to_numpy(dtype=float) if u > 0 else np.zeros((c, 0), dtype=float)
    f = st.session_state.df_f.to_numpy(dtype=float).reshape((c, 1))
    l_vec = st.session_state.df_l["Merjeno l_i"].to_numpy(dtype=float).reshape((n, 1))

    if u > 0:
        x0_vec = st.session_state.df_x0["Približna vrednost x₀"].to_numpy(dtype=float).reshape((u, 1))
        imena_x = [str(x).strip() if str(x).strip() else f"x_{k+1}" for k, x in enumerate(st.session_state.df_x0["Ime neznanke"])]
    else:
        x0_vec = np.zeros((0, 1))
        imena_x = []

    # Stohastični model
    if tip_u == "STD_ODKLONI":
        col_name = [col for col in st.session_state.df_l.columns if col != "Merjeno l_i"][0]
        stds = st.session_state.df_l[col_name].to_numpy(dtype=float)
        if np.any(stds <= 0):
            st.error("Vsi standardni odkloni opazovanj σ_i morajo biti strogo pozitivni (> 0).")
            return False
        Sigma = np.diag(stds ** 2)
        Q = Sigma / sigma0_sq
        P = np.linalg.inv(Q)
    elif tip_u == "DIAGONALNA_P":
        col_name = [col for col in st.session_state.df_l.columns if col != "Merjeno l_i"][0]
        ps = st.session_state.df_l[col_name].to_numpy(dtype=float)
        if np.any(ps <= 0):
            st.error("Vse uteži opazovanj p_i morajo biti strogo pozitivne (> 0).")
            return False
        P = np.diag(ps)
        Q = np.linalg.inv(P)
        Sigma = sigma0_sq * Q
    else:
        Sigma = st.session_state.df_Sigma.to_numpy(dtype=float)
        if not np.allclose(Sigma, Sigma.T, atol=1e-8):
            st.error("Kovariančna matrika Σ mora biti simetrična (σ_ij = σ_ji)!")
            return False
        if np.any(np.linalg.eigvalsh(Sigma) <= 0):
            st.error("Kovariančna matrika Σ ni pozitivno definitna!")
            return False
        Q = Sigma / sigma0_sq
        P = np.linalg.inv(Q)

    # Reševanje glede na izbrani model
    N = None
    if metoda == "POSREDNA":
        if u == 0 or c != n or n <= u:
            st.error(f"Neveljavne dimenzije za posredno izravnavo (zahtevano: u >= 1, c = n, n > u; trenutno n={n}, c={c}, u={u}).")
            return False
        if not np.allclose(A, np.eye(n), atol=1e-6):
            if np.linalg.matrix_rank(A) < n:
                st.error("Matrika A ni obrnljiva! Uporabite SPLOŠNO ali POGOJNO izravnavo.")
                return False
            A_inv = np.linalg.inv(A)
            B = A_inv @ B
            f = A_inv @ f
            A = np.eye(n)

        r = n - u
        N = B.T @ P @ B
        if np.linalg.matrix_rank(N) < u:
            st.error("Matrika normalnih enačb N = Bᵀ·P·B je singularna! Preverite koeficiente v matriki B.")
            return False
        Q_dd = np.linalg.inv(N)
        t = B.T @ P @ f
        Delta = Q_dd @ t
        v = f - B @ Delta
        l_hat = l_vec + v
        x_hat = x0_vec + Delta
        Q_vv = Q - B @ Q_dd @ B.T
        Q_ll_hat = B @ Q_dd @ B.T
        R_mat = Q_vv @ P
        k_kor = None

    elif metoda == "POGOJNA":
        if c >= n:
            st.error(f"Število pogojnih enačb (c={c}) mora biti manjše od števila opazovanj (n={n})!")
            return False
        if np.linalg.matrix_rank(A) < c:
            st.error("Matrika A nima polnega ranga! Preverite linearno neodvisnost pogojev.")
            return False
        r = c
        Qe = A @ Q @ A.T
        Pe = np.linalg.inv(Qe)
        k_kor = Pe @ f
        v = Q @ A.T @ k_kor
        l_hat = l_vec + v
        Q_dd, Delta, x_hat = None, None, None
        Q_vv = Q @ A.T @ Pe @ A @ Q
        Q_ll_hat = Q - Q_vv
        R_mat = Q_vv @ P

    else:  # SPLOSNA
        if c <= u or c > n:
            st.error(f"Pri splošnem modelu mora veljati u < c <= n (trenutno u={u}, c={c}, n={n}).")
            return False
        r = c - u
        Qe = A @ Q @ A.T
        if np.linalg.matrix_rank(Qe) < c:
            st.error("Matrika Q_e = A·Q·Aᵀ je singularna!")
            return False
        Pe = np.linalg.inv(Qe)
        if u > 0:
            N = B.T @ Pe @ B
            if np.linalg.matrix_rank(N) < u:
                st.error("Matrika normalnih enačb N = Bᵀ·P_e·B je singularna!")
                return False
            Q_dd = np.linalg.inv(N)
            t = B.T @ Pe @ f
            Delta = Q_dd @ t
            x_hat = x0_vec + Delta
            k_kor = Pe @ (f - B @ Delta)
            Q_kk = Pe @ (np.eye(c) - B @ Q_dd @ B.T @ Pe)
        else:
            Q_dd, Delta, x_hat = None, None, None
            k_kor = Pe @ f
            Q_kk = Pe
        v = Q @ A.T @ k_kor
        l_hat = l_vec + v
        Q_vv = Q @ A.T @ Q_kk @ A @ Q
        Q_ll_hat = Q - Q_vv
        R_mat = Q_vv @ P

    # Statistika modela
    vTPv = float((v.T @ P @ v)[0, 0])
    sigma0_hat_sq = vTPv / r

    Sigma_vv_priori = sigma0_sq * Q_vv
    Sigma_ll_hat_priori = sigma0_sq * Q_ll_hat
    Sigma_dd_priori = sigma0_sq * Q_dd if Q_dd is not None else None
    Sigma_ll_hat_post = sigma0_hat_sq * Q_ll_hat
    Sigma_dd_post = sigma0_hat_sq * Q_dd if Q_dd is not None else None

    # Globalni model test (Chi-kvadrat)
    Y_stat = (r * sigma0_hat_sq) / sigma0_sq
    chi2_low = stats.chi2.ppf(alpha / 2.0, r)
    chi2_high = stats.chi2.ppf(1.0 - alpha / 2.0, r)
    globalni_sprejet = (chi2_low <= Y_stat <= chi2_high)

    ci_var_low = (r * sigma0_hat_sq) / chi2_high
    ci_var_high = (r * sigma0_hat_sq) / chi2_low

    # Kritične vrednosti za Data Snooping & Tau test
    w_crit = stats.norm.ppf(1.0 - alpha0 / 2.0)
    if r > 1:
        t_val = stats.t.ppf(1.0 - alpha0 / 2.0, r - 1)
        tau_crit = (np.sqrt(r) * t_val) / np.sqrt(r - 1.0 + t_val**2)
    else:
        tau_crit = np.nan

    # Zanesljivost po Baardi (Internal & External Reliability - MDB & Delta_bar)
    # delta_0 = w_crit(alpha_0) + z(beta_0)
    z_beta = stats.norm.ppf(beta0)
    delta0 = w_crit + z_beta

    r_diag = np.diag(R_mat)
    r_avg = r / n

    # Korelacijska matrika neznank R_xx
    R_xx = None
    if Q_dd is not None and u > 0:
        std_diag = np.sqrt(np.maximum(0.0, np.diag(Q_dd)))
        outer_std = np.outer(std_diag, std_diag)
        outer_std[outer_std == 0] = 1.0
        R_xx = Q_dd / outer_std
        np.fill_diagonal(R_xx, 1.0)

    # Sestava tekstovnega elaborata
    out = []
    sep = "═" * 100
    subsep = "─" * 100
    out.append(sep)
    out.append(" GEODETSKI IZRAVNALNI RAČUN PO METODI NAJMANJŠIH KVADRATOV (MNK)")
    out.append(f" Izbrani model izravnave: {metoda} IZRAVNAVA")
    out.append(sep)
    out.append("\n1. PARAMETRI MATEMATIČNEGA IN STOHASTIČNEGA MODELA")
    out.append(subsep)
    out.append(f"  Število vseh opazovanj (n)               : {n}")
    out.append(f"  Število enačb v modelu (c)               : {c}")
    out.append(f"  Število neznank v modelu (u)             : {u}")
    out.append(f"  Število nadštevilnih opazovanj (r)       : {r}   (prostostne stopnje)")
    out.append(f"  Povprečna nadštevilnost modela (r̄ = r/n) : {r_avg:.4f} ({r_avg*100:.1f} %)")
    out.append(f"  Vsota diag(R) = Σr_i (kontrola = r)      : {np.sum(r_diag):.4f}")
    out.append(f"  Vsota uteženih kvadratov (vᵀ·P·v)        : {vTPv:.6f}")
    out.append(f"  Referenčna varianca a-priori (σ₀²)       : {sigma0_sq:.6f}   (σ₀ = {np.sqrt(sigma0_sq):.5f})")
    out.append(f"  Referenčna varianca a-posteriori (σ̂₀²)   : {sigma0_hat_sq:.6f}   (σ̂₀ = {np.sqrt(max(0, sigma0_hat_sq)):.5f})")
    out.append(f"  {(1-alpha)*100:.1f}% interval zaupanja za varianco σ₀²   : [{ci_var_low:.6f}, {ci_var_high:.6f}]")
    out.append(f"  Baarda parameter necentralnosti δ₀       : {delta0:.4f}  (α₀={alpha0}, 1-β₀={beta0*100:.0f}%)")

    df_neznanke_res = None
    if Delta is not None and u > 0:
        out.append("\n\n2. OCENJENE VREDNOSTI NEZNANK (x̂ = x₀ + Δ) IN NJIHOVA NATANČNOST")
        out.append(subsep)
        t_95 = stats.t.ppf(1.0 - alpha / 2.0, r)
        out.append(f"  {'Neznanka':<12} {'Pribl. x₀':>12} {'Popravek Δ':>13} {'Ocenjena x̂':>14} {'σ_x̂ (priori)':>14} {'σ̂_x̂ (post.)':>14} {'±CI (95% post)':>15}")
        out.append("  " + "-" * 92)
        rows_nez = []
        for k in range(u):
            q_kk = max(0.0, Q_dd[k, k])
            s_pri = np.sqrt(sigma0_sq * q_kk)
            s_pos = np.sqrt(sigma0_hat_sq * q_kk)
            ci_pol = t_95 * s_pos
            oznaka = f"{imena_x[k]} (Δ_{k+1})"
            out.append(f"  {oznaka:<12} {x0_vec[k,0]:12.5f} {Delta[k,0]:13.6f} {x_hat[k,0]:14.5f} {s_pri:14.5f} {s_pos:14.5f} {'±' + f'{ci_pol:.5f}':>15}")
            rows_nez.append({
                "Neznanka": oznaka,
                "Pribl. x₀": x0_vec[k, 0],
                "Popravek Δ": Delta[k, 0],
                "Ocenjena x̂": x_hat[k, 0],
                "σ_x̂ (priori)": s_pri,
                "σ̂_x̂ (post.)": s_pos,
                "±CI (95%)": ci_pol
            })
        df_neznanke_res = pd.DataFrame(rows_nez)

    out.append("\n\n3. ANALIZA OPAZOVANJ, DATA SNOOPING IN BAARDA ZANESLJIVOST (MDB)")
    out.append(subsep)
    out.append(f"  Kritična vrednost Baarda Data Snooping (α₀={alpha0}, N(0,1)) : w_krit = {w_crit:.3f}")
    out.append(f"  Kritična vrednost Pope Tau-test (α₀={alpha0}, τ(r={r}))      : τ_krit = {tau_crit:.3f}")
    out.append(f"  Parameter zanesljivosti Baarda δ₀ (moč testa 1-β₀={beta0:.2f})  : δ₀ = {delta0:.3f}\n")
    glava_op = (
        f"  {'Op.':<5} {'Merjeno l':>11} {'Popr. v':>11} {'Izravn. l̂':>11} "
        f"{'σ_l':>8} {'σ_v(pri)':>9} {'Nadšt. r_i':>10} "
        f"{'|w_i|':>8} {'|τ_i|':>8} {'MDB ∇₀l':>10} {'Zun. δ̄_i':>9} {'Status'}"
    )
    out.append(glava_op)
    out.append("  " + "-" * 120)

    sumljiva_opazovanja = []
    rows_op = []
    for i in range(n):
        li = l_vec[i, 0]
        vi = v[i, 0]
        li_hat = l_hat[i, 0]
        sig_l = np.sqrt(max(0.0, sigma0_sq * Q[i, i]))
        sig_l_hat = np.sqrt(max(0.0, Sigma_ll_hat_priori[i, i]))
        sig_v_pri = np.sqrt(max(0.0, Sigma_vv_priori[i, i]))
        sig_v_pos = np.sqrt(max(0.0, sigma0_hat_sq * Q_vv[i, i]))
        ri = r_diag[i]

        wi = abs(vi) / sig_v_pri if sig_v_pri > 1e-12 else 0.0
        tau_i = abs(vi) / sig_v_pos if sig_v_pos > 1e-12 else 0.0
        nabla_li_est = -vi / ri if ri > 1e-6 else np.nan

        # Baarda MDB & Zunanja zanesljivost
        if ri > 1e-6:
            mdb_i = sig_l * (delta0 / np.sqrt(ri))
            delta_bar_i = np.sqrt((1.0 - ri) / ri) * delta0
        else:
            mdb_i = np.nan
            delta_bar_i = np.nan

        status = "OK"
        if wi > w_crit and tau_i > (tau_crit if not np.isnan(tau_crit) else 999):
            status = "GROBI POGREŠEK (w & τ)!"
            sumljiva_opazovanja.append((i+1, wi, tau_i, nabla_li_est, ri))
        elif wi > w_crit:
            status = "Outlier (w-test)!"
            sumljiva_opazovanja.append((i+1, wi, tau_i, nabla_li_est, ri))
        elif not np.isnan(tau_crit) and tau_i > tau_crit:
            status = "Outlier (τ-test)!"
            sumljiva_opazovanja.append((i+1, wi, tau_i, nabla_li_est, ri))
        elif ri < 0.10:
            status = "Kritično slaba nadšt. (r_i<0.1)"
        elif ri < 0.30:
            status = "Slaba nadšt. (r_i<0.3)"

        mdb_str = f"{mdb_i:10.4f}" if not np.isnan(mdb_i) else "       Ni"
        dbar_str = f"{delta_bar_i:9.3f}" if not np.isnan(delta_bar_i) else "      Ni"
        out.append(
            f"  {'l_'+str(i+1):<5} {li:11.4f} {vi:11.5f} {li_hat:11.4f} "
            f"{sig_l:8.4f} {sig_v_pri:9.4f} {ri:10.4f} "
            f"{wi:8.3f} {tau_i:8.3f} {mdb_str} {dbar_str} {status}"
        )
        rows_op.append({
            "Op.": f"l_{i+1}",
            "Merjeno l": li,
            "Popravek v": vi,
            "Izravnano l̂": li_hat,
            "σ_l": sig_l,
            "σ_l̂ (pri)": sig_l_hat,
            "σ_v (pri)": sig_v_pri,
            "Nadšt. r_i": ri,
            "|w_i|": wi,
            "|τ_i|": tau_i,
            "Ocena ∇l_i": nabla_li_est,
            "MDB ∇₀l_i": mdb_i,
            "Zun. zan. δ̄_i": delta_bar_i,
            "Status": status
        })
    df_opazovanja_res = pd.DataFrame(rows_op)

    out.append("\n\n4. STATISTIČNI TESTI IN INTERPRETACIJA MODELA")
    out.append(subsep)
    out.append(f"  Testna statistika globalnega testa Y = {Y_stat:.4f}")
    out.append(f"  Kritični interval χ² (α={alpha}): [{chi2_low:.4f}, {chi2_high:.4f}]")
    out.append("  ODLOČITEV: " + ("✔ H₀ NE ZAVRNEMO (Model in natančnost meritev sta usklajena)" if globalni_sprejet else "✖ H₀ ZAVRNEMO (Model ni usklajen ali prisoten grobi pogrešek)!"))

    out.append("\n\n5. MATRIKE KOFAKTORJEV (Q), KOVARIANČNE MATRIKE (Σ) IN MATRIKA NADŠTEVILNOSTI (R)")
    out.append(subsep)
    if N is not None:
        out.append(format_matrika(N, "Matrika normalnih enačb N"))
    if Q_dd is not None:
        out.append(format_matrika(Q_dd, "Matrika kofaktorjev neznank Q_ΔΔ = N⁻¹"))
        out.append(format_matrika(Sigma_dd_priori, "Kovariančna matrika neznank Σ_ΔΔ (a-priori)"))
        out.append(format_matrika(Sigma_dd_post, "Kovariančna matrika neznank Σ̂_ΔΔ (a-posteriori)"))
    out.append(format_matrika(Q_vv, "Matrika kofaktorjev popravkov opazovanj Q_vv"))
    out.append(format_matrika(Q_ll_hat, "Matrika kofaktorjev izravnanih opazovanj Q_l̂l̂"))
    out.append(format_matrika(R_mat, "Matrika nadštevilnosti R = Q_vv · P"))

    elaborat_txt = "\n".join(out)

    st.session_state.zadnji_rezultati = {
        "metoda": metoda, "c": c, "n": n, "u": u, "r": r,
        "sigma0_sq": sigma0_sq, "sigma0_hat_sq": sigma0_hat_sq, "vTPv": vTPv,
        "Y_stat": Y_stat, "chi2_low": chi2_low, "chi2_high": chi2_high,
        "globalni_sprejet": globalni_sprejet,
        "w_crit": w_crit, "tau_crit": tau_crit, "delta0": delta0,
        "N": N, "P": P, "A": A, "B": B, "f": f,
        "Q_dd": Q_dd, "Q_ll_hat": Q_ll_hat, "Q_vv": Q_vv, "R_mat": R_mat, "R_xx": R_xx,
        "x_hat": x_hat, "l_hat": l_hat, "imena_x": imena_x,
        "df_neznanke": df_neznanke_res, "df_opazovanja": df_opazovanja_res,
        "sumljiva_opazovanja": sumljiva_opazovanja,
        "elaborat_txt": elaborat_txt
    }
    return True


def skaliraj_sigma_in_ponovi():
    res = st.session_state.zadnji_rezultati
    if res is None:
        return
    faktor = res["sigma0_hat_sq"] / res["sigma0_sq"]
    if faktor <= 0:
        return
    tip = st.session_state.tip_utezi
    if tip == "STD_ODKLONI":
        col_name = [col for col in st.session_state.df_l.columns if col != "Merjeno l_i"][0]
        st.session_state.df_l[col_name] = st.session_state.df_l[col_name] * np.sqrt(faktor)
    elif tip == "DIAGONALNA_P":
        col_name = [col for col in st.session_state.df_l.columns if col != "Merjeno l_i"][0]
        st.session_state.df_l[col_name] = st.session_state.df_l[col_name] / faktor
    else:
        st.session_state.df_Sigma = st.session_state.df_Sigma * faktor

    izvedi_izravnavo_jedro()
    st.toast(f"🔄 Stohastični model skaliran s faktorjem {faktor:.4f}. Globalni test je zdaj usklajen (Y = r)!")


# =============================================================================
# ELIPSE POGREŠKOV: IZRAČUNI
# =============================================================================
def izracunaj_parametre_2x2(s_x2, s_y2, s_xy, p_zaup):
    s_x = np.sqrt(max(0.0, s_x2))
    s_y = np.sqrt(max(0.0, s_y2))
    rho = s_xy / (s_x * s_y) if (s_x * s_y) > 0 else 0.0

    koren = np.sqrt(((s_x2 - s_y2) ** 2) / 4.0 + s_xy ** 2)
    s_xp = np.sqrt((s_x2 + s_y2) / 2.0 + koren)
    s_yp = np.sqrt(max(0.0, (s_x2 + s_y2) / 2.0 - koren))

    two_theta_rad = np.arctan2(2.0 * s_xy, s_x2 - s_y2)
    if two_theta_rad < 0:
        two_theta_rad += 2.0 * np.pi
    theta_rad = 0.5 * two_theta_rad
    theta_deg = np.degrees(theta_rad)
    theta_gon = theta_deg * (200.0 / 180.0)

    c_mult = np.sqrt(-2.0 * np.log(1.0 - p_zaup))
    a_conf = c_mult * s_xp
    b_conf = c_mult * s_yp
    s_p = np.sqrt(s_x2 + s_y2)  # Helmertov položajni pogrešek
    return s_x, s_y, rho, s_xp, s_yp, theta_rad, theta_deg, theta_gon, c_mult, a_conf, b_conf, s_p


def get_ellipse_points(cx, cy, a, b, theta_rad, n_pts=120):
    """Izračuna točke na obodu rotirane elipse."""
    t = np.linspace(0, 2 * np.pi, n_pts)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    cos_th = np.cos(theta_rad)
    sin_th = np.sin(theta_rad)
    x = cx + a * cos_t * cos_th - b * sin_t * sin_th
    y = cy + a * cos_t * sin_th + b * sin_t * cos_th
    return x, y


# =============================================================================
# HITRI UVOZI IN IZVOZI MATRIK (EXCEL, BESEDILO, JSON)
# =============================================================================
def parse_matrix_from_text(raw_text: str):
    """Razčleni poljubno besedilo ločeno s presledki, tabulatorji, podpičji ali vejicami."""
    lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
    if not lines:
        return None
    matrix = []
    for line in lines:
        # Zamenjaj tabulatorje, podpičja ali več presledkov z enojnim presledkom
        clean = re.sub(r"[\t;,]+", " ", line).strip()
        parts = clean.split()
        row = [float(p.replace(",", ".")) for p in parts]
        matrix.append(row)
    return np.array(matrix, dtype=float)


def export_matrices_to_excel_bytes() -> bytes:
    """Izvozi vse trenutne matrike v Excelov zvezek z več zavihki."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        st.session_state.df_A.to_excel(writer, sheet_name="Matrika_A")
        if not st.session_state.df_B.empty:
            st.session_state.df_B.to_excel(writer, sheet_name="Matrika_B")
        st.session_state.df_f.to_excel(writer, sheet_name="Vektor_f")
        st.session_state.df_l.to_excel(writer, sheet_name="Opazovanja_l")
        st.session_state.df_Sigma.to_excel(writer, sheet_name="Kovarianca_Sigma")
        if not st.session_state.df_x0.empty:
            st.session_state.df_x0.to_excel(writer, sheet_name="Neznanke_x0")
    return output.getvalue()


def import_matrices_from_excel(uploaded_file):
    """Uvozi matrike iz Excelovega zvezka."""
    try:
        xls = pd.ExcelFile(uploaded_file)
        if "Matrika_A" in xls.sheet_names:
            st.session_state.df_A = pd.read_excel(xls, sheet_name="Matrika_A", index_col=0)
        if "Matrika_B" in xls.sheet_names:
            st.session_state.df_B = pd.read_excel(xls, sheet_name="Matrika_B", index_col=0)
        if "Vektor_f" in xls.sheet_names:
            st.session_state.df_f = pd.read_excel(xls, sheet_name="Vektor_f", index_col=0)
        if "Opazovanja_l" in xls.sheet_names:
            st.session_state.df_l = pd.read_excel(xls, sheet_name="Opazovanja_l", index_col=0)
        if "Kovarianca_Sigma" in xls.sheet_names:
            st.session_state.df_Sigma = pd.read_excel(xls, sheet_name="Kovarianca_Sigma", index_col=0)
        if "Neznanke_x0" in xls.sheet_names:
            st.session_state.df_x0 = pd.read_excel(xls, sheet_name="Neznanke_x0", index_col=0)

        # Posodobi dimenzije
        st.session_state.c = len(st.session_state.df_f)
        st.session_state.n = len(st.session_state.df_l)
        st.session_state.u = len(st.session_state.df_x0) if not st.session_state.df_x0.empty else 0
        return True, "Uspešno uvožene vse tabele iz Excela!"
    except Exception as e:
        return False, f"Napaka pri uvozu Excela: {str(e)}"


# =============================================================================
# GLAVNI IZRIS MODULA MNK
# =============================================================================
def render_mnk_module(current_user: dict):
    username = current_user["username"]
    user_profile = auth.get_user_profile(username)
    init_mnk_session_state(user_profile)

    # -------------------------------------------------------------------------
    # UPRAVITELJ SHRANJENIH PROJEKTOV (ZGORAJ)
    # -------------------------------------------------------------------------
    with st.expander("💾 Moje shranjene izravnave (Upravitelj projektov & šolski primeri)", expanded=False):
        c_p1, c_p2 = st.columns([1.2, 1.8], gap="medium")

        saved_adjs = auth.get_user_saved_adjustments(username)
        with c_p1:
            st.markdown("##### 📁 Shranjevanje trenutne izravnave")
            proj_name = st.text_input("Ime projekta / naloge:", placeholder="npr. Vaje MNK - Naloga 3", key="mnk_save_name")
            proj_note = st.text_input("Opomba / opis (opcijsko):", placeholder="npr. Nivelmanska mreža Kranj z 2 navezovalnima točkama", key="mnk_save_note")

            if st.button("💾 Shrani trenutno izravnavo v moj profil", use_container_width=True, type="primary"):
                if not proj_name.strip():
                    st.error("Vnesite ime projekta!")
                else:
                    proj_id = re.sub(r"[^\w\-]+", "_", proj_name.strip().lower())
                    state_dict = {
                        "metoda": st.session_state.metoda,
                        "n": st.session_state.n, "u": st.session_state.u, "c": st.session_state.c,
                        "sigma0_sq": st.session_state.sigma0_sq,
                        "alpha": st.session_state.alpha, "alpha0": st.session_state.alpha0,
                        "beta0": st.session_state.get("beta0", 0.80),
                        "tip_utezi": st.session_state.tip_utezi,
                        "df_A": st.session_state.df_A, "df_B": st.session_state.df_B,
                        "df_f": st.session_state.df_f, "df_l": st.session_state.df_l,
                        "df_Sigma": st.session_state.df_Sigma, "df_x0": st.session_state.df_x0
                    }
                    auth.save_adjustment_project(username, proj_id, proj_name, proj_note, state_dict)
                    st.success(f"Projekt '{proj_name}' je varno shranjen v vaš profil!")
                    st.rerun()

        with c_p2:
            st.markdown("##### 📂 Naloži ali uredi obstoječe izravnave")
            if not saved_adjs:
                st.info("Trenutno še nimate shranjenih projektov. Shranite trenutno izravnavo ali izberite šolski primer spodaj!")
            else:
                izbire = []
                kljuci = list(saved_adjs.keys())
                for k in kljuci:
                    rec = saved_adjs[k]
                    izbire.append(f"{rec['name']} [{rec['metoda']}, n={rec['n']}, u={rec['u']}] ({rec['updated_at']})")

                izbran_idx = st.selectbox("Izberite shranjen projekt:", range(len(izbire)), format_func=lambda i: izbire[i])
                izbran_key = kljuci[izbran_idx]
                izbran_rec = saved_adjs[izbran_key]

                if izbran_rec.get("note"):
                    st.caption(f"📝 **Opomba:** {izbran_rec['note']}")

                col_btn_l, col_btn_u, col_btn_d = st.columns(3)
                with col_btn_l:
                    if st.button("📂 Naloži projekt", use_container_width=True):
                        st.session_state.metoda = izbran_rec["metoda"]
                        st.session_state.n = izbran_rec["n"]
                        st.session_state.u = izbran_rec["u"]
                        st.session_state.c = izbran_rec["c"]
                        st.session_state.sigma0_sq = izbran_rec["sigma0_sq"]
                        st.session_state.alpha = izbran_rec["alpha"]
                        st.session_state.alpha0 = izbran_rec["alpha0"]
                        st.session_state.beta0 = izbran_rec.get("beta0", 0.80)
                        st.session_state.tip_utezi = izbran_rec["tip_utezi"]

                        mats = izbran_rec["matrices"]
                        st.session_state.df_A = auth.deserialize_dataframe(mats["df_A"])
                        st.session_state.df_B = auth.deserialize_dataframe(mats["df_B"])
                        st.session_state.df_f = auth.deserialize_dataframe(mats["df_f"])
                        st.session_state.df_l = auth.deserialize_dataframe(mats["df_l"])
                        st.session_state.df_Sigma = auth.deserialize_dataframe(mats["df_Sigma"])
                        st.session_state.df_x0 = auth.deserialize_dataframe(mats["df_x0"])

                        izvedi_izravnavo_jedro()
                        st.toast(f"Projekt '{izbran_rec['name']}' je naložen in izračunan!")
                        st.rerun()

                with col_btn_u:
                    if st.button("🔄 Prepiši z novim", use_container_width=True):
                        state_dict = {
                            "metoda": st.session_state.metoda,
                            "n": st.session_state.n, "u": st.session_state.u, "c": st.session_state.c,
                            "sigma0_sq": st.session_state.sigma0_sq,
                            "alpha": st.session_state.alpha, "alpha0": st.session_state.alpha0,
                            "beta0": st.session_state.get("beta0", 0.80),
                            "tip_utezi": st.session_state.tip_utezi,
                            "df_A": st.session_state.df_A, "df_B": st.session_state.df_B,
                            "df_f": st.session_state.df_f, "df_l": st.session_state.df_l,
                            "df_Sigma": st.session_state.df_Sigma, "df_x0": st.session_state.df_x0
                        }
                        auth.save_adjustment_project(username, izbran_key, izbran_rec["name"], izbran_rec.get("note", ""), state_dict)
                        st.success(f"Projekt '{izbran_rec['name']}' uspešno posodobljen!")
                        st.rerun()

                with col_btn_d:
                    if st.button("🗑️ Izbriši projekt", use_container_width=True):
                        auth.delete_adjustment_project(username, izbran_key)
                        st.warning(f"Projekt '{izbran_rec['name']}' izbrisan.")
                        st.rerun()

    # -------------------------------------------------------------------------
    # ŠTIRJE GLAVNI ZAVIHKI MNK MODULA
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Izbira modela in Navodila",
        "2️⃣ Pametni vnos enačb in meritev",
        "3️⃣ Elaborat in Statistični testi",
        "4️⃣ Elipse pogreškov in Mreža"
    ])

    # =========================================================================
    # ZAVIHEK 1: NASTAVITVE MODELA IN PRIROČNIK
    # =========================================================================
    with tab1:
        col_nast, col_prirocnik = st.columns([1, 1.2], gap="large")

        with col_nast:
            st.subheader("1. korak: Izbira metode in dimenzij")
            metoda_izbira = st.radio(
                "Katero vrsto izravnave želite izvesti?",
                options=["POSREDNA", "POGOJNA", "SPLOSNA"],
                format_func=lambda x: {
                    "POSREDNA": "Posredna izravnava (v + B·Δ = f) — določitev neznank/koordinat",
                    "POGOJNA": "Pogojna izravnava (A·v = f) — le geometrijski pogoji brez neznank",
                    "SPLOSNA": "Splošni model (A·v + B·Δ = f) — več meritev in neznank v posamezni enačbi"
                }[x],
                index=["POSREDNA", "POGOJNA", "SPLOSNA"].index(st.session_state.metoda)
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                n_new = st.number_input("Št. opazovanj (n)", min_value=1, max_value=60, value=int(st.session_state.n))
            with c2:
                if metoda_izbira == "POGOJNA":
                    u_new = st.number_input("Št. neznank (u)", min_value=0, max_value=0, value=0, disabled=True)
                else:
                    u_new = st.number_input("Št. neznank (u)", min_value=1 if metoda_izbira == "POSREDNA" else 0, max_value=50, value=max(1, int(st.session_state.u)))
            with c3:
                if metoda_izbira == "POSREDNA":
                    c_new = st.number_input("Št. enačb (c = n)", min_value=int(n_new), max_value=int(n_new), value=int(n_new), disabled=True)
                else:
                    c_new = st.number_input("Št. enačb / pogojev (c)", min_value=1, max_value=60, value=min(int(st.session_state.c), int(n_new)-1 if metoda_izbira=="POGOJNA" else 60))

            st.divider()
            st.subheader("Stohastični model in statistični parametri zaupanja")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.session_state.sigma0_sq = st.number_input("Varianca a-priori (σ₀²)", value=float(st.session_state.sigma0_sq), format="%.5f")
            with s2:
                st.session_state.alpha = st.number_input("Stopnja α (Globalni χ²)", value=float(st.session_state.alpha), format="%.4f")
            with s3:
                st.session_state.alpha0 = st.number_input("Stopnja α₀ (Snooping)", value=float(st.session_state.alpha0), format="%.4f")
            with s4:
                st.session_state.beta0 = st.number_input("Moč testa (1-β₀)", value=float(st.session_state.get("beta0", 0.80)), min_value=0.50, max_value=0.99, format="%.2f")

            tip_utezi_new = st.selectbox(
                "Način podajanja natančnosti meritev:",
                options=["STD_ODKLONI", "DIAGONALNA_P", "POLNA_SIGMA"],
                format_func=lambda x: {
                    "STD_ODKLONI": "Standardni odkloni opazovanj (σ_i) [priporočeno]",
                    "DIAGONALNA_P": "Diagonalne uteži opazovanj (p_i)",
                    "POLNA_SIGMA": "Polna kovariančna matrika opazovanj (Σ)"
                }[x],
                index=["STD_ODKLONI", "DIAGONALNA_P", "POLNA_SIGMA"].index(st.session_state.tip_utezi)
            )

            if st.button("🛠️ Pripravi prazno tabelo za izbrane dimenzije", type="primary"):
                st.session_state.metoda = metoda_izbira
                st.session_state.n = int(n_new)
                st.session_state.u = int(u_new)
                st.session_state.c = int(c_new)
                st.session_state.tip_utezi = tip_utezi_new
                init_default_matrices(metoda_izbira, int(c_new), int(n_new), int(u_new), tip_utezi_new)
                st.success("Tabele v zavihku 2 so pripravljene na nove dimenzije!")

        with col_prirocnik:
            st.subheader("📖 Hitri geodetski priročnik & teorija")
            with st.expander("1. Katero metodo izravnave izbrati?", expanded=True):
                st.markdown("""
                * **🔹 A) POSREDNA IZRAVNAVA ($\mathbf{v} + \mathbf{B}\mathbf{\Delta} = \mathbf{f}$):**
                  Standardni geodetski model, ko določamo položaje točk (koordinate $X, Y, H$) iz opazovanj (dolžin, smeri, višinskih razlik). Vsako opazovanje $l_i$ dobi svojo enačbo ($c = n$). Matrika $\mathbf{A} = \mathbf{I}$.
                * **🔹 B) POGOJNA IZRAVNAVA ($\mathbf{A}\mathbf{v} = \mathbf{f}$):**
                  Uporabimo jo, ko ne računamo koordinat ($u = 0$), temveč le izravnavamo meritve z zaprtjem geometrijskih pogojev (npr. vsota kotov v trikotniku $\sum \alpha_i = 200\text{ gon}$ ali zaprtje nivelmanskega poligona).
                * **🔹 C) SPLOŠNI MODEL ($\mathbf{A}\mathbf{v} + \mathbf{B}\mathbf{\Delta} = \mathbf{f}$):**
                  Kombinirani model, ko v eni enačbi sočasno nastopa več različnih meritev in neznanke (npr. prilagajanje premice točkam s pogreški v obeh koordinatah $x$ in $y$).
                """)
            with st.expander("2. Notranja in zunanja zanesljivost (Baarda Reliability)", expanded=True):
                st.markdown("""
                * **Mejna vrednost grobega pogreška ($MDB - \nabla_0 l_i$):**
                  Predstavlja najmanjši grobi pogrešek v meritvi $l_i$, ki ga z verjetnostjo (močjo testa) $1-\beta_0$ še zanesljivo odkrijemo z Data Snoopingom:
                  $$\\nabla_0 l_i = \\sigma_{l_i} \\cdot \\frac{\\delta_0}{\\sqrt{r_i}}$$
                  Kjer je $r_i$ delna nadštevilnost meritve ($0 \\le r_i \\le 1$). Če je $r_i$ majhen ($r_i < 0.3$), je geometrija mreže šibka in grobega pogreška ne moremo odkriti!
                * **Zunanja zanesljivost ($\bar{\delta}_i$):**
                  Vpliv morebitnega neodkritega mejnega grobega pogreška na ocene neznank:
                  $$\\bar{\\delta}_i = \\sqrt{\\frac{1 - r_i}{r_i}} \\cdot \\delta_0$$
                """)
            with st.expander("3. 2D elipse pogreškov in koti v stopinjah ter gonih", expanded=False):
                st.markdown("""
                * **Glavna polos ($a$ oz. $\sigma_{x'}$):** Smer z najmanjšo natančnostjo.
                * **Mala polos ($b$ oz. $\sigma_{y'}$):** Smer z največjo natančnostjo.
                * **Helmertov položajni pogrešek:** $\sigma_P = \sqrt{\sigma_X^2 + \sigma_Y^2} = \sqrt{\sigma_{x'}^2 + \sigma_{y'}^2}$.
                * **Zasuk ($\theta$):** Prikazan v stopinjah ($^\circ$) ter gonih ($1^\circ = \\frac{200}{180}\\text{ gon}$).
                """)

    # =========================================================================
    # ZAVIHEK 2: PAMETNI VNOS ENAČB IN MERITEV
    # =========================================================================
    with tab2:
        metoda = st.session_state.metoda
        c, n, u = st.session_state.c, st.session_state.n, st.session_state.u

        if metoda == "POSREDNA":
            st.info("💡 **POSREDNA IZRAVNAVA ($\mathbf{v} + \mathbf{B}\mathbf{\Delta} = \mathbf{f}$):** Vsaka vrstica predstavlja eno meritev $l_i$. Matrika A je samodejno identiteta.")
        elif metoda == "POGOJNA":
            st.info("💡 **POGOJNA IZRAVNAVA ($\mathbf{A}\mathbf{v} = \mathbf{f}$):** V modelu ni neznank ($u=0$). Vnesite koeficiente pogojev v matriko **A** in odstopanja **f**.")
        else:
            st.info("💡 **SPLOŠNI MODEL ($\mathbf{A}\mathbf{v} + \mathbf{B}\mathbf{\Delta} = \mathbf{f}$):** Vnesite odvode po meritvah (**A**), odvode po neznankah (**B**) in proste člene (**f**).")

        # ---------------------------------------------------------------------
        # DODATNI NAČINI VNOSA (HITRO KOPIRANJE, CSV/EXCEL, KALKULATOR f)
        # ---------------------------------------------------------------------
        with st.expander("📋 Dodatni načini vnosa (Hitro kopiranje iz Excela/MATLAB-a ali CSV / Excel uvoz)", expanded=False):
            col_v1, col_v2 = st.columns([1.2, 1], gap="large")

            with col_v1:
                st.markdown("##### 📥 Hiter tekstovni vnos (Copy-Paste številk)")
                st.caption("Prilepite števila (ločena s presledki, tabulatorji ali podpičji) neposredno iz Excela ali PDF skripte:")
                cilj_matrike = st.selectbox(
                    "Ciljna tabela za prepis:",
                    ["Matrika B (odvodi po neznankah)", "Matrika A (pogojne enačbe)", "Vektor prostih členov f", "Merjena opazovanja l in standardni odkloni"]
                )
                raw_paste = st.text_area("Prilepite vsebino tukaj:", height=110, placeholder="npr.:\n-1.0 0.0\n0.0 -1.0\n-0.7 -0.7")
                if st.button("📥 Pretvori prilepljeno besedilo v tabelo"):
                    parsed = parse_matrix_from_text(raw_paste)
                    if parsed is None or parsed.size == 0:
                        st.error("Besedila ni bilo mogoče pretvoriti v števila!")
                    else:
                        if "Matrika B" in cilj_matrike:
                            if u > 0 and parsed.shape == (c, u):
                                st.session_state.df_B.iloc[:, :] = parsed
                                st.success(f"Matrika B ({c}×{u}) uspešno posodobljena!")
                            elif parsed.ndim == 2:
                                st.session_state.c, st.session_state.u = parsed.shape
                                st.session_state.df_B = pd.DataFrame(parsed, index=[f"Enačba {i+1}" for i in range(parsed.shape[0])], columns=[f"Δ_{k+1}" for k in range(parsed.shape[1])])
                                st.success(f"Matrika B uvožena s prilagojenimi dimenzijami ({parsed.shape[0]}×{parsed.shape[1]})!")
                            st.rerun()

                        elif "Matrika A" in cilj_matrike:
                            if parsed.shape == (c, n):
                                st.session_state.df_A.iloc[:, :] = parsed
                                st.success(f"Matrika A ({c}×{n}) uspešno posodobljena!")
                            elif parsed.ndim == 2:
                                st.session_state.c, st.session_state.n = parsed.shape
                                st.session_state.df_A = pd.DataFrame(parsed, index=[f"Enačba {i+1}" for i in range(parsed.shape[0])], columns=[f"v_{j+1}" for j in range(parsed.shape[1])])
                                st.success(f"Matrika A uvožena s prilagojenimi dimenzijami ({parsed.shape[0]}×{parsed.shape[1]})!")
                            st.rerun()

                        elif "Vektor prostih" in cilj_matrike:
                            vals = parsed.flatten()
                            if len(vals) == c:
                                st.session_state.df_f.iloc[:, 0] = vals
                                st.success(f"Vektor f ({c}) uspešno posodobljen!")
                            else:
                                st.session_state.c = len(vals)
                                st.session_state.df_f = pd.DataFrame({"f (prosti člen)": vals}, index=[f"Enačba {i+1}" for i in range(len(vals))])
                                st.success(f"Vektor f uvožen ({len(vals)} enačb)!")
                            st.rerun()

                        elif "Merjena opazovanja" in cilj_matrike:
                            if parsed.ndim == 2 and parsed.shape[1] >= 2:
                                if parsed.shape[0] == n:
                                    st.session_state.df_l["Merjeno l_i"] = parsed[:, 0]
                                    col_u = [col for col in st.session_state.df_l.columns if col != "Merjeno l_i"][0]
                                    st.session_state.df_l[col_u] = parsed[:, 1]
                                    st.success(f"Opazovanja l in uteži/odkloni posodobljeni ({n} vrstic)!")
                                    st.rerun()
                            elif parsed.ndim == 1 or parsed.shape[1] == 1:
                                vals = parsed.flatten()
                                if len(vals) == n:
                                    st.session_state.df_l["Merjeno l_i"] = vals
                                    st.success(f"Meritve l_i posodobljene ({n} opazovanj)!")
                                    st.rerun()

            with col_v2:
                st.markdown("##### 📁 Excel / CSV uvoz in izvoz")
                st.caption("Izvozite ali uvozite celoten nabor matrik v Excel (.xlsx):")
                excel_bytes = export_matrices_to_excel_bytes()
                st.download_button(
                    label="📥 Prenesi vse tabele v Excel (.xlsx)",
                    data=excel_bytes,
                    file_name="mnk_vhodne_matrike.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

                uploaded_excel = st.file_uploader("📤 Uvozi tabele iz Excela (.xlsx):", type=["xlsx"], key="mnk_excel_uploader")
                if uploaded_excel is not None:
                    ok, msg = import_matrices_from_excel(uploaded_excel)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

                st.divider()
                st.markdown("##### 🧮 Avtomatski izračun prostih členov")
                st.caption("Za linearne modele ($\mathbf{l} + \mathbf{v} = \mathbf{B}\hat{\mathbf{x}}$), kjer je $\mathbf{f} = \mathbf{B}\mathbf{x}_0 - \mathbf{l}$:")
                if st.button("🧮 Izračunaj vektor prostih členov f = B·x₀ - l", use_container_width=True):
                    if u > 0 and st.session_state.df_B.shape == (c, u):
                        try:
                            B_cur = st.session_state.df_B.to_numpy(dtype=float)
                            x0_cur = st.session_state.df_x0["Približna vrednost x₀"].to_numpy(dtype=float).reshape((u, 1))
                            l_cur = st.session_state.df_l["Merjeno l_i"].to_numpy(dtype=float).reshape((n, 1))
                            if c == n:
                                f_calc = (B_cur @ x0_cur) - l_cur
                                st.session_state.df_f.iloc[:, 0] = f_calc.flatten()
                                st.success("Vektor prostih členov f = B·x₀ - l je uspešno izračunan!")
                                st.rerun()
                            else:
                                st.warning("Samodejni izračun f zahteva c = n.")
                        except Exception as e:
                            st.error(f"Napaka pri izračunu: {e}")
                    else:
                        st.warning("Matrika B in vektor x₀ morata biti ustrezno definirana!")

        # ---------------------------------------------------------------------
        # OBSTOJEČI INTERAKTIVNI VNOS MATRIK (ST.DATA_EDITOR) - 100% OHRANJENO!
        # ---------------------------------------------------------------------
        prikazi_vse_A = st.checkbox("Prikaži polno matriko A tudi pri Posredni izravnavi (napredno)", value=False)
        st.subheader(f"1. Vnos matrik modela ({metoda}: c={c}, n={n}, u={u})")
        prikazi_A = (metoda in ["POGOJNA", "SPLOSNA"]) or prikazi_vse_A
        prikazi_B = (metoda in ["POSREDNA", "SPLOSNA"] and u > 0)

        cols_mat = st.columns([2 if prikazi_A else 0.1, 2 if prikazi_B else 0.1, 1])
        if prikazi_A:
            with cols_mat[0]:
                st.markdown("**Matrika A (koeficienti popravkov $v_j$)**")
                st.session_state.df_A = st.data_editor(st.session_state.df_A, use_container_width=True, key="editor_A")
        if prikazi_B:
            with cols_mat[1]:
                st.markdown("**Matrika B (koeficienti neznank $\Delta_k$)**")
                st.session_state.df_B = st.data_editor(st.session_state.df_B, use_container_width=True, key="editor_B")
        with cols_mat[2]:
            st.markdown("**Vektor prostih členov f**")
            st.session_state.df_f = st.data_editor(st.session_state.df_f, use_container_width=True, key="editor_f")

        st.divider()
        st.subheader("2. Merjena opazovanja, natančnost in približne neznanke")
        col_l, col_x0 = st.columns(2)

        with col_l:
            st.markdown("**Merjena opazovanja ($l_i$) in stohastični model**")
            st.session_state.df_l = st.data_editor(st.session_state.df_l, use_container_width=True, key="editor_l")
            if st.session_state.tip_utezi == "POLNA_SIGMA":
                st.markdown("**Polna kovariančna matrika opazovanj $\Sigma$ ($n \\times n$)**")
                st.session_state.df_Sigma = st.data_editor(st.session_state.df_Sigma, use_container_width=True, key="editor_Sigma")

        with col_x0:
            if u > 0:
                st.markdown("**Poimenovanje neznank (za elipse) in približne vrednosti ($x_0$)**")
                st.session_state.df_x0 = st.data_editor(st.session_state.df_x0, use_container_width=True, key="editor_x0")
            else:
                st.caption("V pogojni izravnavi ni neznank ($u = 0$).")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡ IZVEDI IZRAVNAVO IN STATISTIČNE TESTE", type="primary", key="btn_izracun_tab2", use_container_width=True):
            if izvedi_izravnavo_jedro():
                st.success("Izravnava je končana! Odprite zavihek **3️⃣ Elaborat in Statistični testi** ali **4️⃣ Elipse pogreškov**.")

    # =========================================================================
    # ZAVIHEK 3: ELABORAT, STATISTIČNI TESTI IN MASTER'S LEVEL ANALITIKA
    # =========================================================================
    with tab3:
        res = st.session_state.zadnji_rezultati
        if res is None:
            st.warning("Najprej izvedite izravnavo v zavihku 2 ali izberite šolski primer v levem meniju!")
        else:
            # Vrstica s ključnimi metrikami
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Nadštevilnost (r)", f"{res['r']} (r̄ = {res['r']/res['n']*100:.1f}%)")
            m2.metric("σ₀² (a-priori)", f"{res['sigma0_sq']:.5f}")
            m3.metric("σ̂₀² (a-posteriori)", f"{res['sigma0_hat_sq']:.5f}")
            m4.metric("Globalni χ² test (Y)", f"{res['Y_stat']:.3f}",
                      delta="USPEŠEN (H₀ sprejeta)" if res["globalni_sprejet"] else "ZAVRNJEN (H₀ zavrnjena)",
                      delta_color="normal" if res["globalni_sprejet"] else "inverse")

            b1, b2 = st.columns(2)
            with b1:
                if st.button("🔄 Skaliraj kovariančno matriko Σ z σ̂₀² in ponovi izravnavo (str. 230)", use_container_width=True):
                    skaliraj_sigma_in_ponovi()
                    st.rerun()
            with b2:
                st.download_button(
                    label="💾 Prenesi celoten geodetski elaborat (.txt)",
                    data=res["elaborat_txt"],
                    file_name="geodetski_elaborat_MNK.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            # 1. Tabela neznank
            if res["df_neznanke"] is not None:
                st.subheader("1. Ocenjene vrednosti neznank ($\hat{\mathbf{x}} = \mathbf{x}_0 + \mathbf{\Delta}$)")
                st.dataframe(res["df_neznanke"].style.format({
                    "Pribl. x₀": "{:.5f}", "Popravek Δ": "{:.6f}", "Ocenjena x̂": "{:.5f}",
                    "σ_x̂ (priori)": "{:.5f}", "σ̂_x̂ (post.)": "{:.5f}", "±CI (95%)": "±{:.5f}"
                }), use_container_width=True)

            # 2. Tabela opazovanj z MDB in Baarda zanesljivostjo
            st.subheader(f"2. Pregled opazovanj, popravkov in zanesljivosti ($w_{{krit}}={res['w_crit']:.3f}$, $\\tau_{{krit}}={res['tau_crit']:.3f}$, $\\delta_0={res['delta0']:.3f}$)")
            
            def color_redundancy_and_status(row):
                styles = [''] * len(row)
                # Obarvaj slabo nadštevilnost
                try:
                    r_val = float(row["Nadšt. r_i"])
                    if r_val < 0.10:
                        styles[row.index.get_loc("Nadšt. r_i")] = 'background-color: #fee2e2; color: #991b1b; font-weight: bold;'
                    elif r_val < 0.30:
                        styles[row.index.get_loc("Nadšt. r_i")] = 'background-color: #fef3c7; color: #92400e;'
                except Exception:
                    pass
                # Obarvaj grobe pogreške
                status_str = str(row["Status"])
                if "GROBI" in status_str or "Outlier" in status_str:
                    styles[row.index.get_loc("Status")] = 'background-color: #ef4444; color: white; font-weight: bold;'
                return styles

            styled_df_op = res["df_opazovanja"].style.format({
                "Merjeno l": "{:.4f}", "Popravek v": "{:.5f}", "Izravnano l̂": "{:.4f}",
                "σ_l": "{:.4f}", "σ_l̂ (pri)": "{:.4f}", "σ_v (pri)": "{:.4f}",
                "Nadšt. r_i": "{:.4f}", "|w_i|": "{:.3f}", "|τ_i|": "{:.3f}",
                "Ocena ∇l_i": lambda v: f"{v:.4f}" if not np.isnan(v) else "—",
                "MDB ∇₀l_i": lambda v: f"{v:.4f}" if not np.isnan(v) else "—",
                "Zun. zan. δ̄_i": lambda v: f"{v:.3f}" if not np.isnan(v) else "—"
            }).apply(color_redundancy_and_status, axis=1)

            st.dataframe(styled_df_op, use_container_width=True)

            # 3. INTERAKTIVNI GRAFIKONI STATISTIČNIH TESTOV IN NADŠTEVILNOSTI
            st.subheader("3. Vizualna diagnostika: Data Snooping & Delna nadštevilnost")
            col_g1, col_g2 = st.columns(2, gap="medium")

            df_op = res["df_opazovanja"]
            labels = df_op["Op."].tolist()

            with col_g1:
                fig_snoop = go.Figure()
                fig_snoop.add_trace(go.Bar(
                    x=labels, y=df_op["|w_i|"],
                    name="|w_i| (Baarda)",
                    marker_color=["#ef4444" if w > res["w_crit"] else "#3b82f6" for w in df_op["|w_i|"]]
                ))
                if not np.isnan(res["tau_crit"]):
                    fig_snoop.add_trace(go.Bar(
                        x=labels, y=df_op["|τ_i|"],
                        name="|τ_i| (Pope Tau)",
                        marker_color=["#dc2626" if t > res["tau_crit"] else "#93c5fd" for t in df_op["|τ_i|"]]
                    ))
                fig_snoop.add_hline(y=res["w_crit"], line_dash="dash", line_color="#b91c1c",
                                    annotation_text=f"w_krit = {res['w_crit']:.2f}", annotation_position="top right")
                if not np.isnan(res["tau_crit"]):
                    fig_snoop.add_hline(y=res["tau_crit"], line_dash="dot", line_color="#991b1b",
                                        annotation_text=f"τ_krit = {res['tau_crit']:.2f}", annotation_position="bottom right")

                fig_snoop.update_layout(
                    title="Standardizirani popravki (|w_i| in |τ_i|) z mejama",
                    xaxis_title="Opazovanje", yaxis_title="Vrednost testa",
                    barmode="group", height=340, margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_snoop, use_container_width=True)

            with col_g2:
                fig_red = go.Figure()
                colors_r = []
                for r_val in df_op["Nadšt. r_i"]:
                    if r_val < 0.10:
                        colors_r.append("#ef4444")
                    elif r_val < 0.30:
                        colors_r.append("#f59e0b")
                    else:
                        colors_r.append("#10b981")

                fig_red.add_trace(go.Bar(
                    x=labels, y=df_op["Nadšt. r_i"],
                    name="Delna nadštevilnost r_i",
                    marker_color=colors_r
                ))
                r_avg = res["r"] / res["n"]
                fig_red.add_hline(y=r_avg, line_dash="dash", line_color="#1e3a8a",
                                  annotation_text=f"r̄ = {r_avg:.3f}", annotation_position="top right")
                fig_red.add_hline(y=0.30, line_dash="dot", line_color="#f59e0b",
                                  annotation_text="Kritična meja r=0.30", annotation_position="bottom right")

                fig_red.update_layout(
                    title="Delna nadštevilnost opazovanj (r_i)",
                    xaxis_title="Opazovanje", yaxis_title="r_i (0 ≤ r_i ≤ 1)",
                    height=340, yaxis_range=[0, 1.05], margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_red, use_container_width=True)

            # 4. KORELACIJSKA MATRIKA NEZNANK (R_xx)
            if res.get("R_xx") is not None and res["u"] > 1:
                st.subheader("4. Korelacijska matrika ocenjenih neznank ($\mathbf{R}_{xx}$)")
                imena_x = res["imena_x"]
                fig_corr = go.Figure(data=go.Heatmap(
                    z=res["R_xx"],
                    x=imena_x, y=imena_x,
                    colorscale="RdBu_r", zmin=-1, zmax=1,
                    text=np.round(res["R_xx"], 3), texttemplate="%{text}",
                    textfont={"size": 11}
                ))
                fig_corr.update_layout(height=350, margin=dict(l=40, r=40, t=30, b=40))
                st.plotly_chart(fig_corr, use_container_width=True)

            # 5. IZVOZ V LATEX ZA MAGISTRSKO NALOGO / POROČILO
            with st.expander("🎓 Izvoz matrik in tabel v LaTeX (za Overleaf / Magistrsko nalogo)", expanded=False):
                st.markdown("Kopirajte pripravljeno LaTeX kodo neposredno v svojo magistrsko nalogo ali članek:")
                c_lx1, c_lx2 = st.columns(2)
                with c_lx1:
                    st.markdown("**Matrika B (koeficienti neznank):**")
                    st.code(matrix_to_latex(res.get("B")), language="latex")
                    st.markdown("**Kovariančna matrika neznank $\mathbf{\Sigma}_{\Delta\Delta}$:**")
                    Sigma_dd_show = res["sigma0_sq"] * res["Q_dd"] if res.get("Q_dd") is not None else None
                    st.code(matrix_to_latex(Sigma_dd_show), language="latex")
                with c_lx2:
                    st.markdown("**Matrika normalnih enačb N:**")
                    st.code(matrix_to_latex(res.get("N")), language="latex")
                    st.markdown("**Tabela rezultatov neznank:**")
                    if res.get("df_neznanke") is not None:
                        st.code(dataframe_to_latex_table(res["df_neznanke"], "Ocenjene vrednosti neznank MNK"), language="latex")

            # 6. Tekstovni elaborat
            with st.expander("📄 Poglej celoten tekstovni izpis elaborata in vse matrike (Q, Σ, R)", expanded=False):
                st.code(res["elaborat_txt"], language="text")

    # =========================================================================
    # ZAVIHEK 4: ELIPSE POGREŠKOV IN MREŽA (INTERAKTIVNI PLOTLY + MATPLOTLIB)
    # =========================================================================
    with tab4:
        res = st.session_state.zadnji_rezultati
        if res is None:
            st.warning("Najprej izvedite izravnavo v zavihku 2 ali izberite Primer 4 (2D mreža) v levem meniju!")
        else:
            col_ctrl, col_plot = st.columns([1, 1.8], gap="large")

            with col_ctrl:
                st.subheader("Nastavitve izrisa elips")
                privzeti_vir = 0 if (res["Q_dd"] is not None and res["Q_dd"].shape[0] >= 2) else 1
                vir_elipse = st.selectbox(
                    "1. Vir koordinat točk:",
                    options=["NEZNANKE", "IZRAVNANA_OPAZOVANJA"],
                    format_func=lambda x: "V ocenjenih neznankah (x̂)" if x == "NEZNANKE" else "V izravnanih opazovanjih (l̂)",
                    index=privzeti_vir
                )

                Q_mat = res["Q_dd"] if vir_elipse == "NEZNANKE" else res["Q_ll_hat"]
                vektor_vrednosti = res["x_hat"] if vir_elipse == "NEZNANKE" else res["l_hat"]
                imena = res["imena_x"] if vir_elipse == "NEZNANKE" else [f"l̂_{i+1}" for i in range(res["n"])]

                tocke_pari = []
                if Q_mat is not None and Q_mat.shape[0] >= 2:
                    for p in range(Q_mat.shape[0] // 2):
                        ix, iy = 2 * p, 2 * p + 1
                        naziv = f"Točka {p+1} ({imena[ix]}, {imena[iy]})"
                        tocke_pari.append((ix, iy, naziv))

                if not tocke_pari:
                    st.error("V izbranem viru ni vsaj 2 elementov za 2D elipso! Preklopite vir na 'Izravnana opazovanja'.")
                else:
                    nacin_izrisa = st.radio(
                        "2. Način prikaza:",
                        options=["CELOTNA_MREZA", "POSAMEZNA"],
                        format_func=lambda x: "🗺️ Celotna 2D mreža z elipsami točk" if x == "CELOTNA_MREZA" else "📈 Posamezna točka / Relativna elipsa"
                    )

                    idx_a = st.selectbox("Izberite točko A:", range(len(tocke_pari)), format_func=lambda i: tocke_pari[i][2])
                    relativna = st.checkbox("Izračunaj RELATIVNO elipso med A in B", value=False, disabled=(nacin_izrisa == "CELOTNA_MREZA"))
                    idx_b = st.selectbox("Izberite točko B:", range(len(tocke_pari)), index=min(1, len(tocke_pari)-1), format_func=lambda i: tocke_pari[i][2], disabled=not relativna)

                    sigma_tip = st.radio("3. Referenčna varianca:", ["A_PRIORI", "A_POSTERIORI"], format_func=lambda x: "A-priori (σ₀²)" if x == "A_PRIORI" else "A-posteriori (σ̂₀²)", horizontal=True)
                    p_zaup = st.slider("4. Stopnja zaupanja elipse P:", min_value=0.50, max_value=0.99, value=0.95, step=0.01)

                    prikazi_povezave = False
                    merilo_faktor = 1000.0
                    avto_merilo = True
                    if nacin_izrisa == "CELOTNA_MREZA":
                        st.divider()
                        st.markdown("##### 🔍 Povečava elips in povezave mreže")
                        avto_merilo = st.checkbox("Avtomatsko merilo elips", value=True)
                        if not avto_merilo:
                            merilo_faktor = st.slider("Ročni faktor povečave elips (npr. 500x - 5000x):", min_value=1.0, max_value=10000.0, value=1000.0, step=50.0)
                        prikazi_povezave = st.checkbox("🔗 Prikaži povezave / relativne elipse med točkami", value=True)

            with col_plot:
                if tocke_pari:
                    sigma_sq = res["sigma0_sq"] if sigma_tip == "A_PRIORI" else res["sigma0_hat_sq"]
                    Sigma_full = sigma_sq * Q_mat

                    # =========================================================
                    # 1. INTERAKTIVNI PLOTLY PRIKAZ (PRIMARNI POGLED)
                    # =========================================================
                    st.markdown("#### 🌟 Interaktivni geodetski izris (Plotly)")
                    st.caption("Uporabite miško ali S-Pen za povečevanje (Box / Scroll Zoom), premikanje (Pan) in lebdenje (Hover) za točne koordinate in parametre elips.")

                    if nacin_izrisa == "POSAMEZNA":
                        ix_a, iy_a, naziv_a = tocke_pari[idx_a]
                        if not relativna:
                            s_x2, s_y2, s_xy = Sigma_full[ix_a, ix_a], Sigma_full[iy_a, iy_a], Sigma_full[ix_a, iy_a]
                            naslov = f"Absolutna elipsa pogreškov — {naziv_a}"
                        else:
                            ix_b, iy_b, naziv_b = tocke_pari[idx_b]
                            idx_sub = [ix_a, iy_a, ix_b, iy_b]
                            Sigma_sub = Sigma_full[np.ix_(idx_sub, idx_sub)]
                            A_rel = np.array([[-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]])
                            Sigma_rel = A_rel @ Sigma_sub @ A_rel.T
                            s_x2, s_y2, s_xy = Sigma_rel[0, 0], Sigma_rel[1, 1], Sigma_rel[0, 1]
                            naslov = f"Relativna elipsa med {naziv_a} in {naziv_b}"

                        s_x, s_y, rho, s_xp, s_yp, theta_rad, theta_deg, theta_gon, c_mult, a_conf, b_conf, s_p = izracunaj_parametre_2x2(s_x2, s_y2, s_xy, p_zaup)

                        fig_p = go.Figure()

                        # Pravokotnik 2sx x 2sy
                        fig_p.add_trace(go.Scatter(
                            x=[-s_x, s_x, s_x, -s_x, -s_x],
                            y=[-s_y, -s_y, s_y, s_y, -s_y],
                            mode="lines", line=dict(color="#64748b", dash="dot", width=1.5),
                            name="Pravokotnik 2σ_x × 2σ_y"
                        ))

                        # Standardna elipsa (c=1, 39.4%)
                        xe_std, ye_std = get_ellipse_points(0, 0, s_xp, s_yp, theta_rad)
                        fig_p.add_trace(go.Scatter(
                            x=xe_std, y=ye_std, mode="lines", fill="toself",
                            fillcolor="rgba(147, 197, 253, 0.45)", line=dict(color="#1d4ed8", width=2.2),
                            name="Standardna elipsa (c=1, 39.4%)",
                            hoverinfo="text",
                            text=f"Standardna elipsa<br>a = {s_xp:.5f}<br>b = {s_yp:.5f}<br>θ = {theta_deg:.2f}° ({theta_gon:.2f} gon)"
                        ))

                        # Elipsa zaupanja (P%, c_mult)
                        xe_conf, ye_conf = get_ellipse_points(0, 0, a_conf, b_conf, theta_rad)
                        fig_p.add_trace(go.Scatter(
                            x=xe_conf, y=ye_conf, mode="lines",
                            line=dict(color="#dc2626", width=2.0, dash="dash"),
                            name=f"Elipsa zaupanja ({p_zaup*100:.0f}%, c={c_mult:.2f})",
                            hoverinfo="text",
                            text=f"Elipsa zaupanja ({p_zaup*100:.0f}%)<br>a = {a_conf:.5f}<br>b = {b_conf:.5f}<br>θ = {theta_deg:.2f}° ({theta_gon:.2f} gon)"
                        ))

                        # Osi elipse x' in y'
                        dx_xp, dy_xp = a_conf * np.cos(theta_rad), a_conf * np.sin(theta_rad)
                        dx_yp, dy_yp = -b_conf * np.sin(theta_rad), b_conf * np.cos(theta_rad)
                        fig_p.add_trace(go.Scatter(
                            x=[-dx_xp, dx_xp], y=[-dy_xp, dy_xp], mode="lines",
                            line=dict(color="#1e3a8a", dash="dashdot", width=1.5),
                            name="Glavna os x' (velika polos)"
                        ))
                        fig_p.add_trace(go.Scatter(
                            x=[-dx_yp, dx_yp], y=[-dy_yp, dy_yp], mode="lines",
                            line=dict(color="#0284c7", dash="dashdot", width=1.5),
                            name="Stranska os y' (mala polos)"
                        ))

                        # Koordinatni križ
                        max_lim = max(a_conf, b_conf, s_x, s_y, 1e-5) * 1.3
                        fig_p.add_hline(y=0, line_width=0.8, line_color="#475569")
                        fig_p.add_vline(x=0, line_width=0.8, line_color="#475569")

                        fig_p.update_layout(
                            title=naslov,
                            xaxis=dict(title="Pogrešek v smeri osi X [enota]", range=[-max_lim, max_lim]),
                            yaxis=dict(title="Pogrešek v smeri osi Y [enota]", scaleanchor="x", scaleratio=1, range=[-max_lim, max_lim]),
                            height=520, margin=dict(l=40, r=40, t=50, b=40),
                            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)")
                        )
                        st.plotly_chart(fig_p, use_container_width=True, config={"scrollZoom": True})

                        # Prikaz numeričnih parametrov
                        st.info(
                            f"**Natančnost točke in parametri elipse:**\n"
                            f"- $\\sigma_X = {s_x:.5f}$, $\\sigma_Y = {s_y:.5f}$, $\\rho_{{XY}} = {rho:.4f}$\n"
                            f"- **Helmertov položajni pogrešek:** $\\sigma_P = \\sqrt{{\\sigma_X^2 + \\sigma_Y^2}} = {s_p:.5f}$\n"
                            f"- **Standardna elipsa ($c=1$):** $\\sigma_{{x'}} = {s_xp:.5f}$, $\\sigma_{{y'}} = {s_yp:.5f}$\n"
                            f"- **Zasuk polosi $\\theta$:** **{theta_deg:.3f}°** ({theta_deg:.0f}° {int((theta_deg%1)*60)}' {((theta_deg%1)*60%1)*60:.1f}\") oz. **{theta_gon:.3f} gon**\n"
                            f"- **Elipsa zaupanja ({p_zaup*100:.0f}%):** $a = {a_conf:.5f}$, $b = {b_conf:.5f}$ ($c={c_mult:.2f}$)"
                        )

                    else:
                        # CELOTNA 2D MREŽA
                        xs = [float(vektor_vrednosti[ix, 0]) for ix, iy, _ in tocke_pari]
                        ys = [float(vektor_vrednosti[iy, 0]) for ix, iy, _ in tocke_pari]
                        razpon = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
                        max_polos = 0.0
                        podatki_tock = []

                        for ix, iy, naziv in tocke_pari:
                            x_val, y_val = float(vektor_vrednosti[ix, 0]), float(vektor_vrednosti[iy, 0])
                            s_x, s_y, rho, s_xp, s_yp, theta_rad, theta_deg, theta_gon, c_mult, a_conf, b_conf, s_p = izracunaj_parametre_2x2(
                                Sigma_full[ix, ix], Sigma_full[iy, iy], Sigma_full[ix, iy], p_zaup
                            )
                            max_polos = max(max_polos, a_conf)
                            podatki_tock.append((x_val, y_val, naziv, s_x, s_y, s_p, s_xp, s_yp, theta_rad, theta_deg, theta_gon, a_conf, b_conf))

                        if avto_merilo:
                            merilo_elips = (0.15 * razpon) / max_polos if max_polos > 0 else 1.0
                        else:
                            merilo_elips = merilo_faktor

                        fig_m = go.Figure()

                        # Povezave med točkami (vizure mreže)
                        if prikazi_povezave and len(podatki_tock) >= 2:
                            for i in range(len(podatki_tock)):
                                for j in range(i + 1, len(podatki_tock)):
                                    p1 = podatki_tock[i]
                                    p2 = podatki_tock[j]
                                    fig_m.add_trace(go.Scatter(
                                        x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                                        mode="lines", line=dict(color="#cbd5e1", width=1.5, dash="dot"),
                                        hoverinfo="none", showlegend=False
                                    ))
                                    # Relativna elipsa na sredini vizure
                                    ix_1, iy_1 = tocke_pari[i][0], tocke_pari[i][1]
                                    ix_2, iy_2 = tocke_pari[j][0], tocke_pari[j][1]
                                    idx_sub = [ix_1, iy_1, ix_2, iy_2]
                                    Sigma_sub = Sigma_full[np.ix_(idx_sub, idx_sub)]
                                    A_rel = np.array([[-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]])
                                    Sigma_rel = A_rel @ Sigma_sub @ A_rel.T
                                    _, _, _, s_xp_r, s_yp_r, th_rad_r, th_deg_r, th_gon_r, _, a_conf_r, b_conf_r, s_p_r = izracunaj_parametre_2x2(
                                        Sigma_rel[0, 0], Sigma_rel[1, 1], Sigma_rel[0, 1], p_zaup
                                    )
                                    mid_x, mid_y = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
                                    xe_r, ye_r = get_ellipse_points(mid_x, mid_y, a_conf_r * merilo_elips, b_conf_r * merilo_elips, th_rad_r)
                                    fig_m.add_trace(go.Scatter(
                                        x=xe_r, y=ye_r, mode="lines",
                                        line=dict(color="#8b5cf6", width=1.2, dash="dash"),
                                        name=f"Relativna elipsa {i+1}-{j+1}",
                                        hoverinfo="text",
                                        text=f"Relativna elipsa med Točko {i+1} in {j+1}:<br>a_rel = {a_conf_r:.4f}<br>b_rel = {b_conf_r:.4f}<br>θ = {th_deg_r:.2f}° ({th_gon_r:.2f} gon)"
                                    ))

                        # Izris elips za vsako točko
                        for i, p in enumerate(podatki_tock):
                            x_val, y_val, naziv, s_x, s_y, s_p, s_xp, s_yp, theta_rad, theta_deg, theta_gon, a_conf, b_conf = p

                            # Standardna elipsa (c=1)
                            xe_std, ye_std = get_ellipse_points(x_val, y_val, s_xp * merilo_elips, s_yp * merilo_elips, theta_rad)
                            fig_m.add_trace(go.Scatter(
                                x=xe_std, y=ye_std, mode="lines", fill="toself",
                                fillcolor="rgba(147, 197, 253, 0.40)", line=dict(color="#1d4ed8", width=1.5),
                                name="Standardna elipsa (c=1)" if i == 0 else None,
                                showlegend=(i == 0), hoverinfo="text",
                                text=f"{naziv}<br>Standardna elipsa (c=1):<br>σ_x' = {s_xp:.4f}<br>σ_y' = {s_yp:.4f}"
                            ))

                            # Elipsa zaupanja (P%)
                            xe_conf, ye_conf = get_ellipse_points(x_val, y_val, a_conf * merilo_elips, b_conf * merilo_elips, theta_rad)
                            fig_m.add_trace(go.Scatter(
                                x=xe_conf, y=ye_conf, mode="lines",
                                line=dict(color="#dc2626", width=1.5, dash="dash"),
                                name=f"Elipsa zaupanja ({p_zaup*100:.0f}%)" if i == 0 else None,
                                showlegend=(i == 0), hoverinfo="text",
                                text=f"{naziv}<br>Elipsa zaupanja ({p_zaup*100:.0f}%):<br>a = {a_conf:.4f}<br>b = {b_conf:.4f}<br>θ = {theta_deg:.2f}° ({theta_gon:.2f} gon)"
                            ))

                            # Točka v središču
                            fig_m.add_trace(go.Scatter(
                                x=[x_val], y=[y_val], mode="markers+text",
                                marker=dict(size=10, color="#1e3a8a", symbol="circle"),
                                text=[naziv], textposition="top right",
                                name=naziv, showlegend=False,
                                hoverinfo="text",
                                texttemplate=f"<b>{naziv}</b>",
                                hovertext=(
                                    f"<b>{naziv}</b><br>"
                                    f"X = {x_val:.4f}<br>Y = {y_val:.4f}<br>"
                                    f"σ_X = {s_x:.4f}, σ_Y = {s_y:.4f}<br>"
                                    f"Helmertov σ_P = {s_p:.4f}<br>"
                                    f"θ = {theta_deg:.2f}° ({theta_gon:.2f} gon)"
                                )
                            ))

                        rob = razpon * 0.30
                        fig_m.update_layout(
                            title=f"2D Geodetska mreža z elipsami pogreškov (faktor povečave: {merilo_elips:.0f}×)",
                            xaxis=dict(title="Koordinata X [m / enota]", range=[min(xs) - rob, max(xs) + rob]),
                            yaxis=dict(title="Koordinata Y [m / enota]", scaleanchor="x", scaleratio=1, range=[min(ys) - rob, max(ys) + rob]),
                            height=600, margin=dict(l=40, r=40, t=50, b=40),
                            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)")
                        )
                        st.plotly_chart(fig_m, use_container_width=True, config={"scrollZoom": True})

                    # =========================================================
                    # 2. KLASIČNI MATPLOTLIB PRIKAZ ZA IZVOZ V MAGISTRSKO NALOGO
                    # =========================================================
                    with st.expander("🖼️ Klasični prikaz (Matplotlib) za izvoz v magistrsko nalogo (.png / .pdf)", expanded=False):
                        st.caption("Visoko ločljivostni vektorski (.pdf) ali rasterski (.png) izvoz za diplomska/magistrska dela:")
                        fig_mpl, ax_mpl = plt.subplots(figsize=(7, 7), dpi=130)

                        if nacin_izrisa == "POSAMEZNA":
                            ix_a, iy_a, naziv_a = tocke_pari[idx_a]
                            if not relativna:
                                s_x2, s_y2, s_xy = Sigma_full[ix_a, ix_a], Sigma_full[iy_a, iy_a], Sigma_full[ix_a, iy_a]
                                naslov_mpl = f"Absolutna elipsa pogreškov — {naziv_a}"
                            else:
                                ix_b, iy_b, naziv_b = tocke_pari[idx_b]
                                idx_sub = [ix_a, iy_a, ix_b, iy_b]
                                Sigma_sub = Sigma_full[np.ix_(idx_sub, idx_sub)]
                                A_rel = np.array([[-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]])
                                Sigma_rel = A_rel @ Sigma_sub @ A_rel.T
                                s_x2, s_y2, s_xy = Sigma_rel[0, 0], Sigma_rel[1, 1], Sigma_rel[0, 1]
                                naslov_mpl = f"Relativna elipsa med {naziv_a} in {naziv_b}"

                            s_x, s_y, rho, s_xp, s_yp, theta_rad, theta_deg, theta_gon, c_mult, a_conf, b_conf, s_p = izracunaj_parametre_2x2(s_x2, s_y2, s_xy, p_zaup)
                            ax_mpl.axhline(0, color="black", linewidth=0.8)
                            ax_mpl.axvline(0, color="black", linewidth=0.8)
                            ax_mpl.plot([-s_x, s_x, s_x, -s_x, -s_x], [-s_y, -s_y, s_y, s_y, -s_y], linestyle=":", color="dimgray", label="Pravokotnik 2σ_x × 2σ_y")
                            ax_mpl.add_patch(Ellipse((0, 0), 2*s_xp, 2*s_yp, angle=theta_deg, edgecolor="#1d4ed8", facecolor="#93c5fd", alpha=0.55, linewidth=2, label="Standardna elipsa (c=1, 39.4%)"))
                            ax_mpl.add_patch(Ellipse((0, 0), 2*a_conf, 2*b_conf, angle=theta_deg, edgecolor="#dc2626", facecolor="none", linestyle="--", linewidth=1.8, label=f"Elipsa zaupanja ({p_zaup*100:.0f}%, c={c_mult:.2f})"))

                            dx_xp, dy_xp = a_conf * np.cos(theta_rad), a_conf * np.sin(theta_rad)
                            dx_yp, dy_yp = -b_conf * np.sin(theta_rad), b_conf * np.cos(theta_rad)
                            ax_mpl.plot([-dx_xp, dx_xp], [-dy_xp, dy_xp], color="#1e3a8a", linestyle="-.", linewidth=1.1, label="Glavna os x' (σ_x')")
                            ax_mpl.plot([-dx_yp, dx_yp], [-dy_yp, dy_yp], color="#3b82f6", linestyle="-.", linewidth=1.1, label="Glavna os y' (σ_y')")

                            max_lim = max(a_conf, b_conf, s_x, s_y, 1e-6) * 1.35
                            ax_mpl.set_xlim(-max_lim, max_lim)
                            ax_mpl.set_ylim(-max_lim, max_lim)
                            ax_mpl.set_aspect("equal", adjustable="box")
                            ax_mpl.grid(True, linestyle="--", alpha=0.4)
                            ax_mpl.set_xlabel("Pogrešek v smeri osi X [enota]")
                            ax_mpl.set_ylabel("Pogrešek v smeri osi Y [enota]")
                            ax_mpl.set_title(naslov_mpl, fontsize=11, fontweight="bold")
                            ax_mpl.legend(loc="upper right", fontsize=8)
                        else:
                            xs = [float(vektor_vrednosti[ix, 0]) for ix, iy, _ in tocke_pari]
                            ys = [float(vektor_vrednosti[iy, 0]) for ix, iy, _ in tocke_pari]
                            razpon = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
                            max_polos = 0.0
                            podatki_mpl = []
                            for ix, iy, naziv in tocke_pari:
                                x_val, y_val = float(vektor_vrednosti[ix, 0]), float(vektor_vrednosti[iy, 0])
                                s_x, s_y, rho, s_xp, s_yp, theta_rad, theta_deg, theta_gon, c_mult, a_conf, b_conf, s_p = izracunaj_parametre_2x2(
                                    Sigma_full[ix, ix], Sigma_full[iy, iy], Sigma_full[ix, iy], p_zaup
                                )
                                max_polos = max(max_polos, a_conf)
                                podatki_mpl.append((x_val, y_val, naziv, s_xp, s_yp, a_conf, b_conf, theta_deg))

                            merilo_elips = (0.15 * razpon) / max_polos if max_polos > 0 else 1.0
                            for i, (x_val, y_val, naziv, s_xp, s_yp, a_conf, b_conf, theta_deg) in enumerate(podatki_mpl):
                                ax_mpl.scatter(x_val, y_val, color="#1e3a8a", s=50, zorder=5)
                                ax_mpl.annotate(f"  {naziv}", (x_val, y_val), fontsize=9, fontweight="bold", color="#1e3a8a")
                                ax_mpl.add_patch(Ellipse((x_val, y_val), 2*s_xp*merilo_elips, 2*s_yp*merilo_elips, angle=theta_deg, edgecolor="#1d4ed8", facecolor="#93c5fd", alpha=0.55, linewidth=1.8, label="Standardna elipsa (c=1)" if i == 0 else "_nolegend_"))
                                ax_mpl.add_patch(Ellipse((x_val, y_val), 2*a_conf*merilo_elips, 2*b_conf*merilo_elips, angle=theta_deg, edgecolor="#dc2626", facecolor="none", linestyle="--", linewidth=1.4, label=f"Elipsa zaupanja ({p_zaup*100:.0f}%)" if i == 0 else "_nolegend_"))

                            rob = razpon * 0.35
                            ax_mpl.set_xlim(min(xs) - rob, max(xs) + rob)
                            ax_mpl.set_ylim(min(ys) - rob, max(ys) + rob)
                            ax_mpl.set_aspect("equal", adjustable="box")
                            ax_mpl.grid(True, linestyle="--", alpha=0.4)
                            ax_mpl.set_xlabel("Koordinata X [m / enota]")
                            ax_mpl.set_ylabel("Koordinata Y [m / enota]")
                            ax_mpl.set_title(f"Geodetska mreža z elipsami pogreškov ({merilo_elips:.0f}×)", fontsize=11, fontweight="bold")
                            ax_mpl.legend(loc="upper right", fontsize=8)

                        st.pyplot(fig_mpl)

                        # Prenos PNG in PDF
                        buf_png = io.BytesIO()
                        fig_mpl.savefig(buf_png, format="png", dpi=300, bbox_inches="tight")
                        buf_pdf = io.BytesIO()
                        fig_mpl.savefig(buf_pdf, format="pdf", bbox_inches="tight")

                        c_dl1, c_dl2 = st.columns(2)
                        with c_dl1:
                            st.download_button("💾 Prenesi sliko (.png 300 DPI)", data=buf_png.getvalue(), file_name="elipse_pogreskov.png", mime="image/png", use_container_width=True)
                        with c_dl2:
                            st.download_button("💾 Prenesi vektorsko grafiko (.pdf)", data=buf_pdf.getvalue(), file_name="elipse_pogreskov.pdf", mime="application/pdf", use_container_width=True)
                        plt.close(fig_mpl)
