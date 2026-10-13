import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# =============================================================================
# NASTAVITVE STRANI IN SLOGA ZA SAMSUNG GALAXY TAB S11 ULTRA
# =============================================================================
st.set_page_config(
    page_title="Geodetski Izravnalni Račun (MNK)",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Prilagoditev velikosti gumbov in tabel za zaslon na dotik in S Pen
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        height: 3.2em;
        font-weight: bold;
        border-radius: 8px;
    }
    .main-header {
        background-color: #1e3a8a;
        color: white;
        padding: 16px 24px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
    .main-header h2 {
        color: white !important;
        margin: 0;
        font-size: 1.6rem;
    }
    .main-header p {
        color: #bfdbfe;
        margin: 4px 0 0 0;
        font-size: 0.95rem;
    }
</style>
<div class="main-header">
    <h2>📐 GEODETSKI IZRAVNALNI RAČUN IN STATISTIČNA ANALIZA MREŽ</h2>
    <p>Posredna • Pogojna • Splošni model • Data Snooping • Elipse pogreškov (Tab S11 Ultra Touch UI)</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# INICIALIZACIJA STANJA SEJE (SESSION STATE)
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


if "initialized" not in st.session_state:
    st.session_state.metoda = "POSREDNA"
    st.session_state.n = 6
    st.session_state.u = 1
    st.session_state.c = 6
    st.session_state.sigma0_sq = 1.0
    st.session_state.alpha = 0.05
    st.session_state.alpha0 = 0.001
    st.session_state.tip_utezi = "STD_ODKLONI"
    st.session_state.zadnji_rezultati = None
    init_default_matrices("POSREDNA", 6, 6, 1, "STD_ODKLONI")
    st.session_state.initialized = True


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
# MATEMATIČNO JEDRO IZRAVNAVE IN POROČILA
# =============================================================================
def format_matrika(M, naziv, predpona="  "):
    if M is None:
        return f"{naziv}: Ni definirana v tem modelu.\n"
    vrstice = [f"{naziv} ({M.shape[0]}×{M.shape[1]}):"]
    for row in M:
        vrstice.append(predpona + " ".join(f"{val:11.5f}" for val in row))
    return "\n".join(vrstice) + "\n"


def izvedi_izravnavo_jedro():
    metoda = st.session_state.metoda
    c = int(st.session_state.c)
    n = int(st.session_state.n)
    u = int(st.session_state.u)
    sigma0_sq = float(st.session_state.sigma0_sq)
    alpha = float(st.session_state.alpha)
    alpha0 = float(st.session_state.alpha0)
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

    # Reševanje glede na izbrano metodo
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

    # Statistika
    vTPv = float((v.T @ P @ v)[0, 0])
    sigma0_hat_sq = vTPv / r

    Sigma_vv_priori = sigma0_sq * Q_vv
    Sigma_ll_hat_priori = sigma0_sq * Q_ll_hat
    Sigma_dd_priori = sigma0_sq * Q_dd if Q_dd is not None else None
    Sigma_ll_hat_post = sigma0_hat_sq * Q_ll_hat
    Sigma_dd_post = sigma0_hat_sq * Q_dd if Q_dd is not None else None

    Y_stat = (r * sigma0_hat_sq) / sigma0_sq
    chi2_low = stats.chi2.ppf(alpha / 2.0, r)
    chi2_high = stats.chi2.ppf(1.0 - alpha / 2.0, r)
    globalni_sprejet = (chi2_low <= Y_stat <= chi2_high)

    ci_var_low = (r * sigma0_hat_sq) / chi2_high
    ci_var_high = (r * sigma0_hat_sq) / chi2_low

    w_crit = stats.norm.ppf(1.0 - alpha0 / 2.0)
    if r > 1:
        t_val = stats.t.ppf(1.0 - alpha0 / 2.0, r - 1)
        tau_crit = (np.sqrt(r) * t_val) / np.sqrt(r - 1.0 + t_val**2)
    else:
        tau_crit = np.nan

    r_diag = np.diag(R_mat)
    r_avg = r / n

    # Priprava tabel in tekstovnega elaborata
    out = []
    sep = "═" * 96
    subsep = "─" * 96
    out.append(sep)
    out.append(" GEODETSKI IZRAVNALNI RAČUN PO METODI NAJMANJŠIH KVADRATOV (MNK)")
    out.append(f" Izbrani model izravnave: {metoda} IZRAVNAVA")
    out.append(sep)
    out.append("\n1. PARAMETRI MATEMATIČNEGA MODELA")
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

    df_neznanke_res = None
    if Delta is not None and u > 0:
        out.append("\n\n2. OCENJENE VREDNOSTI NEZNANK (x̂ = x₀ + Δ) IN NJIHOVA NATANČNOST")
        out.append(subsep)
        t_95 = stats.t.ppf(1.0 - alpha / 2.0, r)
        out.append(f"  {'Neznanka':<12} {'Pribl. x₀':>12} {'Popravek Δ':>13} {'Ocenjena x̂':>14} {'σ_x̂ (priori)':>14} {'σ̂_x̂ (post.)':>14} {'±CI (95% post)':>15}")
        out.append("  " + "-" * 90)
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

    out.append("\n\n3. OPAZOVANJA, POPRAVKI, NADŠTEVILNOST IN PREGLED OPAZOVANJ (Data Snooping & Tau-test)")
    out.append(subsep)
    out.append(f"  Kritična vrednost Baarda Data Snooping (α₀={alpha0}, N(0,1)) : w_krit = {w_crit:.3f}")
    out.append(f"  Kritična vrednost Pope Tau-test (α₀={alpha0}, τ(r={r}))      : τ_krit = {tau_crit:.3f}\n")
    glava_op = (
        f"  {'Op.':<5} {'Merjeno l':>11} {'Popr. v':>11} {'Izravn. l̂':>11} "
        f"{'σ_l':>9} {'σ_l̂(pri)':>9} {'σ_v(pri)':>9} {'Nadšt. r_i':>10} "
        f"{'|w_i|':>8} {'|τ_i|':>8} {'Ocena ∇l_i':>11} {'Status'}"
    )
    out.append(glava_op)
    out.append("  " + "-" * 116)

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
        nabla_li = -vi / ri if ri > 1e-6 else np.nan

        status = "OK"
        if wi > w_crit and tau_i > (tau_crit if not np.isnan(tau_crit) else 999):
            status = "GROBI POGREŠEK (w & τ)!"
            sumljiva_opazovanja.append((i+1, wi, tau_i, nabla_li, ri))
        elif wi > w_crit:
            status = "Outlier (w-test)!"
            sumljiva_opazovanja.append((i+1, wi, tau_i, nabla_li, ri))
        elif not np.isnan(tau_crit) and tau_i > tau_crit:
            status = "Outlier (τ-test)!"
            sumljiva_opazovanja.append((i+1, wi, tau_i, nabla_li, ri))
        elif ri < 0.10:
            status = "Slaba nadšt. (r_i<0.1)"

        nabla_str = f"{nabla_li:11.4f}" if not np.isnan(nabla_li) else "     Ni nad."
        out.append(
            f"  {'l_'+str(i+1):<5} {li:11.4f} {vi:11.5f} {li_hat:11.4f} "
            f"{sig_l:9.4f} {sig_l_hat:9.4f} {sig_v_pri:9.4f} {ri:10.4f} "
            f"{wi:8.3f} {tau_i:8.3f} {nabla_str} {status}"
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
            "Ocena ∇l_i": nabla_li,
            "Status": status
        })
    df_opazovanja_res = pd.DataFrame(rows_op)

    out.append("\n\n4. STATISTIČNI TESTI IN INTERPRETACIJA")
    out.append(subsep)
    out.append(f"  Testna statistika Y = {Y_stat:.4f}, Kritični interval: [{chi2_low:.4f}, {chi2_high:.4f}]")
    out.append("  ODLOČITEV: " + ("✔ H₀ NE ZAVRNEMO (USPEŠEN)" if globalni_sprejet else "✖ H₀ ZAVRNEMO!"))

    out.append("\n\n5. MATRIKE KOFAKTORJEV (Q), KOVARIANČNE MATRIKE (Σ) IN MATRIKA NADŠTEVILNOSTI (R)")
    out.append(subsep)
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
        "w_crit": w_crit, "tau_crit": tau_crit,
        "Q_dd": Q_dd, "Q_ll_hat": Q_ll_hat, "Q_vv": Q_vv, "R_mat": R_mat,
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
# STRANSKA VRSTICA (SIDEBAR) ZA HITRE AKCIJE NA TABLICI
# =============================================================================
with st.sidebar:
    st.header("🚀 Hitri šolski primeri")
    st.caption("Z enim dotikom naložite in izračunajte primere iz skripte:")
    if st.button("📦 1. Kocka (Posredna)"):
        nalozi_primer_1()
        izvedi_izravnavo_jedro()
    if st.button("📐 2. Kocka (Pogojna)"):
        nalozi_primer_2()
        izvedi_izravnavo_jedro()
    if st.button("📏 3. Premica (Splošni model)"):
        nalozi_primer_3()
        izvedi_izravnavo_jedro()
    if st.button("🗺️ 4. 2D Mreža (Elipse)"):
        nalozi_primer_4()
        izvedi_izravnavo_jedro()

    st.divider()
    st.subheader("⚡ Hiter izračun")
    if st.button("⚡ IZVEDI IZRAVNAVO IN TESTE", type="primary"):
        if izvedi_izravnavo_jedro():
            st.success("Izravnava uspešno izvedena! Poglejte zavihek 3 in 4.")


# =============================================================================
# GLAVNI ZAVIHKI APLIKACIJE
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Izbira modela in Navodila",
    "2️⃣ Pametni vnos enačb in meritev",
    "3️⃣ Elaborat in Statistični testi",
    "4️⃣ Elipse pogreškov in Mreža"
])

# -----------------------------------------------------------------------------
# ZAVIHEK 1: NASTAVITVE MODELA IN PRIROČNIK
# -----------------------------------------------------------------------------
with tab1:
    col_nast, col_prirocnik = st.columns([1, 1.2], gap="large")

    with col_nast:
        st.subheader("1. korak: Izbira metode in dimenzij")
        metoda_izbira = st.radio(
            "Katero vrsto izravnave želite izvesti?",
            options=["POSREDNA", "POGOJNA", "SPLOSNA"],
            format_func=lambda x: {
                "POSREDNA": "Posredna izravnava (v + B·Δ = f) — iščemo koordinate/neznanke",
                "POGOJNA": "Pogojna izravnava (A·v = f) — le geometrijski pogoji brez neznank",
                "SPLOSNA": "Splošni model (A·v + B·Δ = f) — več meritev in neznank v enačbi"
            }[x],
            index=["POSREDNA", "POGOJNA", "SPLOSNA"].index(st.session_state.metoda)
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            n_new = st.number_input("Št. opazovanj (n)", min_value=1, max_value=50, value=int(st.session_state.n))
        with c2:
            if metoda_izbira == "POGOJNA":
                u_new = st.number_input("Št. neznank (u)", min_value=0, max_value=0, value=0, disabled=True)
            else:
                u_new = st.number_input("Št. neznank (u)", min_value=1 if metoda_izbira == "POSREDNA" else 0, max_value=50, value=max(1, int(st.session_state.u)))
        with c3:
            if metoda_izbira == "POSREDNA":
                c_new = st.number_input("Št. enačb (c = n)", min_value=int(n_new), max_value=int(n_new), value=int(n_new), disabled=True)
            else:
                c_new = st.number_input("Št. enačb / pogojev (c)", min_value=1, max_value=50, value=min(int(st.session_state.c), int(n_new)-1 if metoda_izbira=="POGOJNA" else 50))

        st.divider()
        st.subheader("Stohastični model in stopnje tveganja")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.session_state.sigma0_sq = st.number_input("Varianca a-priori (σ₀²)", value=float(st.session_state.sigma0_sq), format="%.5f")
        with s2:
            st.session_state.alpha = st.number_input("Stopnja α (Globalni χ²)", value=float(st.session_state.alpha), format="%.4f")
        with s3:
            st.session_state.alpha0 = st.number_input("Stopnja α₀ (Snooping)", value=float(st.session_state.alpha0), format="%.4f")

        tip_utezi_new = st.selectbox(
            "Način podajanja natančnosti meritev:",
            options=["STD_ODKLONI", "DIAGONALNA_P", "POLNA_SIGMA"],
            format_func=lambda x: {
                "STD_ODKLONI": "Standardni odkloni opazovanj (σ_i)",
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
        st.subheader("📖 Hitri geodetski priročnik")
        with st.expander("1. Katero izravnavo izbrati in kaj pomenijo enačbe?", expanded=True):
            st.markdown("""
            V geodeziji imamo vedno več meritev ($n$), kot je nujno potrebno za izračun neznank ($u$). Razliko imenujemo **nadštevilna opazovanja** ($r = n - u$). Zaradi merskih pogreškov vsaki meritvi $l_i$ dodamo popravek $v_i$, da dobimo izravnano meritev: $\hat{l}_i = l_i + v_i$.

            * **🔹 A) POSREDNA IZRAVNAVA ($\mathbf{v} + \mathbf{B}\mathbf{\Delta} = \mathbf{f}$):**
              Uporabimo jo, ko iščemo neznanke (npr. koordinate točk $X, Y, H$). Vsaka meritev dobi svojo enačbo popravka ($c = n$). Matrika $\mathbf{A} = \mathbf{I}$ je samodejno skrita!
            * **🔹 B) POGOJNA IZRAVNAVA ($\mathbf{A}\mathbf{v} = \mathbf{f}$):**
              Ko ne računamo koordinat ($u = 0$), ampak poznamo geometrijske pogoje (npr. vsota kotov v trikotniku ali višinskih razlik v nivelmanskem vlaku).
            * **🔹 C) SPLOŠNI MODEL ($\mathbf{A}\mathbf{v} + \mathbf{B}\mathbf{\Delta} = \mathbf{f}$):**
              Ko v isti enačbi nastopa več merjenih količin hkrati IN neznanke (npr. prilagajanje premice točkam, kjer smo merili obe koordinati).
            """)
        with st.expander("2. Kako delujejo elipse pogreškov v 2D mreži?", expanded=True):
            st.markdown("""
            Ko v 2D mreži določamo položaj točke $T(X, Y)$, imata koordinati standardna odklona $\sigma_X, \sigma_Y$ ter kovarianco $\sigma_{XY}$.
            * **Velika polos ($\sigma_{x'}$):** Smer najmanjše natančnosti položaja točke.
            * **Mala polos ($\sigma_{y'}$):** Smer največje natančnosti položaja točke.
            * **Zasuk ($\theta$):** Kot zasuka velike polosi glede na os $X$.
            * V **zavihku 4** preprosto izberete točko iz spustnega seznama (brez vpisovanja indeksov) ali izrišete celotno mrežo naenkrat!
            """)


# -----------------------------------------------------------------------------
# ZAVIHEK 2: PAMETNI VNOS ENAČB IN MERITEV (PRILAGOJENO ZA S PEN / DOTIK)
# -----------------------------------------------------------------------------
with tab2:
    metoda = st.session_state.metoda
    c, n, u = st.session_state.c, st.session_state.n, st.session_state.u

    if metoda == "POSREDNA":
        st.info("💡 **OPOMNIK ZA POSREDNO IZRAVNAVO ($v_i + b_{i1}\Delta_1 + \dots = f_i$):** Vsaka vrstica pripada eni meritvi $l_i$. Matrika A je identiteta (skrita). Vnesite matriko odvodov **B** in vektor prostih členov **f**.")
    elif metoda == "POGOJNA":
        st.info("💡 **OPOMNIK ZA POGOJNO IZRAVNAVO ($a_{i1}v_1 + a_{i2}v_2 + \dots = f_i$):** V modelu ni neznank ($u=0$). Vnesite koeficiente pogojev v matriko **A** in odstopanja **f**.")
    else:
        st.info("💡 **OPOMNIK ZA SPLOŠNI MODEL ($\mathbf{A}\mathbf{v} + \mathbf{B}\mathbf{\Delta} = \mathbf{f}$):** Vnesite odvode po meritvah (**A**), odvode po neznankah (**B**) in proste člene (**f**).")

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
    if st.button("⚡ IZVEDI IZRAVNAVO IN STATISTIČNE TESTE", type="primary", key="btn_izracun_tab2"):
        if izvedi_izravnavo_jedro():
            st.success("Izravnava je končana! Odprite zavihek **3️⃣ Elaborat in Statistični testi** ali **4️⃣ Elipse pogreškov**.")


# -----------------------------------------------------------------------------
# ZAVIHEK 3: ELABORAT, NADŠTEVILNOST IN STATISTIČNI TESTI
# -----------------------------------------------------------------------------
with tab3:
    res = st.session_state.zadnji_rezultati
    if res is None:
        st.warning("Najprej izvedite izravnavo v zavihku 2 ali izberite šolski primer v levem meniju!")
    else:
        # Vrstica z metrikami za hiter pregled na zaslonu tablice
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Nadštevilnost (r)", f"{res['r']} (r̄ = {res['r']/res['n']*100:.1f}%)")
        m2.metric("σ₀² (a-priori)", f"{res['sigma0_sq']:.5f}")
        m3.metric("σ̂₀² (a-posteriori)", f"{res['sigma0_hat_sq']:.5f}")
        m4.metric("Globalni χ² test (Y)", f"{res['Y_stat']:.3f}",
                  delta="USPEŠEN (H₀ sprejeta)" if res["globalni_sprejet"] else "ZAVRNJEN (H₀ zavrnjena)",
                  delta_color="normal" if res["globalni_sprejet"] else "inverse")

        # Akcijski gumbi
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🔄 Skaliraj kovariančno matriko Σ z σ̂₀² in ponovi izravnavo (str. 230)"):
                skaliraj_sigma_in_ponovi()
                st.rerun()
        with b2:
            st.download_button(
                label="💾 Prenesi celoten geodetski elaborat (.txt)",
                data=res["elaborat_txt"],
                file_name="geodetski_elaborat_MNK.txt",
                mime="text/plain"
            )

        if res["df_neznanke"] is not None:
            st.subheader("Ocenjene vrednosti neznank ($\hat{\mathbf{x}} = \mathbf{x}_0 + \mathbf{\Delta}$)")
            st.dataframe(res["df_neznanke"].style.format({
                "Pribl. x₀": "{:.5f}", "Popravek Δ": "{:.6f}", "Ocenjena x̂": "{:.5f}",
                "σ_x̂ (priori)": "{:.5f}", "σ̂_x̂ (post.)": "{:.5f}", "±CI (95%)": "±{:.5f}"
            }), use_container_width=True)

        st.subheader(f"Pregled opazovanj, popravkov in odkrivanje grobih pogreškov ($w_{{krit}}={res['w_crit']:.3f}$, $\\tau_{{krit}}={res['tau_crit']:.3f}$)")
        st.dataframe(res["df_opazovanja"].style.format({
            "Merjeno l": "{:.4f}", "Popravek v": "{:.5f}", "Izravnano l̂": "{:.4f}",
            "σ_l": "{:.4f}", "σ_l̂ (pri)": "{:.4f}", "σ_v (pri)": "{:.4f}",
            "Nadšt. r_i": "{:.4f}", "|w_i|": "{:.3f}", "|τ_i|": "{:.3f}", "Ocena ∇l_i": "{:.4f}"
        }), use_container_width=True)

        with st.expander("📄 Poglej celoten tekstovni izpis elaborata in vse matrike (Q, Σ, R)", expanded=True):
            st.code(res["elaborat_txt"], language="text")


# -----------------------------------------------------------------------------
# ZAVIHEK 4: ELIPSE POGREŠKOV IN IZRIS MREŽE
# -----------------------------------------------------------------------------
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

    c_mult = np.sqrt(-2.0 * np.log(1.0 - p_zaup))
    a_conf = c_mult * s_xp
    b_conf = c_mult * s_yp
    return s_x, s_y, rho, s_xp, s_yp, theta_rad, theta_deg, c_mult, a_conf, b_conf


with tab4:
    res = st.session_state.zadnji_rezultati
    if res is None:
        st.warning("Najprej izvedite izravnavo v zavihku 2 ali izberite Primer 4 (2D mreža) v stranski vrstici!")
    else:
        col_ctrl, col_plot = st.columns([1, 1.8], gap="large")

        with col_ctrl:
            st.subheader("Nastavitve izrisa elips")
            privzeti_vir = 0 if (res["Q_dd"] is not None and res["Q_dd"].shape[0] >= 2) else 1
            vir_elipse = st.selectbox(
                "1. Kje se nahajajo koordinate točk?",
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
                    tocke_pari.append((ix, iy, f"Točka {p+1} ({imena[ix]}, {imena[iy]})"))

            if not tocke_pari:
                st.error("V izbranem viru ni vsaj 2 elementov za 2D elipso! Preklopite vir na 'Izravnana opazovanja'.")
            else:
                nacin_izrisa = st.radio(
                    "2. Način izrisa:",
                    options=["POSAMEZNA", "CELOTNA_MREZA"],
                    format_func=lambda x: "📈 Elipsa izbrane točke (ali relativna)" if x == "POSAMEZNA" else "🗺️ Vse točke mreže z elipsami hkrati"
                )

                idx_a = st.selectbox("Izberite točko A:", range(len(tocke_pari)), format_func=lambda i: tocke_pari[i][2])
                relativna = st.checkbox("Izračunaj RELATIVNO elipso med A in B", value=False, disabled=(nacin_izrisa == "CELOTNA_MREZA"))
                idx_b = st.selectbox("Izberite točko B:", range(len(tocke_pari)), index=min(1, len(tocke_pari)-1), format_func=lambda i: tocke_pari[i][2], disabled=not relativna)

                sigma_tip = st.radio("3. Referenčna varianca:", ["A_PRIORI", "A_POSTERIORI"], format_func=lambda x: "A-priori (σ₀²)" if x == "A_PRIORI" else "A-posteriori (σ̂₀²)", horizontal=True)
                p_zaup = st.slider("4. Stopnja zaupanja elipse P:", min_value=0.50, max_value=0.99, value=0.95, step=0.01)

        with col_plot:
            if tocke_pari:
                sigma_sq = res["sigma0_sq"] if sigma_tip == "A_PRIORI" else res["sigma0_hat_sq"]
                Sigma_full = sigma_sq * Q_mat

                fig, ax = plt.subplots(figsize=(7, 7), dpi=110)

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

                    s_x, s_y, rho, s_xp, s_yp, theta_rad, theta_deg, c_mult, a_conf, b_conf = izracunaj_parametre_2x2(s_x2, s_y2, s_xy, p_zaup)

                    ax.axhline(0, color="black", linewidth=0.8)
                    ax.axvline(0, color="black", linewidth=0.8)
                    ax.plot([-s_x, s_x, s_x, -s_x, -s_x], [-s_y, -s_y, s_y, s_y, -s_y], linestyle=":", color="dimgray", label="Pravokotnik 2σ_x × 2σ_y")

                    ax.add_patch(Ellipse((0, 0), 2*s_xp, 2*s_yp, angle=theta_deg, edgecolor="#1d4ed8", facecolor="#93c5fd", alpha=0.55, linewidth=2, label="Standardna elipsa (c=1, 39.4%)"))
                    ax.add_patch(Ellipse((0, 0), 2*a_conf, 2*b_conf, angle=theta_deg, edgecolor="#dc2626", facecolor="none", linestyle="--", linewidth=1.8, label=f"Elipsa zaupanja ({p_zaup*100:.0f}%, c={c_mult:.2f})"))

                    dx_xp, dy_xp = a_conf * np.cos(theta_rad), a_conf * np.sin(theta_rad)
                    dx_yp, dy_yp = -b_conf * np.sin(theta_rad), b_conf * np.cos(theta_rad)
                    ax.plot([-dx_xp, dx_xp], [-dy_xp, dy_xp], color="#1e3a8a", linestyle="-.", linewidth=1.1, label="Glavna os x' (σ_x')")
                    ax.plot([-dx_yp, dx_yp], [-dy_yp, dy_yp], color="#3b82f6", linestyle="-.", linewidth=1.1, label="Glavna os y' (σ_y')")

                    max_lim = max(a_conf, b_conf, s_x, s_y, 1e-6) * 1.35
                    ax.set_xlim(-max_lim, max_lim)
                    ax.set_ylim(-max_lim, max_lim)
                    ax.set_aspect("equal", adjustable="box")
                    ax.grid(True, linestyle="--", alpha=0.4)
                    ax.set_xlabel("Pogrešek v smeri osi X [enota]")
                    ax.set_ylabel("Pogrešek v smeri osi Y [enota]")
                    ax.set_title(naslov, fontsize=11, fontweight="bold")
                    ax.legend(loc="upper right", fontsize=8)
                    st.pyplot(fig)

                    with col_ctrl:
                        st.info(
                            f"**Parametri elipse:**\n"
                            f"- $\\sigma_x = {s_x:.5f}$, $\\sigma_y = {s_y:.5f}$, $\\rho_{{xy}} = {rho:.4f}$\n"
                            f"- **Standardna ($c=1$):** $\\sigma_{{x'}} = {s_xp:.5f}$, $\\sigma_{{y'}} = {s_yp:.5f}$, $\\theta = {theta_deg:.2f}^\\circ$\n"
                            f"- **Zaupanja ({p_zaup*100:.0f}%):** $a = {a_conf:.5f}$, $b = {b_conf:.5f}$"
                        )
                else:
                    xs = [float(vektor_vrednosti[ix, 0]) for ix, iy, _ in tocke_pari]
                    ys = [float(vektor_vrednosti[iy, 0]) for ix, iy, _ in tocke_pari]
                    razpon = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
                    max_polos = 0.0
                    podatki_tock = []

                    for ix, iy, naziv in tocke_pari:
                        x_val, y_val = float(vektor_vrednosti[ix, 0]), float(vektor_vrednosti[iy, 0])
                        s_x, s_y, rho, s_xp, s_yp, theta_rad, theta_deg, c_mult, a_conf, b_conf = izracunaj_parametre_2x2(
                            Sigma_full[ix, ix], Sigma_full[iy, iy], Sigma_full[ix, iy], p_zaup
                        )
                        max_polos = max(max_polos, a_conf)
                        podatki_tock.append((x_val, y_val, naziv, s_xp, s_yp, a_conf, b_conf, theta_deg))

                    merilo_elips = (0.15 * razpon) / max_polos if max_polos > 0 else 1.0
                    for i, (x_val, y_val, naziv, s_xp, s_yp, a_conf, b_conf, theta_deg) in enumerate(podatki_tock):
                        ax.scatter(x_val, y_val, color="#1e3a8a", s=50, zorder=5)
                        ax.annotate(f"  {naziv}", (x_val, y_val), fontsize=9, fontweight="bold", color="#1e3a8a")
                        ax.add_patch(Ellipse((x_val, y_val), 2*s_xp*merilo_elips, 2*s_yp*merilo_elips, angle=theta_deg, edgecolor="#1d4ed8", facecolor="#93c5fd", alpha=0.55, linewidth=1.8, label="Standardna elipsa (c=1)" if i == 0 else "_nolegend_"))
                        ax.add_patch(Ellipse((x_val, y_val), 2*a_conf*merilo_elips, 2*b_conf*merilo_elips, angle=theta_deg, edgecolor="#dc2626", facecolor="none", linestyle="--", linewidth=1.4, label=f"Elipsa zaupanja ({p_zaup*100:.0f}%)" if i == 0 else "_nolegend_"))

                    rob = razpon * 0.35
                    ax.set_xlim(min(xs) - rob, max(xs) + rob)
                    ax.set_ylim(min(ys) - rob, max(ys) + rob)
                    ax.set_aspect("equal", adjustable="box")
                    ax.grid(True, linestyle="--", alpha=0.4)
                    ax.set_xlabel("Koordinata X [m / enota]")
                    ax.set_ylabel("Koordinata Y [m / enota]")
                    ax.set_title(f"Geodetska mreža z elipsami pogreškov (elipse povečane {merilo_elips:.0f}×)", fontsize=11, fontweight="bold")
                    ax.legend(loc="upper right", fontsize=8)
                    st.pyplot(fig)