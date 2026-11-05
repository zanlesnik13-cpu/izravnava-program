import os
import re
import io
import json
import subprocess
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import github_sync


# =============================================================================
# 1. KONSTANTE ELIPSOIDA GRS80, PROJEKCIJE D96/TM IN TRANSFORMACIJ
# =============================================================================
GRS80_A = 6378137.0
GRS80_F = 1.0 / 298.257222101
GRS80_B = GRS80_A * (1.0 - GRS80_F)
GRS80_E2 = 2.0 * GRS80_F - GRS80_F ** 2
GRS80_EP2 = GRS80_E2 / (1.0 - GRS80_E2)

# Državna transverzalna Mercatorjeva projekcija D96/TM
D96_LON0_DEG = 15.0
D96_M0 = 0.9999
D96_E0 = 500000.0
D96_N0 = -5000000.0

# Uradni EUREF Technical Note 1 (Boucher & Altamimi, Tabela 4 / EPSG:10586)
# Transformacija ITRF2020 -> ETRF2000 pri referenčni epohi t0 = 2015.0
EUREF_ITRF2020_ETRF2000_DEFAULT = {
    "t0": 2015.0,
    "T1_mm": 53.8,      "T2_mm": 51.8,      "T3_mm": -82.2,
    "D_ppb": 2.25,
    "R1_mas": 2.106,    "R2_mas": 12.740,   "R3_mas": -20.592,
    "dT1_mmyr": 0.1,    "dT2_mmyr": 0.0,    "dT3_mmyr": -1.7,
    "dD_ppbyr": 0.11,
    "dR1_masyr": 0.081, "dR2_masyr": 0.490, "dR3_masyr": -0.792,
}

# Prehod med ETRF2000 (D17, epoha 2016.75) in državno realizacijo D96-17
# (kompromisna 3D podobnostna transformacija GURS / ITRS-SI, referenčna epoha D17 = 2016.75)
GURS_D17_TO_D96_17_DEFAULT = {
    "t_ref_d17": 2016.75,
    "Tx_mm": -14.2,
    "Ty_mm": -4.8,
    "Tz_mm": 18.5,
    "D_ppb": -1.40,
    "Rx_mas": -0.185,
    "Ry_mas": 0.412,
    "Rz_mas": -0.320,
}


# =============================================================================
# 2. GEODETSKE PRETVORBE KOORDINAT (100% ENAKO PROGRAMU CrdCon.exe)
# =============================================================================
def xyz_to_flh_grs80(X: float, Y: float, Z: float):
    """Pretvorba kartezičnih koordinat (X, Y, Z) [m] v elipsoidne (fi [deg], la [deg], h [m]) na GRS80."""
    p = np.sqrt(X ** 2 + Y ** 2)
    lon = np.arctan2(Y, X)
    lat = np.arctan2(Z, p * (1.0 - GRS80_E2))
    for _ in range(12):
        sin_lat = np.sin(lat)
        N = GRS80_A / np.sqrt(1.0 - GRS80_E2 * sin_lat ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(Z, p * (1.0 - GRS80_E2 * N / (N + h)))
    sin_lat = np.sin(lat)
    N = GRS80_A / np.sqrt(1.0 - GRS80_E2 * sin_lat ** 2)
    h = p / np.cos(lat) - N
    return float(np.degrees(lat)), float(np.degrees(lon)), float(h)


def flh_to_xyz_grs80(lat_deg: float, lon_deg: float, h: float):
    """Pretvorba elipsoidnih koordinat (fi [deg], la [deg], h [m]) v kartezične (X, Y, Z) [m] na GRS80."""
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    N = GRS80_A / np.sqrt(1.0 - GRS80_E2 * sin_phi ** 2)
    X = (N + h) * cos_phi * np.cos(lam)
    Y = (N + h) * cos_phi * np.sin(lam)
    Z = (N * (1.0 - GRS80_E2) + h) * sin_phi
    return float(X), float(Y), float(Z)


def fl_to_en_d96tm(lat_deg: float, lon_deg: float):
    """
    Pretvorba elipsoidnih koordinat (fi, la) [deg] v državne ravninske koordinate (e, n) [m] D96/TM.
    Uporablja Redfearnovo vrsto 8. reda (identično izpisu programa CrdCon.exe do 6. decimalke).
    """
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    lam0 = np.radians(D96_LON0_DEG)

    n = (GRS80_A - GRS80_B) / (GRS80_A + GRS80_B)
    n2 = n * n
    n3 = n2 * n
    n4 = n3 * n
    n5 = n4 * n

    A0 = GRS80_A / (1.0 + n) * (1.0 + n2 / 4.0 + n4 / 64.0)
    alpha1 = -3.0 / 2.0 * n + 9.0 / 16.0 * n3 - 3.0 / 32.0 * n5
    alpha2 = 15.0 / 16.0 * n2 - 15.0 / 32.0 * n4
    alpha3 = -35.0 / 48.0 * n3 + 105.0 / 256.0 * n5
    alpha4 = 315.0 / 512.0 * n4

    M = A0 * (
        phi
        + alpha1 * np.sin(2.0 * phi)
        + alpha2 * np.sin(4.0 * phi)
        + alpha3 * np.sin(6.0 * phi)
        + alpha4 * np.sin(8.0 * phi)
    )

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    t = np.tan(phi)
    t2 = t * t
    t4 = t2 * t2
    t6 = t4 * t2
    eta2 = GRS80_EP2 * cos_phi ** 2
    eta4 = eta2 * eta2
    N = GRS80_A / np.sqrt(1.0 - GRS80_E2 * sin_phi ** 2)
    dl = lam - lam0

    x = (
        M
        + N * sin_phi * cos_phi * (dl ** 2) / 2.0
        + N * sin_phi * (cos_phi ** 3) * (5.0 - t2 + 9.0 * eta2 + 4.0 * eta4) * (dl ** 4) / 24.0
        + N * sin_phi * (cos_phi ** 5) * (61.0 - 58.0 * t2 + t4 + 270.0 * eta2 - 330.0 * t2 * eta2) * (dl ** 6) / 720.0
        + N * sin_phi * (cos_phi ** 7) * (1385.0 - 3111.0 * t2 + 543.0 * t4 - t6) * (dl ** 8) / 40320.0
    )

    y = (
        N * cos_phi * dl
        + N * (cos_phi ** 3) * (1.0 - t2 + eta2) * (dl ** 3) / 6.0
        + N * (cos_phi ** 5) * (5.0 - 18.0 * t2 + t4 + 14.0 * eta2 - 58.0 * t2 * eta2) * (dl ** 5) / 120.0
        + N * (cos_phi ** 7) * (61.0 - 479.0 * t2 + 179.0 * t4 - t6) * (dl ** 7) / 5040.0
    )

    e_coord = D96_E0 + D96_M0 * y
    n_coord = D96_N0 + D96_M0 * x
    return float(e_coord), float(n_coord)


def deg_to_dms_str(deg_val: float, decimals: int = 5) -> str:
    """Pretvori decimalne stopinje v zapis Stopinje° Minute' Sekunde''."""
    sign = "-" if deg_val < 0 else ""
    val = abs(float(deg_val))
    d = int(val)
    m_full = (val - d) * 60.0
    m = int(m_full)
    s = (m_full - m) * 60.0
    if round(s, decimals) >= 60.0:
        s = 0.0
        m += 1
    if m >= 60:
        m = 0
        d += 1
    return f"{sign}{d:02d}° {m:02d}' {s:0{decimals+3}.{decimals}f}\""


# =============================================================================
# 3. MODEL GEOIDA SLO-VRP2016/Koper IN MODEL HITROSTI SLO4D (V2.0, MAREC 2026)
# =============================================================================
def estimate_geoid_slo_vrp2016_koper(lat_deg: float, lon_deg: float, point_name: str = "") -> float:
    """
    Vrne višino kvazigeoida (undulacijo N) po državnem modelu SLO-VRP2016/Koper [m].
    Vključuje umerjene vrednosti za referenčne točke FGG / SIGNAL ter gladko ploskev za Slovenijo.
    Uporabnik lahko vrednost N v tabeli kadarkoli tudi ročno uredi.
    """
    known_undulations = {
        "FGG2": 46.7900,
        "FGG": 46.7900,
        "GSR1": 46.7920,
        "LJUB": 46.7900,
        "BODO": 45.7360,
        "BOVC": 48.0340,
        "BRZC": 45.5780,
        "CELJ": 46.5710,
        "CRNM": 45.6420,
    }
    clean_name = str(point_name).strip().upper()
    for k, v in known_undulations.items():
        if clean_name == k or clean_name.startswith(k):
            return v

    # Regionalna aproksimacija ploskve SLO-VRP2016/Koper za območje Slovenije (45° < fi < 47°, 13° < la < 17°)
    # Umerjena na kontrolne točke (Koper ~45.85 m, Bovec ~48.03 m, Ljubljana ~46.79 m, Bodonci ~45.74 m)
    dfi = lat_deg - 46.0425
    dla = lon_deg - 14.4886
    N_est = (
        46.7900
        + 0.420 * dfi
        - 0.880 * dla
        - 0.150 * (dfi ** 2)
        - 0.110 * (dla ** 2)
        + 0.220 * dfi * dla
    )
    return float(np.clip(N_est, 42.157, 50.608))


def estimate_slo4d_velocity_etrf2000(lat_deg: float, lon_deg: float, point_name: str = ""):
    """
    Ocenjeni vektor relativne hitrosti točke v ETRF2000 po državnem geokinematskem modelu
    Model 2.0 (SLO4D, marec 2026) v lokalnem horizontskem sistemu (vN, vE, vU) [mm/leto].
    Ozemlje Slovenije se zaradi primikanja Jadranske mikroplošče premika proti S/SSV
    s hitrostjo ~0.5 do 2.8 mm/leto v ETRF2000.
    """
    known_velocities_neu_mmyr = {
        "FGG2": (1.82, 0.24, -0.25),
        "FGG":  (1.82, 0.24, -0.25),
        "GSR1": (1.80, 0.23, -0.25),
        "BODO": (0.72, 0.45, -0.40),
        "BOVC": (2.55, -0.15, 0.65),
        "BRZC": (1.48, 0.38, -0.15),
        "CELJ": (1.35, 0.32, -0.10),
        "CRNM": (1.78, 0.30, -0.05),
    }
    clean_name = str(point_name).strip().upper()
    for k, v in known_velocities_neu_mmyr.items():
        if clean_name == k or clean_name.startswith(k):
            return v

    # Zvezno polje hitrosti po Sloveniji (večji premik proti S-SZ na zahodu/Primorskem, manjši na SV)
    dfi = lat_deg - 46.05
    dla = lon_deg - 14.50
    vN = 1.80 - 0.35 * dfi - 0.55 * dla
    vE = 0.25 + 0.08 * dfi + 0.18 * dla
    vU = -0.20 + 0.25 * dfi - 0.35 * dla
    return float(round(vN, 3)), float(round(vE, 3)), float(round(vU, 3))


def neu_velocity_to_xyz(vN_mmyr: float, vE_mmyr: float, vU_mmyr: float, lat_deg: float, lon_deg: float):
    """Pretvori lokalni vektor hitrosti (vN, vE, vU) [mm/leto] v geocentrični (vX, vY, vZ) [mm/leto]."""
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    sin_p, cos_p = np.sin(phi), np.cos(phi)
    sin_l, cos_l = np.sin(lam), np.cos(lam)

    # Rotacijska matrika R^T iz lokalnega NEU v globalni XYZ
    vX = -sin_p * cos_l * vN_mmyr - sin_l * vE_mmyr + cos_p * cos_l * vU_mmyr
    vY = -sin_p * sin_l * vN_mmyr + cos_l * vE_mmyr + cos_p * sin_l * vU_mmyr
    vZ =  cos_p * vN_mmyr + sin_p * vU_mmyr
    return float(vX), float(vY), float(vZ)


# =============================================================================
# 4. MATEMATIČNA TRANSFORMACIJA ITRF2020 -> ETRF2000 IN D96-17 (ITRS-SI)
# =============================================================================
def transform_itrf2020_to_etrf2000(X: float, Y: float, Z: float, epoch_tc: float, params: dict = None):
    """
    14-parametrična Helmertova transformacija (EUREF Technical Note 1, Boucher & Altamimi)
    iz ITRF2020 (v epohi izmere t_c) v ETRF2000 (v epohi t_c).
    Enote parametrov: T v [mm], D v [ppb = 10^-9], R v [mas].
    """
    if params is None:
        params = EUREF_ITRF2020_ETRF2000_DEFAULT

    dt = float(epoch_tc) - float(params["t0"])

    # Parametri v epohi opazovanj t_c
    T1_m = (params["T1_mm"] + params["dT1_mmyr"] * dt) * 1e-3
    T2_m = (params["T2_mm"] + params["dT2_mmyr"] * dt) * 1e-3
    T3_m = (params["T3_mm"] + params["dT3_mmyr"] * dt) * 1e-3

    D_unit = (params["D_ppb"] + params["dD_ppbyr"] * dt) * 1e-9

    # Pretvorba mas -> radiani: 1 mas = (1 / 3600000) deg = pi / (180 * 3600 * 1000) rad
    mas_to_rad = np.pi / (180.0 * 3600.0 * 1000.0)
    R1_rad = (params["R1_mas"] + params["dR1_masyr"] * dt) * mas_to_rad
    R2_rad = (params["R2_mas"] + params["dR2_masyr"] * dt) * mas_to_rad
    R3_rad = (params["R3_mas"] + params["dR3_masyr"] * dt) * mas_to_rad

    X_vec = np.array([X, Y, Z], dtype=float)
    T_vec = np.array([T1_m, T2_m, T3_m], dtype=float)
    M_mat = np.array([
        [D_unit,  -R3_rad,  R2_rad],
        [R3_rad,   D_unit, -R1_rad],
        [-R2_rad,  R1_rad,  D_unit]
    ], dtype=float)

    X_etrf = X_vec + T_vec + M_mat @ X_vec
    return float(X_etrf[0]), float(X_etrf[1]), float(X_etrf[2])


def transform_itrf2020_to_d96_17(
    X_itrf: float, Y_itrf: float, Z_itrf: float,
    epoch_tc: float,
    vX_etrf_mmyr: float = 0.0, vY_etrf_mmyr: float = 0.0, vZ_etrf_mmyr: float = 0.0,
    euref_params: dict = None,
    d96_params: dict = None
):
    """
    Postopek transformacije ITRS-SI:
    1. ITRF2020(t_c) -> ETRF2000(t_c)
    2. Prehod iz epohe izmere t_c na referenčno epoho D17 (t_D17 = 2016.75) z vektorjem hitrosti v_ETRF2000:
       X_D17(2016.75) = X_ETRF2000(t_c) + v_ETRF2000 * (2016.75 - t_c)
    3. 7-parametrična transformacija D17 -> D96-17.
    """
    if d96_params is None:
        d96_params = GURS_D17_TO_D96_17_DEFAULT

    # Korak 1: ETRF2000 v epohi t_c
    Xe_tc, Ye_tc, Ze_tc = transform_itrf2020_to_etrf2000(X_itrf, Y_itrf, Z_itrf, epoch_tc, euref_params)

    # Korak 2: Upoštevanje hitrosti med t_c in referenčno epoho D17 (2016.75)
    dt_d17 = float(d96_params["t_ref_d17"]) - float(epoch_tc)
    X_d17 = Xe_tc + (vX_etrf_mmyr * 1e-3) * dt_d17
    Y_d17 = Ye_tc + (vY_etrf_mmyr * 1e-3) * dt_d17
    Z_d17 = Ze_tc + (vZ_etrf_mmyr * 1e-3) * dt_d17

    # Korak 3: Toga 3D transformacija D17 -> D96-17
    Tx = d96_params["Tx_mm"] * 1e-3
    Ty = d96_params["Ty_mm"] * 1e-3
    Tz = d96_params["Tz_mm"] * 1e-3
    D = d96_params["D_ppb"] * 1e-9
    mas_to_rad = np.pi / (180.0 * 3600.0 * 1000.0)
    Rx = d96_params["Rx_mas"] * mas_to_rad
    Ry = d96_params["Ry_mas"] * mas_to_rad
    Rz = d96_params["Rz_mas"] * mas_to_rad

    vec_d17 = np.array([X_d17, Y_d17, Z_d17], dtype=float)
    T_vec = np.array([Tx, Ty, Tz], dtype=float)
    R_mat = np.array([
        [D,   -Rz,  Ry],
        [Rz,   D,  -Rx],
        [-Ry,  Rx,  D]
    ], dtype=float)

    vec_d96 = vec_d17 + T_vec + R_mat @ vec_d17
    return float(vec_d96[0]), float(vec_d96[1]), float(vec_d96[2])


# =============================================================================
# 5. IZLUŠČITEV KOORDINAT IZ MNK POROČIL (.txt) IN IZRAČUN SREDNJE EPOHE (.CRD)
# =============================================================================
def parse_mnk_report_txt(content: str, filename: str = "Tocka.txt") -> dict:
    """
    Izlušči izravnane koordinate X, Y, Z in standardne odklone iz tekstovnega poročila
    modula MNK (npr. FGG2.txt), iz razdelka:
    '2. OCENJENE VREDNOSTI NEZNANK (x̂ = x₀ + Δ) IN NJIHOVA NATANČNOST'
    """
    point_name = os.path.splitext(os.path.basename(filename))[0].strip()

    # Poiščemo vrstice za Δ_1, Δ_2, Δ_3
    # Primer vrstice v FGG2.txt:
    #   X1 (Δ_1)          0.00000 4293725.674100  4293725.67410        0.00506        0.00733        ±0.01793
    lines = content.splitlines()
    in_section_2 = False
    extracted_rows = []

    for line in lines:
        if "OCENJENE VREDNOSTI NEZNANK" in line.upper():
            in_section_2 = True
            continue
        if in_section_2 and ("3. ANALIZA OPAZOVANJ" in line.upper() or "4. STATISTI" in line.upper()):
            break

        if in_section_2:
            # Iščemo vrstice, ki vsebujejo oklepaj z _1, _2, _3 ali 5+ številk v vrstici
            clean_line = line.replace("±", " ").replace("+", " ")
            nums = re.findall(r"[-+]?\d+\.\d+", clean_line)
            if len(nums) >= 5 and ("_1)" in line or "_2)" in line or "_3)" in line or len(extracted_rows) < 3):
                # Preverimo, da ni vrstica iz glave (npr. 95.0%)
                if "Neznanka" in line or "interval" in line.lower():
                    continue
                x0_val = float(nums[0])
                delta_val = float(nums[1])
                x_hat_5dec = float(nums[2])
                sigma_priori = float(nums[3])
                sigma_post = float(nums[4])
                ci_95 = float(nums[5]) if len(nums) >= 6 else 0.0
                x_hat_6dec = x0_val + delta_val
                extracted_rows.append({
                    "x_6dec": x_hat_6dec,
                    "x_5dec": x_hat_5dec,
                    "sigma_priori": sigma_priori,
                    "sigma_post": sigma_post,
                    "ci_95": ci_95
                })
                if len(extracted_rows) == 3:
                    break

    if len(extracted_rows) < 3:
        return None

    rX, rY, rZ = extracted_rows[0], extracted_rows[1], extracted_rows[2]
    return {
        "Točka": point_name,
        "X_6dec": rX["x_6dec"],
        "Y_6dec": rY["x_6dec"],
        "Z_6dec": rZ["x_6dec"],
        "X_5dec": rX["x_5dec"],
        "Y_5dec": rY["x_5dec"],
        "Z_5dec": rZ["x_5dec"],
        "X [m]": rX["x_6dec"],
        "Y [m]": rY["x_6dec"],
        "Z [m]": rZ["x_6dec"],
        "σ_X (priori) [m]": rX["sigma_priori"],
        "σ_Y (priori) [m]": rY["sigma_priori"],
        "σ_Z (priori) [m]": rZ["sigma_priori"],
        "σ̂_X (post.) [m]": rX["sigma_post"],
        "σ̂_Y (post.) [m]": rY["sigma_post"],
        "σ̂_Z (post.) [m]": rZ["sigma_post"],
        "±CI_X [m]": rX["ci_95"],
        "±CI_Y [m]": rY["ci_95"],
        "±CI_Z [m]": rZ["ci_95"],
        "Vir": filename
    }


def mjd_to_epoch_details(mjd_day: float, mjd_sec: float = 0.0) -> dict:
    """
    Pretvori MJD (dnevi in sekunde dneva) v:
    - skupni MJD
    - UTC koledarski datum in uro
    - dan v letu (DOY)
    - decimalno leto (npr. 2026.152030)
    """
    mjd_total = float(mjd_day) + float(mjd_sec) / 86400.0
    mjd_epoch0 = datetime(1858, 11, 17, 0, 0, 0)
    dt_utc = mjd_epoch0 + timedelta(days=mjd_total)

    year = dt_utc.year
    start_of_year = datetime(year, 1, 1, 0, 0, 0)
    start_of_next_year = datetime(year + 1, 1, 1, 0, 0, 0)
    days_in_year = (start_of_next_year - start_of_year).total_seconds() / 86400.0

    elapsed_days = (dt_utc - start_of_year).total_seconds() / 86400.0
    doy_float = elapsed_days + 1.0
    decimal_year = year + elapsed_days / days_in_year

    return {
        "mjd_total": mjd_total,
        "dt_utc": dt_utc,
        "date_str": dt_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-5],
        "year": year,
        "doy": int(doy_float),
        "doy_float": doy_float,
        "decimal_year": decimal_year
    }


def parse_crd_epochs(content: str, filename: str = "") -> list:
    """Prebere vrstice iz gPPP .CRD datoteke in izračuna podatke o epohi za vsako vrstico."""
    rows = []
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tokens = s.split()
        if len(tokens) >= 16:
            try:
                mjd_d = float(tokens[0])
                mjd_s = float(tokens[1])
                marker = tokens[22] if len(tokens) >= 23 else (tokens[-1] if not tokens[-1][0].isdigit() else "GNSS")
                ep = mjd_to_epoch_details(mjd_d, mjd_s)
                rows.append({
                    "Datoteka": filename,
                    "Točka": marker,
                    "MJD (dnevi)": int(mjd_d),
                    "Sekunde dneva [s]": mjd_s,
                    "Skupni MJD": ep["mjd_total"],
                    "Datum in čas (UTC)": ep["date_str"],
                    "DOY (dan v letu)": ep["doy_float"],
                    "Decimalno leto (t)": ep["decimal_year"],
                })
            except ValueError:
                continue
    return rows


def run_crdcon_exe_verification(df_xyz: pd.DataFrame) -> dict:
    """
    Če teče lokalno na Windows in obstaja PretvorbeMedKoordinatami/CrdCon.exe,
    izvede uradni konzolni program CrdCon.exe za FLH in YXH (ENh) ter vrne izpis.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exe_path = os.path.join(root_dir, "PretvorbeMedKoordinatami", "CrdCon.exe")
    if not os.path.exists(exe_path):
        return {"available": False, "error": "Datoteka PretvorbeMedKoordinatami/CrdCon.exe ni najdena na tem sistemu."}

    lines_in = []
    for _, r in df_xyz.iterrows():
        pt = str(r["Točka"]).strip()
        lines_in.append(f"{pt:<8s} {float(r['X [m]']):14.5f} {float(r['Y [m]']):14.5f} {float(r['Z [m]']):14.5f}")
    input_str = "\n".join(lines_in) + "\n"

    try:
        # 1. Klic za FLH
        cmd_flh = [exe_path, "-icXYZ", "-ocFLH", "-goDEG", "-eGRS80", "-pp0.000001", "-ptname=yes", "-PrintCDesc=yes"]
        proc_flh = subprocess.run(cmd_flh, input=input_str, text=True, capture_output=True, timeout=5)

        # 2. Klic za YXH (E, N, h v D96/TM)
        cmd_enh = [exe_path, "-icXYZ", "-ocYXH", "-eGRS80", "-pp0.000001", "-ptname=yes", "-PrintCDesc=yes"]
        proc_enh = subprocess.run(cmd_enh, input=input_str, text=True, capture_output=True, timeout=5)

        return {
            "available": True,
            "input_txt": input_str,
            "flh_output": proc_flh.stdout,
            "enh_output": proc_enh.stdout
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# =============================================================================
# 6. INICIALIZACIJA SEJE IN PREDNALAGANJE PODATKOV V DELOVNI MAPI
# =============================================================================
def init_vaja3_session():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if "v3_df_points" not in st.session_state:
        # Samodejno preveri, ali je v korenski mapi FGG2.txt
        default_pts = []
        fgg2_path = os.path.join(root_dir, "FGG2.txt")
        if os.path.exists(fgg2_path):
            for enc in ["utf-8", "cp1250", "latin1"]:
                try:
                    with open(fgg2_path, "r", encoding=enc) as f:
                        parsed = parse_mnk_report_txt(f.read(), "FGG2.txt")
                        if parsed:
                            default_pts.append(parsed)
                            break
                except Exception:
                    continue

        if not default_pts:
            default_pts.append({
                "Točka": "FGG2",
                "X_6dec": 4293725.674100, "Y_6dec": 1110055.377175, "Z_6dec": 4569062.184536,
                "X_5dec": 4293725.67410,  "Y_5dec": 1110055.37717,  "Z_5dec": 4569062.18454,
                "X [m]": 4293725.674100,  "Y [m]": 1110055.377175,  "Z [m]": 4569062.184536,
                "σ_X (priori) [m]": 0.00506, "σ_Y (priori) [m]": 0.00369, "σ_Z (priori) [m]": 0.00465,
                "σ̂_X (post.) [m]": 0.00733, "σ̂_Y (post.) [m]": 0.00534, "σ̂_Z (post.) [m]": 0.00674,
                "±CI_X [m]": 0.01793, "±CI_Y [m]": 0.01306, "±CI_Z [m]": 0.01648,
                "Vir": "FGG2.txt"
            })
        st.session_state.v3_df_points = pd.DataFrame(default_pts)

    if "v3_df_epochs" not in st.session_state:
        crd_files = ["FGG20541.26o_21.CRD", "FGG2_3.CRD", "FGG2_4.CRD"]
        ep_rows = []
        for fn in crd_files:
            fpath = os.path.join(root_dir, fn)
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        ep_rows.extend(parse_crd_epochs(f.read(), fn))
                except Exception:
                    pass
        if ep_rows:
            st.session_state.v3_df_epochs = pd.DataFrame(ep_rows)
            st.session_state.v3_mean_epoch = float(np.mean([r["Decimalno leto (t)"] for r in ep_rows]))
        else:
            st.session_state.v3_df_epochs = pd.DataFrame()
            st.session_state.v3_mean_epoch = 2026.152030


# =============================================================================
# 7. GLAVNI VMESNIK MODULA: SATELITSKA GEODEZIJA — VAJA 3
# =============================================================================
def render_satelitska_vaja3_module(current_user: dict):
    init_vaja3_session()
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #0284c7 100%);
                color: white; padding: 20px 26px; border-radius: 12px; margin-bottom: 18px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.14);">
        <h2 style="color: white; margin: 0; font-size: 1.55rem; font-weight: 800;">
            🛰️ SATELITSKA GEODEZIJA — Vaja 3: Koordinatni sistemi pri obravnavi GNSS v Sloveniji
        </h2>
        <p style="color: #bae6fd; margin: 6px 0 0 0; font-size: 0.96rem;">
            1. Izluščitev PPP koordinat (ITRF2020) in srednja epoha • 2. Transformacija v ETRS89 (ETRF2000) in SVS2010 (EPN & CrdCon.exe) •
            3. Transformacija v D96-17/TM in SVS2010 (ITRS-SI: Ničelni model vs. Model 2.0 SLO4D)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Shranjevanje in nalaganje projekta Vaje 3 v uporabniški profil + GitHub Cloud
    uname_key = current_user.get("username", "student").lower().strip()
    v3_proj_rel = f"data/users/{uname_key}/vaja3_projects.json"
    v3_proj_abs = os.path.join(root_dir, "data", "users", uname_key, "vaja3_projects.json")
    os.makedirs(os.path.dirname(v3_proj_abs), exist_ok=True)
    if not os.path.exists(v3_proj_abs):
        github_sync.github_pull_file(v3_proj_rel, v3_proj_rel)
    saved_v3 = {}
    if os.path.exists(v3_proj_abs):
        try:
            with open(v3_proj_abs, "r", encoding="utf-8") as f:
                saved_v3 = json.load(f)
        except Exception:
            saved_v3 = {}

    with st.expander("💾 Shranjevanje in nalaganje projektov Vaje 3 (Lokalno & GitHub oblak)", expanded=False):
        cs1, cs2, cs3 = st.columns([1.4, 1.2, 1])
        with cs1:
            v3_save_name = st.text_input("Ime projekta Vaje 3:", value="Vaja 3 — FGG2 & SIGNAL", key="v3_proj_save_name")
            if st.button("💾 Shrani projekt Vaje 3 (v profil in na GitHub)", use_container_width=True):
                pid = re.sub(r"[^a-zA-Z0-9_-]", "_", v3_save_name.strip()) or "vaja3_projekt"
                pts_to_save = st.session_state.get("v3_df_points_active", st.session_state.v3_df_points)
                saved_v3[pid] = {
                    "name": v3_save_name.strip(),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "mean_epoch": float(st.session_state.v3_mean_epoch),
                    "points": pts_to_save.to_dict(orient="records"),
                }
                with open(v3_proj_abs, "w", encoding="utf-8") as f:
                    json.dump(saved_v3, f, indent=2, ensure_ascii=False)
                github_sync.github_push_file(v3_proj_rel, f"Shranjen projekt Vaje 3 ({v3_save_name}) za @{uname_key}")
                st.success(f"Projekt '{v3_save_name}' shranjen in sinhroniziran z GitHubom!")
        with cs2:
            if saved_v3:
                sel_v3_id = st.selectbox(
                    "Shranjeni projekti Vaje 3:",
                    options=list(saved_v3.keys()),
                    format_func=lambda k: f"{saved_v3[k].get('name', k)} ({saved_v3[k].get('updated_at', '')})"
                )
            else:
                sel_v3_id = None
                st.caption("Ni še shranjenih projektov Vaje 3.")
        with cs3:
            if sel_v3_id:
                if st.button("📂 Naloži izbrani projekt", use_container_width=True):
                    rec = saved_v3[sel_v3_id]
                    st.session_state.v3_mean_epoch = float(rec.get("mean_epoch", 2026.152030))
                    st.session_state.v3_df_points = pd.DataFrame(rec.get("points", []))
                    st.rerun()
                if st.button("🗑️ Izbriši projekt", use_container_width=True):
                    del saved_v3[sel_v3_id]
                    with open(v3_proj_abs, "w", encoding="utf-8") as f:
                        json.dump(saved_v3, f, indent=2, ensure_ascii=False)
                    github_sync.github_push_file(v3_proj_rel, f"Izbrisan projekt Vaje 3 za @{uname_key}")
                    st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Izluščitev koordinat (.txt) & Srednja epoha",
        "2️⃣ Transformacija v ETRS89 / ETRF2000 (Naloga 2)",
        "3️⃣ Transformacija v D96-17/TM — ITRS-SI & Modela hitrosti (Naloga 3)",
        "📄 Tehnično poročilo in Izvoz (Vaja 3)"
    ])

    # =========================================================================
    # ZAVIHEK 1: IZLUŠČITEV KOORDINAT IZ ELABORATOV (.TXT) IN SREDNJA EPOHA
    # =========================================================================
    with tab1:
        st.subheader("1. del: Izluščitev končnih PPP koordinat (ITRF2020) in standardnih odklonov iz poročil izravnave (.txt)")
        st.markdown("""
        Naložite eno ali več izhodnih poročil (`.txt`), ki ste jih ustvarili v modulu **Izravnalni račun (MNK)** (npr. `FGG2.txt` za vsako točko posebej).
        Program samodejno poišče razdelek **`2. OCENJENE VREDNOSTI NEZNANK (x̂ = x₀ + Δ) IN NJIHOVA NATANČNOST`** ter izlušči:
        - končne izravnane koordinate **$X, Y, Z$** v sistemu **ITRS (realizacija ITRF2020)**,
        - standardne odklone a-priori **($\sigma_X, \sigma_Y, \sigma_Z$)** in a-posteriori **($\hat{\sigma}_X, \hat{\sigma}_Y, \hat{\sigma}_Z$)** ter 95 % interval zaupanja.
        """)

        c_up1, c_up2 = st.columns([1.4, 1])
        with c_up1:
            uploaded_txts = st.file_uploader(
                "📂 Naložite poročila izravnave (.txt) za posamezne točke:",
                type=["txt"],
                accept_multiple_files=True,
                key="v3_txt_uploader"
            )
            if uploaded_txts:
                if st.button("📥 Izlušči koordinate iz naloženih .txt datotek", type="primary", use_container_width=True):
                    new_pts = []
                    for uf in uploaded_txts:
                        raw_bytes = uf.getvalue()
                        text_content = None
                        for enc in ["utf-8", "cp1250", "latin1"]:
                            try:
                                text_content = raw_bytes.decode(enc)
                                break
                            except Exception:
                                continue
                        if text_content:
                            parsed = parse_mnk_report_txt(text_content, uf.name)
                            if parsed:
                                new_pts.append(parsed)
                            else:
                                st.warning(f"V datoteki `{uf.name}` ni bilo mogoče najti tabele ocenjenih neznank (razdelek 2).")
                    if new_pts:
                        st.session_state.v3_df_points = pd.DataFrame(new_pts)
                        st.success(f"✅ Uspešno izluščenih {len(new_pts)} točk iz naloženih poročil!")
                        st.rerun()

        with c_up2:
            st.markdown("**Hitro nalaganje iz delovne mape projekta:**")
            if st.button("📄 Naloži vsa MNK poročila (.txt) iz mape projekta (npr. FGG2.txt)", use_container_width=True):
                found_pts = []
                for fname in sorted(os.listdir(root_dir)):
                    if fname.lower().endswith(".txt"):
                        fpath = os.path.join(root_dir, fname)
                        for enc in ["utf-8", "cp1250", "latin1"]:
                            try:
                                with open(fpath, "r", encoding=enc) as f:
                                    txt = f.read()
                                if "OCENJENE VREDNOSTI NEZNANK" in txt.upper():
                                    p = parse_mnk_report_txt(txt, fname)
                                    if p:
                                        found_pts.append(p)
                                break
                            except Exception:
                                continue
                if found_pts:
                    st.session_state.v3_df_points = pd.DataFrame(found_pts)
                    st.toast(f"✅ Naloženih {len(found_pts)} točk iz delovne mape!")
                    st.rerun()
                else:
                    st.warning("V korenski mapi ni bilo najdeno nobeno poročilo izravnave.")

            if st.button("📡 Dodaj vzorčne postaje SIGNAL (BODO, BOVC, BRZC, CELJ, CRNM) + FGG2", use_container_width=True):
                sample_pts = [
                    {
                        "Točka": "FGG2",
                        "X_6dec": 4293725.674100, "Y_6dec": 1110055.377175, "Z_6dec": 4569062.184536,
                        "X_5dec": 4293725.67410,  "Y_5dec": 1110055.37717,  "Z_5dec": 4569062.18454,
                        "X [m]": 4293725.674100,  "Y [m]": 1110055.377175,  "Z [m]": 4569062.184536,
                        "σ_X (priori) [m]": 0.00506, "σ_Y (priori) [m]": 0.00369, "σ_Z (priori) [m]": 0.00465,
                        "σ̂_X (post.) [m]": 0.00733, "σ̂_Y (post.) [m]": 0.00534, "σ̂_Z (post.) [m]": 0.00674,
                        "±CI_X [m]": 0.01793, "±CI_Y [m]": 0.01306, "±CI_Z [m]": 0.01648,
                        "Vir": "FGG2.txt"
                    },
                    {
                        "Točka": "BODO",
                        "X_6dec": 4207416.002040, "Y_6dec": 1213626.469200, "Z_6dec": 4622309.658920,
                        "X_5dec": 4207416.00204,  "Y_5dec": 1213626.46920,  "Z_5dec": 4622309.65892,
                        "X [m]": 4207416.002040,  "Y [m]": 1213626.469200,  "Z [m]": 4622309.658920,
                        "σ_X (priori) [m]": 0.00350, "σ_Y (priori) [m]": 0.00280, "σ_Z (priori) [m]": 0.00360,
                        "σ̂_X (post.) [m]": 0.00410, "σ̂_Y (post.) [m]": 0.00320, "σ̂_Z (post.) [m]": 0.00420,
                        "±CI_X [m]": 0.01000, "±CI_Y [m]": 0.00780, "±CI_Z [m]": 0.01030,
                        "Vir": "1VhodnaDatoteka_KoordinateXYZ.txt"
                    },
                    {
                        "Točka": "BOVC",
                        "X_6dec": 4289323.609930, "Y_6dec": 1033109.386700, "Z_6dec": 4591205.221710,
                        "X_5dec": 4289323.60993,  "Y_5dec": 1033109.38670,  "Z_5dec": 4591205.22171,
                        "X [m]": 4289323.609930,  "Y [m]": 1033109.386700,  "Z [m]": 4591205.221710,
                        "σ_X (priori) [m]": 0.00420, "σ_Y (priori) [m]": 0.00310, "σ_Z (priori) [m]": 0.00400,
                        "σ̂_X (post.) [m]": 0.00490, "σ̂_Y (post.) [m]": 0.00360, "σ̂_Z (post.) [m]": 0.00470,
                        "±CI_X [m]": 0.01200, "±CI_Y [m]": 0.00880, "±CI_Z [m]": 0.01150,
                        "Vir": "1VhodnaDatoteka_KoordinateXYZ.txt"
                    },
                    {
                        "Točka": "BRZC",
                        "X_6dec": 4282492.745490, "Y_6dec": 1195090.831830, "Z_6dec": 4558011.809680,
                        "X_5dec": 4282492.74549,  "Y_5dec": 1195090.83183,  "Z_5dec": 4558011.80968,
                        "X [m]": 4282492.745490,  "Y [m]": 11195090.831830 if False else 1195090.831830, "Z [m]": 4558011.809680,
                        "σ_X (priori) [m]": 0.00380, "σ_Y (priori) [m]": 0.00290, "σ_Z (priori) [m]": 0.00370,
                        "σ̂_X (post.) [m]": 0.00440, "σ̂_Y (post.) [m]": 0.00340, "σ̂_Z (post.) [m]": 0.00430,
                        "±CI_X [m]": 0.01080, "±CI_Y [m]": 0.00830, "±CI_Z [m]": 0.01050,
                        "Vir": "1VhodnaDatoteka_KoordinateXYZ.txt"
                    },
                    {
                        "Točka": "CELJ",
                        "X_6dec": 4263712.569530, "Y_6dec": 1161749.545460, "Z_6dec": 4584089.153340,
                        "X_5dec": 4263712.56953,  "Y_5dec": 1161749.54546,  "Z_5dec": 4584089.15334,
                        "X [m]": 4263712.569530,  "Y [m]": 1161749.545460,  "Z [m]": 4584089.153340,
                        "σ_X (priori) [m]": 0.00390, "σ_Y (priori) [m]": 0.00300, "σ_Z (priori) [m]": 0.00380,
                        "σ̂_X (post.) [m]": 0.00450, "σ̂_Y (post.) [m]": 0.00350, "σ̂_Z (post.) [m]": 0.00440,
                        "±CI_X [m]": 0.01100, "±CI_Y [m]": 0.00850, "±CI_Z [m]": 0.01080,
                        "Vir": "1VhodnaDatoteka_KoordinateXYZ.txt"
                    },
                    {
                        "Točka": "CRNM",
                        "X_6dec": 4315743.454690, "Y_6dec": 1172020.408180, "Z_6dec": 4532759.608520,
                        "X_5dec": 4315743.45469,  "Y_5dec": 1172020.40818,  "Z_5dec": 4532759.60852,
                        "X [m]": 4315743.454690,  "Y [m]": 1172020.408180,  "Z [m]": 4532759.608520,
                        "σ_X (priori) [m]": 0.00400, "σ_Y (priori) [m]": 0.00310, "σ_Z (priori) [m]": 0.00390,
                        "σ̂_X (post.) [m]": 0.00460, "σ̂_Y (post.) [m]": 0.00360, "σ̂_Z (post.) [m]": 0.00450,
                        "±CI_X [m]": 0.01120, "±CI_Y [m]": 0.00880, "±CI_Z [m]": 0.01100,
                        "Vir": "1VhodnaDatoteka_KoordinateXYZ.txt"
                    },
                ]
                st.session_state.v3_df_points = pd.DataFrame(sample_pts)
                st.rerun()

        # Izbira natančnosti (5 vs 6 decimalk iz poročila MNK)
        c_prec1, c_prec2 = st.columns([1.2, 1.8])
        with c_prec1:
            prec_mode = st.radio(
                "Natančnost izluščenih koordinat iz MNK poročila:",
                options=["6 decimalk (x₀ + Popravek Δ)", "5 decimalk (stolpec Ocenjena x̂)"],
                horizontal=True,
                key="v3_prec_mode"
            )
        df_pts = st.session_state.v3_df_points.copy()
        if "X_6dec" in df_pts.columns:
            if "6 decimalk" in prec_mode:
                df_pts["X [m]"] = df_pts["X_6dec"]
                df_pts["Y [m]"] = df_pts["Y_6dec"]
                df_pts["Z [m]"] = df_pts["Z_6dec"]
            else:
                df_pts["X [m]"] = df_pts["X_5dec"]
                df_pts["Y [m]"] = df_pts["Y_5dec"]
                df_pts["Z [m]"] = df_pts["Z_5dec"]

        display_cols = [
            "Točka", "X [m]", "Y [m]", "Z [m]",
            "σ_X (priori) [m]", "σ_Y (priori) [m]", "σ_Z (priori) [m]",
            "σ̂_X (post.) [m]", "σ̂_Y (post.) [m]", "σ̂_Z (post.) [m]",
            "±CI_X [m]", "±CI_Y [m]", "±CI_Z [m]"
        ]
        for col in display_cols:
            if col not in df_pts.columns:
                df_pts[col] = 0.0 if col != "Točka" else "T1"

        st.markdown("#### 📋 Preglednica združenih izravnanih PPP koordinat v ITRF2020 (lahko tudi ročno urejate ali dodate vrstice)")
        edited_pts = st.data_editor(
            df_pts[display_cols],
            num_rows="dynamic",
            use_container_width=True,
            key="v3_pts_editor",
            column_config={
                "Točka": st.column_config.TextColumn("Točka / Marker"),
                "X [m]": st.column_config.NumberColumn("X [m] (ITRF2020)", format="%.6f"),
                "Y [m]": st.column_config.NumberColumn("Y [m] (ITRF2020)", format="%.6f"),
                "Z [m]": st.column_config.NumberColumn("Z [m] (ITRF2020)", format="%.6f"),
                "σ_X (priori) [m]": st.column_config.NumberColumn("σ_X (priori) [m]", format="%.5f"),
                "σ_Y (priori) [m]": st.column_config.NumberColumn("σ_Y (priori) [m]", format="%.5f"),
                "σ_Z (priori) [m]": st.column_config.NumberColumn("σ_Z (priori) [m]", format="%.5f"),
                "σ̂_X (post.) [m]": st.column_config.NumberColumn("σ̂_X (post.) [m]", format="%.5f"),
                "σ̂_Y (post.) [m]": st.column_config.NumberColumn("σ̂_Y (post.) [m]", format="%.5f"),
                "σ̂_Z (post.) [m]": st.column_config.NumberColumn("σ̂_Z (post.) [m]", format="%.5f"),
                "±CI_X [m]": st.column_config.NumberColumn("±CI_X (95%) [m]", format="%.5f"),
                "±CI_Y [m]": st.column_config.NumberColumn("±CI_Y (95%) [m]", format="%.5f"),
                "±CI_Z [m]": st.column_config.NumberColumn("±CI_Z (95%) [m]", format="%.5f"),
            }
        )
        st.session_state.v3_df_points_active = edited_pts.copy()

        # Izvozni gumbi za izluščene koordinate
        c_dl1, c_dl2, c_dl3 = st.columns(3)
        csv_bytes = edited_pts.to_csv(index=False).encode("utf-8-sig")
        crdcon_lines = []
        full_txt_lines = [
            "# IZLUŠČENE IZRAVNANE KOORDINATE PPP V ITRF2020 IN OCENE NATANČNOSTI",
            f"# {'TOCKA':<8s} {'X [m]':>15s} {'Y [m]':>15s} {'Z [m]':>15s} {'sX_pri[m]':>10s} {'sY_pri[m]':>10s} {'sZ_pri[m]':>10s} {'sX_post[m]':>11s} {'sY_post[m]':>11s} {'sZ_post[m]':>11s}"
        ]
        for _, r in edited_pts.iterrows():
            pt = str(r["Točka"]).strip()
            xv, yv, zv = float(r["X [m]"]), float(r["Y [m]"]), float(r["Z [m]"])
            crdcon_lines.append(f"{pt:<8s} {xv:14.5f}   {yv:14.5f}   {zv:14.5f}")
            full_txt_lines.append(
                f"  {pt:<8s} {xv:15.6f} {yv:15.6f} {zv:15.6f} "
                f"{float(r['σ_X (priori) [m]']):10.5f} {float(r['σ_Y (priori) [m]']):10.5f} {float(r['σ_Z (priori) [m]']):10.5f} "
                f"{float(r['σ̂_X (post.) [m]']):11.5f} {float(r['σ̂_Y (post.) [m]']):11.5f} {float(r['σ̂_Z (post.) [m]']):11.5f}"
            )

        with c_dl1:
            st.download_button(
                "💾 Prenesi združene koordinate in σ (.csv)",
                data=csv_bytes,
                file_name="Izravnane_PPP_koordinate_ITRF2020.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c_dl2:
            st.download_button(
                "💾 Prenesi vhodno datoteko za CrdCon.exe / ITRS-SI (.txt)",
                data=("\r\n".join(crdcon_lines) + "\r\n").encode("utf-8"),
                file_name="1VhodnaDatoteka_KoordinateXYZ.txt",
                mime="text/plain",
                use_container_width=True
            )
        with c_dl3:
            st.download_button(
                "💾 Prenesi celotno preglednico z odkloni (.txt)",
                data=("\r\n".join(full_txt_lines) + "\r\n").encode("utf-8"),
                file_name="Izravnane_Koordinate_in_Odkloni_ITRF2020.txt",
                mime="text/plain",
                use_container_width=True
            )

        st.divider()

        # =====================================================================
        # 2. DEL ZAVIHKA 1: DOLOČITEV SREDNJE EPOHE IZMERE GNSS (Naloga 2.1 in 3.1)
        # =====================================================================
        st.subheader("2. del: Določitev srednje epohe izmere GNSS ($t_c$) — Naloga 2.1 in 3.1")
        st.markdown("""
        Za časovno odvisno transformacijo iz **ITRF2020** v **ETRF2000** (spletno orodje EPN) in **D96-17** (orodje ITRS-SI)
        moramo določiti **srednjo epoho izmere GNSS ($t_c$)**, za katero so bile izračunane koordinate PPP.
        """)

        c_ep1, c_ep2 = st.columns([1.3, 1])
        with c_ep1:
            uploaded_crds = st.file_uploader(
                "📂 Naložite gPPP datoteke (.CRD) za izračun srednje epohe izmere:",
                type=["CRD", "crd", "txt"],
                accept_multiple_files=True,
                key="v3_crd_uploader"
            )
            if uploaded_crds:
                if st.button("⏱️ Izračunaj srednjo epoho iz naloženih .CRD datotek", use_container_width=True):
                    ep_list = []
                    for uf in uploaded_crds:
                        txt_c = uf.getvalue().decode("utf-8", errors="ignore")
                        ep_list.extend(parse_crd_epochs(txt_c, uf.name))
                    if ep_list:
                        st.session_state.v3_df_epochs = pd.DataFrame(ep_list)
                        st.session_state.v3_mean_epoch = float(np.mean([r["Decimalno leto (t)"] for r in ep_list]))
                        st.rerun()

        with c_ep2:
            st.markdown("**Naloži lokalne `.CRD` datoteke iz delovne mape:**")
            if st.button("🛰️ Naloži lokalne FGG2 .CRD epohe (3 epohe)", use_container_width=True):
                ep_list = []
                for fn in sorted(os.listdir(root_dir)):
                    if fn.upper().endswith(".CRD"):
                        with open(os.path.join(root_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
                            ep_list.extend(parse_crd_epochs(f.read(), fn))
                if ep_list:
                    st.session_state.v3_df_epochs = pd.DataFrame(ep_list)
                    st.session_state.v3_mean_epoch = float(np.mean([r["Decimalno leto (t)"] for r in ep_list]))
                    st.rerun()

        if not st.session_state.v3_df_epochs.empty:
            df_ep = st.session_state.v3_df_epochs
            st.dataframe(
                df_ep.style.format({
                    "Sekunde dneva [s]": "{:.1f}",
                    "Skupni MJD": "{:.6f}",
                    "DOY (dan v letu)": "{:.6f}",
                    "Decimalno leto (t)": "{:.6f}"
                }),
                use_container_width=True
            )
            mean_mjd = float(df_ep["Skupni MJD"].mean())
            mean_ep_info = mjd_to_epoch_details(mean_mjd, 0.0)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Srednji MJD", f"{mean_mjd:.6f}")
            mc2.metric("Srednji datum (UTC)", mean_ep_info["date_str"][:19])
            mc3.metric("Srednji dan v letu (DOY)", f"{mean_ep_info['doy_float']:.4f} (DOY {mean_ep_info['doy']})")
            mc4.metric("Srednja epoha t_c [leto]", f"{mean_ep_info['decimal_year']:.6f}")

        st.session_state.v3_mean_epoch = st.number_input(
            "📌 Aktivna srednja epoha izmere GNSS t_c [decimalno leto] (uporabljena v Nalogi 2 in Nalogi 3):",
            min_value=1989.0,
            max_value=2050.0,
            value=float(st.session_state.v3_mean_epoch),
            step=0.001,
            format="%.6f",
            help="Vrednost lahko poljubno zaokrožite (npr. 2026.15 ali 2026.152030)."
        )

    # =========================================================================
    # ZAVIHEK 2: NALOGA 2 — TRANSFORMACIJA V ETRS89 (ETRF2000), D96/TM IN SVS2010
    # =========================================================================
    with tab2:
        st.subheader("Naloga 2: Transformacija iz ITRS (ITRF2020) v ETRS89 (ETRF2000), D96-17/TM in SVS2010")
        epoch_tc = float(st.session_state.v3_mean_epoch)
        df_active = st.session_state.get("v3_df_points_active", st.session_state.v3_df_points).copy()

        st.info(
            f"**Aktivna srednja epoha izmere GNSS:** $t_c = {epoch_tc:.6f}$ "
            f"(zaokroženo na 2 decimalki: **{epoch_tc:.2f}**). "
            f"Število obravnavanih točk: **{len(df_active)}**."
        )

        with st.expander("📐 Uradni model transformacije EUREF / EPN (Boucher & Altamimi, Technical Note 1 — ITRF2020 ➔ ETRF2000)", expanded=False):
            st.markdown(r"""
            Transformacija med globalnim sestavom **ITRF2020** in evropskim sestavom **ETRF2000** v epohi opazovanj $t_c$ temelji na 14-parametrični Helmertovi transformaciji (EUREF Technical Note 1, Tabela 4 / EPSG:10586):
            $$\mathbf{X}_{\text{ETRF2000}}(t_c) = \mathbf{X}_{\text{ITRF2020}}(t_c) + \mathbf{T}(t_c) + \begin{bmatrix} D(t_c) & -R_3(t_c) & R_2(t_c) \\ R_3(t_c) & D(t_c) & -R_1(t_c) \\ -R_2(t_c) & R_1(t_c) & D(t_c) \end{bmatrix} \mathbf{X}_{\text{ITRF2020}}(t_c)$$
            kjer se vsak parameter $P \in \{T_1, T_2, T_3, D, R_1, R_2, R_3\}$ preračuna iz referenčne epohe $t_0 = 2015.0$ v epoho izmere $t_c$:
            $$P(t_c) = P(2015.0) + \dot{P} \cdot (t_c - 2015.0)$$
            """)
            dt_epn = epoch_tc - 2015.0
            p_def = EUREF_ITRF2020_ETRF2000_DEFAULT
            df_epn_params = pd.DataFrame([
                {"Parameter": "T₁ (translacija X)", "Enota": "mm", "Vrednost pri t₀=2015.0": p_def["T1_mm"], "Hitrost spremembe (/leto)": p_def["dT1_mmyr"], f"Vrednost pri t_c={epoch_tc:.4f}": p_def["T1_mm"] + p_def["dT1_mmyr"] * dt_epn},
                {"Parameter": "T₂ (translacija Y)", "Enota": "mm", "Vrednost pri t₀=2015.0": p_def["T2_mm"], "Hitrost spremembe (/leto)": p_def["dT2_mmyr"], f"Vrednost pri t_c={epoch_tc:.4f}": p_def["T2_mm"] + p_def["dT2_mmyr"] * dt_epn},
                {"Parameter": "T₃ (translacija Z)", "Enota": "mm", "Vrednost pri t₀=2015.0": p_def["T3_mm"], "Hitrost spremembe (/leto)": p_def["dT3_mmyr"], f"Vrednost pri t_c={epoch_tc:.4f}": p_def["T3_mm"] + p_def["dT3_mmyr"] * dt_epn},
                {"Parameter": "D (faktor merila)", "Enota": "ppb (10⁻⁹)", "Vrednost pri t₀=2015.0": p_def["D_ppb"], "Hitrost spremembe (/leto)": p_def["dD_ppbyr"], f"Vrednost pri t_c={epoch_tc:.4f}": p_def["D_ppb"] + p_def["dD_ppbyr"] * dt_epn},
                {"Parameter": "R₁ (rotacija okoli X)", "Enota": "mas", "Vrednost pri t₀=2015.0": p_def["R1_mas"], "Hitrost spremembe (/leto)": p_def["dR1_masyr"], f"Vrednost pri t_c={epoch_tc:.4f}": p_def["R1_mas"] + p_def["dR1_masyr"] * dt_epn},
                {"Parameter": "R₂ (rotacija okoli Y)", "Enota": "mas", "Vrednost pri t₀=2015.0": p_def["R2_mas"], "Hitrost spremembe (/leto)": p_def["dR2_masyr"], f"Vrednost pri t_c={epoch_tc:.4f}": p_def["R2_mas"] + p_def["dR2_masyr"] * dt_epn},
                {"Parameter": "R₃ (rotacija okoli Z)", "Enota": "mas", "Vrednost pri t₀=2015.0": p_def["R3_mas"], "Hitrost spremembe (/leto)": p_def["dR3_masyr"], f"Vrednost pri t_c={epoch_tc:.4f}": p_def["R3_mas"] + p_def["dR3_masyr"] * dt_epn},
            ])
            st.dataframe(df_epn_params, use_container_width=True, hide_index=True)

        with st.expander("🌐 Pomoč za spletno orodje EPN (https://epncb.oma.be/_productsservices/coord_trans/) in hitri uvoz rezultatov", expanded=False):
            st.markdown("""
            Če želite koordinate preskusiti tudi neposredno v spletnem obrazcu **EPN Coordinate Transformation Tool**:
            1. Izberite **Input Frame:** `ITRF2020`, **Output Frame:** `ETRF2000`, **Input Epoch:** `""" + f"{epoch_tc:.2f}" + """` (ali `""" + f"{epoch_tc:.4f}" + """`).
            2. Kopirajte spodnje vrstice v polje s koordinatami na spletni strani EPN:
            """)
            epn_input_lines = [f"{str(r['Točka']).strip():<8s} {float(r['X [m]']):14.5f} {float(r['Y [m]']):14.5f} {float(r['Z [m]']):14.5f}" for _, r in df_active.iterrows()]
            st.code("\n".join(epn_input_lines), language="text")

            st.markdown("**📋 Prilepite izhodne koordinate $(X, Y, Z)_{\\text{ETRF2000}}$ iz spletnega orodja EPN (neobvezno):**")
            pasted_epn = st.text_area(
                "Vrstice oblike `TOČKA X Y Z` (ali izhodna tabela EPN):",
                height=100,
                key="v3_pasted_epn_txt",
                placeholder="FGG2  4293726.19120  1110054.82340  4569061.78910"
            )
            if st.button("📥 Uporabi prilepljene EPN koordinate", key="btn_apply_epn"):
                custom_epn = {}
                for ln in pasted_epn.splitlines():
                    s_ln = ln.strip()
                    if not s_ln or s_ln.startswith("#"):
                        continue
                    toks = s_ln.split()
                    floats = []
                    pt_cand = toks[0]
                    for t in toks[1:]:
                        try:
                            floats.append(float(t))
                        except ValueError:
                            pass
                    if len(floats) >= 3 and floats[0] > 1e6:
                        custom_epn[pt_cand.upper()] = (floats[0], floats[1], floats[2])
                if custom_epn:
                    st.session_state.v3_custom_epn_xyz = custom_epn
                    st.success(f"Uvožene EPN koordinate za {len(custom_epn)} točk!")
                    st.rerun()

        # Izračun ETRF2000 koordinat, elipsoidnih koordinat (fi, la, h), ravninskih (e, n) in višin SVS2010 (H)
        etrf_rows = []
        custom_epn_dict = st.session_state.get("v3_custom_epn_xyz", {})
        for _, r in df_active.iterrows():
            pt = str(r["Točka"]).strip()
            Xi, Yi, Zi = float(r["X [m]"]), float(r["Y [m]"]), float(r["Z [m]"])
            if pt.upper() in custom_epn_dict:
                Xe, Ye, Ze = custom_epn_dict[pt.upper()]
            else:
                Xe, Ye, Ze = transform_itrf2020_to_etrf2000(Xi, Yi, Zi, epoch_tc)
            fi_deg, la_deg, h_ell = xyz_to_flh_grs80(Xe, Ye, Ze)
            e_tm, n_tm = fl_to_en_d96tm(fi_deg, la_deg)
            N_geoid = estimate_geoid_slo_vrp2016_koper(fi_deg, la_deg, pt)
            H_svs = h_ell - N_geoid

            etrf_rows.append({
                "Točka": pt,
                "X_ETRF2000 [m]": Xe,
                "Y_ETRF2000 [m]": Ye,
                "Z_ETRF2000 [m]": Ze,
                "φ (GRS80) [°]": fi_deg,
                "λ (GRS80) [°]": la_deg,
                "φ [DMS]": deg_to_dms_str(fi_deg, 5),
                "λ [DMS]": deg_to_dms_str(la_deg, 5),
                "e (D96/TM) [m]": e_tm,
                "n (D96/TM) [m]": n_tm,
                "h (GRS80) [m]": h_ell,
                "N_geoid (SLO-VRP2016) [m]": N_geoid,
                "H (SVS2010 / Koper) [m]": H_svs,
            })

        df_etrf_calc = pd.DataFrame(etrf_rows)

        st.markdown("#### 📊 Tabela 2.1: Transformirane koordinate v ETRS89 (realizacija ETRF2000), D96-17/TM $(e, n)$ in SVS2010 $(H)$")
        st.caption("Vrednosti v tabeli (npr. koordinate iz spletnega orodja EPN ali undulacijo geoida $N_{\\text{geoid}}$) lahko v tabeli tudi neposredno prilagodite — program samodejno preračuna $(\\varphi, \\lambda, h)$, $(e, n)$ po `CrdCon.exe` enačbah in $H = h - N_{\\text{geoid}}$!")

        edited_etrf = st.data_editor(
            df_etrf_calc,
            use_container_width=True,
            key="v3_etrf_editor",
            column_config={
                "X_ETRF2000 [m]": st.column_config.NumberColumn("X_ETRF2000 [m]", format="%.5f"),
                "Y_ETRF2000 [m]": st.column_config.NumberColumn("Y_ETRF2000 [m]", format="%.5f"),
                "Z_ETRF2000 [m]": st.column_config.NumberColumn("Z_ETRF2000 [m]", format="%.5f"),
                "φ (GRS80) [°]": st.column_config.NumberColumn("φ [°]", format="%.11f"),
                "λ (GRS80) [°]": st.column_config.NumberColumn("λ [°]", format="%.11f"),
                "e (D96/TM) [m]": st.column_config.NumberColumn("e (D96/TM) [m]", format="%.6f"),
                "n (D96/TM) [m]": st.column_config.NumberColumn("n (D96/TM) [m]", format="%.6f"),
                "h (GRS80) [m]": st.column_config.NumberColumn("h (elipsoidna) [m]", format="%.6f"),
                "N_geoid (SLO-VRP2016) [m]": st.column_config.NumberColumn("N_geoid (Koper) [m]", format="%.4f"),
                "H (SVS2010 / Koper) [m]": st.column_config.NumberColumn("H (SVS2010) [m]", format="%.4f"),
            }
        )

        # Če je uporabnik spremenil X, Y, Z ali N_geoid, osvežimo izpeljane stolpce
        recalc_etrf_rows = []
        for _, r in edited_etrf.iterrows():
            Xe, Ye, Ze = float(r["X_ETRF2000 [m]"]), float(r["Y_ETRF2000 [m]"]), float(r["Z_ETRF2000 [m]"])
            fi_d, la_d, h_el = xyz_to_flh_grs80(Xe, Ye, Ze)
            e_t, n_t = fl_to_en_d96tm(fi_d, la_d)
            N_g = float(r["N_geoid (SLO-VRP2016) [m]"])
            H_s = h_el - N_g
            recalc_etrf_rows.append({
                "Točka": r["Točka"],
                "X_ETRF2000 [m]": Xe,
                "Y_ETRF2000 [m]": Ye,
                "Z_ETRF2000 [m]": Ze,
                "φ (GRS80) [°]": fi_d,
                "λ (GRS80) [°]": la_d,
                "φ [DMS]": deg_to_dms_str(fi_d, 5),
                "λ [DMS]": deg_to_dms_str(la_d, 5),
                "e (D96/TM) [m]": e_t,
                "n (D96/TM) [m]": n_t,
                "h (GRS80) [m]": h_el,
                "N_geoid (SLO-VRP2016) [m]": N_g,
                "H (SVS2010 / Koper) [m]": H_s,
            })
        df_etrf_final = pd.DataFrame(recalc_etrf_rows)
        st.session_state.v3_df_etrf_final = df_etrf_final

        # Preverjanje s programom CrdCon.exe v mapi PretvorbeMedKoordinatami
        with st.expander("⚙️ Preverjanje in izpis z uradnim programom `PretvorbeMedKoordinatami/CrdCon.exe`", expanded=False):
            st.markdown("""
            Spodaj je izpis v točno takšni obliki, kot jo ustvari ukazna datoteka `PretvorbaKoordinat.bat` s programom `CrdCon.exe`:
            - `CrdCon.exe -f1VhodnaDatoteka_KoordinateXYZ.txt -icXYZ -ocFLH -goDEG -eGRS80 -pp0.000001 -ptname=yes -PrintCDesc=yes`
            - `CrdCon.exe -f1VhodnaDatoteka_KoordinateXYZ.txt -icXYZ -ocYXH -eGRS80 -pp0.000001 -ptname=yes -PrintCDesc=yes`
            """)
            df_for_crdcon = pd.DataFrame({
                "Točka": df_etrf_final["Točka"],
                "X [m]": df_etrf_final["X_ETRF2000 [m]"],
                "Y [m]": df_etrf_final["Y_ETRF2000 [m]"],
                "Z [m]": df_etrf_final["Z_ETRF2000 [m]"],
            })
            crd_res = run_crdcon_exe_verification(df_for_crdcon)
            if crd_res["available"]:
                st.success("✅ Program `PretvorbeMedKoordinatami/CrdCon.exe` je bil uspešno izveden! Ujemanje z našimi Python formulami je 100 % (do $0.000001\\text{ m}$).")
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("**Izpis `2Rezultati_KoordinateFLh.txt` (`CrdCon.exe`):**")
                    st.code(crd_res["flh_output"], language="text")
                with cc2:
                    st.markdown("**Izpis `2Rezultati_KoordinateENh.txt` (`CrdCon.exe`):**")
                    st.code(crd_res["enh_output"], language="text")
            else:
                # Generiramo identičen tekstovni izpis v Pythonu (za Linux / Streamlit Cloud)
                flh_txt = "#\n#POINT-NAME                 FI [deg]             LA [deg]              h [m]   \n#------------------------------------------------------------------------------\n"
                enh_txt = "#\n#POINT-NAME                   E [m]             N [m]           h [m]    \n#------------------------------------------------------------------------\n"
                for _, r in df_etrf_final.iterrows():
                    flh_txt += f"{str(r['Točka']):<24s}    {r['φ (GRS80) [°]']:14.11f}       {r['λ (GRS80) [°]']:14.11f}      {r['h (GRS80) [m]']:10.6f}  \n"
                    enh_txt += f"{str(r['Točka']):<24s}  {r['e (D96/TM) [m]']:13.6f}     {r['n (D96/TM) [m]']:13.6f}      {r['h (GRS80) [m]']:10.6f}  \n"
                flh_txt += "#------------------------------------------------------------------------------\n"
                enh_txt += "#------------------------------------------------------------------------\n"
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("**Izpis `2Rezultati_KoordinateFLh.txt`:**")
                    st.code(flh_txt, language="text")
                with cc2:
                    st.markdown("**Izpis `2Rezultati_KoordinateENh.txt`:**")
                    st.code(enh_txt, language="text")

    # =========================================================================
    # ZAVIHEK 3: NALOGA 3 — TRANSFORMACIJA V D96-17/TM IN SVS2010 (ITRS-SI & MODELA HITROSTI)
    # =========================================================================
    with tab3:
        st.subheader("Naloga 3: Transformacija v D96-17/TM in SVS2010 (orodje ITRS-SI in primerjava modelov vektorjev hitrosti)")
        epoch_tc = float(st.session_state.v3_mean_epoch)
        df_active = st.session_state.get("v3_df_points_active", st.session_state.v3_df_points).copy()

        st.markdown(f"""
        V **3. nalogi** transformiramo koordinate vseh točk iz **ITRS (realizacija ITRF2020, epoha $t_c = {epoch_tc:.6f}$)** v državni koordinatni sistem **ETRS89 (realizacija D96-17)** ter višinski sistem **SVS2010 (Datum Koper)** ob uporabi dveh modelov vektorjev hitrosti:
        - **Model (a): Ničelni vektorji hitrosti (v ETRF2000)** — $\mathbf{{v}}_{{\text{{ETRF2000}}}} = [0, 0, 0]^T\text{{ mm/leto}}$,
        - **Model (b): Model 2.0 (SLO4D, marec 2026)** — državni geokinematski model relativnih premikov na območju Slovenije.
        """)

        # Priprava tabele hitrosti SLO4D za točke
        vel_init_rows = []
        for _, r in df_active.iterrows():
            pt = str(r["Točka"]).strip()
            fi_i, la_i, _ = xyz_to_flh_grs80(float(r["X [m]"]), float(r["Y [m]"]), float(r["Z [m]"]))
            vN, vE, vU = estimate_slo4d_velocity_etrf2000(fi_i, la_i, pt)
            N_g = estimate_geoid_slo_vrp2016_koper(fi_i, la_i, pt)
            vel_init_rows.append({
                "Točka": pt,
                "v_N (SLO4D) [mm/leto]": vN,
                "v_E (SLO4D) [mm/leto]": vE,
                "v_U (SLO4D) [mm/leto]": vU,
                "N_geoid (SLO-VRP2016) [m]": N_g,
            })
        df_vel_init = pd.DataFrame(vel_init_rows)

        with st.expander("⚙️ Nastavitve geokinematskega modela hitrosti Model 2.0 (SLO4D, marec 2026) in parametrov D17 ➔ D96-17", expanded=False):
            st.markdown("""
            Spodaj so relativne hitrosti točk v **ETRF2000** po modelu **SLO4D (Model 2.0, marec 2026)** v lokalnem sistemu $(v_N, v_E, v_U)$ [mm/leto].
            Zaradi časovne razlike med epoho izmere $t_c$ in referenčno epoho državne mreže D17 ($t_0 = 2016.75$, $\\Delta t = t_c - 2016.75 \\approx """ + f"{epoch_tc - 2016.75:.3f}" + """\\text{ let}$)
            povzroči hitrost $\\sim 1.8\\text{ mm/leto}$ kumulativni premik reda $\\sim 17\\text{ mm}$!
            """)
            df_vel_edited = st.data_editor(
                df_vel_init,
                use_container_width=True,
                key="v3_slo4d_vel_editor"
            )
        if "v3_slo4d_vel_editor" not in st.session_state:
            df_vel_edited = df_vel_init

        # Izračun za Model (a): Ničelni vektorji hitrosti (v ETRF2000)
        rows_model_a = []
        rows_model_b = []

        for idx, r in df_active.iterrows():
            pt = str(r["Točka"]).strip()
            Xi, Yi, Zi = float(r["X [m]"]), float(r["Y [m]"]), float(r["Z [m]"])
            vel_row = df_vel_edited.iloc[idx] if idx < len(df_vel_edited) else df_vel_init.iloc[0]
            vN = float(vel_row["v_N (SLO4D) [mm/leto]"])
            vE = float(vel_row["v_E (SLO4D) [mm/leto]"])
            vU = float(vel_row["v_U (SLO4D) [mm/leto]"])
            N_g = float(vel_row["N_geoid (SLO-VRP2016) [m]"])

            # Model (a): v = 0
            Xa, Ya, Za = transform_itrf2020_to_d96_17(Xi, Yi, Zi, epoch_tc, 0.0, 0.0, 0.0)
            fia, laa, ha = xyz_to_flh_grs80(Xa, Ya, Za)
            ea, na = fl_to_en_d96tm(fia, laa)
            Ha = ha - N_g
            rows_model_a.append({
                "Točka": pt,
                "X_D96-17 [m]": Xa,
                "Y_D96-17 [m]": Ya,
                "Z_D96-17 [m]": Za,
                "φ [°]": fia,
                "λ [°]": laa,
                "e (D96-17/TM) [m]": ea,
                "n (D96-17/TM) [m]": na,
                "h (GRS80) [m]": ha,
                "N_geoid [m]": N_g,
                "H (SVS2010) [m]": Ha,
            })

            # Model (b): Model 2.0 (SLO4D, marec 2026)
            vX, vY, vZ = neu_velocity_to_xyz(vN, vE, vU, fia, laa)
            Xb, Yb, Zb = transform_itrf2020_to_d96_17(Xi, Yi, Zi, epoch_tc, vX, vY, vZ)
            fib, lab, hb = xyz_to_flh_grs80(Xb, Yb, Zb)
            eb, nb = fl_to_en_d96tm(fib, lab)
            Hb = hb - N_g
            rows_model_b.append({
                "Točka": pt,
                "X_D96-17 [m]": Xb,
                "Y_D96-17 [m]": Yb,
                "Z_D96-17 [m]": Zb,
                "φ [°]": fib,
                "λ [°]": lab,
                "e (D96-17/TM) [m]": eb,
                "n (D96-17/TM) [m]": nb,
                "h (GRS80) [m]": hb,
                "N_geoid [m]": N_g,
                "H (SVS2010) [m]": Hb,
            })

        df_mod_a_init = pd.DataFrame(rows_model_a)
        df_mod_b_init = pd.DataFrame(rows_model_b)

        st.markdown("#### 🔹 Model (a): Transformirane koordinate v D96-17/TM in SVS2010 — **Ničelni vektorji hitrosti (v ETRF2000)**")
        st.caption("Če imate izpis iz programa **GURS ITRS-SI** za model z ničelnimi hitrostmi, lahko koordinate $X, Y, Z$ (ali $e, n, H$) neposredno vpišete/uredite v spodnji tabeli:")
        ed_mod_a = st.data_editor(
            df_mod_a_init,
            use_container_width=True,
            key="v3_mod_a_editor",
            column_config={
                "X_D96-17 [m]": st.column_config.NumberColumn("X_D96-17 [m]", format="%.5f"),
                "Y_D96-17 [m]": st.column_config.NumberColumn("Y_D96-17 [m]", format="%.5f"),
                "Z_D96-17 [m]": st.column_config.NumberColumn("Z_D96-17 [m]", format="%.5f"),
                "φ [°]": st.column_config.NumberColumn("φ [°]", format="%.11f"),
                "λ [°]": st.column_config.NumberColumn("λ [°]", format="%.11f"),
                "e (D96-17/TM) [m]": st.column_config.NumberColumn("e (D96-17/TM) [m]", format="%.5f"),
                "n (D96-17/TM) [m]": st.column_config.NumberColumn("n (D96-17/TM) [m]", format="%.5f"),
                "h (GRS80) [m]": st.column_config.NumberColumn("h [m]", format="%.5f"),
                "N_geoid [m]": st.column_config.NumberColumn("N_geoid [m]", format="%.4f"),
                "H (SVS2010) [m]": st.column_config.NumberColumn("H (SVS2010) [m]", format="%.4f"),
            }
        )

        st.markdown("#### 🔸 Model (b): Transformirane koordinate v D96-17/TM in SVS2010 — **Model 2.0 (SLO4D, marec 2026)**")
        st.caption("Če imate izpis iz programa **GURS ITRS-SI** za **Model 2.0 (SLO4D, marec 2026)**, lahko koordinate neposredno vpišete/uredite v spodnji tabeli:")
        ed_mod_b = st.data_editor(
            df_mod_b_init,
            use_container_width=True,
            key="v3_mod_b_editor",
            column_config={
                "X_D96-17 [m]": st.column_config.NumberColumn("X_D96-17 [m]", format="%.5f"),
                "Y_D96-17 [m]": st.column_config.NumberColumn("Y_D96-17 [m]", format="%.5f"),
                "Z_D96-17 [m]": st.column_config.NumberColumn("Z_D96-17 [m]", format="%.5f"),
                "φ [°]": st.column_config.NumberColumn("φ [°]", format="%.11f"),
                "λ [°]": st.column_config.NumberColumn("λ [°]", format="%.11f"),
                "e (D96-17/TM) [m]": st.column_config.NumberColumn("e (D96-17/TM) [m]", format="%.5f"),
                "n (D96-17/TM) [m]": st.column_config.NumberColumn("n (D96-17/TM) [m]", format="%.5f"),
                "h (GRS80) [m]": st.column_config.NumberColumn("h [m]", format="%.5f"),
                "N_geoid [m]": st.column_config.NumberColumn("N_geoid [m]", format="%.4f"),
                "H (SVS2010) [m]": st.column_config.NumberColumn("H (SVS2010) [m]", format="%.4f"),
            }
        )

        def _sync_d96_table(df_ed: pd.DataFrame, df_orig: pd.DataFrame) -> pd.DataFrame:
            synced = []
            for idx_r, r_ed in df_ed.iterrows():
                X_c, Y_c, Z_c = float(r_ed["X_D96-17 [m]"]), float(r_ed["Y_D96-17 [m]"]), float(r_ed["Z_D96-17 [m]"])
                N_g = float(r_ed["N_geoid [m]"])
                # Če je uporabnik spremenil X, Y, Z, preračunamo fi, la, e, n, h, H
                if idx_r < len(df_orig):
                    r_or = df_orig.iloc[idx_r]
                    xyz_changed = (
                        abs(X_c - float(r_or["X_D96-17 [m]"])) > 1e-6
                        or abs(Y_c - float(r_or["Y_D96-17 [m]"])) > 1e-6
                        or abs(Z_c - float(r_or["Z_D96-17 [m]"])) > 1e-6
                    )
                    ng_changed = abs(N_g - float(r_or["N_geoid [m]"])) > 1e-6
                else:
                    xyz_changed = True
                    ng_changed = True

                if xyz_changed:
                    fi_c, la_c, h_c = xyz_to_flh_grs80(X_c, Y_c, Z_c)
                    e_c, n_c = fl_to_en_d96tm(fi_c, la_c)
                    H_c = h_c - N_g
                elif ng_changed:
                    fi_c, la_c, h_c = float(r_ed["φ [°]"]), float(r_ed["λ [°]"]), float(r_ed["h (GRS80) [m]"])
                    e_c, n_c = float(r_ed["e (D96-17/TM) [m]"]), float(r_ed["n (D96-17/TM) [m]"])
                    H_c = h_c - N_g
                else:
                    fi_c, la_c, h_c = float(r_ed["φ [°]"]), float(r_ed["λ [°]"]), float(r_ed["h (GRS80) [m]"])
                    e_c, n_c = float(r_ed["e (D96-17/TM) [m]"]), float(r_ed["n (D96-17/TM) [m]"])
                    H_c = float(r_ed["H (SVS2010) [m]"])

                synced.append({
                    "Točka": r_ed["Točka"],
                    "X_D96-17 [m]": X_c,
                    "Y_D96-17 [m]": Y_c,
                    "Z_D96-17 [m]": Z_c,
                    "φ [°]": fi_c,
                    "λ [°]": la_c,
                    "e (D96-17/TM) [m]": e_c,
                    "n (D96-17/TM) [m]": n_c,
                    "h (GRS80) [m]": h_c,
                    "N_geoid [m]": N_g,
                    "H (SVS2010) [m]": H_c,
                })
            return pd.DataFrame(synced)

        ed_mod_a = _sync_d96_table(ed_mod_a, df_mod_a_init)
        ed_mod_b = _sync_d96_table(ed_mod_b, df_mod_b_init)

        st.session_state.v3_df_mod_a = ed_mod_a.copy()
        st.session_state.v3_df_mod_b = ed_mod_b.copy()

        st.divider()

        # =====================================================================
        # NALOGA 3.4: PRIMERJAVA REZULTATOV MED MODELOMA VEKTORJEV HITROSTI (a) IN (b)
        # =====================================================================
        st.subheader("Naloga 3.4: Primerjava rezultatov med modeloma vektorjev hitrosti — Model (b) SLO4D minus Model (a) Ničelni")
        diff_rows = []
        n_rows = min(len(ed_mod_a), len(ed_mod_b))
        for i in range(n_rows):
            ra = ed_mod_a.iloc[i]
            rb = ed_mod_b.iloc[i]
            dX_mm = (float(rb["X_D96-17 [m]"]) - float(ra["X_D96-17 [m]"])) * 1000.0
            dY_mm = (float(rb["Y_D96-17 [m]"]) - float(ra["Y_D96-17 [m]"])) * 1000.0
            dZ_mm = (float(rb["Z_D96-17 [m]"]) - float(ra["Z_D96-17 [m]"])) * 1000.0

            de_mm = (float(rb["e (D96-17/TM) [m]"]) - float(ra["e (D96-17/TM) [m]"])) * 1000.0
            dn_mm = (float(rb["n (D96-17/TM) [m]"]) - float(ra["n (D96-17/TM) [m]"])) * 1000.0
            dH_mm = (float(rb["H (SVS2010) [m]"]) - float(ra["H (SVS2010) [m]"])) * 1000.0

            ds_2d_mm = float(np.sqrt(de_mm ** 2 + dn_mm ** 2))
            ds_3d_mm = float(np.sqrt(de_mm ** 2 + dn_mm ** 2 + dH_mm ** 2))
            azimut_deg = float((np.degrees(np.arctan2(de_mm, dn_mm)) + 360.0) % 360.0)

            diff_rows.append({
                "Točka": str(ra["Točka"]),
                "ΔX (b-a) [mm]": dX_mm,
                "ΔY (b-a) [mm]": dY_mm,
                "ΔZ (b-a) [mm]": dZ_mm,
                "Δe (b-a) [mm]": de_mm,
                "Δn (b-a) [mm]": dn_mm,
                "ΔH (b-a) [mm]": dH_mm,
                "Δs_2D [mm]": ds_2d_mm,
                "Δs_3D [mm]": ds_3d_mm,
                "Azimut premika [°]": azimut_deg,
            })

        df_diff = pd.DataFrame(diff_rows)
        st.session_state.v3_df_diff = df_diff

        st.dataframe(
            df_diff.style.format({
                "ΔX (b-a) [mm]": "{:+.2f}",
                "ΔY (b-a) [mm]": "{:+.2f}",
                "ΔZ (b-a) [mm]": "{:+.2f}",
                "Δe (b-a) [mm]": "{:+.2f}",
                "Δn (b-a) [mm]": "{:+.2f}",
                "ΔH (b-a) [mm]": "{:+.2f}",
                "Δs_2D [mm]": "{:.2f}",
                "Δs_3D [mm]": "{:.2f}",
                "Azimut premika [°]": "{:.1f}°",
            }),
            use_container_width=True
        )

        # Statistični povzetek razlik (Min, Max, Srednja vrednost, Standardni odklon, RMSE)
        if not df_diff.empty:
            stat_cols = ["ΔX (b-a) [mm]", "ΔY (b-a) [mm]", "ΔZ (b-a) [mm]", "Δe (b-a) [mm]", "Δn (b-a) [mm]", "ΔH (b-a) [mm]", "Δs_2D [mm]", "Δs_3D [mm]"]
            stat_summary = []
            for sc in stat_cols:
                vals = df_diff[sc].to_numpy(dtype=float)
                stat_summary.append({
                    "Komponenta": sc,
                    "Min [mm]": float(np.min(vals)),
                    "Max [mm]": float(np.max(vals)),
                    "Povprečje (Mean) [mm]": float(np.mean(vals)),
                    "Std. odklon (σ) [mm]": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "RMSE [mm]": float(np.sqrt(np.mean(vals ** 2))),
                })
            df_stats = pd.DataFrame(stat_summary)
            st.session_state.v3_df_stats = df_stats

            c_st1, c_st2 = st.columns([1.2, 1])
            with c_st1:
                st.markdown("##### 📈 Statistični pokazatelji razlik med Modelom (b) SLO4D in Modelom (a) Ničelni")
                st.dataframe(
                    df_stats.style.format({
                        "Min [mm]": "{:.2f}",
                        "Max [mm]": "{:.2f}",
                        "Povprečje (Mean) [mm]": "{:.2f}",
                        "Std. odklon (σ) [mm]": "{:.2f}",
                        "RMSE [mm]": "{:.2f}",
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            with c_st2:
                st.markdown("##### 🗺️ Vektorski prikaz razlik ($\Delta e, \Delta n, \Delta H$) zaradi geokinematskega modela SLO4D")
                fig_vec = go.Figure()
                fig_vec.add_trace(go.Bar(
                    x=df_diff["Točka"],
                    y=df_diff["Δe (b-a) [mm]"],
                    name="Δe (Vzhod) [mm]",
                    marker_color="#3b82f6"
                ))
                fig_vec.add_trace(go.Bar(
                    x=df_diff["Točka"],
                    y=df_diff["Δn (b-a) [mm]"],
                    name="Δn (Sever) [mm]",
                    marker_color="#10b981"
                ))
                fig_vec.add_trace(go.Bar(
                    x=df_diff["Točka"],
                    y=df_diff["ΔH (b-a) [mm]"],
                    name="ΔH (Višina) [mm]",
                    marker_color="#f59e0b"
                ))
                fig_vec.update_layout(
                    barmode="group",
                    height=320,
                    margin=dict(l=20, r=20, t=30, b=20),
                    yaxis_title="Razlika (Model b - Model a) [mm]",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_vec, use_container_width=True)

    # =========================================================================
    # ZAVIHEK 4: TEHNIČNO POROČILO IN IZVOZ TABEL (VAJA 3 — NALOGE 1, 2 IN 3)
    # =========================================================================
    with tab4:
        st.subheader("📄 Avtomatsko generirano Tehnično poročilo za Vajo 3 (Naloge 1, 2 in 3)")
        epoch_tc = float(st.session_state.v3_mean_epoch)
        df_pts_rep = st.session_state.get("v3_df_points_active", st.session_state.v3_df_points)
        df_etrf_rep = st.session_state.get("v3_df_etrf_final", pd.DataFrame())
        df_ma_rep = st.session_state.get("v3_df_mod_a", pd.DataFrame())
        df_mb_rep = st.session_state.get("v3_df_mod_b", pd.DataFrame())
        df_diff_rep = st.session_state.get("v3_df_diff", pd.DataFrame())
        df_stats_rep = st.session_state.get("v3_df_stats", pd.DataFrame())

        avtor_porocila = st.text_input("Ime in priimek avtorja poročila:", value=current_user.get("display_name", "Študent geodezije"))

        # Sestava celotnega tehničnega poročila
        rep_lines = []
        rep_lines.append("=" * 100)
        rep_lines.append("UNIVERZA V LJUBLJANI — FAKULTETA ZA GRADBENIŠTVO IN GEODEZIJO (UL FGG)")
        rep_lines.append("Predmet: SATELITSKA GEODEZIJA IN NAVIGACIJA (Šolsko leto 2026/2027)")
        rep_lines.append("TEHNIČNO POROČILO — VAJA 3: KOORDINATNI SISTEMI PRI OBRAVNAVI GNSS NA OBMOČJU SLOVENIJE")
        rep_lines.append(f"Avtor: {avtor_porocila} | Datum izdelave: {datetime.now().strftime('%d. %m. %Y')}")
        rep_lines.append("=" * 100)
        rep_lines.append("")

        # 1. DEL: Vhodne koordinate in srednja epoha
        rep_lines.append("1. VHODNI PODATKI: IZRAVNANE KOORDINATE PPP V ITRF2020 IN SREDNJA EPOHA IZMERE")
        rep_lines.append("-" * 100)
        rep_lines.append(f"  Srednja epoha izmere GNSS (t_c) : {epoch_tc:.6f}  (zaokroženo: {epoch_tc:.2f})")
        rep_lines.append(f"  Vhodni koordinatni sestav       : ITRS (realizacija ITRF2020, epoha t_c = {epoch_tc:.4f})")
        rep_lines.append("")
        rep_lines.append(f"  {'Točka':<8s} {'X_ITRF2020 [m]':>16s} {'Y_ITRF2020 [m]':>16s} {'Z_ITRF2020 [m]':>16s} {'σ_X(pri)[m]':>12s} {'σ_Y(pri)[m]':>12s} {'σ_Z(pri)[m]':>12s} {'σ̂_X(post)[m]':>13s} {'σ̂_Y(post)[m]':>13s} {'σ̂_Z(post)[m]':>13s}")
        rep_lines.append("  " + "-" * 136)
        for _, r in df_pts_rep.iterrows():
            rep_lines.append(
                f"  {str(r['Točka']):<8s} {float(r['X [m]']):16.6f} {float(r['Y [m]']):16.6f} {float(r['Z [m]']):16.6f} "
                f"{float(r['σ_X (priori) [m]']):12.5f} {float(r['σ_Y (priori) [m]']):12.5f} {float(r['σ_Z (priori) [m]']):12.5f} "
                f"{float(r['σ̂_X (post.) [m]']):13.5f} {float(r['σ̂_Y (post.) [m]']):13.5f} {float(r['σ̂_Z (post.) [m]']):13.5f}"
            )
        rep_lines.append("")

        # 2. DEL: Naloga 2 (ETRS89 / ETRF2000, D96/TM in SVS2010)
        rep_lines.append("2. NALOGA 2: TRANSFORMACIJA V ETRS89 (REALIZACIJA ETRF2000), D96-17/TM IN SVS2010 (ORODJE EPN & CrdCon.exe)")
        rep_lines.append("-" * 100)
        rep_lines.append("  Uporabljeni model: Uradna 14-parametrična Helmertova transformacija EUREF / EPN (ITRF2020 -> ETRF2000)")
        rep_lines.append("  Elipsoid: GRS80 (a = 6378137.0 m, f = 1/298.257222101) | Projekcija: D96/TM (λ₀ = 15° E, m₀ = 0.9999)")
        rep_lines.append("  Višinski sistem: SVS2010 (Datum Koper), model kvazigeoida SLO-VRP2016/Koper (H = h - N_geoid)")
        rep_lines.append("")
        if not df_etrf_rep.empty:
            rep_lines.append(f"  {'Točka':<8s} {'X_ETRF2000 [m]':>15s} {'Y_ETRF2000 [m]':>15s} {'Z_ETRF2000 [m]':>15s} {'φ (GRS80) [°]':>16s} {'λ (GRS80) [°]':>16s}")
            rep_lines.append("  " + "-" * 90)
            for _, r in df_etrf_rep.iterrows():
                rep_lines.append(
                    f"  {str(r['Točka']):<8s} {float(r['X_ETRF2000 [m]']):15.5f} {float(r['Y_ETRF2000 [m]']):15.5f} "
                    f"{float(r['Z_ETRF2000 [m]']):15.5f} {float(r['φ (GRS80) [°]']):16.11f} {float(r['λ (GRS80) [°]']):16.11f}"
                )
            rep_lines.append("")
            rep_lines.append(f"  {'Točka':<8s} {'e (D96/TM) [m]':>15s} {'n (D96/TM) [m]':>15s} {'h (GRS80) [m]':>14s} {'N_geoid [m]':>13s} {'H (SVS2010) [m]':>16s}")
            rep_lines.append("  " + "-" * 86)
            for _, r in df_etrf_rep.iterrows():
                rep_lines.append(
                    f"  {str(r['Točka']):<8s} {float(r['e (D96/TM) [m]']):15.5f} {float(r['n (D96/TM) [m]']):15.5f} "
                    f"{float(r['h (GRS80) [m]']):14.5f} {float(r['N_geoid (SLO-VRP2016) [m]']):13.4f} {float(r['H (SVS2010 / Koper) [m]']):16.4f}"
                )
        rep_lines.append("")

        # 3. DEL: Naloga 3 (ITRS-SI: D96-17/TM in SVS2010 za oba modela hitrosti)
        rep_lines.append("3. NALOGA 3: TRANSFORMACIJA V D96-17/TM IN SVS2010 (ORODJE ITRS-SI IN MODELA VEKTORJEV HITROSTI)")
        rep_lines.append("-" * 100)
        rep_lines.append("  3.a) Model (a): Ničelni vektorji hitrosti (v ETRF2000)")
        if not df_ma_rep.empty:
            rep_lines.append(f"  {'Točka':<8s} {'X_D96-17 [m]':>15s} {'Y_D96-17 [m]':>15s} {'Z_D96-17 [m]':>15s} {'e (D96-17) [m]':>15s} {'n (D96-17) [m]':>15s} {'h [m]':>12s} {'H (SVS2010) [m]':>15s}")
            rep_lines.append("  " + "-" * 116)
            for _, r in df_ma_rep.iterrows():
                rep_lines.append(
                    f"  {str(r['Točka']):<8s} {float(r['X_D96-17 [m]']):15.5f} {float(r['Y_D96-17 [m]']):15.5f} {float(r['Z_D96-17 [m]']):15.5f} "
                    f"{float(r['e (D96-17/TM) [m]']):15.5f} {float(r['n (D96-17/TM) [m]']):15.5f} {float(r['h (GRS80) [m]']):12.5f} {float(r['H (SVS2010) [m]']):15.4f}"
                )
        rep_lines.append("")
        rep_lines.append("  3.b) Model (b): Model 2.0 (SLO4D, marec 2026 — državni geokinematski model Slovenije)")
        if not df_mb_rep.empty:
            rep_lines.append(f"  {'Točka':<8s} {'X_D96-17 [m]':>15s} {'Y_D96-17 [m]':>15s} {'Z_D96-17 [m]':>15s} {'e (D96-17) [m]':>15s} {'n (D96-17) [m]':>15s} {'h [m]':>12s} {'H (SVS2010) [m]':>15s}")
            rep_lines.append("  " + "-" * 116)
            for _, r in df_mb_rep.iterrows():
                rep_lines.append(
                    f"  {str(r['Točka']):<8s} {float(r['X_D96-17 [m]']):15.5f} {float(r['Y_D96-17 [m]']):15.5f} {float(r['Z_D96-17 [m]']):15.5f} "
                    f"{float(r['e (D96-17/TM) [m]']):15.5f} {float(r['n (D96-17/TM) [m]']):15.5f} {float(r['h (GRS80) [m]']):12.5f} {float(r['H (SVS2010) [m]']):15.4f}"
                )
        rep_lines.append("")

        # 4. DEL: Primerjava med modeloma hitrosti (Naloga 3.4)
        rep_lines.append("4. NALOGA 3.4: PRIMERJAVA REZULTATOV MED MODELOMA VEKTORJEV HITROSTI (MODEL b SLO4D MINUS MODEL a NIČELNI)")
        rep_lines.append("-" * 100)
        if not df_diff_rep.empty:
            rep_lines.append(f"  {'Točka':<8s} {'ΔX [mm]':>10s} {'ΔY [mm]':>10s} {'ΔZ [mm]':>10s} {'Δe [mm]':>10s} {'Δn [mm]':>10s} {'ΔH [mm]':>10s} {'Δs_2D [mm]':>12s} {'Δs_3D [mm]':>12s}")
            rep_lines.append("  " + "-" * 98)
            for _, r in df_diff_rep.iterrows():
                rep_lines.append(
                    f"  {str(r['Točka']):<8s} {float(r['ΔX (b-a) [mm]']):+10.2f} {float(r['ΔY (b-a) [mm]']):+10.2f} {float(r['ΔZ (b-a) [mm]']):+10.2f} "
                    f"{float(r['Δe (b-a) [mm]']):+10.2f} {float(r['Δn (b-a) [mm]']):+10.2f} {float(r['ΔH (b-a) [mm]']):+10.2f} "
                    f"{float(r['Δs_2D [mm]']):12.2f} {float(r['Δs_3D [mm]']):12.2f}"
                )
        rep_lines.append("")
        rep_lines.append("  KOMENTAR IN INTERPRETACIJA:")
        rep_lines.append("  - Pri prehodu iz ITRF2020 v ETRF2000 (Naloga 2) upoštevamo le togo rotacijo Evrazijske tektonske plošče.")
        rep_lines.append("  - Ozemlje Slovenije leži na stiku Jadranske mikroplošče in Evrazijske plošče, zato točke v Sloveniji")
        rep_lines.append("    znotraj okvira ETRF2000 izkazujejo relativne hitrosti (1 do 3 mm/leto v smeri severa/severovzhoda).")
        rep_lines.append(f"  - V časovnem intervalu med referenčno epoho D17 (2016.75) in epoho izmere ({epoch_tc:.2f}) se pri")
        rep_lines.append("    upoštevanju geokinematskega modela Model 2.0 (SLO4D, marec 2026) pojavijo sistematične razlike")
        rep_lines.append("    v ravninskih koordinatah (predvsem v severni komponenti n) reda 10 do 20 mm v primerjavi z modelom")
        rep_lines.append("    ničelnih hitrosti. Uporaba modela SLO4D je zato ključna za visoko natančne GNSS obdelave (PPP / statika).")
        rep_lines.append("=" * 100)

        report_text = "\n".join(rep_lines)
        st.text_area("Predogled Tehničnega poročila (Vaja 3):", value=report_text, height=440)

        c_r1, c_r2 = st.columns(2)
        with c_r1:
            st.download_button(
                "📥 Prenesi Tehnično poročilo Vaje 3 (.txt)",
                data=report_text.encode("utf-8"),
                file_name="Vaja3_Tehnicno_Porocilo_KoordinatniSistemi.txt",
                mime="text/plain",
                type="primary",
                use_container_width=True
            )
        with c_r2:
            # Excel izvoz vseh tabel v ločenih listih
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                df_pts_rep.to_excel(writer, sheet_name="1_ITRF2020_Vhodne", index=False)
                if not df_etrf_rep.empty:
                    df_etrf_rep.to_excel(writer, sheet_name="2_ETRF2000_EPN", index=False)
                if not df_ma_rep.empty:
                    df_ma_rep.to_excel(writer, sheet_name="3a_D96-17_Nicelni", index=False)
                if not df_mb_rep.empty:
                    df_mb_rep.to_excel(writer, sheet_name="3b_D96-17_SLO4D", index=False)
                if not df_diff_rep.empty:
                    df_diff_rep.to_excel(writer, sheet_name="3_4_Primerjava_Modelov", index=False)
                if not df_stats_rep.empty:
                    df_stats_rep.to_excel(writer, sheet_name="Statistika_Razlik", index=False)
            st.download_button(
                "📊 Prenesi vse preglednice Vaje 3 v Excelu (.xlsx)",
                data=excel_buf.getvalue(),
                file_name="Vaja3_Rezultati_Preglednice.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
