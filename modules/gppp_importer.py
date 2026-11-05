import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st


def parse_gppp_text(content: str, filename: str = "") -> list[dict]:
    """Razčleni vsebino gPPP koordinatne datoteke (*.CRD, *.txt, *.out) v seznam zapisov epoh."""
    lines = content.strip().splitlines()
    records = []
    
    for line in lines:
        line_str = line.strip()
        if not line_str or line_str.startswith("#"):
            continue
        tokens = line_str.split()
        if len(tokens) >= 21:
            try:
                # Standardni format gPPP z 23 žetoni (ali 22, če MJD ni razdeljen)
                # 0: MJD_day, 1: MJD_sec, ..., 10: #OGAL, 11: S0, 12: GTEST, 13: X, 14: Y, 15: Z,
                # 16: Q11, 17: Q22, 18: Q33, 19: Q12, 20: Q13, 21: Q23, 22: MARKER
                if len(tokens) >= 23:
                    mjd = f"{tokens[0]} {tokens[1]}"
                    s0 = float(tokens[11])
                    gtest = float(tokens[12])
                    x_val = float(tokens[13])
                    y_val = float(tokens[14])
                    z_val = float(tokens[15])
                    q11 = float(tokens[16])
                    q22 = float(tokens[17])
                    q33 = float(tokens[18])
                    q12 = float(tokens[19])
                    q13 = float(tokens[20])
                    q23 = float(tokens[21])
                    marker = tokens[22]
                else:
                    mjd = f"{tokens[0]} {tokens[1]}"
                    s0 = float(tokens[10])
                    gtest = float(tokens[11])
                    x_val = float(tokens[12])
                    y_val = float(tokens[13])
                    z_val = float(tokens[14])
                    q11 = float(tokens[15])
                    q22 = float(tokens[16])
                    q33 = float(tokens[17])
                    q12 = float(tokens[18])
                    q13 = float(tokens[19])
                    q23 = float(tokens[20])
                    marker = tokens[21] if len(tokens) > 21 else "GNSS"

                records.append({
                    "filename": filename,
                    "mjd": mjd,
                    "s0": s0,
                    "gtest": gtest,
                    "X": x_val,
                    "Y": y_val,
                    "Z": z_val,
                    "Q11": q11,
                    "Q22": q22,
                    "Q33": q33,
                    "Q12": q12,
                    "Q13": q13,
                    "Q23": q23,
                    "marker": marker
                })
            except Exception:
                continue

    return records


def get_available_local_gppp_samples() -> list[Path]:
    """Poišče lokalne vzorčne gPPP datoteke (*.CRD) v korenskem imeniku ali mapi viri/."""
    candidates = []
    root = Path.cwd()
    for pattern in ["*.CRD", "*.crd"]:
        candidates.extend(list(root.glob(pattern)))
        viri_dir = root / "viri"
        if viri_dir.exists():
            candidates.extend(list(viri_dir.glob(pattern)))
    
    # Sortiraj po imenu in odstrani morebitne podvojitve
    unique_paths = sorted(list({p.resolve(): p for p in candidates}.values()))
    return unique_paths


def load_local_gppp_samples() -> list[dict]:
    """Naloži lokalne vzorčne gPPP datoteke (FGG2)."""
    samples = get_available_local_gppp_samples()
    all_recs = []
    for s in samples:
        try:
            content = s.read_text(encoding="utf-8")
        except Exception:
            try:
                content = s.read_text(encoding="latin-1")
            except Exception:
                continue
        recs = parse_gppp_text(content, filename=s.name)
        all_recs.extend(recs)
    return all_recs


def build_gppp_matrices(
    epochs: list[dict],
    konvencija: str = "KONVENCIJA_1",
    x0_mode: str = "NULA",
    custom_x0: tuple = (0.0, 0.0, 0.0),
    multiply_s0: bool = False
) -> dict:
    """Sestavi sistem matrik (POSREDNA izravnava) za k epoh GNSS gPPP."""
    k = len(epochs)
    n = 3 * k
    c = 3 * k
    u = 3

    # Določitev približnih vrednosti x0
    if x0_mode == "NULA":
        x0_vec = np.zeros((3, 1))
    elif x0_mode == "PRVA_EPOHA":
        x0_vec = np.array([epochs[0]["X"], epochs[0]["Y"], epochs[0]["Z"]]).reshape((3, 1))
    elif x0_mode == "SREDINA":
        mean_x = float(np.mean([ep["X"] for ep in epochs]))
        mean_y = float(np.mean([ep["Y"] for ep in epochs]))
        mean_z = float(np.mean([ep["Z"] for ep in epochs]))
        x0_vec = np.array([mean_x, mean_y, mean_z]).reshape((3, 1))
    else:  # ROCNO
        x0_vec = np.array([custom_x0[0], custom_x0[1], custom_x0[2]]).reshape((3, 1))

    marker_name = epochs[0].get("marker", "GNSS")

    # Matrika A = I_(3k)
    A_mat = np.eye(n)
    df_A = pd.DataFrame(
        A_mat,
        index=[f"Enačba {i+1}" for i in range(c)],
        columns=[f"v_{j+1}" for j in range(n)]
    )

    # Matrika B (3k x 3): naloženi bloki -I (Konvencija 1) ali +I (Konvencija 2)
    B_mat = np.zeros((n, 3))
    sign_B = -1.0 if konvencija == "KONVENCIJA_1" else 1.0
    for e in range(k):
        B_mat[3*e : 3*e+3, :] = sign_B * np.eye(3)

    df_B = pd.DataFrame(
        B_mat,
        index=[f"Enačba {i+1}" for i in range(c)],
        columns=[f"Δ_{j+1}" for j in range(u)]
    )

    # Vektor opazovanj l (3k x 1)
    l_vec = np.zeros((n, 1))
    imena_vrstic_l = []
    for e, ep in enumerate(epochs):
        idx = 3 * e
        l_vec[idx, 0] = ep["X"]
        l_vec[idx + 1, 0] = ep["Y"]
        l_vec[idx + 2, 0] = ep["Z"]
        imena_vrstic_l.extend([
            f"l_{idx+1}",
            f"l_{idx+2}",
            f"l_{idx+3}"
        ])

    df_l = pd.DataFrame({
        "Merjeno l_i": l_vec.flatten(),
        "p_i (utež)": np.ones(n)
    }, index=imena_vrstic_l)

    # Vektor prostih členov f (3k x 1)
    # Konvencija 1: f = x0 - l (ko je x0=0, je f = -l)
    # Konvencija 2: f = l - x0 (ko je x0=0, je f = +l)
    x0_tiled = np.tile(x0_vec, (k, 1))
    if konvencija == "KONVENCIJA_1":
        f_vec = x0_tiled - l_vec
    else:
        f_vec = l_vec - x0_tiled

    df_f = pd.DataFrame(
        {"f (prosti člen)": f_vec.flatten()},
        index=[f"Enačba {i+1}" for i in range(c)]
    )

    # Polna kovariančna matrika Sigma (3k x 3k)
    Sigma_mat = np.zeros((n, n))
    for e, ep in enumerate(epochs):
        scale = (ep["s0"] ** 2) if multiply_s0 else 1.0
        q11 = ep["Q11"] * scale
        q22 = ep["Q22"] * scale
        q33 = ep["Q33"] * scale
        q12 = ep["Q12"] * scale
        q13 = ep["Q13"] * scale
        q23 = ep["Q23"] * scale

        block = np.array([
            [q11, q12, q13],
            [q12, q22, q23],
            [q13, q23, q33]
        ])
        Sigma_mat[3*e : 3*e+3, 3*e : 3*e+3] = block

    df_Sigma = pd.DataFrame(
        Sigma_mat,
        index=imena_vrstic_l,
        columns=imena_vrstic_l
    )

    # Neznanke x0
    imena_x0 = [f"X1", f"Y1", f"X2"] if marker_name == "FGG2" else [f"X_{marker_name}", f"Y_{marker_name}", f"Z_{marker_name}"]
    df_x0 = pd.DataFrame({
        "Ime neznanke": imena_x0,
        "Približna vrednost x₀": x0_vec.flatten()
    }, index=[f"Δ_{j+1}" for j in range(u)])

    return {
        "df_A": df_A,
        "df_B": df_B,
        "df_f": df_f,
        "df_l": df_l,
        "df_Sigma": df_Sigma,
        "df_x0": df_x0,
        "n": n, "c": c, "u": u,
        "metoda": "POSREDNA",
        "tip_utezi": "POLNA_SIGMA"
    }
