import os
import json
import hashlib
import binascii
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd

# =============================================================================
# POTI DO PODATKOV IN INICIALIZACIJA
# =============================================================================
DATA_DIR = Path("data")
USERS_DIR = DATA_DIR / "users"
AUTH_FILE = DATA_DIR / "auth_users.json"


def init_storage():
    """Zagotovi obstoj map in datoteke z uporabniki."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    gitkeep = USERS_DIR / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

    if not AUTH_FILE.exists():
        with open(AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)


# =============================================================================
# VARNO ZGOSČEVANJE GESEL (PBKDF2-HMAC-SHA256)
# =============================================================================
def hash_password(password: str, salt: bytes = None) -> tuple[str, str]:
    """Zgosti geslo z uporabo PBKDF2-HMAC-SHA256 in 100.000 iteracij."""
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    salt_hex = binascii.hexlify(salt).decode("ascii")
    hash_hex = binascii.hexlify(dk).decode("ascii")
    return salt_hex, hash_hex


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    """Preveri ujemanje gesla."""
    try:
        salt = binascii.unhexlify(salt_hex.encode("ascii"))
        _, check_hash = hash_password(password, salt)
        return check_hash == hash_hex
    except Exception:
        return False


# =============================================================================
# UPRAVLJANJE UPORABNIKOV
# =============================================================================
def get_all_users() -> dict:
    init_storage()
    try:
        with open(AUTH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_all_users(users: dict):
    init_storage()
    with open(AUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def get_user_dir(username: str) -> Path:
    user_dir = USERS_DIR / username.lower().strip()
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def register_user(username: str, display_name: str, password: str, role: str = "Študent / Geodet") -> tuple[bool, str]:
    username = username.strip()
    if not username:
        return False, "Uporabniško ime ne sme biti prazno."
    if len(username) < 3:
        return False, "Uporabniško ime mora imeti vsaj 3 znake."
    if not password or len(password) < 4:
        return False, "Geslo mora imeti vsaj 4 znake."

    users = get_all_users()
    key = username.lower()
    if key in users:
        return False, f"Uporabniško ime '{username}' je že zasedeno. Izberite drugo."

    salt_hex, hash_hex = hash_password(password)
    users[key] = {
        "username": username,
        "display_name": display_name.strip() if display_name.strip() else username,
        "role": role,
        "salt": salt_hex,
        "hash": hash_hex,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_all_users(users)

    # Ustvari uporabniško mapo in privzet profil
    user_dir = get_user_dir(username)
    profile = {
        "username": username,
        "display_name": display_name.strip() if display_name.strip() else username,
        "role": role,
        "default_sigma0_sq": 1.0,
        "default_alpha": 0.05,
        "default_alpha0": 0.001,
        "default_beta0": 0.80,
        "preferred_units": "m",
        "angle_units": "gon",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(user_dir / "profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    with open(user_dir / "saved_adjustments.json", "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2, ensure_ascii=False)

    return True, "Profil uspešno ustvarjen! Zdaj se lahko prijavite."


def authenticate_user(username: str, password: str) -> tuple[bool, dict | None, str]:
    users = get_all_users()
    key = username.strip().lower()
    if key not in users:
        return False, None, "Uporabniško ime ne obstaja."

    user_record = users[key]
    if verify_password(password, user_record["salt"], user_record["hash"]):
        user_info = {
            "username": user_record["username"],
            "display_name": user_record.get("display_name", user_record["username"]),
            "role": user_record.get("role", "Študent / Geodet")
        }
        return True, user_info, "Prijava uspešna."
    return False, None, "Napačno geslo."


def get_user_profile(username: str) -> dict:
    user_dir = get_user_dir(username)
    profile_file = user_dir / "profile.json"
    if profile_file.exists():
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "username": username,
        "display_name": username,
        "role": "Študent / Geodet",
        "default_sigma0_sq": 1.0,
        "default_alpha": 0.05,
        "default_alpha0": 0.001,
        "default_beta0": 0.80,
        "preferred_units": "m",
        "angle_units": "gon"
    }


def save_user_profile(username: str, profile_data: dict):
    user_dir = get_user_dir(username)
    with open(user_dir / "profile.json", "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2, ensure_ascii=False)


# =============================================================================
# SHRANJEVANJE IN NALAGANJE PROJEKTOV IZRAVNAV
# =============================================================================
def get_user_saved_adjustments(username: str) -> dict:
    user_dir = get_user_dir(username)
    adj_file = user_dir / "saved_adjustments.json"
    if adj_file.exists():
        try:
            with open(adj_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_user_saved_adjustments(username: str, data: dict):
    user_dir = get_user_dir(username)
    adj_file = user_dir / "saved_adjustments.json"
    with open(adj_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def serialize_dataframe(df: pd.DataFrame | None) -> dict | None:
    if df is None:
        return None
    return df.to_dict(orient="split")


def deserialize_dataframe(d: dict | None) -> pd.DataFrame | None:
    if d is None:
        return None
    try:
        return pd.DataFrame(data=d["data"], index=d["index"], columns=d["columns"])
    except Exception:
        return None


def save_adjustment_project(username: str, project_id: str, name: str, note: str, state_dict: dict) -> bool:
    adjustments = get_user_saved_adjustments(username)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    record = {
        "id": project_id,
        "name": name.strip(),
        "note": note.strip(),
        "updated_at": now_str,
        "metoda": state_dict.get("metoda", "POSREDNA"),
        "n": int(state_dict.get("n", 6)),
        "u": int(state_dict.get("u", 1)),
        "c": int(state_dict.get("c", 6)),
        "sigma0_sq": float(state_dict.get("sigma0_sq", 1.0)),
        "alpha": float(state_dict.get("alpha", 0.05)),
        "alpha0": float(state_dict.get("alpha0", 0.001)),
        "beta0": float(state_dict.get("beta0", 0.80)),
        "tip_utezi": state_dict.get("tip_utezi", "STD_ODKLONI"),
        "matrices": {
            "df_A": serialize_dataframe(state_dict.get("df_A")),
            "df_B": serialize_dataframe(state_dict.get("df_B")),
            "df_f": serialize_dataframe(state_dict.get("df_f")),
            "df_l": serialize_dataframe(state_dict.get("df_l")),
            "df_Sigma": serialize_dataframe(state_dict.get("df_Sigma")),
            "df_x0": serialize_dataframe(state_dict.get("df_x0"))
        }
    }
    adjustments[project_id] = record
    save_user_saved_adjustments(username, adjustments)
    return True


def delete_adjustment_project(username: str, project_id: str) -> bool:
    adjustments = get_user_saved_adjustments(username)
    if project_id in adjustments:
        del adjustments[project_id]
        save_user_saved_adjustments(username, adjustments)
        return True
    return False


def export_user_backup_json(username: str) -> str:
    profile = get_user_profile(username)
    adjustments = get_user_saved_adjustments(username)
    package = {
        "version": "1.0",
        "app": "Geodetski Studio MNK",
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "profile": profile,
        "saved_adjustments": adjustments
    }
    return json.dumps(package, indent=2, ensure_ascii=False)


def import_user_backup_json(username: str, json_str: str) -> tuple[bool, str]:
    try:
        data = json.loads(json_str)
        if "saved_adjustments" not in data:
            return False, "Neveljavna datoteka varnostne kopije (manjka 'saved_adjustments')."
        
        # Posodobi obstoječe ali uvozi
        existing = get_user_saved_adjustments(username)
        imported_adj = data["saved_adjustments"]
        existing.update(imported_adj)
        save_user_saved_adjustments(username, existing)

        if "profile" in data:
            prof = data["profile"]
            prof["username"] = username  # ohrani trenutnega uporabnika
            save_user_profile(username, prof)

        count = len(imported_adj)
        return True, f"Uspešno uvoženih {count} projektov in nastavitve profila!"
    except Exception as e:
        return False, f"Napaka pri branju JSON datoteke: {str(e)}"


# =============================================================================
# PRIJAVNI / REGISTRACIJSKI ZASLON
# =============================================================================
def render_auth_screen():
    """Izriše prijavno / registracijsko pristajalno stran."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #0369a1 100%);
                color: white; padding: 26px 30px; border-radius: 12px; margin-bottom: 24px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.15);">
        <div style="display: flex; align-items: center; gap: 16px;">
            <span style="font-size: 2.8rem;">📐</span>
            <div>
                <h1 style="color: white; margin: 0; font-size: 2.0rem; font-weight: 800; letter-spacing: -0.5px;">
                    GEODETSKI STUDIO
                </h1>
                <p style="color: #bfdbfe; margin: 4px 0 0 0; font-size: 1.05rem;">
                    Platforma za geodetski izravnalni račun (MNK), analizo mrež, transformacije in deformacije
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        st.markdown("""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px 24px 10px 24px; margin-bottom: 20px;">
            <p style="font-size: 1.05rem; font-weight: 600; color: #1e3a8a; margin: 0 0 6px 0;">
                👋 Dobrodošli v platformi za geodete in študente!
            </p>
            <p style="color: #475569; font-size: 0.92rem; margin: 0;">
                Za dostop do svojih shranjenih izravnav, prilagojenih nastavitev in projektov se prijavite ali ustvarite nov profil.
            </p>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔑 Prijava", "📝 Registracija novega uporabnika"])

        with tab_login:
            st.markdown("#### Prijava v obstoječi profil")
            with st.form("form_login"):
                login_user = st.text_input("Uporabniško ime:", placeholder="npr. jan_geodet")
                login_pass = st.text_input("Geslo:", type="password", placeholder="••••••••")
                btn_login = st.form_submit_button("🔑 Prijavi se v Geodetski Studio", use_container_width=True)

                if btn_login:
                    success, user_info, msg = authenticate_user(login_user, login_pass)
                    if success:
                        st.session_state["current_user"] = user_info
                        st.success(f"Pozdravljeni, {user_info['display_name']}!")
                        st.rerun()
                    else:
                        st.error(msg)

        with tab_register:
            st.markdown("#### Ustvarite nov profil")
            with st.form("form_register"):
                reg_user = st.text_input("Uporabniško ime (brez presledkov):", placeholder="npr. marko_novak")
                reg_name = st.text_input("Ime in priimek / Naziv:", placeholder="npr. Marko Novak, mag. inž. geod.")
                reg_role = st.selectbox(
                    "Vaša geodetska vloga:",
                    ["Študent geodezije (FGG / magistrski študij)", "Pooblaščeni inženir geodezije (PI)", "Raziskovalec / Asistent", "Geodet operater"]
                )
                c_p1, c_p2 = st.columns(2)
                with c_p1:
                    reg_pass1 = st.text_input("Geslo:", type="password", placeholder="vsaj 4 znaki")
                with c_p2:
                    reg_pass2 = st.text_input("Ponovite geslo:", type="password", placeholder="ponovite geslo")

                btn_reg = st.form_submit_button("📝 Ustvari profil in se prijavi", use_container_width=True)
                if btn_reg:
                    if reg_pass1 != reg_pass2:
                        st.error("Gesli se ne ujemata!")
                    else:
                        ok, reg_msg = register_user(reg_user, reg_name, reg_pass1, reg_role)
                        if ok:
                            # Avtomatska prijava po uspešni registraciji
                            _, user_info, _ = authenticate_user(reg_user, reg_pass1)
                            st.session_state["current_user"] = user_info
                            st.success(f"Dobrodošli, {user_info['display_name']}! Vaš profil je pripravljen.")
                            st.rerun()
                        else:
                            st.error(reg_msg)

        st.caption("🔒 Vaši podatki in projekti so ločeno shranjeni v vaši varni uporabniški mapi. Gesla so šifrirana s PBKDF2-HMAC-SHA256.")
