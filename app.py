import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
ENV_PATH = os.path.join(PROJECT_DIR, ".env.local")
import sys
import gc
import math
import copy
import json
import time
import subprocess
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import auth
import backend
import streamlib
import github_sync
import ai_coach_pro

# Zagotovi obstoj map za Streamlit Cloud in enkrat ob zagonu strežnika preveri GitHub obnovo
backend.init_storage()
if "github_boot_restored" not in st.session_state:
    st.session_state["github_boot_restored"] = True
    try:
        github_sync.github_restore_all_users_on_startup(force=False)
    except Exception:
        pass

# ==========================================
# KONFIGURACIJA STRANI & PROFIL KOLESARJA
# ==========================================
st.set_page_config(
    page_title="Garmin Kolesarski Trener & Analytics",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""<style>
@media (max-width: 768px) {
  .block-container, div[data-testid="stMainBlockContainer"] { padding: 0.5rem 0.6rem !important; }
  div[data-baseweb="tab-list"] { overflow-x: auto !important; flex-wrap: nowrap !important; -webkit-overflow-scrolling: touch; scrollbar-width: none; }
  div[data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
  div[data-baseweb="tab-list"] button { flex: 0 0 auto !important; min-height: 46px !important; padding: 0.5rem 0.9rem !important; white-space: nowrap !important; }
  div[data-testid="stMetric"] { padding: 0.15rem 0 !important; }
  div[data-testid="stMetricValue"] { font-size: 1.15rem !important; }
  div[data-testid="stMetricValue"] > div { overflow: visible !important; text-overflow: clip !important; white-space: normal !important; }
  div[data-testid="stMetricLabel"], div[data-testid="stMetricDelta"] { font-size: 0.75rem !important; white-space: normal !important; }
  .modebar-container { display: none !important; }
  .js-plotly-plot .plotly .nsewdrag, .js-plotly-plot .plotly .main-svg { touch-action: pan-y !important; }
  h1 { font-size: 1.5rem !important; } h2 { font-size: 1.3rem !important; } h3 { font-size: 1.1rem !important; }
}
</style>""", unsafe_allow_html=True)

if not getattr(st.plotly_chart, "_responsive", False):
    _orig_plotly_chart = st.plotly_chart

    def _responsive_plotly_chart(fig, *args, **kwargs):
        """Kompaktni robovi, vodoravna legenda in brez nadležne orodne vrstice (mobilna optimizacija)."""
        try:
            is_map = any(getattr(t, "type", "") in ("scattermap", "scattermapbox") for t in fig.data)
            if not is_map:
                fig.update_layout(margin=dict(l=10, r=10, t=35, b=20))
                fig.update_xaxes(automargin=True)
                fig.update_yaxes(automargin=True)
                if getattr(fig.layout.legend, "orientation", None) != "h":
                    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            cfg = dict(kwargs.get("config") or {})
            cfg.setdefault("displaylogo", False)
            cfg.setdefault("modeBarButtonsToRemove", ["select2d", "lasso2d"])
            kwargs["config"] = cfg
        except Exception:
            pass
        return _orig_plotly_chart(fig, *args, **kwargs)

    _responsive_plotly_chart._responsive = True
    st.plotly_chart = _responsive_plotly_chart

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
ENV_PATH = os.path.join(PROJECT_DIR, ".env.local")

def get_secret(key: str, default: str = "") -> str:
    """Pridobi skrivni ključ (Streamlit secrets -> os.environ -> .env.local -> default)."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    val = os.environ.get(key)
    if val:
        return val.strip()
    for _ef in (".env.local", ".env"):
        _read_env_fn = getattr(ai_coach_pro, "read_env_file_key", None)
        _ev = _read_env_fn(os.path.join(PROJECT_DIR, _ef), key) if callable(_read_env_fn) else ""
        if _ev:
            return _ev
    return default

def call_gemini_coach(prompt_text: str, api_key: str = None) -> str:
    """Pošlje analizo in vprašanja v Gemini API z uporabo hibridne logike ključev (Flash vs. Pro)."""
    cfg_now = load_user_config()
    if api_key and api_key.strip():
        cfg_now = dict(cfg_now, gemini_api_key=api_key.strip())
    _build_sys_fn = getattr(ai_coach_pro, "build_coach_system_prompt", None)
    sys_ins = _build_sys_fn(cfg_now, get_current_display_name(), user_dir=get_active_user_dir()) if callable(_build_sys_fn) else FALLBACK_INSTRUCTIONS
    _call_hyb_fn = getattr(ai_coach_pro, "call_gemini_hybrid", None)
    if callable(_call_hyb_fn):
        ok, ans, used_model = _call_hyb_fn(
            user_prompt=prompt_text,
            user_cfg=cfg_now,
            project_dir=PROJECT_DIR,
            system_instruction=sys_ins,
        )
        st.session_state["_last_coach_model_used"] = used_model
        _tok_fn = getattr(ai_coach_pro, "get_last_token_usage", None)
        st.session_state["_last_coach_token_usage"] = _tok_fn() if callable(_tok_fn) else None
        return ans
    return "Napaka: modul ai_coach_pro ni ustrezno naložen."

def get_current_user():
    """Vrne uporabniško ime trenutno prijavljenega uporabnika."""
    return st.session_state.get("current_user")

def get_current_display_name():
    """Vrne prikazno ime trenutno prijavljenega uporabnika."""
    return st.session_state.get("display_name") or get_current_user() or "Športnik"

def get_active_user_dir():
    """Vrne izolirano mapo trenutno prijavljenega uporabnika."""
    u = get_current_user()
    if u:
        return auth.get_user_data_dir(u)
    return DATA_DIR

# Cilji in tekme so shranjeni za vsakega uporabnika posebej v config.json ("goals").

# Gosta mreža trajanj za Krivuljo moči (Intervals.icu MMP) in Krivuljo tempa
STANDARD_MMP_DURATIONS = [
    1, 2, 3, 5, 10, 15, 20, 30, 45, 60,
    90, 120, 150, 180, 240, 300, 360, 420, 480, 540,
    600, 720, 900, 1080, 1200, 1500, 1800, 2100, 2400, 2700,
    3000, 3600, 4500, 5400, 7200, 9000, 10800,
    14400, 18000, 21600, 28800
]

DURATION_LABELS = {
    1: "1s", 2: "2s", 3: "3s", 5: "5s", 10: "10s", 15: "15s", 20: "20s", 30: "30s", 45: "45s",
    60: "1m", 90: "1m30s", 120: "2m", 150: "2m30s", 180: "3m", 240: "4m", 300: "5m",
    360: "6m", 420: "7m", 480: "8m", 540: "9m", 600: "10m", 720: "12m", 900: "15m",
    1080: "18m", 1200: "20m", 1500: "25m", 1800: "30m", 2100: "35m", 2400: "40m",
    2700: "45m", 3000: "50m", 3600: "60m", 4500: "75m", 5400: "90m", 7200: "120m",
    9000: "150m", 10800: "180m", 14400: "240m", 18000: "5h", 21600: "6h", 28800: "8h"
}

# ==========================================
# UPRAVLJANJE NASTAVITEV & SEZON (config.json)
# ==========================================
FALLBACK_MODEL_OPTIONS_PRO = [
    "gemini-4-hydrogen",
    "gemini-3.8-flash",
    "gemini-3.8-pro",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "Poljuben model (vnesi ime modela)"
]

FALLBACK_INSTRUCTIONS = """Ti si moj osebni trener za vzdržljivostne športe (cestno kolesarstvo / trenažer / tek). Tvoja naloga je dolgoročno izboljševati mojo zmogljivost in mi pomagati sprejemati dobre odločitve glede treninga, regeneracije, dirk in periodizacije.

# MOJ PROFIL ŠPORTNIKA
- Višina: [Vpiši svojo višino]
- Teža med sezono / trenutna teža: [Vpiši svojo težo]
- Šport in oprema: [Vpiši svoj šport in opremo]
- Tip športnika: [Vpiši svoj tip športnika]
- Glavni fokus: [Vpiši svoj glavni fokus]

# TRENUTNA PRIPRAVLJENOST IN CONE
- Trenutni FTP zunaj / na trenažerju (oz. Pragovni tempo za tek): [Vpiši svoje vrednosti]
- Osebni rekordi (5m / 20m / 30m / 40m): [Vpiši svoje rekorde]
- Posebnosti pri conah: [Vpiši svoje cone]

# KAJ MI NAJBOLJ USTREZA IN MOJE SLABOSTI
- Kateri intervali mi prinašajo največ napredka: [Vpiši svoje podatke]
- Ciljni tedenski volumen v urah: [Vpiši ciljne ure/teden]
- Občutljivost na vročino / mraz in poškodbe: [Vpiši svoje opombe]

# MOJI KRATKOROČNI IN DOLGOROČNI CILJI (TEKME)
- Glavne dirke/tekme, datumi in ciljni vati / W/kg / časi: [Vpiši svoje cilje]"""

def load_user_config():
    u = get_current_user() or "default"
    cache_key = f"_cached_user_cfg_{u}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    u_dir = get_active_user_dir()
    cfg_path = os.path.join(u_dir, "config.json")
    _model_opts = getattr(ai_coach_pro, "MODEL_OPTIONS_PRO", FALLBACK_MODEL_OPTIONS_PRO)
    defaults = {
        "garmin_api_key": get_secret("GARMIN_PERSONAL_API_KEY", ""),
        "gemini_api_key": "",
        "claude_api_key": "",
        "personal_gemini_key_set": False,
        "ai_model_choice": _model_opts[0] if _model_opts else FALLBACK_MODEL_OPTIONS_PRO[0],
        "ai_custom_model": "",
        "custom_coach_instructions": getattr(ai_coach_pro, "DEFAULT_COACH_TEMPLATE", getattr(ai_coach_pro, "DEFAULT_KOLESARSKI_TRENING_INSTRUCTIONS", FALLBACK_INSTRUCTIONS)),
        "ftp_outdoor": 320,
        "ftp_indoor": 305,
        "weight_kg": 65.5,
        "max_hr": 202,
        "rhr_base": 48,
        "sport_profile": "🚴♂️ Kolesar (Cesta / Trenažer / MTB)",
        "threshold_pace": "",
        "lthr": 0,
        "vdot_ref_dist": "Brez",
        "vdot_ref_time": "",
        "run_cadence": 180,
        "goals": [],
        "prefs": {"include_questions": True},
        "seasons": {
            "42_days": {"name": "42 Dnevi", "type": "rolling_42", "start": "", "end": "", "color": "#7e57c2", "weight_kg": 65.5},
            "season_current": {"name": "Ta sezona", "type": "fixed", "start": "2025-11-01", "end": "2026-10-31", "color": "#d81b60", "weight_kg": 65.5},
            "season_previous": {"name": "Prejšnja sezona", "type": "fixed", "start": "2024-11-01", "end": "2025-10-31", "color": "#f4511e", "weight_kg": 65.5},
            "season_2024": {"name": "Sezona 2024", "type": "fixed", "start": "2023-10-27", "end": "2024-11-07", "color": "#43a047", "weight_kg": 65.5},
            "season_2023": {"name": "Sezona 2023", "type": "fixed", "start": "2022-11-14", "end": "2023-10-26", "color": "#fdd835", "weight_kg": 65.5}
        }
    }
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                defaults.update(data)
                _get_m_key = getattr(ai_coach_pro, "get_master_gemini_key", lambda _p: "")
                if not defaults.get("personal_gemini_key_set") and defaults.get("gemini_api_key") == _get_m_key(PROJECT_DIR):
                    defaults["gemini_api_key"] = ""
                if "seasons" not in defaults or not defaults["seasons"]:
                    defaults["seasons"] = {
                        "42_days": {"name": "42 Dnevi", "type": "rolling_42", "start": "", "end": "", "color": "#7e57c2", "weight_kg": defaults.get("weight_kg", 65.5)},
                        "season_current": {"name": "Ta sezona", "type": "fixed", "start": "2025-11-01", "end": "2026-10-31", "color": "#d81b60", "weight_kg": defaults.get("weight_kg", 65.5)},
                        "season_previous": {"name": "Prejšnja sezona", "type": "fixed", "start": "2024-11-01", "end": "2025-10-31", "color": "#f4511e", "weight_kg": defaults.get("weight_kg", 65.5)},
                        "season_2024": {"name": "Sezona 2024", "type": "fixed", "start": "2023-10-27", "end": "2024-11-07", "color": "#43a047", "weight_kg": defaults.get("weight_kg", 65.5)},
                        "season_2023": {"name": "Sezona 2023", "type": "fixed", "start": "2022-11-14", "end": "2023-10-26", "color": "#fdd835", "weight_kg": defaults.get("weight_kg", 65.5)}
                    }
        except Exception:
            pass
    st.session_state[cache_key] = defaults
    return defaults

def save_user_config(cfg):
    u_dir = get_active_user_dir()
    cfg_path = os.path.join(u_dir, "config.json")
    try:
        os.makedirs(u_dir, exist_ok=True)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        u = get_current_user()
        st.session_state[f"_cached_user_cfg_{u or 'default'}"] = cfg
        if u:
            try:
                github_sync.github_push_file(
                    f"data/users/{u}/config.json",
                    commit_message=f"Update config for {u}",
                    async_push=True,
                )
            except Exception:
                pass
        return True
    except Exception:
        return False

# ==========================================
# SAMODEJNO SHRANJEVANJE DNEVNIKA (user_journal.json)
# ==========================================
def load_user_journal():
    u = get_current_user() or "default"
    cache_key = f"_cached_user_journal_{u}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    u_dir = get_active_user_dir()
    j_path = os.path.join(u_dir, "user_journal.json")
    defaults = {"morning_entries": {}, "workout_entries": {}}
    if os.path.exists(j_path):
        try:
            with open(j_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                defaults.update(data)
        except Exception:
            pass
    st.session_state[cache_key] = defaults
    return defaults

def save_user_journal(journal):
    u_dir = get_active_user_dir()
    j_path = os.path.join(u_dir, "user_journal.json")
    try:
        os.makedirs(u_dir, exist_ok=True)
        with open(j_path, "w", encoding="utf-8") as f:
            json.dump(journal, f, indent=2, ensure_ascii=False)
        u = get_current_user()
        st.session_state[f"_cached_user_journal_{u or 'default'}"] = journal
        if u:
            try:
                github_sync.github_push_file(
                    f"data/users/{u}/user_journal.json",
                    commit_message=f"Update user_journal for {u}",
                    async_push=True,
                )
            except Exception:
                pass
        return True
    except Exception:
        return False

# ==========================================
# ČASOVNI PRETVORNIK ZA SPANEC (CET / CEST)
# ==========================================
def format_sleep_times(sleep_obj):
    """Pretvori GMT čas spanca v lokalni slovenski čas (Europe/Ljubljana), z avtomatskim upoštevanjem zimskega/poletnega časa."""
    if not sleep_obj:
        return "Ni zabeleženo"
    gmt_s = sleep_obj.get("sleepStartTimestampGMT")
    gmt_e = sleep_obj.get("sleepEndTimestampGMT")
    if gmt_s and gmt_e:
        try:
            lj_tz = ZoneInfo("Europe/Ljubljana")
            dt_s = datetime.fromtimestamp(gmt_s / 1000.0, tz=timezone.utc).astimezone(lj_tz)
            dt_e = datetime.fromtimestamp(gmt_e / 1000.0, tz=timezone.utc).astimezone(lj_tz)
            return f"{dt_s.strftime('%H:%M')} – {dt_e.strftime('%H:%M')}"
        except Exception:
            pass
    loc_s = sleep_obj.get("sleepStartTimestampLocal")
    loc_e = sleep_obj.get("sleepEndTimestampLocal")
    if loc_s and loc_e:
        try:
            dt_s = datetime.fromtimestamp(loc_s / 1000.0, tz=timezone.utc)
            dt_e = datetime.fromtimestamp(loc_e / 1000.0, tz=timezone.utc)
            return f"{dt_s.strftime('%H:%M')} – {dt_e.strftime('%H:%M')}"
        except Exception:
            pass
    return "Ni zabeleženo"

# ==========================================
# POMNILNIŠKO OPTIMIZIRANO NALAGANJE TOKOV (LAZY STREAMS)
# ==========================================
from streamlib import LazyStream


# ==========================================
# NALAGANJE PODATKOV (IZOLIRANO PO UPORABNIKU)
# ==========================================
@st.cache_data(ttl=600)
def load_all_data(username=None):
    user_dir = auth.get_user_data_dir(username) if username else DATA_DIR
    splits_dir = os.path.join(user_dir, "splits")
    streams_dir = os.path.join(user_dir, "streams")
    mmp_dir = os.path.join(user_dir, "mmp")

    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(streams_dir, exist_ok=True)
    os.makedirs(mmp_dir, exist_ok=True)

    def read_json(fname, default):
        p = os.path.join(user_dir, fname)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return default

    activities_dict = read_json("activities.json", {})
    dailies_dict = read_json("dailies.json", {})
    sleep_dict = read_json("sleep.json", {})
    hrv_dict = read_json("hrv.json", {})
    weight_dict = read_json("weight.json", {})
    hr_zones_list = read_json("heart_rate_zones.json", [])

    splits_dict = {}
    if os.path.exists(splits_dir):
        for fname in os.listdir(splits_dir):
            if fname.endswith(".json"):
                aid = fname.replace(".json", "")
                try:
                    with open(os.path.join(splits_dir, fname), "r", encoding="utf-8") as f:
                        splits_dict[aid] = json.load(f)
                except Exception:
                    pass

    # Hitri predpomnilnik povzetkov tokov (_stream_summaries_cache.json), da ne beremo večmegabajtnih streamov ob vsakem zagonu
    summaries_cache_path = os.path.join(user_dir, "_stream_summaries_cache.json")
    summaries_cache = {}
    cache_dirty = False
    if os.path.exists(summaries_cache_path):
        try:
            with open(summaries_cache_path, "r", encoding="utf-8") as cf:
                summaries_cache = json.load(cf) or {}
        except Exception:
            summaries_cache = {}

    def read_summary_stub(fpath, aid):
        nonlocal cache_dirty
        try:
            st_stat = os.stat(fpath)
            mtime_key = f"{int(st_stat.st_mtime)}_{st_stat.st_size}"
            cached_entry = summaries_cache.get(aid)
            if isinstance(cached_entry, dict) and cached_entry.get("_mtime_key") == mtime_key:
                stub = dict(cached_entry)
                stub.pop("_mtime_key", None)
                return stub

            with open(fpath, "r", encoding="utf-8") as f:
                raw_s = json.load(f)
            act_meta = activities_dict.get(aid) or {}
            coords_list = raw_s.get("coordinates") or raw_s.get("coords") or raw_s.get("polyline") or []
            stub = {
                "summary": raw_s.get("summary", {}),
                "repairTried": raw_s.get("repairTried", False),
                "activityId": aid,
                "startTimeLocal": raw_s.get("startTimeLocal") or act_meta.get("startTimeLocal") or "",
                "activityName": raw_s.get("activityName") or act_meta.get("activityName") or "",
                "n": raw_s.get("n") or (raw_s.get("summary") or {}).get("totalPoints") or 0,
                "hasCoordinates": bool(isinstance(coords_list, list) and len(coords_list) >= 2),
            }
            cache_entry = dict(stub)
            cache_entry["_mtime_key"] = mtime_key
            summaries_cache[aid] = cache_entry
            cache_dirty = True
            del raw_s
            return stub
        except Exception:
            return {"summary": {}, "repairTried": False, "activityId": aid}

    streams_dict = {}
    if os.path.exists(streams_dir):
        for fname in os.listdir(streams_dir):
            if fname.endswith(".json"):
                aid = fname.replace(".json", "")
                fpath = os.path.join(streams_dir, fname)
                try:
                    streams_dict[aid] = LazyStream(aid, fpath, read_summary_stub(fpath, aid))
                except Exception:
                    pass

    if cache_dirty:
        try:
            with open(summaries_cache_path, "w", encoding="utf-8") as cf:
                json.dump(summaries_cache, cf, ensure_ascii=False)
        except Exception:
            pass

    mmp_dict = {}
    if os.path.exists(mmp_dir):
        for fname in os.listdir(mmp_dir):
            if fname.endswith(".json"):
                aid = fname.replace(".json", "")
                try:
                    with open(os.path.join(mmp_dir, fname), "r", encoding="utf-8") as f:
                        mmp_dict[aid] = json.load(f)
                except Exception:
                    pass

    intraday_dir = os.path.join(user_dir, "intraday")
    os.makedirs(intraday_dir, exist_ok=True)
    intraday_dict = {}
    if os.path.exists(intraday_dir):
        for fname in os.listdir(intraday_dir):
            if fname.endswith(".json"):
                d_k = fname.replace(".json", "")
                try:
                    with open(os.path.join(intraday_dir, fname), "r", encoding="utf-8") as f:
                        intraday_dict[d_k] = json.load(f)
                except Exception:
                    pass

    gc.collect()

    return {
        "activities": activities_dict,
        "dailies": dailies_dict,
        "sleep": sleep_dict,
        "hrv": hrv_dict,
        "weight": weight_dict,
        "hr_zones": hr_zones_list,
        "splits": splits_dict,
        "streams": streams_dict,
        "mmp": mmp_dict,
        "intraday": intraday_dict,
    }

# ==========================================
# AVTENTIKACIJSKA STRAN (LOGIN & REGISTRACIJA)
# ==========================================
def render_auth_page():
    col_spacer1, col_center, col_spacer2 = st.columns([1, 2, 1])
    with col_center:
        st.markdown(
            "<div style='text-align: center; margin-top: 25px; margin-bottom: 25px;'>"
            "<h1 style='margin-bottom: 5px;'>🚴 Garmin Kolesarski Trener</h1>"
            "<h4 style='color: #94a3b8; font-weight: normal; margin-top: 0;'>"
            "Zavarovan večuporabniški sistem &bull; Analiza moči &bull; Priprava na tekme in cilje"
            "</h4>"
            "</div>",
            unsafe_allow_html=True
        )

        tab_login, tab_register = st.tabs(["🔑 Prijava", "📝 Registracija"])

        with tab_login:
            st.markdown("#### Prijava v svoj račun")
            with st.form("login_form"):
                u_in = st.text_input("Uporabniško ime:", placeholder="npr. kolesar1").strip()
                p_in = st.text_input("Geslo:", type="password", placeholder="Vnesite svoje geslo")
                submit_login = st.form_submit_button("Prijavi se", use_container_width=True)

                if submit_login:
                    if not u_in or not p_in:
                        st.error("Prosimo, vnesite uporabniško ime in geslo.")
                    else:
                        ok, user_or_err = auth.authenticate_user(u_in, p_in)
                        if ok:
                            st.session_state["current_user"] = user_or_err["username"].lower()
                            st.session_state["display_name"] = user_or_err["display_name"]
                            st.success(f"Pozdravljen, {user_or_err['display_name']}!")
                            st.rerun()
                        else:
                            st.error(f"Prijava ni uspela: {user_or_err}")

        with tab_register:
            st.markdown("#### Ustvari nov kolesarski račun")
            has_legacy = auth.has_legacy_data()
            all_users = auth.load_auth_users()
            is_first_user = len(all_users) == 0

            with st.form("register_form"):
                reg_u = st.text_input("Uporabniško ime:", placeholder="npr. kolesar1 (vsaj 3 znaki)").strip()
                reg_name = st.text_input("Prikazno ime (Ime Športnika):", placeholder="npr. Janez Novak").strip()
                reg_p1 = st.text_input("Geslo:", type="password", placeholder="Vsaj 5 znakov")
                reg_p2 = st.text_input("Ponovite geslo:", type="password")

                claim_box = False
                if has_legacy:
                    claim_box = st.checkbox(
                        "📥 Prevzemi obstoječe zgodovinske podatke in nastavitve (1.144 aktivnosti)",
                        value=is_first_user,
                        help="Če je to vaš primarni račun, bo celotna dosedanja zgodovina aktivnosti, spanja, krivulje moči in dnevnika varno prenesena v vaš novi profil."
                    )

                submit_reg = st.form_submit_button("Ustvari nov račun", use_container_width=True)

                if submit_reg:
                    if not reg_u or not reg_name or not reg_p1:
                        st.error("Prosimo, izpolnite vsa polja.")
                    elif reg_p1 != reg_p2:
                        st.error("Vneseni gesli se ne ujemata.")
                    else:
                        ok, msg = auth.register_user(reg_u, reg_name, reg_p1, claim_legacy=claim_box)
                        if ok:
                            st.success("Račun je bil uspešno ustvarjen! Samodejna prijava...")
                            st.session_state["current_user"] = reg_u.lower()
                            st.session_state["display_name"] = reg_name
                            st.rerun()
                        else:
                            st.error(f"Registracija ni uspela: {msg}")

def get_sport_category(type_key, act_name=""):
    tk = (type_key or "").lower()
    nm = (act_name or "").lower()
    if any(k in tk for k in ["road", "biking", "cycling", "gravel", "mountain"]):
        if any(k in tk for k in ["indoor", "virtual"]) or any(k in nm for k in ["zwift", "trainer", "trenažer", "rouvy", "indoor"]):
            return "Trenažer / Indoor"
        return "Cestno kolo"
    if any(k in tk for k in ["run", "trail_run", "treadmill", "jogging"]):
        return "Tek"
    if any(k in tk for k in ["walk", "hike", "mountaineer"]):
        return "Hoja / Pohodništvo"
    if any(k in tk for k in ["swim", "pool", "open_water"]):
        return "Plavanje"
    if any(k in tk for k in ["strength", "gym", "fitness", "bouldering", "climb"]):
        return "Moč / Fitnes / Plezanje"
    return "Drugo"

def is_indoor_cycling(act_obj):
    t_key = act_obj.get("activityType", {}).get("typeKey", "").lower()
    nm = act_obj.get("activityName", "").lower()
    if t_key in ["virtual_ride", "indoor_cycling"]:
        return True
    if any(k in nm for k in ["zwift", "trainer", "trenažer", "rouvy", "mywhoosh", "indoor"]):
        return True
    return False

def get_power_zones(ftp):
    return {
        "Z1": ("Regeneracija (<55%)", 0, round(ftp * 0.55)),
        "Z2": ("Baza (55–75%)", round(ftp * 0.55), round(ftp * 0.75)),
        "Z3": ("Tempo (75–90%)", round(ftp * 0.75), round(ftp * 0.90)),
        "SST": ("Sweet Spot (88–94%)", round(ftp * 0.88), round(ftp * 0.94)),
        "Z4": ("Threshold (90–105%)", round(ftp * 0.90), round(ftp * 1.05)),
        "Z5": ("VO2 Max (105–120%)", round(ftp * 1.05), round(ftp * 1.20)),
        "Z6": ("Anaerobno (>120%)", round(ftp * 1.20), 1600)
    }

def compute_fixed_10m_blocks(p_arr, hr_arr=None, cad_arr=None, spd_arr=None, temp_arr=None, fallback_temp=None, block_sec=600, ele_arr=None):
    """Izračuna fiksne 10-minutne bloke (600 s) za kolesarsko vadbo, vključno s temperaturo (°C)."""
    elapsed_sec = len(p_arr) if p_arr is not None else 0
    if elapsed_sec == 0:
        return []
    num_blocks = int(math.ceil(elapsed_sec / block_sec))
    blocks = []
    p_np = np.array(p_arr, dtype=float)
    hr_np = np.array(hr_arr, dtype=float) if (hr_arr is not None and len(hr_arr) == elapsed_sec) else None
    cad_np = np.array(cad_arr, dtype=float) if (cad_arr is not None and len(cad_arr) == elapsed_sec) else None
    spd_np = np.array(spd_arr, dtype=float) if (spd_arr is not None and len(spd_arr) == elapsed_sec) else None
    tmp_np = np.array(temp_arr, dtype=float) if (temp_arr is not None and len(temp_arr) == elapsed_sec) else None
    ele_np = np.array(ele_arr, dtype=float) if (ele_arr is not None and len(ele_arr) == elapsed_sec) else None

    for b in range(num_blocks):
        bs = b * block_sec
        be = min((b + 1) * block_sec, elapsed_sec)
        bp = p_np[bs:be]
        bhr = hr_np[bs:be] if hr_np is not None else np.array([])
        bcad = cad_np[bs:be] if cad_np is not None else np.array([])
        bspd = spd_np[bs:be] if spd_np is not None else np.array([])

        if len(bspd) > 0 and np.any(bspd > 0.5):
            b_m = bspd > 0.5
            bmp = bp[b_m]
            bmhr = bhr[b_m] if len(bhr) == len(b_m) else bhr
            bmcad = bcad[b_m] if len(bcad) == len(b_m) else bcad
        else:
            bmp = bp
            bmhr = bhr
            bmcad = bcad

        b_avg_w = float(np.mean(bmp)) if len(bmp) > 0 else 0.0
        b_max_w = float(np.max(bp)) if len(bp) > 0 else 0.0
        val_hr = bmhr[bmhr > 0] if len(bmhr) > 0 else np.array([])
        b_avg_hr = float(np.mean(val_hr)) if len(val_hr) > 0 else 0.0
        b_max_hr = float(np.max(val_hr)) if len(val_hr) > 0 else 0.0
        b_min_hr = float(np.min(val_hr)) if len(val_hr) > 0 else 0.0
        val_cad = bmcad[bmcad > 0] if len(bmcad) > 0 else np.array([])
        b_avg_cad = float(np.mean(val_cad)) if len(val_cad) > 0 else 0.0

        if len(bp) >= 30:
            r_roll = np.convolve(bp, np.ones(30) / 30, mode="valid")
            b_np = float((np.mean(r_roll ** 4)) ** 0.25)
        else:
            b_np = b_avg_w

        b_ef = round(b_np / b_avg_hr, 2) if b_avg_hr > 0 else 0.0
        b_vi = round(b_np / b_avg_w, 2) if b_avg_w > 0 else 1.0
        b_elev_gain = int(round(float(np.sum(np.clip(np.diff(ele_np[bs:be]), 0, 2.8))))) if (ele_np is not None and len(ele_np[bs:be]) > 1) else 0
        b_coast = round((np.sum(bmp == 0) / len(bmp)) * 100, 1) if len(bmp) > 0 else 0.0

        b_temp = fallback_temp
        if tmp_np is not None:
            bt = tmp_np[bs:be]
            vt = bt[~np.isnan(bt) & (bt > -45) & (bt < 65)]
            if len(vt) > 0:
                b_temp = round(float(np.mean(vt)), 1)

        bm_ = max(1, int(block_sec // 60))
        label = f"{b * bm_}–{min((b + 1) * bm_, int(math.ceil(elapsed_sec / 60)))} min"
        dur_str = f"{len(bmp) // 60}:{len(bmp) % 60:02d}"

        blocks.append({
            "label": label,
            "movingTime": dur_str,
            "avgW": round(b_avg_w),
            "np": round(b_np),
            "maxW": round(b_max_w),
            "vi": b_vi,
            "minHR": round(b_min_hr) if b_min_hr > 0 else "--",
            "elevGain": b_elev_gain,
            "avgHR": round(b_avg_hr) if b_avg_hr > 0 else "--",
            "maxHR": round(b_max_hr) if b_max_hr > 0 else "--",
            "avgCad": round(b_avg_cad) if b_avg_cad > 0 else "--",
            "ef": b_ef,
            "coastingPct": b_coast,
            "avgTemp": b_temp,
        })
    return blocks


def compute_run_time_blocks(sf, block_sec=600, fallback_temp=None):
    """Razrez teka/hoje na časovne odseke poljubne dolžine (tempo, HR, kadenca, vzpon, temperatura)."""
    sf = sf or {}
    spd = np.nan_to_num(np.array(sf.get("speed") or [], dtype=float))
    n = len(spd)
    if n == 0:
        return []

    def _arr(k):
        a = np.array(sf.get(k) or [], dtype=float)
        return a if len(a) == n else None

    hr, cad, ele, tmp = _arr("heartRate"), _arr("cadence"), _arr("elevation"), _arr("temperature")
    hr = np.nan_to_num(hr) if hr is not None else None
    cad = np.nan_to_num(cad) if cad is not None else None
    ele_s = pd.Series(np.nan_to_num(ele)).rolling(5, min_periods=1, center=True).mean().to_numpy() if ele is not None else None
    bm = max(1, int(block_sec // 60))
    total_min = int(math.ceil(n / 60))
    out = []
    for b in range(int(math.ceil(n / block_sec))):
        s, e = b * block_sec, min((b + 1) * block_sec, n)
        v = spd[s:e]
        mv = v > 0.5
        mov_s = int(mv.sum())
        dist_m = float(v[mv].sum())
        pace = (mov_s / (dist_m / 1000.0)) if dist_m > 0 and mov_s > 0 else 0
        h = hr[s:e][hr[s:e] > 0] if hr is not None else np.array([])
        c = cad[s:e][cad[s:e] > 0] if cad is not None else np.array([])
        c_avg = float(np.mean(c)) if len(c) else 0.0
        if 0 < c_avg < 120:
            c_avg *= 2
        asc = 0.0
        if ele_s is not None and e - s > 1:
            d = np.diff(ele_s[s:e])
            asc = float(d[d > 0].sum())
        t_avg = fallback_temp
        if tmp is not None:
            tv = tmp[s:e]
            tv = tv[~np.isnan(tv) & (tv > -45) & (tv < 65)]
            if len(tv):
                t_avg = round(float(np.mean(tv)), 1)
        out.append({
            "label": f"{b * bm}–{min((b + 1) * bm, total_min)} min",
            "movingTime": f"{mov_s // 60}:{mov_s % 60:02d}",
            "dist_km": dist_m / 1000.0,
            "pace_sec": pace,
            "pace_str": fmt_pace(pace) if pace > 0 else "--",
            "avgHR": round(float(np.mean(h))) if len(h) else None,
            "maxHR": round(float(np.max(h))) if len(h) else None,
            "avgCad": round(c_avg) if c_avg > 0 else None,
            "ascent": round(asc),
            "avgTemp": t_avg,
        })
    return out


# ==========================================
# POMOŽNE FUNKCIJE: ŠPORTI, LAPI, ZEMLJEVID, GRAFI, CILJI, REKORDI
# ==========================================
SPORT_FILTERS = ["Vsi športi", "Kolesarjenje", "Tek", "Plavanje", "Hoja / Pohodništvo", "Moč / Ostalo"]


def sport_group(type_key):
    tk = (type_key or "").lower()
    if any(k in tk for k in ["cycl", "bik", "virtual_ride"]):
        return "Kolesarjenje"
    if "run" in tk or "treadmill" in tk:
        return "Tek"
    if "swim" in tk:
        return "Plavanje"
    if any(k in tk for k in ["walk", "hik", "mountaineer"]):
        return "Hoja / Pohodništvo"
    return "Moč / Ostalo"


def fmt_dur(sec):
    sec = int(round(sec or 0))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_pace(sec):
    if not sec or sec <= 0 or not np.isfinite(sec):
        return "--"
    return f"{int(sec // 60)}:{int(round(sec % 60)) % 60:02d}"


def extract_activity_temp_summary(act_obj, sf=None, split_obj=None):
    """Vrne (avg_temp, min_temp, max_temp, temp_str) iz 1 Hz toka, summaryDTO ali povzetka aktivnosti."""
    summ = (sf or {}).get("summary") or {}
    sdto = (split_obj or {}).get("summaryDTO") or {}
    t_arr = (sf or {}).get("temperature") or []
    avg_t = summ.get("avgTemp")
    min_t = summ.get("minTemp")
    max_t = summ.get("maxTemp")
    if avg_t is None and len(t_arr) > 0:
        arr = np.asarray(t_arr, dtype=float)
        vt = arr[~np.isnan(arr) & (arr > -45) & (arr < 65)]
        if len(vt) > 0:
            avg_t = round(float(np.mean(vt)), 1)
            min_t = round(float(np.min(vt)), 1)
            max_t = round(float(np.max(vt)), 1)
    if avg_t is None:
        avg_t = sdto.get("averageTemperature") or (act_obj or {}).get("averageTemperature")
    if min_t is None:
        min_t = sdto.get("minTemperature") if sdto.get("minTemperature") is not None else (act_obj or {}).get("minTemperature")
    if max_t is None:
        max_t = sdto.get("maxTemperature") if sdto.get("maxTemperature") is not None else (act_obj or {}).get("maxTemperature")
    if avg_t is None and min_t is not None and max_t is not None:
        try:
            avg_t = round((float(min_t) + float(max_t)) / 2.0, 1)
        except Exception:
            pass
    if avg_t is not None or min_t is not None or max_t is not None:
        a_s = f"{float(avg_t):.1f} °C" if avg_t is not None else "--"
        mn_s = f"{float(min_t):.1f} °C" if min_t is not None else "--"
        mx_s = f"{float(max_t):.1f} °C" if max_t is not None else "--"
        return avg_t, min_t, max_t, f"Povprečna: {a_s} (Min: {mn_s} | Max: {mx_s})"
    return None, None, None, "Ni zabeležena (naprava brez senzorja ali trenažer)"


def normalize_laps(laps, sf=None, act_obj=None, split_obj=None, weight_kg=None):
    """Pretvori surove Garmin lape (m, s) v prikazne vrednosti (km, mm:ss, tempo, W/kg, EF, temperatura)."""
    out = []
    t_stream = np.asarray((sf or {}).get("temperature") or [], dtype=float)
    avg_act_t, min_act_t, max_act_t, _ = extract_activity_temp_summary(act_obj, sf, split_obj)
    w_kg = float(weight_kg) if (weight_kg is not None and float(weight_kg) > 0) else None

    def _lap_dur_s(item):
        val = (
            item.get("duration")
            if item.get("duration") is not None
            else (
                item.get("movingDuration")
                if item.get("movingDuration") is not None
                else (item.get("elapsedDuration") if item.get("elapsedDuration") is not None else item.get("duration_s"))
            )
        )
        return int(round(float(val or 0)))

    cum_s = 0
    total_dur = sum(_lap_dur_s(x) for x in (laps or []) if isinstance(x, dict))
    for i, lp in enumerate(laps or []):
        if not isinstance(lp, dict):
            continue
        d = lp.get("distance") or 0
        dur = _lap_dur_s(lp)
        pace = (dur / (d / 1000.0)) if d and d > 0 else 0
        s0, s1 = cum_s, cum_s + max(1, dur)
        cum_s = s1
        lp_temp = lp.get("avgTemperature") if lp.get("avgTemperature") is not None else lp.get("averageTemperature")
        if lp_temp is None and len(t_stream) > s0:
            seg_t = t_stream[s0:min(s1, len(t_stream))]
            vt = seg_t[~np.isnan(seg_t) & (seg_t > -45) & (seg_t < 65)]
            if len(vt) > 0:
                lp_temp = round(float(np.mean(vt)), 1)
        if lp_temp is None and avg_act_t is not None:
            # Če imamo min in max temperaturo aktivnosti, rahlo interpoliraj ali uporabi povprečno temperaturo
            if min_act_t is not None and max_act_t is not None and total_dur > 0:
                frac = ((s0 + s1) / 2.0) / float(total_dur)
                lp_temp = round(float(min_act_t) + (float(max_act_t) - float(min_act_t)) * min(1.0, max(0.0, frac * 0.6 + 0.2)), 1)
            else:
                lp_temp = round(float(avg_act_t), 1)
        avg_p = lp.get("avgPower") if lp.get("avgPower") is not None else lp.get("averagePower")
        norm_p = lp.get("normalizedPower") if lp.get("normalizedPower") is not None else lp.get("normPower")
        avg_hr = lp.get("avgHR") if lp.get("avgHR") is not None else lp.get("averageHR")
        max_hr = lp.get("maxHR")
        avg_cad = (
            lp.get("avgCadence")
            if lp.get("avgCadence") is not None
            else (
                lp.get("averageBikeCadence")
                if lp.get("averageBikeCadence") is not None
                else (lp.get("averageRunCadence") if lp.get("averageRunCadence") is not None else lp.get("averageCadence"))
            )
        )
        ascent = (
            lp.get("totalAscent")
            if lp.get("totalAscent") is not None
            else (lp.get("elevationGain") if lp.get("elevationGain") is not None else lp.get("ascent"))
        )
        wkg_val = lp.get("wkg")
        if wkg_val is None and w_kg and avg_p is not None:
            try:
                wkg_val = round(float(avg_p) / w_kg, 2)
            except Exception:
                wkg_val = None
        ef_val = lp.get("ef")
        p_for_ef = norm_p if (norm_p is not None and float(norm_p or 0) > 0) else avg_p
        if ef_val is None and p_for_ef is not None and avg_hr is not None:
            try:
                if float(p_for_ef) > 0 and float(avg_hr) > 0:
                    ef_val = round(float(p_for_ef) / float(avg_hr), 2)
            except Exception:
                ef_val = None
        out.append({
            "lapIndex": lp.get("lapIndex", i + 1),
            "distance_km": d / 1000.0,
            "duration_s": dur,
            "duration_str": fmt_dur(dur),
            "avgPower": avg_p,
            "normPower": norm_p,
            "wkg": wkg_val,
            "avgHR": avg_hr,
            "maxHR": max_hr,
            "avgCadence": avg_cad,
            "pace_str": fmt_pace(pace) if pace > 0 else "--",
            "ascent": ascent,
            "ef": ef_val,
            "avgTemp": lp_temp,
        })
    return out


def hr_fallback_series(n, act_obj, laps):
    """Če v toku ni HR, zgradi zvezno vrsto iz povprečij po lapih ali celotnega povprečja (HR ne sme izginiti)."""
    avg = act_obj.get("averageHR") or 0
    arr = np.zeros(max(n, 0))
    if n <= 0:
        return arr, ""
    if laps:
        pos = 0
        for lp in laps:
            d = int(lp.get("duration_s") or 0)
            h = lp.get("avgHR") or 0
            if d > 0 and h > 0:
                arr[pos:pos + d] = h
            pos += d
        if arr.max() > 0:
            if avg:
                arr[arr == 0] = avg
            return arr, "povprečje po lapih"
    if avg > 0:
        arr[:] = avg
        return arr, "povprečje aktivnosti"
    return arr, ""


def _rolling(a, w):
    return pd.Series(a).rolling(window=w, min_periods=1).mean().to_numpy()


def km_splits_from_stream(sf, act_obj=None, split_obj=None):
    """Pravi 1 km razrez iz 1 Hz toka (hitrost -> razdalja), vključno s temperaturo."""
    spd = np.nan_to_num(np.array((sf or {}).get("speed") or [], dtype=float))
    if len(spd) < 60 or spd.sum() < 1000:
        return []
    hr = np.array(sf.get("heartRate") or [], dtype=float)
    cad = np.array(sf.get("cadence") or [], dtype=float)
    ele = np.array(sf.get("elevation") or [], dtype=float)
    tmp = np.array(sf.get("temperature") or [], dtype=float)
    avg_act_t, _, _, _ = extract_activity_temp_summary(act_obj, sf, split_obj)
    cum = np.cumsum(spd)
    n_km = int(cum[-1] // 1000)
    bounds = [min(int(np.searchsorted(cum, k * 1000)), len(cum) - 1) for k in range(1, n_km + 1)]
    rem = cum[-1] - n_km * 1000
    if rem >= 100:
        bounds.append(len(cum) - 1)
    res, prev = [], 0
    off_gain = float((act_obj or {}).get("elevationGain") or 0.0)
    _, cum_gain, _, _, _, _ = streamlib.compute_elevation_metrics(ele, spd, official_gain=off_gain)
    cg0 = np.concatenate([[0.0], cum_gain]) if len(cum_gain) == len(spd) else None
    for idx, j in enumerate(bounds, 1):
        dist_m = 1000.0 if (idx <= n_km) else rem
        sec = max(1, j + 1 - prev)
        pace = sec / (dist_m / 1000.0)
        seg_hr = hr[prev:j + 1] if len(hr) == len(spd) else np.array([])
        seg_cad = cad[prev:j + 1] if len(cad) == len(spd) else np.array([])
        v_hr = seg_hr[seg_hr > 0]
        v_cad = seg_cad[seg_cad > 0]
        asc = float(cg0[min(j + 1, len(cg0) - 1)] - cg0[prev]) if (cg0 is not None and j > prev) else 0.0
        t_val = avg_act_t
        if len(tmp) == len(spd):
            st_seg = tmp[prev:j + 1]
            vt = st_seg[~np.isnan(st_seg) & (st_seg > -45) & (st_seg < 65)]
            if len(vt):
                t_val = round(float(np.mean(vt)), 1)
        res.append({
            "km": idx, "distance_km": round(dist_m / 1000.0, 2), "duration_str": fmt_dur(sec),
            "pace_sec": pace, "pace_str": fmt_pace(pace),
            "avgHR": round(float(np.mean(v_hr))) if len(v_hr) else None,
            "avgCad": round(float(np.mean(v_cad))) if len(v_cad) else None,
            "elev": round(asc),
            "avgTemp": t_val,
        })
        prev = j + 1
    return res


def get_km_splits(act_obj, sf, laps_raw, split_obj=None):
    """1 km razrez: najprej iz toka, sicer iz Garmin lapov (če so ~1 km). Nikoli izmišljenih podatkov."""
    sp = km_splits_from_stream(sf, act_obj=act_obj, split_obj=split_obj)
    if sp:
        return sp
    laps = normalize_laps(laps_raw, sf=sf, act_obj=act_obj, split_obj=split_obj)
    if len(laps) >= 2 and all(0.85 <= l["distance_km"] <= 1.15 for l in laps[:-1]):
        res = []
        for l in laps:
            pace = l["duration_s"] / l["distance_km"] if l["distance_km"] > 0 else 0
            res.append({
                "km": l["lapIndex"], "distance_km": round(l["distance_km"], 2), "duration_str": l["duration_str"],
                "pace_sec": pace, "pace_str": fmt_pace(pace), "avgHR": round(l["avgHR"]) if l["avgHR"] else None,
                "avgCad": round(l["avgCadence"]) if l["avgCadence"] else None, "elev": round(l["ascent"] or 0),
                "avgTemp": l.get("avgTemp"),
            })
        return res
    return []


def ensure_activity_details(aid, act_obj, raw, force=False, user_cfg=None):
    """
    Zagotovi, da ima izbran trening (tudi starejši od 14 dni ali večletni) shranjen 1 Hz tok z GPS ter kroge.
    Če datoteke že obstajajo lokalno na disku, jih naloži v hipu BREZ mrežnega klica v Garmin!
    Če manjkajo ali so nepopolne, jih samodejno prenese iz Garmina (/details + /splits + FIT fallback) in trajno predpomni.
    """
    aid = str(aid)
    user = get_current_user()
    if not user:
        return
    user_dir = auth.get_user_data_dir(user)
    stream_file = os.path.join(user_dir, "streams", f"{aid}.json")
    split_file = os.path.join(user_dir, "splits", f"{aid}.json")
    mmp_file = os.path.join(user_dir, "mmp", f"{aid}.json")

    stream = (raw.get("streams") or {}).get(aid)
    split = (raw.get("splits") or {}).get(aid) or {}

    # 1. Če v RAM-u še ni naloženo v celoti, najprej hitro preveri lokalni disk (0 mrežnih klicev!)
    if (not stream or (isinstance(stream, LazyStream) and not stream._loaded)) and os.path.exists(stream_file):
        try:
            with open(stream_file, "r", encoding="utf-8") as f:
                disk_stream = json.load(f)
            if disk_stream:
                stream = disk_stream
                raw.setdefault("streams", {})[aid] = disk_stream
        except Exception:
            pass
    if not split and os.path.exists(split_file):
        try:
            with open(split_file, "r", encoding="utf-8") as f:
                disk_split = json.load(f)
            if disk_split:
                split = disk_split
                raw.setdefault("splits", {})[aid] = disk_split
        except Exception:
            pass

    stream_ok = False
    try:
        has_series_or_blocks = bool(
            (stream or {}).get("speed")
            or (stream or {}).get("heartRate")
            or (stream or {}).get("power")
            or (stream or {}).get("coordinates")
            or ((stream or {}).get("summary") or {}).get("blocks")
        )
        stream_ok = bool(stream) and streamlib.stream_is_valid(stream, act_obj) and has_series_or_blocks
    except Exception:
        stream_ok = False
    expected_laps = int((act_obj or {}).get("lapCount") or 0)
    sp_laps = split.get("laps") if isinstance(split, dict) else None
    sp_dtos = split.get("lapDTOs") if isinstance(split, dict) else None
    has_nonempty_laps = bool((isinstance(sp_laps, list) and len(sp_laps) > 0) or (isinstance(sp_dtos, list) and len(sp_dtos) > 0))
    checked_zero = bool(isinstance(split, dict) and split.get("splitsChecked") and expected_laps == 0)
    count_incomplete = bool(expected_laps > 1 and isinstance(sp_laps, list) and len(sp_laps) < expected_laps)
    split_ok = bool(split and (has_nonempty_laps or checked_zero) and not count_incomplete)
    if stream_ok and split_ok and not force:
        return

    flag = f"details_checked_{user}_{aid}"
    if st.session_state.get(flag) and not force:
        return
    st.session_state[flag] = True
    try:
        cfg_to_use = user_cfg if isinstance(user_cfg, dict) else load_user_config()
        with st.spinner("⏳ Prenašam podrobne odseke in zemljevid za ta starejši trening iz Garmina..."):
            payload, split_p, msg = backend.ensure_full_activity_details(
                activity_id=aid,
                username=user,
                config=cfg_to_use,
                force=force,
            )
        if payload:
            coords_list = payload.get("coordinates") or payload.get("coords") or payload.get("polyline") or []
            stub = {
                "summary": payload.get("summary", {}),
                "repairTried": payload.get("repairTried", True),
                "activityId": aid,
                "startTimeLocal": payload.get("startTimeLocal") or (act_obj or {}).get("startTimeLocal") or "",
                "activityName": payload.get("activityName") or (act_obj or {}).get("activityName") or "",
                "n": payload.get("n") or (payload.get("summary") or {}).get("totalPoints") or 0,
                "hasCoordinates": bool(isinstance(coords_list, list) and len(coords_list) >= 2),
            }
            ls_obj = LazyStream(aid, stream_file, stub)
            ls_obj.update(payload)
            ls_obj._loaded = True
            raw.setdefault("streams", {})[aid] = ls_obj
        if split_p:
            raw.setdefault("splits", {})[aid] = split_p
            if isinstance(act_obj, dict) and isinstance(split_p.get("laps"), list) and split_p["laps"]:
                act_obj["laps"] = split_p["laps"]
        if payload or split_p:
            try:
                st.cache_data.clear()
            except Exception:
                pass
        if os.path.exists(mmp_file):
            try:
                with open(mmp_file, "r", encoding="utf-8") as mf:
                    raw.setdefault("mmp", {})[aid] = json.load(mf)
            except Exception:
                pass
        if not payload and not split_p:
            st.warning(f"Podrobnosti treninga ni bilo mogoče pridobiti ({msg}).")
        elif force:
            coords_cnt = len((payload or {}).get("coordinates") or [])
            laps_cnt = len((split_p or {}).get("laps") or [])
            blocks_cnt = len(((payload or {}).get("summary") or {}).get("blocks") or [])
            st.success(
                f"✅ Podrobnosti za trening #{aid} uspešno prenesene! "
                f"(GPS točk: {coords_cnt}, uradni krogi: {laps_cnt}, 10-min bloki: {blocks_cnt})"
            )
    except Exception as e:
        st.warning(f"Samodejni prenos podrobnosti ni uspel: {e}")


def run_batch_backfill_older_workouts(raw, user_cfg, batch_size=20, container=None):
    """
    Poišče do `batch_size` (20) starejših treningov (kolo, tek, hoja, pohodništvo), ki jim na disku
    manjka podroben 1-sekundni tok / zemljevid (`streams/{id}.json`) ali krogi (`splits/{id}.json`),
    in jih zaporedno prenese z napredkom `st.progress` brez ponastavitve zavihka!
    """
    import pull
    user = get_current_user()
    if not user:
        return 0
    target_ui = container if container is not None else st
    user_dir = auth.get_user_data_dir(user)
    streams_dir = os.path.join(user_dir, "streams")
    splits_dir = os.path.join(user_dir, "splits")
    mmp_dir = os.path.join(user_dir, "mmp")
    os.makedirs(streams_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(mmp_dir, exist_ok=True)

    acts_all = list((raw.get("activities") or {}).values())
    acts_all.sort(key=lambda x: x.get("startTimeLocal", ""), reverse=True)

    candidates = []
    sport_keys = ("cycl", "bik", "virtual_ride", "run", "walk", "hik", "mountaineer", "trail")
    for a in acts_all:
        aid = str(a.get("activityId") or "")
        if not aid:
            continue
        t_str = str(a.get("activityType") or "").lower()
        if not any(k in t_str for k in sport_keys):
            continue
        sf_path = os.path.join(streams_dir, f"{aid}.json")
        sp_path = os.path.join(splits_dir, f"{aid}.json")
        has_stream = os.path.exists(sf_path) and os.path.getsize(sf_path) > 120
        has_split = os.path.exists(sp_path) and os.path.getsize(sp_path) > 50
        if not has_stream or not has_split:
            candidates.append(a)
            if len(candidates) >= batch_size:
                break

    if not candidates:
        target_ui.success("✅ Vsi treningi v zgodovini že imajo prenesene zemljevide, 1-sekundne tokove in odseke!")
        return 0

    prog_bar = target_ui.progress(0.0, text=f"Pripravljam prenos za {len(candidates)} starejših treningov...")
    status_box = target_ui.empty()

    custom_key = (user_cfg or {}).get("garmin_api_key") if isinstance(user_cfg, dict) else None
    pull.set_target_user(username=user, custom_api_key=custom_key)
    try:
        tok = pull.get_access_token()
    except Exception as e:
        prog_bar.empty()
        target_ui.error(f"Napaka pri pridobivanju Garmin žetona: {e}")
        return 0

    ok_count = 0
    total_c = len(candidates)
    for idx, act_item in enumerate(candidates, 1):
        aid = str(act_item.get("activityId"))
        d_s = (act_item.get("startTimeLocal") or "")[:10]
        nm_s = act_item.get("activityName") or "Trening"
        prog_bar.progress((idx - 1) / float(total_c), text=f"⏳ [{idx}/{total_c}] Prenašam zemljevid in odseke: {d_s} – {nm_s}...")
        try:
            payload, split_p, _ = pull.ensure_full_activity_details(
                activity_id=aid,
                username=user,
                config=user_cfg,
                token=tok,
                force=False,
                sync_cloud=True,
            )
            if payload:
                sf_path = os.path.join(streams_dir, f"{aid}.json")
                ls = LazyStream(aid, sf_path, {"summary": payload.get("summary", {}), "repairTried": True, "activityId": aid})
                raw.setdefault("streams", {})[aid] = ls
            if split_p:
                raw.setdefault("splits", {})[aid] = split_p
            mf_path = os.path.join(mmp_dir, f"{aid}.json")
            if os.path.exists(mf_path):
                try:
                    with open(mf_path, "r", encoding="utf-8") as mf:
                        raw.setdefault("mmp", {})[aid] = json.load(mf)
                except Exception:
                    pass
            if payload or split_p:
                ok_count += 1
        except Exception as ex_b:
            status_box.caption(f"Opomba pri {aid}: {ex_b}")

    prog_bar.progress(1.0, text=f"✅ Končano! Prenesenih {ok_count}/{total_c} starejših treningov.")
    try:
        github_sync.github_push_file(f"data/users/{user}/activities.json", async_push=True)
    except Exception:
        pass
    status_box.success(f"🎉 Uspešno preneseni zemljevidi, lapi in 10-minutni odseki za **{ok_count}** starejših treningov!")
    return ok_count


def render_activity_map(act_obj, sf, aid):
    """
    Zemljevid trase (Plotly + OpenStreetMap, brez API ključa):
    - Samodejno pobere GPS koordinate iz vseh ključev (coordinates, coords, polyline)
    - Izrisuje polno debelo črto (width=4) za kolesarjenje (#2563eb) in tek/hojo/pohodništvo (#FF4B4B)
    - Zaznava Start (zelena) in Cilj (rdeča)
    """
    try:
        raw_coords = (sf or {}).get("coordinates") or (sf or {}).get("coords") or (sf or {}).get("polyline") or []
        coords = streamlib.clean_coords(raw_coords)
        partial = False
        if len(coords) < 2:
            pts = []
            for la, lo in [(act_obj.get("startLatitude"), act_obj.get("startLongitude")),
                           (act_obj.get("endLatitude"), act_obj.get("endLongitude"))]:
                if la and lo:
                    pts.append([la, lo])
            coords = streamlib.clean_coords(pts)
            partial = True
        if not coords:
            return
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        c_lat, c_lon = (min(lats) + max(lats)) / 2.0, (min(lons) + max(lons)) / 2.0
        d_lat = max(max(lats) - min(lats), 0.002) * 1.25
        d_lon = max(max(lons) - min(lons), 0.002) * 1.25
        zoom = min(16.0, math.log2(984.0 / d_lon), math.log2(720.0 * max(0.2, math.cos(math.radians(c_lat))) / d_lat))
        zoom = max(2.0, zoom)

        # Barva trase: Rdeča (#FF4B4B) za tek, trail, pohodništvo, hojo; Modra (#2563eb) za kolesarjenje
        sp_type = (str(act_obj.get("activityType", {})) + " " + str(act_obj.get("activityName", ""))).lower()
        is_run_walk = any(k in sp_type for k in ["run", "tek", "walk", "hik", "pohod", "hoja", "trail"])
        line_color = "#FF4B4B" if is_run_walk else "#2563eb"

        use_new = hasattr(go, "Scattermap")
        Sc = go.Scattermap if use_new else go.Scattermapbox
        fig = go.Figure()
        if len(coords) >= 2 and not partial:
            fig.add_trace(Sc(lat=lats, lon=lons, mode="lines", line=dict(width=4, color=line_color), name="Trasa", hoverinfo="skip"))
        elif len(coords) >= 2 and partial:
            fig.add_trace(Sc(lat=lats, lon=lons, mode="lines", line=dict(width=2, dash="dash", color=line_color), name="Zračna linija", hoverinfo="skip"))
        fig.add_trace(Sc(lat=[lats[0]], lon=[lons[0]], mode="markers", marker=dict(size=14, color="#16a34a"), name="Start"))
        fig.add_trace(Sc(lat=[lats[-1]], lon=[lons[-1]], mode="markers", marker=dict(size=14, color="#dc2626"), name="Cilj"))
        layout = dict(style="open-street-map", center=dict(lat=c_lat, lon=c_lon), zoom=zoom)
        fig.update_layout(**{("map" if use_new else "mapbox"): layout})
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02, x=0))
        st.markdown("#### 🗺️ Trasa vadbe (OpenStreetMap)")
        if partial:
            st.caption("Podrobna GPS trasa še ni shranjena za to aktivnost – prikazana sta le začetek (zelena) in konec (rdeča). "
                       "Pritisnite **🛠️ Popravi / preveri tokove** v stranski vrstici za ponoven prenos.")
        st.plotly_chart(fig, use_container_width=True, key=f"map_{aid}")
    except Exception:
        pass


def _hr_for_chart(sf, n, act_obj, laps):
    hr = np.nan_to_num(np.array((sf or {}).get("heartRate") or [], dtype=float))
    if len(hr) == n and np.any(hr > 0):
        return hr, ""
    arr, lbl = hr_fallback_series(n, act_obj, laps)
    return arr, lbl


def _temp_for_chart(sf, n, act_obj, laps, split_obj=None):
    """Vrne (temp_series, temp_label) za prikaz temperature skozi trening."""
    t_arr = np.asarray((sf or {}).get("temperature") or [], dtype=float)
    if len(t_arr) == n and np.any(~np.isnan(t_arr)):
        return t_arr, ""
    avg_t, min_t, max_t, _ = extract_activity_temp_summary(act_obj, sf, split_obj)
    if avg_t is not None and n > 0:
        if min_t is not None and max_t is not None and float(max_t) > float(min_t):
            # Gladka krivulja od min proti max/avg skozi čas vadbe
            x = np.linspace(0.0, 1.0, n)
            curve = float(min_t) + (float(max_t) - float(min_t)) * np.sin(np.pi * np.clip(x * 0.85 + 0.1, 0, 1))
            curve = curve + (float(avg_t) - float(np.mean(curve)))
            return curve, "ocena iz min/povp/max"
        return np.full(n, float(avg_t)), "povprečna temperatura"
    return np.array([]), ""


def render_cycling_chart(aid, sf, act_obj, laps, blocks_list, ftp, split_obj=None):
    st.markdown("#### 📈 Interaktivna kolesarska analitika (Moč, HR, Kumulativni vzpon, Naklon % in Temperatura)")
    p = np.nan_to_num(np.array((sf or {}).get("power") or [], dtype=float))
    spd = np.nan_to_num(np.array((sf or {}).get("speed") or [], dtype=float))
    cad = np.nan_to_num(np.array((sf or {}).get("cadence") or [], dtype=float))
    ele = np.nan_to_num(np.array((sf or {}).get("elevation") or [], dtype=float))
    n = max(len(p), len(spd), len(cad), len(ele), len((sf or {}).get("heartRate") or []))
    if n == 0:
        st.info("Časovni tok (1 Hz) za to aktivnost še ni prenesen – prikazani so povzetki po 10-minutnih blokih.")
        if blocks_list:
            xs = [b * 10 + 5 for b in range(len(blocks_list))]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=[b["np"] for b in blocks_list], mode="lines+markers", name="NP (W)", line=dict(color="#3b82f6", width=3)))
            hrs = [b["avgHR"] if isinstance(b["avgHR"], (int, float)) and b["avgHR"] > 0 else None for b in blocks_list]
            if any(h is not None for h in hrs):
                fig.add_trace(go.Scatter(x=xs, y=hrs, mode="lines+markers", name="Avg HR (bpm)", yaxis="y2", line=dict(color="#ef4444", width=2.5)))
            fig.update_layout(height=380, hovermode="x unified", xaxis_title="Čas (min)", yaxis_title="Moč (W)",
                              yaxis2=dict(title="Srčni utrip (bpm)", overlaying="y", side="right", showgrid=False))
            st.plotly_chart(fig, use_container_width=True, key=f"cyc_blk_{aid}")
        return

    hr, hr_lbl = _hr_for_chart(sf, n, act_obj, laps)
    temp_arr, temp_lbl = _temp_for_chart(sf, n, act_obj, laps, split_obj=split_obj)
    off_gain = float(act_obj.get("elevationGain") or 0.0)
    ele_smooth, cum_gain, slope_pct, _, _, _ = streamlib.compute_elevation_metrics(
        ele if len(ele) == n else None, spd if len(spd) == n else np.ones(n), official_gain=off_gain
    )

    opts = [
        "30s Moč (W)",
        "Trenutna moč - 1s (W)",
        "Srčni utrip (bpm)",
        "Hitrost (km/h)",
        "Kadenca (rpm)",
        "Kumulativni višinski metri (Vzpon v m)",
        "Profil nadmorske višine (m)",
        "Naklon klanca (%)",
        "Temperatura (°C)",
    ]
    default = [
        "30s Moč (W)",
        "Srčni utrip (bpm)",
        "Kumulativni višinski metri (Vzpon v m)",
        "Profil nadmorske višine (m)",
        "Naklon klanca (%)",
    ]
    if len(temp_arr) == n:
        default.append("Temperatura (°C)")

    sel = st.multiselect("Prikaz na grafu:", opts, default=default, key=f"cyc_traces_{aid}")
    t_min = np.arange(n) / 60.0
    has_p = len(p) == n and np.any(p > 0)

    from plotly.subplots import make_subplots
    want_bottom = any(k in sel for k in ["Kumulativni višinski metri (Vzpon v m)", "Profil nadmorske višine (m)", "Naklon klanca (%)", "Temperatura (°C)"])
    if want_bottom:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
            row_heights=[0.56, 0.44],
            specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
            subplot_titles=("Moč, Srčni utrip, Hitrost in Kadenca", "Nadmorska višina (m), Kumulativni vzpon (+m), Naklon (%) in Temperatura (°C)")
        )
    else:
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    if "Trenutna moč - 1s (W)" in sel and has_p:
        fig.add_trace(go.Scatter(x=t_min, y=p, name="Moč 1s (W)", mode="lines",
                                 line=dict(color="rgba(59,130,246,0.28)", width=1), hovertemplate="%{y:.0f} W"),
                      row=1, col=1, secondary_y=False)
    if "30s Moč (W)" in sel and has_p:
        fig.add_trace(go.Scatter(x=t_min, y=_rolling(p, 30), name="30s povprečna moč (W)", mode="lines",
                                 line=dict(color="#2563eb", width=2.5), hovertemplate="%{y:.0f} W"),
                      row=1, col=1, secondary_y=False)
    if "Srčni utrip (bpm)" in sel and np.any(hr > 0):
        hr_plot = _rolling(hr, 10) if not hr_lbl else hr
        fig.add_trace(go.Scatter(x=t_min, y=hr_plot, name="Srčni utrip (bpm)" + (f" – {hr_lbl}" if hr_lbl else ""),
                                 mode="lines", line=dict(color="#ef4444", width=2.1, dash="dot" if hr_lbl else "solid"),
                                 hovertemplate="%{y:.0f} bpm"),
                      row=1, col=1, secondary_y=True)
    if "Hitrost (km/h)" in sel and len(spd) == n:
        fig.add_trace(go.Scatter(x=t_min, y=_rolling(spd * 3.6, 10), name="Hitrost (km/h)", mode="lines",
                                 line=dict(color="#06b6d4", width=1.8), hovertemplate="%{y:.1f} km/h"),
                      row=1, col=1, secondary_y=True)
    if "Kadenca (rpm)" in sel and len(cad) == n and np.any(cad > 0):
        fig.add_trace(go.Scatter(x=t_min, y=_rolling(cad, 10), name="Kadenca (rpm)", mode="lines",
                                 line=dict(color="#a855f7", width=1.6), hovertemplate="%{y:.0f} rpm"),
                      row=1, col=1, secondary_y=True)
    if has_p and ftp:
        fig.add_hline(y=ftp, line_dash="dash", line_color="#f59e0b", annotation_text=f"FTP {ftp} W",
                      annotation_position="top left", row=1, col=1)

    if want_bottom:
        if "Profil nadmorske višine (m)" in sel and len(ele_smooth) == n and np.any(ele_smooth != 0):
            fig.add_trace(go.Scatter(
                x=t_min, y=np.round(ele_smooth, 1), name="Profil nadmorske višine (m)", mode="lines",
                fill="tozeroy", line=dict(color="#64748b", width=1.8), fillcolor="rgba(100,116,139,0.20)",
                hovertemplate="%{y:.0f} m n.v."
            ), row=2, col=1, secondary_y=False)
        if "Kumulativni višinski metri (Vzpon v m)" in sel and len(cum_gain) == n and np.any(cum_gain > 0):
            fig.add_trace(go.Scatter(
                x=t_min, y=np.round(cum_gain, 1), name="Kumulativni vzpon (+m)", mode="lines",
                line=dict(color="#16a34a", width=2.6), hovertemplate="+%{y:.0f} m vzpona"
            ), row=2, col=1, secondary_y=False)
        if "Naklon klanca (%)" in sel and len(slope_pct) == n:
            fig.add_trace(go.Scatter(
                x=t_min, y=np.round(slope_pct, 1), name="Naklon klanca (%)", mode="lines",
                line=dict(color="#ea580c", width=1.8), hovertemplate="Naklon: %{y:+.1f} %"
            ), row=2, col=1, secondary_y=True)
        if "Temperatura (°C)" in sel and len(temp_arr) == n:
            fig.add_trace(go.Scatter(
                x=t_min, y=np.round(temp_arr, 1),
                name="Temperatura (°C)" + (f" – {temp_lbl}" if temp_lbl else ""), mode="lines",
                line=dict(color="#0284c7", width=2.0, dash="dot" if temp_lbl else "solid"),
                hovertemplate="Temp: %{y:.1f} °C"
            ), row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Višina / Vzpon (m)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Naklon (%) / Temp (°C)", row=2, col=1, secondary_y=True, showgrid=False)
        fig.update_xaxes(title_text="Čas vadbe (min)", row=2, col=1)
    else:
        fig.update_xaxes(title_text="Čas vadbe (min)", row=1, col=1)

    fig.update_yaxes(title_text="Moč (W)", rangemode="tozero", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="HR (bpm) / km/h / rpm", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_layout(
        height=720 if want_bottom else 480,
        hovermode="x unified",
        margin=dict(l=50, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
    )
    if hr_lbl:
        st.caption(f"ℹ️ Podrobni HR tok ni na voljo – prikazan je srčni utrip ({hr_lbl}).")
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True}, key=f"cyc_chart_{aid}")


def render_running_chart(aid, sf, act_obj, laps, split_obj=None):
    """Sekundni graf za tek/hojo: glajen tempo (obrnjena os), HR, kadenca, kumulativni vzpon, višinski profil, naklon %, temperatura."""
    spd = np.nan_to_num(np.array((sf or {}).get("speed") or [], dtype=float))
    n = len(spd)
    if n < 30 or float(np.max(spd)) <= 0:
        return False
    cad = np.nan_to_num(np.array((sf or {}).get("cadence") or [], dtype=float))
    ele = np.nan_to_num(np.array((sf or {}).get("elevation") or [], dtype=float))
    hr, hr_lbl = _hr_for_chart(sf, n, act_obj, laps)
    temp_arr, temp_lbl = _temp_for_chart(sf, n, act_obj, laps, split_obj=split_obj)
    off_gain = float(act_obj.get("elevationGain") or 0.0)
    ele_smooth, cum_gain, slope_pct, _, _, _ = streamlib.compute_elevation_metrics(
        ele if len(ele) == n else None, spd, official_gain=off_gain
    )

    st.markdown("#### 🏃‍♂️ Interaktivna analitika teka / hoje (Tempo, HR, Kumulativni vzpon, Naklon % in Temperatura)")
    c1, c2 = st.columns([3, 1])
    with c1:
        opts = [
            "Tempo (min/km)",
            "Srčni utrip (bpm)",
            "Kadenca (spm)",
            "Kumulativni višinski metri (Vzpon v m)",
            "Profil nadmorske višine (m)",
            "Naklon klanca (%)",
            "Temperatura (°C)",
        ]
        default = [
            "Tempo (min/km)",
            "Srčni utrip (bpm)",
            "Kumulativni višinski metri (Vzpon v m)",
            "Profil nadmorske višine (m)",
            "Naklon klanca (%)",
        ]
        if len(temp_arr) == n:
            default.append("Temperatura (°C)")
        sel = st.multiselect("Prikaz na grafu:", opts, default=default, key=f"run_traces_{aid}")
    with c2:
        win = 15 if st.radio("Glajenje tempa:", ["15 s", "30 s"], horizontal=True, key=f"run_smooth_{aid}") == "15 s" else 30

    t_min = np.arange(n) / 60.0
    from plotly.subplots import make_subplots
    want_bottom = any(k in sel for k in ["Kumulativni višinski metri (Vzpon v m)", "Profil nadmorske višine (m)", "Naklon klanca (%)", "Temperatura (°C)"])
    if want_bottom:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
            row_heights=[0.56, 0.44],
            specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
            subplot_titles=("Tempo (min/km), Srčni utrip in Kadenca", "Nadmorska višina (m), Kumulativni vzpon (+m), Naklon (%) in Temperatura (°C)")
        )
    else:
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    if "Tempo (min/km)" in sel:
        sm = _rolling(spd, win)
        pace = np.where(sm > 0.5, 1000.0 / np.maximum(sm, 0.01), np.nan)
        pace = np.clip(pace, 150, 1200)
        valid = pace[~np.isnan(pace)]
        lo, hi = (float(np.percentile(valid, 2)), float(np.percentile(valid, 98))) if len(valid) else (240.0, 420.0)
        ticks = list(range(int(lo // 30) * 30, int(hi // 30) * 30 + 61, 30))
        fig.add_trace(go.Scatter(x=t_min, y=pace, name=f"Tempo ({win}s povp.)", mode="lines",
                                 line=dict(color="#ea580c", width=2.5),
                                 customdata=[fmt_pace(x) for x in pace], hovertemplate="%{customdata} /km"),
                      row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Tempo (min/km) – hitreje zgoraj", autorange="reversed",
                         tickmode="array", tickvals=ticks, ticktext=[fmt_pace(x) for x in ticks],
                         row=1, col=1, secondary_y=False)
    if "Srčni utrip (bpm)" in sel and np.any(hr > 0):
        hr_plot = _rolling(hr, 10) if not hr_lbl else hr
        fig.add_trace(go.Scatter(x=t_min, y=hr_plot, name="Srčni utrip (bpm)" + (f" – {hr_lbl}" if hr_lbl else ""),
                                 mode="lines", line=dict(color="#ef4444", width=2.1, dash="dot" if hr_lbl else "solid"),
                                 hovertemplate="%{y:.0f} bpm"),
                      row=1, col=1, secondary_y=True)
    if "Kadenca (spm)" in sel and len(cad) == n and np.any(cad > 0):
        fig.add_trace(go.Scatter(x=t_min, y=_rolling(cad, 15), name="Kadenca (spm)", mode="lines",
                                 line=dict(color="#8b5cf6", width=1.6), hovertemplate="%{y:.0f} spm"),
                      row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="HR (bpm) / Kadenca (spm)", row=1, col=1, secondary_y=True, showgrid=False)

    if want_bottom:
        if "Profil nadmorske višine (m)" in sel and len(ele_smooth) == n and np.any(ele_smooth != 0):
            fig.add_trace(go.Scatter(
                x=t_min, y=np.round(ele_smooth, 1), name="Profil nadmorske višine (m)", mode="lines",
                fill="tozeroy", line=dict(color="#64748b", width=1.8), fillcolor="rgba(100,116,139,0.20)",
                hovertemplate="%{y:.0f} m n.v."
            ), row=2, col=1, secondary_y=False)
        if "Kumulativni višinski metri (Vzpon v m)" in sel and len(cum_gain) == n and np.any(cum_gain > 0):
            fig.add_trace(go.Scatter(
                x=t_min, y=np.round(cum_gain, 1), name="Kumulativni vzpon (+m)", mode="lines",
                line=dict(color="#16a34a", width=2.6), hovertemplate="+%{y:.0f} m vzpona"
            ), row=2, col=1, secondary_y=False)
        if "Naklon klanca (%)" in sel and len(slope_pct) == n:
            fig.add_trace(go.Scatter(
                x=t_min, y=np.round(slope_pct, 1), name="Naklon klanca (%)", mode="lines",
                line=dict(color="#ea580c", width=1.8), hovertemplate="Naklon: %{y:+.1f} %"
            ), row=2, col=1, secondary_y=True)
        if "Temperatura (°C)" in sel and len(temp_arr) == n:
            fig.add_trace(go.Scatter(
                x=t_min, y=np.round(temp_arr, 1),
                name="Temperatura (°C)" + (f" – {temp_lbl}" if temp_lbl else ""), mode="lines",
                line=dict(color="#0284c7", width=2.0, dash="dot" if temp_lbl else "solid"),
                hovertemplate="Temp: %{y:.1f} °C"
            ), row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Višina / Vzpon (m)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Naklon (%) / Temp (°C)", row=2, col=1, secondary_y=True, showgrid=False)
        fig.update_xaxes(title_text="Čas vadbe (min)", row=2, col=1)
    else:
        fig.update_xaxes(title_text="Čas vadbe (min)", row=1, col=1)

    fig.update_layout(
        height=700 if want_bottom else 480,
        hovermode="x unified",
        margin=dict(l=50, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
    )
    if hr_lbl:
        st.caption(f"ℹ️ Podrobni HR tok ni na voljo – prikazan je srčni utrip ({hr_lbl}).")
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True}, key=f"run_chart_{aid}")
    return True


# ---------- Izpis Markdown povzetkov z opcijskimi vprašanji ----------
def _incq_changed(k):
    v = bool(st.session_state.get(k, True))
    for kk in ("incq_morning", "incq_workout", "incq_weekly"):
        st.session_state[kk] = v
    cfg = load_user_config()
    cfg.setdefault("prefs", {})["include_questions"] = v
    save_user_config(cfg)


def show_prompt(base_text, questions_text, title, key):
    k = f"incq_{key}"
    if k not in st.session_state:
        st.session_state[k] = bool(load_user_config().get("prefs", {}).get("include_questions", True))
    inc = st.checkbox("☑️ Vključi vprašanja za AI trenerja na koncu izpisa", key=k, on_change=_incq_changed, args=(k,))
    final = (base_text.rstrip() + "\n\n" + questions_text.strip() + "\n") if inc else base_text.rstrip() + "\n"
    st.markdown(f"##### {title}")
    st.code(final, language="markdown")
    return final


def q_morning(planned):
    return f"""**VPRAŠANJA ZA TRENERJA:**
1. Kakšna je ocena moje pripravljenosti za današnji dan glede na kombinacijo nočnih metrik in subjektivnih občutkov?
2. Ali je planirani trening ({planned}) primeren glede na današnjo pripravljenost in nedavno obremenitev, ali predlagaš prilagoditev intenzivnosti/trajanja (npr. lahkoten Z2 ali dodaten počitek)?
3. Kateri vidik regeneracije danes najbolj izstopa?"""


Q_CYCLING = """**VPRAŠANJA ZA TRENERJA:**
1. Kako ocenjuješ kakovost treninga glede na ciljno območje in aerobno učinkovitost (EF)?
2. Ali je stopnja srčnega drifta (decoupling) v pričakovanih mejah glede na trajanje in pogoje?
3. Kakšen trening svetuješ naslednjič in kako naj ta vpliva na regeneracijo?"""

Q_RUN = """**VPRAŠANJA ZA TRENERJA:**
1. Kako ocenjuješ kakovost treninga (tempo, razporeditev napora po kilometrih, srčni utrip, kadenca)?
2. Ali je bil napor primeren za moje cilje in trenutno stanje regeneracije?
3. Kakšen trening svetuješ naslednjič in koliko regeneracije potrebujem?"""

Q_SWIM = """**VPRAŠANJA ZA TRENERJA:**
1. Kako ocenjuješ tehniko in učinkovitost (SWOLF, tempo na 100 m)?
2. Kakšen trening plavanja svetuješ naslednjič?"""

Q_OTHER = """**VPRAŠANJA ZA TRENERJA:**
1. Kako ocenjuješ kakovost in obremenitev te vadbe?
2. Kako naj ta trening vpliva na načrt naslednjih dni?"""


Q_WEEKLY = """**VPRAŠANJA ZA TRENERJA:**
1. Kako ocenjuješ razmerje med aerobno bazo (Z1/Z2), intenzivnostjo (Z3/Z4/Z5) in regeneracijo v tem obdobju v primerjavi s prejšnjim tednom/blokom?
2. Ali podatki o spanju, nočnem HRV in jutranjem RHR kažejo na dobro absorpcijo treninga ali na kopičenje utrujenosti?
3. Kakšen naj bo strateški poudarek v naslednjem tednu (volumen, intenzivnost, regeneracija) glede na moje cilje?"""


# ---------- Cilji in tekme ----------
def get_goals(cfg):
    g = cfg.get("goals")
    return g if isinstance(g, list) else []


def goal_days_left(g):
    try:
        return (datetime.strptime(g.get("date", ""), "%Y-%m-%d").date() - datetime.now().date()).days
    except Exception:
        return None


def goals_inline(cfg):
    parts = []
    for g in get_goals(cfg):
        dl = goal_days_left(g)
        if dl is not None and dl >= 0:
            parts.append(f"{g.get('name')} ({g.get('date')}, {g.get('metric') or 'brez metrike'}, prioriteta {str(g.get('priority', 'B'))[:1]})")
    return "; ".join(parts) if parts else "brez določenih ciljev"


def nearest_goal_with(cfg, field):
    best = None
    for g in get_goals(cfg):
        dl = goal_days_left(g)
        if dl is None or dl < 0 or not g.get(field):
            continue
        if best is None or dl < goal_days_left(best):
            best = g
    return best


# ---------- Hipnogram spanca, Nočni HR/HRV in Dnevni potek (Intraday) ----------
def _synthesize_sleep_intervals(sleep_obj, hr_pts, hrv_pts, start_ms, end_ms):
    """Če Garmin Personal API ne vrne surovega 'sleepLevels' seznama, zgradi fiziološko usklajen
    hipnogram faz spanca iz točnih trajanj faz (Globok, Lahek, REM, Budnost), števila prebujanj
    ter dejanskega poteka nočnega srčnega utripa in HRV v 5-minutnih oknih."""
    deep_s = float(sleep_obj.get("deepSleepSeconds") or 0)
    light_s = float(sleep_obj.get("lightSleepSeconds") or 0)
    rem_s = float(sleep_obj.get("remSleepSeconds") or 0)
    awake_s = float(sleep_obj.get("awakeSleepSeconds") or 0)
    tot_s = deep_s + light_s + rem_s + awake_s
    if tot_s <= 0 or end_ms <= start_ms:
        return []

    win_ms = 5 * 60 * 1000  # 5-minutna okna
    n_bins = max(12, int(round((end_ms - start_ms) / win_ms)))
    bin_dur_s = (end_ms - start_ms) / 1000.0 / n_bins

    n_deep = int(round(deep_s / max(bin_dur_s, 1.0)))
    n_rem = int(round(rem_s / max(bin_dur_s, 1.0)))
    n_awake = int(round(awake_s / max(bin_dur_s, 1.0)))
    if awake_s > 60 and n_awake == 0:
        n_awake = 1

    # Prilagodi vsoto
    while n_deep + n_rem + n_awake > n_bins:
        if n_rem > 1:
            n_rem -= 1
        elif n_deep > 1:
            n_deep -= 1
        else:
            n_awake = max(0, n_awake - 1)
    n_light = max(0, n_bins - n_deep - n_rem - n_awake)

    # Povprečni HR in HRV v vsakem 5-min oknu
    hr_bins = np.zeros(n_bins, dtype=float)
    hrv_bins = np.zeros(n_bins, dtype=float)
    all_hrs = [v for _, v in hr_pts if v and v > 0]
    base_hr = float(np.median(all_hrs)) if all_hrs else float(sleep_obj.get("avgHeartRate") or 52.0)
    all_hrvs = [v for _, v in hrv_pts if v and v > 0]
    base_hrv = float(np.median(all_hrvs)) if all_hrvs else 65.0

    for i in range(n_bins):
        t0 = start_ms + i * win_ms
        t1 = t0 + win_ms
        h_vals = [v for ts, v in hr_pts if t0 <= ts < t1 and v and v > 0]
        v_vals = [v for ts, v in hrv_pts if t0 <= ts < t1 and v and v > 0]
        hr_bins[i] = float(np.mean(h_vals)) if h_vals else base_hr
        hrv_bins[i] = float(np.mean(v_vals)) if v_vals else base_hrv

    # Normaliziraj z-score
    hr_std = max(1.5, float(np.std(hr_bins)))
    hrv_std = max(3.0, float(np.std(hrv_bins)))
    hr_z = (hr_bins - base_hr) / hr_std
    hrv_z = (hrv_bins - base_hrv) / hrv_std

    stages = ["LIGHT"] * n_bins
    used = set()

    # 1. Budnost (Awake): vrhovi srčnega utripa / prehodi
    if n_awake > 0:
        awake_scores = []
        for i in range(n_bins):
            pos = i / float(max(n_bins - 1, 1))
            edge_bonus = 0.4 if (i == 0 or i == n_bins - 1) else 0.0
            awake_scores.append((hr_z[i] - 0.5 * hrv_z[i] + edge_bonus, i))
        awake_scores.sort(reverse=True)
        for _, idx in awake_scores[:n_awake]:
            stages[idx] = "AWAKE"
            used.add(idx)

    # 2. Globoki spanec (Deep / N3): prevladuje v prvi polovici noči, nizek HR, stabilen HRV, ~90-minutni cikli
    if n_deep > 0:
        deep_scores = []
        for i in range(n_bins):
            if i in used:
                continue
            pos = i / float(max(n_bins - 1, 1))
            cycle_phase = math.sin(2.0 * math.pi * (i * bin_dur_s) / 5400.0)  # 90-min cikel
            early_weight = 1.6 * math.exp(-2.2 * pos)
            score = early_weight + 0.65 * max(0.0, cycle_phase) - 0.55 * hr_z[i] + 0.25 * hrv_z[i]
            deep_scores.append((score, i))
        deep_scores.sort(reverse=True)
        for _, idx in deep_scores[:n_deep]:
            stages[idx] = "DEEP"
            used.add(idx)

    # 3. REM spanec: prevladuje v drugi polovici noči, na koncu 90-minutnih ciklov
    if n_rem > 0:
        rem_scores = []
        for i in range(n_bins):
            if i in used:
                continue
            pos = i / float(max(n_bins - 1, 1))
            cycle_phase = -math.sin(2.0 * math.pi * (i * bin_dur_s) / 5400.0)
            late_weight = 1.4 * (pos ** 0.85)
            score = late_weight + 0.75 * max(0.0, cycle_phase) + 0.25 * hr_z[i] - 0.20 * hrv_z[i]
            rem_scores.append((score, i))
        rem_scores.sort(reverse=True)
        for _, idx in rem_scores[:n_rem]:
            stages[idx] = "REM"
            used.add(idx)

    # Združi sosednja enaka stanja v zvezne intervale
    intervals = []
    cur_st = stages[0]
    cur_t0 = start_ms
    for i in range(1, n_bins):
        t_boundary = int(round(start_ms + i * ((end_ms - start_ms) / float(n_bins))))
        if stages[i] != cur_st:
            intervals.append({"stage": cur_st, "start_ms": cur_t0, "end_ms": t_boundary})
            cur_st = stages[i]
            cur_t0 = t_boundary
    intervals.append({"stage": cur_st, "start_ms": cur_t0, "end_ms": int(end_ms)})
    return intervals


def _parse_iso_to_ms(iso_str, is_local=False):
    if not iso_str:
        return None
    try:
        s = str(iso_str).replace("Z", "")
        dt = datetime.fromisoformat(s)
        if is_local:
            dt = dt.replace(tzinfo=ZoneInfo("Europe/Ljubljana"))
        else:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _ms_to_lj_dt(ts_ms):
    try:
        return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).astimezone(ZoneInfo("Europe/Ljubljana")).replace(tzinfo=None)
    except Exception:
        return None


def render_morning_wellness_charts(sel_m_date, m_day, raw_data, user_cfg):
    """Izriše 3 vizualne Garmin grafe pod Jutranjim pregledom stanja:
    1. Hipnogram faz spanja + vodoravni povzetek faz (ure/min in %)
    2. Nočni srčni utrip in HRV skozi noč
    3. Potek čez dan (Srčni utrip, Stres 0-100 in Body Battery čez dan)"""
    st.markdown("---")
    st.markdown("### 📊 Vizualna analiza spanca, nočne regeneracije in poteka dneva")

    sleep_obj = (raw_data.get("sleep") or {}).get(sel_m_date) or {}
    daily_obj = (raw_data.get("dailies") or {}).get(sel_m_date) or {}
    hrv_obj = (raw_data.get("hrv") or {}).get(sel_m_date) or {}
    intraday_map = raw_data.get("intraday") or {}
    intra = intraday_map.get(sel_m_date) or {}

    # Če za izbrani datum še nimamo predpomnjenega intraday zapisa ali 15-min korakov, samodejno prenesi iz Garmina
    _needs_intra_fetch = (
        not intra
        or not intra.get("heartRateValues")
        or "hourlySteps" not in intra
        or "intraday_steps" not in intra
    )
    if _needs_intra_fetch:
        _intra_flag = f"intra_fetched_{get_current_user()}_{sel_m_date}"
        if not st.session_state.get(_intra_flag):
            st.session_state[_intra_flag] = True
            try:
                import pull as _pull_mod
                _cur_u = get_current_user()
                _custom_k = (user_cfg or {}).get("garmin_api_key") if isinstance(user_cfg, dict) else None
                _pull_mod.set_target_user(username=_cur_u, custom_api_key=_custom_k)
                _fetched_intra = _pull_mod.fetch_intraday_day(sel_m_date, username=_cur_u) or {}
                if _fetched_intra:
                    intra = _fetched_intra
                    intraday_map[sel_m_date] = intra
                    try:
                        _sl_path = os.path.join(_pull_mod.DATA_DIR, "sleep.json")
                        if os.path.exists(_sl_path):
                            with open(_sl_path, "r", encoding="utf-8") as _sf_sl:
                                _disk_sl = json.load(_sf_sl)
                            if isinstance(_disk_sl, dict) and isinstance(_disk_sl.get(sel_m_date), dict):
                                sleep_obj = _disk_sl[sel_m_date]
                                if isinstance(st.session_state.get("raw_data"), dict):
                                    st.session_state["raw_data"].setdefault("sleep", {})[sel_m_date] = sleep_obj
                                    st.session_state["raw_data"].setdefault("intraday", {})[sel_m_date] = intra
                        st.cache_data.clear()
                    except Exception:
                        pass
            except Exception:
                pass

    # Časovno okno spanca
    s_start_ms = sleep_obj.get("sleepStartTimestampGMT") or sleep_obj.get("autoSleepStartTimestampGMT")
    s_end_ms = sleep_obj.get("sleepEndTimestampGMT") or sleep_obj.get("autoSleepEndTimestampGMT")

    # Priprava HR in HRV točk
    hr_raw = intra.get("heartRateValues") or []
    hr_pts = []
    for item in hr_raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and item[0] and item[1] and item[1] > 0:
            hr_pts.append((int(item[0]), float(item[1])))
    hr_pts.sort(key=lambda x: x[0])

    hrv_raw = intra.get("hrvReadings") or []
    hrv_pts = []
    for item in hrv_raw:
        if isinstance(item, dict):
            val = item.get("hrvValue")
            ts = _parse_iso_to_ms(item.get("readingTimeGMT"), is_local=False) or _parse_iso_to_ms(item.get("readingTimeLocal"), is_local=True)
            if ts and val and val > 0:
                hrv_pts.append((ts, float(val)))
    hrv_pts.sort(key=lambda x: x[0])

    if (not s_start_ms or not s_end_ms) and hrv_pts:
        s_start_ms = hrv_pts[0][0]
        s_end_ms = hrv_pts[-1][0]

    # ---------------------------------------------------------
    # 1. HIPNOGRAM FAZ SPANJA & VODORAVNI POVZETEK FAZ
    # ---------------------------------------------------------
    st.markdown("#### 1. Hipnogram faz spanja (Kako so si sledile faze spanja skozi noč)")
    deep_sec = float(sleep_obj.get("deepSleepSeconds") or 0)
    light_sec = float(sleep_obj.get("lightSleepSeconds") or 0)
    rem_sec = float(sleep_obj.get("remSleepSeconds") or 0)
    awake_sec = float(sleep_obj.get("awakeSleepSeconds") or 0)
    total_stage_sec = deep_sec + light_sec + rem_sec + awake_sec

    if total_stage_sec > 0 and s_start_ms and s_end_ms and s_end_ms > s_start_ms:
        raw_levels = intra.get("sleepLevels") or sleep_obj.get("sleepLevels") or []
        intervals = []
        if raw_levels:
            level_map = {0: "DEEP", 1: "LIGHT", 2: "REM", 3: "AWAKE", "DEEP": "DEEP", "LIGHT": "LIGHT", "REM": "REM", "AWAKE": "AWAKE"}
            for lv in raw_levels:
                if isinstance(lv, dict):
                    t0 = _parse_iso_to_ms(lv.get("startGMT"), False) or lv.get("startTimestampGMT")
                    t1 = _parse_iso_to_ms(lv.get("endGMT"), False) or lv.get("endTimestampGMT")
                    st_code = level_map.get(lv.get("activityLevel"), "LIGHT")
                    if t0 and t1 and t1 > t0:
                        intervals.append({"stage": st_code, "start_ms": int(t0), "end_ms": int(t1)})

        if not intervals:
            sleep_hr_pts = [(ts, v) for ts, v in hr_pts if s_start_ms <= ts <= s_end_ms]
            intervals = _synthesize_sleep_intervals(sleep_obj, sleep_hr_pts, hrv_pts, int(s_start_ms), int(s_end_ms))

        stage_meta = {
            "DEEP":  {"label": "Globok spanec (Deep)", "y": 0, "color": "#1d4ed8"},
            "LIGHT": {"label": "Lahek spanec (Light)", "y": 1, "color": "#38bdf8"},
            "REM":   {"label": "REM faza",             "y": 2, "color": "#c084fc"},
            "AWAKE": {"label": "Budnost (Awake)",      "y": 3, "color": "#fb923c"},
        }

        fig_hyp = go.Figure()
        # Stepped povezovalna črta za hipnogram + barvni stolpci po fazah
        step_x, step_y = [], []
        for iv in intervals:
            dt0 = _ms_to_lj_dt(iv["start_ms"])
            dt1 = _ms_to_lj_dt(iv["end_ms"])
            y_v = stage_meta.get(iv["stage"], stage_meta["LIGHT"])["y"]
            if dt0 and dt1:
                step_x.extend([dt0, dt1])
                step_y.extend([y_v, y_v])

        # Dodaj barvne pravokotnike za vsako fazo (Gantt / Garmin slog)
        for st_key, meta in stage_meta.items():
            xs, ys, texts = [], [], []
            for iv in intervals:
                if iv["stage"] != st_key:
                    continue
                dt0 = _ms_to_lj_dt(iv["start_ms"])
                dt1 = _ms_to_lj_dt(iv["end_ms"])
                dur_m = max(1, int(round((iv["end_ms"] - iv["start_ms"]) / 60000.0)))
                if dt0 and dt1:
                    xs.extend([dt0, dt1, None])
                    ys.extend([meta["y"], meta["y"], None])
                    t_lbl = f"<b>{meta['label']}</b><br>{dt0.strftime('%H:%M')} – {dt1.strftime('%H:%M')} ({dur_m} min)"
                    texts.extend([t_lbl, t_lbl, ""])
            if xs:
                fig_hyp.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    name=meta["label"],
                    line=dict(color=meta["color"], width=18),
                    hoverinfo="text", text=texts,
                ))

        if step_x:
            fig_hyp.add_trace(go.Scatter(
                x=step_x, y=step_y, mode="lines",
                name="Prehodi faz",
                line=dict(color="rgba(148,163,184,0.45)", width=1.2, shape="hv"),
                hoverinfo="skip", showlegend=False,
            ))

        t_bed = _ms_to_lj_dt(s_start_ms)
        t_wake = _ms_to_lj_dt(s_end_ms)
        bed_wake_title = f"{t_bed.strftime('%H:%M')} – {t_wake.strftime('%H:%M')}" if (t_bed and t_wake) else ""

        fig_hyp.update_layout(
            title=f"Hipnogram faz spanja ({bed_wake_title})",
            height=300,
            margin=dict(l=20, r=20, t=42, b=25),
            xaxis=dict(title="Ura ponoči", tickformat="%H:%M", gridcolor="#334155"),
            yaxis=dict(
                title="Faza",
                tickvals=[0, 1, 2, 3],
                ticktext=["Globok (Deep)", "Lahek (Light)", "REM", "Budnost"],
                range=[-0.6, 3.6],
                gridcolor="#334155",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="closest",
        )
        st.plotly_chart(fig_hyp, use_container_width=True, key=f"hypnogram_{sel_m_date}")

        # Vodoravni povzetek (Stacked Bar) z urami/minutami in %
        def _fmt_hm(sec_v):
            m_tot = int(round(sec_v / 60.0))
            return f"{m_tot // 60}h {m_tot % 60:02d}m" if m_tot >= 60 else f"{m_tot} min"

        fig_bar_sl = go.Figure()
        for st_key, sec_v in [("DEEP", deep_sec), ("LIGHT", light_sec), ("REM", rem_sec), ("AWAKE", awake_sec)]:
            if sec_v <= 0:
                continue
            meta = stage_meta[st_key]
            pct = (sec_v / total_stage_sec) * 100.0
            hm_s = _fmt_hm(sec_v)
            fig_bar_sl.add_trace(go.Bar(
                y=["Razrez spanca"],
                x=[ round(sec_v / 60.0, 1) ],
                name=f"{meta['label']}: {hm_s} ({pct:.0f}%)",
                orientation="h",
                marker_color=meta["color"],
                text=[f"{hm_s} ({pct:.0f}%)"],
                textposition="inside",
                hovertemplate=f"<b>{meta['label']}</b>: {hm_s} ({pct:.1f}%)<extra></extra>",
            ))
        fig_bar_sl.update_layout(
            barmode="stack",
            height=115,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(title="Skupaj minut", showgrid=False),
            yaxis=dict(showticklabels=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_bar_sl, use_container_width=True, key=f"sleep_bar_{sel_m_date}")
    else:
        st.info("Podrobni podatki o fazah spanja za ta datum niso na voljo.")

    # ---------------------------------------------------------
    # 2. NOČNI SRČNI UTRIP IN HRV SKOZI NOČ
    # ---------------------------------------------------------
    st.markdown("#### 2. Nočni srčni utrip in HRV skozi noč")
    sleep_hr = [( _ms_to_lj_dt(ts), v ) for ts, v in hr_pts if s_start_ms and s_end_ms and (s_start_ms - 900000) <= ts <= (s_end_ms + 900000)]
    sleep_hr = [(dt, v) for dt, v in sleep_hr if dt is not None]
    sleep_hrv = [( _ms_to_lj_dt(ts), v ) for ts, v in hrv_pts]
    sleep_hrv = [(dt, v) for dt, v in sleep_hrv if dt is not None]

    if sleep_hr or sleep_hrv:
        from plotly.subplots import make_subplots
        fig_nhr = make_subplots(specs=[[{"secondary_y": True}]])
        if sleep_hr:
            fig_nhr.add_trace(
                go.Scatter(
                    x=[x[0] for x in sleep_hr],
                    y=[x[1] for x in sleep_hr],
                    mode="lines",
                    name="Nočni srčni utrip (bpm)",
                    line=dict(color="#ef4444", width=2.3),
                    hovertemplate="<b>%{y:.0f} bpm</b><extra>Nočni HR</extra>",
                ),
                secondary_y=False,
            )
            min_pt = min(sleep_hr, key=lambda x: x[1])
            fig_nhr.add_annotation(
                x=min_pt[0], y=min_pt[1],
                text=f"Najnižji: {min_pt[1]:.0f} bpm ({min_pt[0].strftime('%H:%M')})",
                showarrow=True, arrowhead=2, ax=0, ay=-28,
                bgcolor="rgba(15,23,42,0.85)", font=dict(color="#fca5a5", size=11),
            )
        if sleep_hrv:
            fig_nhr.add_trace(
                go.Scatter(
                    x=[x[0] for x in sleep_hrv],
                    y=[x[1] for x in sleep_hrv],
                    mode="lines+markers",
                    name="Nočni HRV (ms - 5 min okna)",
                    line=dict(color="#a855f7", width=2.2),
                    marker=dict(size=4, color="#c084fc"),
                    hovertemplate="<b>%{y:.0f} ms</b><extra>Nočni HRV</extra>",
                ),
                secondary_y=True,
            )
        avg_nhr_v = sleep_obj.get("avgHeartRate")
        if avg_nhr_v:
            fig_nhr.add_hline(y=avg_nhr_v, line_dash="dot", line_color="#f87171", opacity=0.65,
                              annotation_text=f"Povp. nočni HR: {avg_nhr_v:.0f} bpm", annotation_position="top left")

        fig_nhr.update_layout(
            title="Dinamika srčnega utripa (bpm) in variabilnosti utripa HRV (ms) med spanjem",
            height=340,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=42, b=25),
            xaxis=dict(title="Ura ponoči", tickformat="%H:%M", gridcolor="#334155"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_nhr.update_yaxes(title_text="Srčni utrip (bpm)", secondary_y=False, gridcolor="#334155")
        fig_nhr.update_yaxes(title_text="HRV (ms)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_nhr, use_container_width=True, key=f"night_hr_hrv_{sel_m_date}")
    else:
        # Čist povzetek iz dnevnih vrednosti, če za ta starejši dan ni 2-minutnega zapisa
        c_fb1, c_fb2, c_fb3 = st.columns(3)
        c_fb1.metric("Povprečni nočni HR", f"{m_day.get('avg_sleep_hr') or '--'} bpm")
        c_fb2.metric("Jutranji RHR", f"{m_day.get('rhr') or '--'} bpm")
        c_fb3.metric("Nočni HRV povprečje", f"{m_day.get('hrv_night') or '--'} ms", m_day.get("hrv_status") or "")
        st.caption("Podroben 2-minutni nočni tok za ta datum še ni bil prenesen. Kliknite '🔄 Osveži podatke iz Garmina' za zadnjih 7 dni.")

    # ---------------------------------------------------------
    # 3. POTEK ČEZ DAN (UTRIP, STRES IN BODY BATTERY ČEZ DAN)
    # ---------------------------------------------------------
    st.markdown("#### 3. Potek čez dan (Srčni utrip, Stres 0–100 in Body Battery čez dan)")
    try:
        day_start_dt = datetime.strptime(sel_m_date, "%Y-%m-%d")
        day_end_dt = day_start_dt + timedelta(hours=23, minutes=59, seconds=59)
    except Exception:
        day_start_dt, day_end_dt = None, None

    day_hr = []
    if day_start_dt:
        for ts, v in hr_pts:
            dt = _ms_to_lj_dt(ts)
            if dt and day_start_dt <= dt <= day_end_dt:
                day_hr.append((dt, v))

    # Body Battery točke (združimo iz dnevnih ključnih točk, spanca in HR dinamike)
    bb_raw = intra.get("bodyBatteryValuesArray") or []
    bb_pts = []
    for item in bb_raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and item[0] and item[1] is not None:
            dt = _ms_to_lj_dt(item[0])
            if dt and (not day_start_dt or (day_start_dt - timedelta(hours=4)) <= dt <= day_end_dt):
                bb_pts.append((dt, float(item[1])))
    bb_pts.sort(key=lambda x: x[0])

    if day_hr:
        from plotly.subplots import make_subplots
        rhr_v = float(m_day.get("rhr") or user_cfg.get("rhr_base", 46))
        avg_stress_day = float(m_day.get("stress") or 26.0)

        # Izračunaj časovno serijo stresa (0-100) iz odstopanja srčnega utripa od RHR in aktivnosti v tem dnevu
        # (med aktivnostjo je stres izključen/siv, v mirovanju modro <=25, oranžno/rdeče >25)
        act_windows = []
        for a in (raw_data.get("activities") or {}).values():
            st_l = a.get("startTimeLocal", "")
            if st_l.startswith(sel_m_date):
                try:
                    adt0 = datetime.fromisoformat(st_l)
                    adur = float(a.get("elapsedDuration") or a.get("duration") or 0)
                    act_windows.append((adt0, adt0 + timedelta(seconds=adur), a.get("activityName", "Trening")))
                except Exception:
                    pass

        stress_dt, stress_val, stress_colors = [], [], []
        # Zgladi HR za oceno avtonomnega stresa
        hr_arr_d = np.array([v for _, v in day_hr], dtype=float)
        hr_smooth_d = pd.Series(hr_arr_d).rolling(3, min_periods=1, center=True).mean().to_numpy()

        raw_st_list = []
        for (dt, _), h_s in zip(day_hr, hr_smooth_d):
            in_act = any(w0 <= dt <= w1 for w0, w1, _ in act_windows)
            in_sleep = False
            if s_start_ms and s_end_ms:
                dt_bed = _ms_to_lj_dt(s_start_ms)
                dt_wake = _ms_to_lj_dt(s_end_ms)
                if dt_bed and dt_wake and dt_bed <= dt <= dt_wake:
                    in_sleep = True
            if in_act:
                raw_st_list.append(None)
            elif in_sleep:
                st_v = max(4.0, min(32.0, 12.0 + (h_s - rhr_v) * 1.4))
                raw_st_list.append(st_v)
            else:
                st_v = max(8.0, min(95.0, 16.0 + (h_s - rhr_v) * 1.65))
                raw_st_list.append(st_v)

        # Umeri povprečje na Garmin uradni dnevni stres (m_day['stress'])
        valid_st = [v for v in raw_st_list if v is not None]
        if valid_st and avg_stress_day > 0:
            scale_st = avg_stress_day / max(1.0, float(np.mean(valid_st)))
        else:
            scale_st = 1.0

        for (dt, _), sv in zip(day_hr, raw_st_list):
            stress_dt.append(dt)
            if sv is None:
                stress_val.append(0)
                stress_colors.append("rgba(148,163,184,0.25)")  # Med treningom
            else:
                v_cal = float(np.clip(sv * scale_st, 5.0, 98.0))
                stress_val.append(round(v_cal))
                if v_cal <= 25:
                    stress_colors.append("#38bdf8")  # Počitek (Garmin modra)
                elif v_cal <= 50:
                    stress_colors.append("#fbbf24")  # Nizek stres
                elif v_cal <= 75:
                    stress_colors.append("#f97316")  # Srednji stres (oranžna)
                else:
                    stress_colors.append("#ef4444")  # Visok stres (rdeča)

        # Če ima Body Battery le ključne točke, jih zvezno interpoliraj na mrežo dneva
        bb_dt_grid, bb_val_grid = [], []
        if len(bb_pts) >= 2:
            xp = [x[0].timestamp() for x in bb_pts]
            yp = [x[1] for x in bb_pts]
            for dt, _ in day_hr:
                ts_q = dt.timestamp()
                if ts_q <= xp[0]:
                    bb_v = yp[0]
                elif ts_q >= xp[-1]:
                    bb_v = yp[-1]
                else:
                    bb_v = float(np.interp(ts_q, xp, yp))
                bb_dt_grid.append(dt)
                bb_val_grid.append(round(bb_v, 1))
        elif m_day.get("bb_max") and m_day.get("bb_min"):
            # Sintetična krivulja polnjenja ponoči in praznjenja čez dan iz bb_min / bb_max
            bb_max_v = float(m_day["bb_max"])
            bb_min_v = float(m_day["bb_min"])
            wake_dt = _ms_to_lj_dt(s_end_ms) if s_end_ms else day_start_dt.replace(hour=7, minute=0)
            for dt, _ in day_hr:
                if wake_dt and dt <= wake_dt:
                    frac = max(0.0, min(1.0, (dt - day_start_dt).total_seconds() / max(3600.0, (wake_dt - day_start_dt).total_seconds())))
                    bb_v = bb_min_v + (bb_max_v - bb_min_v) * (0.4 + 0.6 * frac)
                else:
                    frac = max(0.0, min(1.0, (dt - (wake_dt or day_start_dt)).total_seconds() / (16.0 * 3600.0)))
                    bb_v = bb_max_v - (bb_max_v - bb_min_v) * frac
                bb_dt_grid.append(dt)
                bb_val_grid.append(round(bb_v, 1))

        fig_intra = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
            subplot_titles=(
                "Srčni utrip čez dan (Intraday HR - 2 min)",
                "Stres čez dan (0–25 modra = počitek, 26–100 oranžna/rdeča = stres)",
                "Body Battery čez dan (polnjenje ponoči in praznjenje čez dan)",
            ),
        )

        # Panel 1: Intraday HR
        fig_intra.add_trace(
            go.Scatter(
                x=[x[0] for x in day_hr],
                y=[x[1] for x in day_hr],
                mode="lines",
                name="Srčni utrip (bpm)",
                line=dict(color="#ef4444", width=2.0),
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.12)",
                hovertemplate="<b>%{y:.0f} bpm</b><extra>HR čez dan</extra>",
            ),
            row=1, col=1,
        )
        if m_day.get("rhr"):
            fig_intra.add_hline(y=m_day["rhr"], line_dash="dot", line_color="#38bdf8",
                                annotation_text=f"RHR ({m_day['rhr']} bpm)", row=1, col=1)

        # Panel 2: Intraday Stress (barvni stolpci kot v Garmin Connect)
        fig_intra.add_trace(
            go.Bar(
                x=stress_dt,
                y=stress_val,
                name="Stres (0–100)",
                marker_color=stress_colors,
                hovertemplate="<b>Stres: %{y}</b><extra></extra>",
            ),
            row=2, col=1,
        )
        fig_intra.add_hline(y=25, line_dash="dot", line_color="#38bdf8", opacity=0.6, row=2, col=1)

        # Panel 3: Body Battery
        if bb_dt_grid:
            fig_intra.add_trace(
                go.Scatter(
                    x=bb_dt_grid,
                    y=bb_val_grid,
                    mode="lines",
                    name="Body Battery (0–100)",
                    line=dict(color="#0ea5e9", width=2.8),
                    fill="tozeroy",
                    fillcolor="rgba(14,165,233,0.18)",
                    hovertemplate="<b>Body Battery: %{y:.0f}</b><extra></extra>",
                ),
                row=3, col=1,
            )

        fig_intra.update_layout(
            height=640,
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=25),
        )
        fig_intra.update_yaxes(title_text="bpm", row=1, col=1, gridcolor="#334155")
        fig_intra.update_yaxes(title_text="0–100", range=[0, 100], row=2, col=1, gridcolor="#334155")
        fig_intra.update_yaxes(title_text="BB (0–100)", range=[0, 105], row=3, col=1, gridcolor="#334155")
        fig_intra.update_xaxes(title_text="Ura v dnevu", tickformat="%H:%M", row=3, col=1, gridcolor="#334155")
        st.plotly_chart(fig_intra, use_container_width=True, key=f"intraday_3p_{sel_m_date}")
    else:
        # Čist povzetek dneva (Min / Avg / Max), če za starejši datum ni podrobnega intraday toka
        hr_d_obj = daily_obj.get("heartRate") if isinstance(daily_obj.get("heartRate"), dict) else {}
        st_d_obj = daily_obj.get("stress") if isinstance(daily_obj.get("stress"), dict) else {}
        bb_d_obj = daily_obj.get("bodyBattery") if isinstance(daily_obj.get("bodyBattery"), dict) else {}
        c_d1, c_d2, c_d3 = st.columns(3)
        c_d1.metric(
            "Srčni utrip v dnevu (RHR / Max)",
            f"{m_day.get('rhr') or '--'} / {hr_d_obj.get('maxValue') or '--'} bpm",
            f"Povp. dnevni HR: {m_day.get('daily_avg_hr') or '--'} bpm",
            delta_color="off",
        )
        c_d2.metric(
            "Dnevni stres (Povp / Max)",
            f"{m_day.get('stress') or '--'} / {st_d_obj.get('maxLevel') or '--'}",
            "0–25 počitek | 26–50 nizek | 51–75 srednji",
            delta_color="off",
        )
        c_d3.metric(
            "Body Battery (Min → Max)",
            f"{m_day.get('bb_min') or '--'} → {m_day.get('bb_max') or '--'}",
            f"+{bb_d_obj.get('chargedValue') or '--'} polnjenje / -{bb_d_obj.get('drainedValue') or '--'} poraba",
            delta_color="off",
        )


# ---------- Rekordi na razdaljah, krivulja tempa, VDOT ----------
DIST_LABELS = {
    "100": "100 m",
    "200": "200 m",
    "400": "400 m",
    "600": "600 m",
    "800": "800 m",
    "1000": "1 km",
    "1500": "1,5 km",
    "1609": "1 milja (1,61 km)",
    "2000": "2 km",
    "3000": "3 km",
    "4000": "4 km",
    "5000": "5 km",
    "6000": "6 km",
    "8000": "8 km",
    "10000": "10 km",
    "12000": "12 km",
    "15000": "15 km",
    "18000": "18 km",
    "20000": "20 km",
    "21097": "Polmaraton (21,1 km)",
    "25000": "25 km",
    "30000": "30 km",
    "40000": "40 km",
    "50000": "50 km",
    "100000": "100 km",
}

RUN_DISTS = [
    100, 200, 400, 600, 800, 1000, 1500, 1609, 2000, 3000, 4000, 5000,
    6000, 8000, 10000, 12000, 15000, 18000, 20000, 21097, 25000, 30000
]
CYC_DISTS = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 100000]
PACE_CURVE_DURS = [
    10, 20, 30, 45, 60, 90, 120, 180, 240, 300, 360, 480, 600,
    720, 900, 1200, 1500, 1800, 2400, 2700, 3600, 4500, 5400, 7200, 10800
]


def gather_best_efforts(raw, groups=None, start=None, end=None):
    cache_k = (
        f"_cached_gbe_{id(raw)}_{len(raw.get('activities') or {})}_{len(raw.get('streams') or {})}_"
        f"{tuple(groups) if groups else None}_{start}_{end}"
    )
    if cache_k in st.session_state:
        return st.session_state[cache_k]
    acts = raw.get("activities", {})
    best_dist, pace_env, asc1h = {}, {}, None
    for aid, s in raw.get("streams", {}).items():
        summ = dict.get(s, "summary") if isinstance(s, LazyStream) else s.get("summary")
        summ = summ or {}
        if not summ.get("bestDist") and not summ.get("bestPace") and not summ.get("maxAscent1h"):
            continue
        a = acts.get(aid, {})
        s_stl = dict.get(s, "startTimeLocal") if isinstance(s, LazyStream) else s.get("startTimeLocal")
        d_str = (s_stl or a.get("startTimeLocal") or "")[:10]
        if groups and sport_group(a.get("activityType", {}).get("typeKey")) not in groups:
            continue
        if start or end:
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d").date()
            except Exception:
                continue
            if (start and d < start) or (end and d > end):
                continue
        s_nm = dict.get(s, "activityName") if isinstance(s, LazyStream) else s.get("activityName")
        name = s_nm or a.get("activityName") or ""
        for D, e in (summ.get("bestDist") or {}).items():
            if D not in best_dist or e["sec"] < best_dist[D]["sec"]:
                best_dist[D] = dict(e, date=d_str, name=name, aid=aid)
        for dur, p in (summ.get("bestPace") or {}).items():
            if dur not in pace_env or p < pace_env[dur]["pace"]:
                pace_env[dur] = {"pace": p, "date": d_str, "name": name, "aid": aid}
        m = summ.get("maxAscent1h")
        s_n = dict.get(s, "n") if isinstance(s, LazyStream) else s.get("n")
        dur_act = float(a.get("movingDuration") or a.get("duration") or s_n or 0)
        gain_act = float(a.get("elevationGain") or 0)
        # Varnostna omejitev: 1h vzpon ne more presegati uradnega vzpona celotne vožnje ali 2200 m/h
        if m and dur_act >= 2700:
            if gain_act > 0 and m > gain_act * 1.05:
                m = gain_act
            if 50 <= m <= 2200 and (asc1h is None or m > asc1h["m"]):
                asc1h = {
                    "m": round(float(m), 1),
                    "date": d_str,
                    "name": name,
                    "aid": aid,
                    "vam30": summ.get("maxVam30m"),
                    "vam20": summ.get("maxVam20m"),
                }
    res_tuple = (best_dist, pace_env, asc1h)
    st.session_state[cache_k] = res_tuple
    return res_tuple


def _fastest_map(a):
    """Garmin 'fastestSplit_<m>' polja (sekunde) -> {str(razdalja): sekunde}."""
    out = {}
    for k, v in a.items():
        if isinstance(k, str) and k.startswith("fastestSplit_") and v:
            try:
                D = int(k.split("_")[1])
                v = float(v)
            except Exception:
                continue
            if D in (21098, 21097, 21100):
                D = 21097
            if D in (1609, 1610):
                D = 1609
            if v > 0:
                out[str(D)] = v
    return out


def gather_best_efforts_full(raw, groups=None, start=None, end=None):
    """Najboljši napori iz 1 Hz tokov + gost izračun vmesnih razdalj (100 m – 30 km) in trajanj (10 s – 180 min)
    iz Garmin povzetkov (fastestSplit_*), krogov, maxSpeed in povprečnega tempa.
    Vrne (best_dist, pace_env, asc1h); vsak zapis ima polje 'src'."""
    cache_k = (
        f"_cached_gbef_{id(raw)}_{len(raw.get('activities') or {})}_{len(raw.get('streams') or {})}_{len(raw.get('splits') or {})}_"
        f"{tuple(groups) if groups else None}_{start}_{end}"
    )
    if cache_k in st.session_state:
        return st.session_state[cache_k]
    bd_orig, pe_orig, asc_orig = gather_best_efforts(raw, groups, start, end)
    bd = {k: dict(v) for k, v in bd_orig.items()}
    pe = {k: dict(v) for k, v in pe_orig.items()}
    asc = dict(asc_orig) if asc_orig else None
    for e in bd.values():
        e.setdefault("src", "tok 1 Hz")
    for e in pe.values():
        e.setdefault("src", "tok 1 Hz")
    if asc:
        asc.setdefault("src", "tok 1 Hz")
    acts = raw.get("activities", {}) or {}
    splits = raw.get("splits", {}) or {}
    streams = raw.get("streams", {}) or {}

    for aid, a in acts.items():
        try:
            grp = sport_group((a.get("activityType") or {}).get("typeKey"))
            if groups and grp not in groups:
                continue
            d_str = (a.get("startTimeLocal") or "")[:10]
            if start or end:
                try:
                    d = datetime.strptime(d_str, "%Y-%m-%d").date()
                except Exception:
                    continue
                if (start and d < start) or (end and d > end):
                    continue
            if grp not in ("Tek", "Hoja / Pohodništvo", "Kolesarjenje"):
                continue
            dist = float(a.get("distance") or 0)
            dur = float(a.get("movingDuration") or a.get("duration") or 0)
            if dist <= 50 or dur <= 10:
                continue
            hr = a.get("averageHR")
            name = a.get("activityName") or ""
            gain = float(a.get("elevationGain") or 0)
            max_spd = float(a.get("maxSpeed") or 0)
            avg_spd = dist / dur if dur > 0 else 0.0

            exact = _fastest_map(a)
            laps = []
            for lp in ((splits.get(str(aid)) or {}).get("laps") or []):
                ld, lt = float(lp.get("distance") or 0), float(lp.get("movingDuration") or lp.get("duration") or 0)
                if ld > 20 and lt > 5:
                    laps.append((ld, lt, lp.get("avgHR") or lp.get("averageHR")))

            targets = CYC_DISTS if grp == "Kolesarjenje" else RUN_DISTS

            def offer(D, sec, src, hr_):
                key = str(D)
                if not sec or sec <= 0:
                    return
                pace_chk = sec / (float(D) / 1000.0)
                # Varnostni filter proti nerealnim vrednostim
                if grp == "Tek" and (pace_chk < 105 or pace_chk > 1200):
                    return
                if grp == "Hoja / Pohodništvo" and (pace_chk < 180 or pace_chk > 2400):
                    return
                if grp == "Kolesarjenje" and (pace_chk < 30 or pace_chk > 600):
                    return
                if key not in bd or sec < bd[key]["sec"]:
                    bd[key] = {
                        "sec": round(float(sec), 1),
                        "date": d_str,
                        "name": name,
                        "aid": str(aid),
                        "asc": gain * min(1.0, float(D) / max(dist, 1.0)),
                        "hr": hr_,
                        "src": src,
                    }

            # Sidrne točke razdalja -> čas (v sekundah) za to aktivnost
            anchors = {}
            for Ds, sec in exact.items():
                anchors[float(Ds)] = (float(sec), "Garmin najhitrejši odsek")
            for ld, lt, _ in laps:
                if ld not in anchors or lt < anchors[ld][0]:
                    anchors[float(ld)] = (float(lt), "krog (Garmin lap)")
            if dist > 0 and dur > 0:
                if dist not in anchors or dur < anchors[dist][0]:
                    anchors[float(dist)] = (float(dur), "povpr. tempo")

            # Dodaj kratko sidro (100 m) iz maxSpeed ali najhitrejšega kroga/1km
            if grp != "Kolesarjenje" and 100.0 not in anchors and dist >= 100:
                if 1000.0 in anchors:
                    # Riegel eksponent ~1.05 za kratke razdalje
                    t100 = anchors[1000.0][0] * ((100.0 / 1000.0) ** 1.05)
                    anchors[100.0] = (t100, "ocena (iz 1 km)")
                elif max_spd > avg_spd and max_spd < 11.0:
                    eff_spd = 0.72 * max_spd + 0.28 * avg_spd
                    if eff_spd > 0.5:
                        anchors[100.0] = (100.0 / eff_spd, "ocena (vrh hitrosti)")

            # Zagotovi monotonost sider: čas mora naraščati z razdaljo, tempo pa ne sme padati hitreje kot fizikalno smiselno
            sorted_d = sorted(anchors.keys())
            clean_d, clean_t, clean_src = [], [], []
            for d_k in sorted_d:
                t_k, s_k = anchors[d_k]
                if t_k <= 0:
                    continue
                if clean_d and t_k <= clean_t[-1]:
                    t_k = clean_t[-1] * (d_k / clean_d[-1]) ** 1.03
                clean_d.append(d_k)
                clean_t.append(t_k)
                clean_src.append(s_k)

            # Ponudi vse ciljne razdalje do dolžine te aktivnosti
            has_1hz_bd = bool(((streams.get(str(aid)) or {}).get("summary") or {}).get("bestDist"))
            for D in targets:
                if D > dist * 1.015:
                    continue
                if str(D) in exact:
                    offer(D, exact[str(D)], "Garmin najhitrejši odsek", hr)
                    continue
                if has_1hz_bd:
                    continue
                # Če imamo krog ki pokriva razdaljo D
                cand_laps = [(lt * (D / ld), lhr or hr) for ld, lt, lhr in laps if ld >= D * 0.99]
                if len(clean_d) >= 2 and clean_d[0] <= D <= clean_d[-1]:
                    log_t = float(np.interp(math.log(D), [math.log(x) for x in clean_d], [math.log(y) for y in clean_t]))
                    est_sec = math.exp(log_t)
                    if cand_laps:
                        best_lp_sec, best_lp_hr = min(cand_laps, key=lambda x: x[0])
                        if best_lp_sec < est_sec:
                            offer(D, best_lp_sec, "krog (ocena)", best_lp_hr)
                        else:
                            offer(D, est_sec, "interpolacija odsekov", hr)
                    else:
                        offer(D, est_sec, "interpolacija odsekov", hr)
                elif cand_laps:
                    best_lp_sec, best_lp_hr = min(cand_laps, key=lambda x: x[0])
                    offer(D, best_lp_sec, "krog (ocena)", best_lp_hr)
                elif dist >= D * 0.99 and dur > 0:
                    # Riegel skaliranje iz celotne aktivnosti za krajše razdalje
                    exp_r = 1.04 if grp == "Tek" else 1.02
                    est_sec = dur * ((float(D) / dist) ** exp_r)
                    offer(D, est_sec, "povpr. tempo (ocena)", hr)

            # Krivulja tempa po času (10 s – 180 min) za tek in hojo
            has_1hz_pe = bool(((streams.get(str(aid)) or {}).get("summary") or {}).get("bestPace"))
            if grp != "Kolesarjenje" and not has_1hz_pe:
                pts = {}
                for d_k, t_k in zip(clean_d, clean_t):
                    if d_k > 0 and t_k > 0:
                        pts[t_k] = t_k / (d_k / 1000.0)
                if 10.0 not in pts and dur >= 10:
                    if pts:
                        first_t = min(pts.keys())
                        pts[10.0] = pts[first_t] * ((10.0 / max(first_t, 10.0)) ** 0.04)
                pts = {k: v for k, v in pts.items() if 105 < v < 2200}
                if pts:
                    ks = sorted(pts.keys())
                    for d_dur in PACE_CURVE_DURS:
                        if d_dur > dur * 1.02:
                            continue
                        if d_dur <= ks[0]:
                            v = pts[ks[0]] * ((d_dur / max(ks[0], 1.0)) ** 0.03)
                        elif d_dur >= ks[-1]:
                            v = pts[ks[-1]]
                        else:
                            v = float(np.exp(np.interp(math.log(d_dur), [math.log(k) for k in ks], [math.log(pts[k]) for k in ks])))
                        key = str(d_dur)
                        if key not in pe or v < pe[key]["pace"]:
                            pe[key] = {
                                "pace": round(float(v), 2),
                                "date": d_str,
                                "name": name,
                                "aid": str(aid),
                                "src": "ocena (odseki/krogi)",
                            }
            elif grp == "Kolesarjenje":
                # Rezervni izračun 1h vzpona za kolesarske vožnje brez 1 Hz toka (če vožnja traja vsaj 45 min)
                has_1hz_asc = bool(((streams.get(str(aid)) or {}).get("summary") or {}).get("maxAscent1h"))
                if not has_1hz_asc and dur >= 2700 and gain >= 100:
                    # Za vožnje >= 60 min je v najhujši uri zbranih vsaj povprečni višinski metri na uro (ali v celotni vožnji, če je okoli 1h)
                    if dur <= 3600:
                        m_1h = gain
                    else:
                        # Preveri še kroge (laps), če kateri od krogov/kombinacije krogov pokaže večji VAM
                        m_1h = gain / (dur / 3600.0)
                    if 50 <= m_1h <= 2200 and (asc is None or m_1h > asc["m"]):
                        asc = {
                            "m": round(float(m_1h), 1),
                            "date": d_str,
                            "name": name,
                            "aid": str(aid),
                            "src": "povprečje vožnje (ocena)",
                        }
        except Exception:
            continue
    res_full = (bd, pe, asc)
    st.session_state[cache_k] = res_full
    return res_full


def cycling_wkg_records(raw, start=None, end=None, weight=64.0, durs=(5, 15, 30, 60, 180, 300, 600, 900, 1200, 1800, 2700, 3600)):
    """Najboljše moči (W in W/kg) iz MMP datotek ali Garmin maxAvgPower_* polj."""
    best = {}
    for aid, a in (raw.get("activities", {}) or {}).items():
        if sport_group((a.get("activityType") or {}).get("typeKey")) != "Kolesarjenje":
            continue
        d_str = (a.get("startTimeLocal") or "")[:10]
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d").date()
        except Exception:
            continue
        if (start and d < start) or (end and d > end):
            continue
        mm = ((raw.get("mmp", {}) or {}).get(str(aid)) or {}).get("mmp") or {}
        for dd in durs:
            w = mm.get(str(dd)) or a.get(f"maxAvgPower_{dd}") or 0
            if w and w > best.get(dd, {}).get("w", 0):
                best[dd] = {"w": float(w), "date": d_str, "name": a.get("activityName", ""), "wkg": float(w) / weight}
    return best


def _mmss_to_sec(s):
    try:
        parts = [int(x) for x in str(s).strip().split(":")]
        sec = 0
        for x in parts:
            sec = sec * 60 + x
        return sec
    except Exception:
        return 0


def vdot_from(dist_m, t_sec):
    """Jack Daniels & Jimmy Gilbert VDOT formula."""
    t = float(t_sec) / 60.0
    if t <= 0 or dist_m <= 0:
        return 0.0
    v = float(dist_m) / t  # m/min
    vo2 = -4.60 + 0.182258 * v + 0.000104 * (v * v)
    frac = 0.8 + 0.1894393 * math.exp(-0.012778 * t) + 0.2989558 * math.exp(-0.1932605 * t)
    if frac <= 0:
        return 0.0
    return vo2 / frac


def velocity_at_vo2(vo2_target):
    """Inverzna Daniels kvadratna enačba: iz ciljnega VO2 (ml/kg/min) izračuna hitrost v (m/min)."""
    if vo2_target <= 0:
        return 0.0
    # 0.000104 * v^2 + 0.182258 * v - (vo2_target + 4.60) = 0
    a_c = 0.000104
    b_c = 0.182258
    c_c = -(float(vo2_target) + 4.60)
    disc = b_c * b_c - 4.0 * a_c * c_c
    if disc <= 0:
        return 0.0
    return (-b_c + math.sqrt(disc)) / (2.0 * a_c)


def daniels_pace_sec_km(vdot, intensity_pct):
    """Vrne priporočen trenažni tempo (sek/km) pri danem % VDOT po Danielsovi formuli."""
    v_m_min = velocity_at_vo2(float(vdot) * float(intensity_pct))
    if v_m_min <= 0:
        return 0.0
    return 1000.0 / (v_m_min / 60.0)


def daniels_training_paces(vdot):
    """Vrne seznam Danielsovih trenažnih con tempa za dani VDOT."""
    if not vdot or vdot <= 15:
        return []
    e_fast = daniels_pace_sec_km(vdot, 0.72)
    e_slow = daniels_pace_sec_km(vdot, 0.62)
    m_pace = predict_time(vdot, 42195) / 42.195
    t_pace = daniels_pace_sec_km(vdot, 0.88)
    i_pace = daniels_pace_sec_km(vdot, 0.98)
    r_pace = max(90.0, i_pace - 15.0)
    return [
        {
            "Cona": "🟢 Easy (E) – Lahek aerobni / regeneracijski tek",
            "% VDOT": "62–72 %",
            "Priporočen tempo": f"{fmt_pace(e_fast)} – {fmt_pace(e_slow)} /km",
            "Namen": "Aerobna baza, kapilarizacija, dolgi teki in ogrevanje",
        },
        {
            "Cona": "🔵 Marathon (M) – Maratonski tempo",
            "% VDOT": "~80–83 %",
            "Priporočen tempo": f"{fmt_pace(m_pace)} /km",
            "Namen": "Specifična vzdržljivost in ekonomičnost pri tekmovalnem tempu",
        },
        {
            "Cona": "🟡 Threshold (T) – Laktatni prag (Tempo tek)",
            "% VDOT": "88 %",
            "Priporočen tempo": f"{fmt_pace(t_pace)} /km",
            "Namen": "Dvig laktatnega praga (npr. 20 min zvezno ali 3–4 × 2 km)",
        },
        {
            "Cona": "🟠 Interval (I) – VO2max intervali",
            "% VDOT": "98–100 %",
            "Priporočen tempo": f"{fmt_pace(i_pace)} /km",
            "Namen": "Maksimalna poraba kisika (npr. 5 × 1000 m ali 4 × 4 min)",
        },
        {
            "Cona": "🔴 Repetition (R) – Hitrost in ekonomija teka",
            "% VDOT": "105–110 %",
            "Priporočen tempo": f"{fmt_pace(r_pace)} /km ({int(round(r_pace * 0.4))} s / 400 m)",
            "Namen": "Nevromišična moč, biomehanika in hitrost (200 m – 400 m ponovitve)",
        },
    ]


def predict_time(vdot, dist_m):
    lo, hi = 60.0, 6 * 3600.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if vdot_from(dist_m, mid) > vdot:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def gap_pace(pace_sec, asc, dist_m):
    grade = (asc or 0) / dist_m * 100.0 if dist_m else 0.0
    return pace_sec / (1.0 + 0.035 * max(0.0, grade))


def duration_value(raw, aid, m_obj, dur, kind="power"):
    """Vrednost za poljubno trajanje: iz 1 Hz toka (točno) ali log-log interpolacija iz shranjenih MMP/tempo točk."""
    sf = raw.get("streams", {}).get(aid) or {}
    if kind == "power":
        p = np.nan_to_num(np.array(sf.get("power") or [], dtype=float))
        if len(p) >= dur and np.any(p > 0):
            cs = np.concatenate([[0.0], np.cumsum(p)])
            return float(np.max(cs[dur:] - cs[:-dur]) / dur), False
        pts = {int(k): v for k, v in (m_obj.get("mmp") or {}).items() if v}
    else:
        spd = np.nan_to_num(np.array(sf.get("speed") or [], dtype=float))
        if len(spd) >= dur and np.any(spd > 0.3):
            cs = np.concatenate([[0.0], np.cumsum(spd)])
            m = float(np.max(cs[dur:] - cs[:-dur]) / dur)
            return (1000.0 / m if m > 0.3 else None), False
        pts = {int(k): v for k, v in ((sf.get("summary") or {}).get("bestPace") or {}).items() if v}
    if len(pts) < 2:
        return None, True
    ks = sorted(pts)
    if dur < ks[0] or dur > ks[-1]:
        return None, True
    return float(np.exp(np.interp(math.log(dur), [math.log(k) for k in ks], [math.log(pts[k]) for k in ks]))), True


# ---------- Športni profili, tekaške cone, rTSS ----------
SPORT_PROFILES = [
    "🚴♂️ Kolesar (Cesta / Trenažer / MTB)",
    "🏃♂️ Tekač (Cesta / Trail / Atletska steza)",
    "🏅 Oboje / Večšportnik (Kolo + Tek / Triatlon)",
]
VDOT_REF_DISTS = {"Brez": 0, "5 km": 5000, "10 km": 10000, "Polmaraton": 21097, "Maraton": 42195}


def parse_hms(s):
    sec = _mmss_to_sec(s)
    return sec if sec and sec > 0 else None


def fmt_hms(sec):
    sec = int(round(sec))
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def estimate_vdot(cfg, bd=None):
    """VDOT iz referenčnega časa, sicer iz threshold tempa (88 % VO2max), sicer iz najboljšega 1–21 km napora."""
    try:
        dm = VDOT_REF_DISTS.get(cfg.get("vdot_ref_dist", "Brez"), 0)
        t = parse_hms(cfg.get("vdot_ref_time", ""))
        if dm and t:
            return vdot_from(dm, t)
        tp = parse_hms(cfg.get("threshold_pace", ""))
        if tp:
            v = 1000.0 / (tp / 60.0)
            return (-4.60 + 0.182258 * v + 0.000104 * v * v) / 0.88
        if bd:
            c = [vdot_from(int(D), e["sec"]) for D, e in bd.items() if 1000 <= int(D) <= 21097 and e.get("sec", 0) >= 150]
            if c:
                return max(c)
    except Exception:
        pass
    return None


def effective_threshold_pace(cfg, bd=None):
    """Threshold tempo (sek/km): ročni vnos, sicer iz VDOT (Daniels 88 %)."""
    tp = parse_hms(cfg.get("threshold_pace", ""))
    if tp:
        return float(tp)
    dm = VDOT_REF_DISTS.get(cfg.get("vdot_ref_dist", "Brez"), 0)
    t = parse_hms(cfg.get("vdot_ref_time", ""))
    vd = vdot_from(dm, t) if (dm and t) else None
    if vd and vd > 15:
        return daniels_pace_sec_km(vd, 0.88)
    return None


def running_pace_zones(t_pace):
    r = [(1.18, None, "Z1 (Regeneracijski tek)", "> 118 % T-Pace"),
         (1.08, 1.18, "Z2 (Aerobni / Pogovorni tek)", "108–118 % T-Pace"),
         (1.02, 1.08, "Z3 (Tempo / Maratonski ritem)", "102–108 % T-Pace"),
         (0.97, 1.02, "Z4 (Laktatni prag / Threshold)", "97–102 % T-Pace"),
         (None, 0.97, "Z5 (VO2max intervali / Hitrost)", "< 97 % T-Pace")]
    rows = []
    for lo, hi, name, pct in r:
        if hi is None:
            rng = f"počasneje od {fmt_pace(t_pace * lo)} /km"
        elif lo is None:
            rng = f"hitreje od {fmt_pace(t_pace * hi)} /km"
        else:
            rng = f"{fmt_pace(t_pace * hi)} – {fmt_pace(t_pace * lo)} /km"
        rows.append({"Cona": name, "% T-Pace": pct, "Tempo": rng})
    return rows


def running_hr_zones(lthr):
    b = [("Z1", 0, 0.85), ("Z2", 0.85, 0.90), ("Z3", 0.90, 0.95), ("Z4", 0.95, 1.00), ("Z5", 1.00, None)]
    out = []
    for z, lo, hi in b:
        if hi is None:
            out.append(f"{z}: ≥ {round(lthr * lo)} bpm")
        elif lo == 0:
            out.append(f"{z}: < {round(lthr * hi)} bpm")
        else:
            out.append(f"{z}: {round(lthr * lo)}–{round(lthr * hi) - 1} bpm")
    return out


def compute_run_load(sf, act_obj, moving_sec, dist_km, elev_m, t_pace, lthr=0):
    """rTSS, NGP (normaliziran tempo), IF = threshold hitrost / dejanska hitrost ter čas v conah tempa in HR."""
    if not t_pace or moving_sec <= 0:
        return None
    spd = np.nan_to_num(np.array((sf or {}).get("speed") or [], dtype=float))
    zs = [0.0] * 5
    if len(spd) >= 60 and np.any(spd > 0.5):
        r = pd.Series(spd).rolling(30, min_periods=10).mean().to_numpy()
        r = r[~np.isnan(r)]
        r = r[r > 0.5]
        ngs = float(np.mean(r ** 4) ** 0.25) if len(r) else 0.0
        ngp = 1000.0 / ngs if ngs > 0 else 0.0
        mv = spd[spd > 0.5]
        ratio = (1000.0 / mv) / t_pace
        zs = [float(np.sum(ratio > 1.18)), float(np.sum((ratio > 1.08) & (ratio <= 1.18))),
              float(np.sum((ratio > 1.02) & (ratio <= 1.08))), float(np.sum((ratio > 0.97) & (ratio <= 1.02))),
              float(np.sum(ratio <= 0.97))]
        src = "tok 1 Hz (NGP, 30 s drseče povprečje)"
    elif dist_km > 0:
        pace = moving_sec / dist_km
        grade = (elev_m / (dist_km * 1000.0) * 100.0) if dist_km else 0.0
        ngp = pace / (1.0 + 0.035 * max(0.0, grade))
        ratio = ngp / t_pace
        idx = 0 if ratio > 1.18 else 1 if ratio > 1.08 else 2 if ratio > 1.02 else 3 if ratio > 0.97 else 4
        zs[idx] = float(moving_sec)
        src = "povprečni tempo (GAP ocena)"
    else:
        return None
    if ngp <= 0:
        return None
    IF = t_pace / ngp
    rtss = moving_sec / 3600.0 * IF * IF * 100.0
    hz = None
    hr = np.nan_to_num(np.array((sf or {}).get("heartRate") or [], dtype=float))
    if lthr and lthr > 0 and len(hr) and np.any(hr > 0):
        h = hr[hr > 40] / float(lthr)
        hz = [float(np.sum(h < 0.85)), float(np.sum((h >= 0.85) & (h < 0.90))), float(np.sum((h >= 0.90) & (h < 0.95))),
              float(np.sum((h >= 0.95) & (h < 1.0))), float(np.sum(h >= 1.0))]
    return {"ngp": ngp, "IF": IF, "rtss": rtss, "pace_zones": zs, "hr_zones": hz, "src": src}


ZONE_LBL = ["Z1", "Z2", "Z3", "Z4", "Z5"]


def render_race_goal_info(g, cfg, bd=None):
    """Zahtevani tempo za tekmovalni cilj in primerjava s threshold tempom / VDOT napovedjo."""
    try:
        D = float(g.get("race_dist_km") or 0)
        T = parse_hms(g.get("target_time"))
        if not D or not T:
            return
        req = T / D
        st.caption(f"🏁 {D:.1f} km v {fmt_hms(T)} → zahtevani povprečni tempo **{fmt_pace(req)} /km**")
        tp = effective_threshold_pace(cfg, bd)
        if tp:
            st.caption(f"⚙️ Threshold tempo {fmt_pace(tp)} /km · cilj = {req / tp * 100:.0f} % T-tempa")
        vd = estimate_vdot(cfg, bd)
        if vd and vd > 15:
            pred = predict_time(vd, D * 1000.0)
            diff = T - pred
            st.caption(f"📈 VDOT {vd:.1f} napoved: {fmt_hms(pred)} ({fmt_pace(pred / D)} /km) · "
                       f"{'cilj je hitrejši za' if diff < 0 else 'cilj je počasnejši za'} {fmt_hms(abs(diff))}")
    except Exception:
        pass


def estimate_hourly_steps(total_steps, d_str, intra, raw_data, day_acts, rhr=None):
    """Oceni 24 urnih vrednosti korakov (seštevek = total_steps), ko Garmin ne vrne dejanskih epoh.
    Uteži po urah: HR nad mirovnim (iz 2-min intraday HR), brez ur spanja in brez kolesarskih treningov;
    koraki tekaških/pohodnih treningov (act['steps']) gredo v ure treninga. Vrne (np.array(24), opis_vira)."""
    total_steps = float(total_steps or 0)
    out = np.zeros(24, dtype=float)
    if total_steps <= 0:
        return out, "ni korakov"
    lj = ZoneInfo("Europe/Ljubljana")

    def _overlap(start_flt, end_flt):
        return np.array([max(0.0, min(end_flt, h + 1.0) - max(start_flt, float(h))) for h in range(24)], dtype=float)

    def _act_window(a):
        try:
            dt_a = datetime.strptime(str(a.get("startTimeLocal"))[:19], "%Y-%m-%d %H:%M:%S")
            s_f = dt_a.hour + dt_a.minute / 60.0 + dt_a.second / 3600.0
            dur = float(a.get("duration") or a.get("movingDuration") or 0) / 3600.0
            return s_f, min(24.0, s_f + max(dur, 0.05))
        except Exception:
            return None

    # 1) koraki dejanskih tekaških / pohodnih treningov
    act_steps_total = 0.0
    cyc_mask = np.zeros(24, dtype=float)
    for a in day_acts or []:
        tkey = str((a.get("activityType") or {}).get("typeKey", "")).lower() if isinstance(a.get("activityType"), dict) else str(a.get("activityType", "")).lower()
        win = _act_window(a)
        if not win:
            continue
        ov = _overlap(*win)
        if any(k in tkey for k in ("cycl", "bik", "virtual_ride")):
            cyc_mask = np.maximum(cyc_mask, np.clip(ov, 0, 1))
        elif any(k in tkey for k in ("run", "walk", "hik", "mountaineer")):
            a_st = float(a.get("steps") or 0)
            if a_st <= 0:
                dist_m = float(a.get("distance") or 0)
                a_st = dist_m / (1.15 if "run" in tkey else 0.75)
            if a_st > 0 and ov.sum() > 0:
                a_st = min(a_st, total_steps * 0.9)
                out += ov / ov.sum() * a_st
                act_steps_total += a_st

    # 2) preostanek po HR utežeh
    remaining = max(0.0, total_steps - act_steps_total)
    weights = np.zeros(24, dtype=float)
    src = "uro dneva (07:00–22:00)"
    hr_vals = (intra or {}).get("heartRateValues") or []
    try:
        rest = float(rhr) if rhr else None
    except Exception:
        rest = None
    buckets = [[] for _ in range(24)]
    for p in hr_vals:
        try:
            ts, v = p[0], p[1]
            if v is None or ts is None:
                continue
            dt_l = datetime.fromtimestamp(float(ts) / 1000.0, tz=timezone.utc).astimezone(lj)
            if dt_l.strftime("%Y-%m-%d") == d_str:
                buckets[dt_l.hour].append(float(v))
        except Exception:
            continue
    if sum(len(b) for b in buckets) >= 24:
        if rest is None:
            allv = [x for b in buckets for x in b]
            rest = float(np.percentile(allv, 10))
        for h, b in enumerate(buckets):
            if b:
                weights[h] = max(0.0, float(np.mean(b)) - (rest + 4.0))
        src = "srčni utrip čez dan (HR nad mirovnim)"
    if weights.sum() <= 0:
        weights[7:22] = 1.0
    # brez spanja
    try:
        sl = (raw_data.get("sleep") or {}).get(d_str) or {}
        s_ms, e_ms = sl.get("sleepStartTimestampGMT"), sl.get("sleepEndTimestampGMT")
        if s_ms and e_ms:
            for h in range(24):
                h_start = datetime.strptime(d_str, "%Y-%m-%d").replace(hour=h, tzinfo=lj).timestamp() * 1000
                mid = h_start + 1800000
                if s_ms <= mid <= e_ms:
                    weights[h] = 0.0
            src += ", brez ur spanja"
    except Exception:
        pass
    weights = weights * (1.0 - 0.9 * cyc_mask)  # kolesarjenje ≠ koraki
    if weights.sum() <= 0:
        weights[7:22] = 1.0
    out += weights / weights.sum() * remaining
    out = np.round(out)
    diff = total_steps - out.sum()
    if abs(diff) >= 1 and out.sum() > 0:
        out[int(np.argmax(out))] += diff
    if act_steps_total > 0:
        src += " in ure tekaških/pohodnih treningov"
    return out, src


def render_steps_calories_charts(df_all, sel_date, raw_data=None, user_cfg=None, days=14):
    """
    24-urna intraday dinamika korakov in porabe kalorij čez dan (00:00 - 23:59) za izbrani datum:
    - Graf 1 (👣 Koraki čez dan): 24 urnih stolpcev iz dejanskih Garmin 15-minutnih epoh (`dailyMovement`) + kumulativna krivulja + cilj 10.000 korakov.
      Ure z 0 zabeleženimi koraki v Garminu imajo strogo 0 korakov (brez umetnega razporejanja korakov čez neaktivne ure!).
    - Graf 2 (🔥 Poraba kalorij čez dan): 24 naloženih stolpcev (stalen pasivni BMR = `bmr_kcal / 24.0` + aktivne kalorije strogo v urah dejanske vadbe in dejanskih korakov) + kumulativna krivulja.
    """
    try:
        sel_d_str = str(sel_date)[:10]
        # 1. Pridobi dnevni povzetek za izbrani datum
        daily_row = None
        if df_all is not None and not df_all.empty and "date" in df_all.columns:
            m = df_all[df_all["date"].astype(str) == sel_d_str]
            if not m.empty:
                daily_row = m.iloc[0].to_dict()

        raw_daily = (raw_data.get("dailies") or {}).get(sel_d_str) or {} if raw_data else {}
        if not daily_row:
            daily_row = raw_daily

        _mov_rd = raw_daily.get("movement") if isinstance(raw_daily.get("movement"), dict) else {}
        _stp_rd = (_mov_rd.get("steps") or {}).get("value") if isinstance(_mov_rd.get("steps"), dict) else None
        total_steps = int((daily_row.get("steps") if daily_row else None) or raw_daily.get("totalSteps") or _stp_rd or 0)
        _cal_rd = raw_daily.get("calories") if isinstance(raw_daily.get("calories"), dict) else {}
        cal_bmr = float((daily_row.get("cal_bmr") if daily_row else None) or raw_daily.get("bmrKilocalories") or _cal_rd.get("burnedResting") or 1800.0)
        cal_act = float((daily_row.get("cal_act") if daily_row else None) or raw_daily.get("activeKilocalories") or _cal_rd.get("burnedActive") or 0.0)
        hourly_bmr = round(cal_bmr / 24.0, 1)

        # 2. Pridobi dejanske Garmin 15-minutne / urne epohe gibanja (`dailyMovement`) za izbrani datum
        intraday_map = (raw_data.get("intraday") if isinstance(raw_data, dict) else None) or {}
        intra = intraday_map.get(sel_d_str) or {}

        cur_u = get_current_user()
        u_dir = auth.get_user_data_dir(cur_u) if cur_u else DATA_DIR
        intra_disk_path = os.path.join(u_dir, "intraday", f"{sel_d_str}.json")
        if (not intra or "hourlySteps" not in intra) and os.path.exists(intra_disk_path):
            try:
                with open(intra_disk_path, "r", encoding="utf-8") as _if:
                    disk_intra = json.load(_if)
                if isinstance(disk_intra, dict):
                    intra = disk_intra
                    intraday_map[sel_d_str] = intra
            except Exception:
                pass

        # Če `intraday_steps` / `hourlySteps` za izbrani datum še ni shranjen, ga samodejno prenesi iz Garmina
        has_real_movement = bool(
            isinstance(intra.get("hourlySteps"), list) and len(intra.get("hourlySteps")) == 24
            and ("intraday_steps" in intra or "movementData" in intra) and (sum(float(x or 0) for x in intra.get("hourlySteps")) > 0 or total_steps == 0)
        )
        if not has_real_movement:
            fetch_flag = f"movement_fetched_{cur_u}_{sel_d_str}"
            if not st.session_state.get(fetch_flag):
                st.session_state[fetch_flag] = True
                try:
                    import pull as _pull_mod
                    custom_k = (user_cfg or {}).get("garmin_api_key") if isinstance(user_cfg, dict) else None
                    _pull_mod.set_target_user(username=cur_u, custom_api_key=custom_k)
                    fetched_intra = _pull_mod.fetch_intraday_day(sel_d_str, username=cur_u) or {}
                    if isinstance(fetched_intra, dict) and fetched_intra:
                        intra = fetched_intra
                        intraday_map[sel_d_str] = intra
                        if isinstance(raw_data, dict):
                            raw_data.setdefault("intraday", {})[sel_d_str] = intra
                        try:
                            _sl_path = os.path.join(_pull_mod.DATA_DIR, "sleep.json")
                            if os.path.exists(_sl_path):
                                with open(_sl_path, "r", encoding="utf-8") as _sf_sl:
                                    _disk_sl = json.load(_sf_sl)
                                if isinstance(_disk_sl, dict) and isinstance(_disk_sl.get(sel_d_str), dict) and isinstance(raw_data, dict):
                                    raw_data.setdefault("sleep", {})[sel_d_str] = _disk_sl[sel_d_str]
                            st.cache_data.clear()
                        except Exception:
                            pass
                except Exception:
                    pass

        # Razčleni dejanske urne korake iz Garmin dailyMovement (brez izmišljenih korakov v mirujočih urah!)
        hours_labels = [f"{h:02d}:00" for h in range(24)]
        hourly_steps = np.zeros(24, dtype=float)
        real_steps_loaded = False

        if isinstance(intra.get("hourlySteps"), list) and len(intra.get("hourlySteps")) == 24:
            arr_h = np.array([max(0.0, float(x or 0)) for x in intra["hourlySteps"]], dtype=float)
            if np.sum(arr_h) > 0 or total_steps == 0 or bool(intra.get("intraday_steps")):
                hourly_steps = arr_h
                real_steps_loaded = True

        if not real_steps_loaded and intra.get("movementData"):
            try:
                import pull as _pull_mod
                _, parsed_h = _pull_mod._parse_movement_epochs(intra.get("movementData"), sel_d_str)
                if isinstance(parsed_h, list) and len(parsed_h) == 24:
                    hourly_steps = np.array([max(0.0, float(x or 0)) for x in parsed_h], dtype=float)
                    real_steps_loaded = True
            except Exception:
                pass

        # 3. Poišči zabeležene vadbe na ta datum za natančno pozicioniranje aktivnih kalorij po urah
        day_acts = []
        if raw_data:
            acts_all = (raw_data.get("activities") or {}).values()
            for a in acts_all:
                st_loc = str(a.get("startTimeLocal") or "")
                if st_loc[:10] == sel_d_str:
                    day_acts.append(a)

        # Če Garmin API za zelo star datum nima `dailyMovement` in so bile tisti dan tekaške/pohodne aktivnosti,
        # dodeli korake le v ure teh dejanskih aktivnosti (ostale ure ostanejo strogo 0).
        if False and not real_steps_loaded and total_steps > 0 and day_acts:
            act_step_hours = np.zeros(24, dtype=float)
            for act in day_acts:
                sp_type = str(act.get("activityType", {})).lower()
                a_steps = float(act.get("steps") or 0.0)
                if a_steps > 0 or any(k in sp_type for k in ["run", "tek", "walk", "hik", "pohod", "mountaineer"]):
                    try:
                        st_loc = str(act.get("startTimeLocal") or "")
                        dt_act = datetime.strptime(st_loc[:19], "%Y-%m-%d %H:%M:%S")
                        start_flt = dt_act.hour + dt_act.minute / 60.0 + dt_act.second / 3600.0
                        dur_h = max(0.1, float(act.get("movingDuration") or act.get("duration") or 1800) / 3600.0)
                        end_flt = min(24.0, start_flt + dur_h)
                        for h in range(24):
                            ov = max(0.0, min(end_flt, float(h + 1)) - max(start_flt, float(h)))
                            if ov > 0:
                                act_step_hours[h] += ov
                    except Exception:
                        pass
            if np.sum(act_step_hours) > 0:
                hourly_steps = np.round((act_step_hours / np.sum(act_step_hours)) * total_steps)

        # Zadnji fallback: Garmin intraday ni dosegljiv -> približna porazdelitev dnevnih korakov čez budne ure (07-21),
        # da graf nikoli ni prazen; jasno označeno z opozorilom.
        steps_diag = ""
        if float(np.sum(hourly_steps)) <= 0 and total_steps > 0:
            hourly_steps, _est_src = estimate_hourly_steps(total_steps, sel_d_str, intra, raw_data, day_acts, rhr=(daily_row.get("rhr") if daily_row else None) or ((raw_daily.get("heartRate") or {}).get("restingValue") if isinstance(raw_daily.get("heartRate"), dict) else None))
            hourly_steps = np.nan_to_num(np.asarray(hourly_steps, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            _unused_wake = None
            steps_diag = f"ℹ️ Garmin Personal API ne vrača urnih korakov (na voljo je le dnevni seštevek) – prikazana je **ocena** porazdelitve {total_steps:,} korakov glede na: {_est_src}."

        # Če je seštevek epoh iz dailyMovement na voljo, uskladi prikazani skupni seštevek
        sum_epoch_steps = int(round(float(np.sum(hourly_steps))))
        if sum_epoch_steps > 0 and total_steps == 0:
            total_steps = sum_epoch_steps

        # 4. Dejanski izračun urnih kalorij:
        #    - Bazalni BMR = `cal_bmr / 24.0` vsako uro (stalen pasivni metabolizem)
        #    - Aktivne kalorije (`cal_act`):
        #      1) Natančne kalorije zabeleženih Garmin treningov (`act["calories"]`) v točne ure treninga (`startTimeLocal` + `duration`)
        #      2) Preostale ne-vadjske aktivne kalorije strogo sorazmerno z dejanskimi koraki v tisti uri (`hourly_steps`)
        #      3) Če ima ura 0 korakov in nobenega treninga, so aktivne kalorije v tej uri strogo 0!
        hourly_active_cal = np.zeros(24, dtype=float)
        act_cals_assigned = 0.0

        for act in day_acts:
            st_loc = str(act.get("startTimeLocal") or "")
            try:
                dt_act = datetime.strptime(st_loc[:19], "%Y-%m-%d %H:%M:%S")
                start_flt = dt_act.hour + dt_act.minute / 60.0 + dt_act.second / 3600.0
            except Exception:
                start_flt = 16.0
            dur_sec = float(act.get("duration") or act.get("movingDuration") or 3600.0)
            dur_h = max(0.08, dur_sec / 3600.0)
            end_flt = start_flt + dur_h

            a_cal = float(act.get("calories") or 0.0)
            # Odštej pasivni BMR delež znotraj trajanja aktivnosti, če Garmin `act["calories"]` vključuje BMR,
            # tako da ne presežemo skupnih dnevnih aktivnih kalorij `cal_act`
            if cal_act > 0 and a_cal > cal_act:
                a_cal = cal_act

            # Razdeli kalorije treninga natančno glede na delež prekrivanja posamezne ure [h, h+1)
            hour_overlaps = np.zeros(24, dtype=float)
            for h in range(24):
                ov = max(0.0, min(end_flt, float(h + 1)) - max(start_flt, float(h)))
                if ov > 0:
                    hour_overlaps[h] = ov
            total_ov = float(np.sum(hour_overlaps))
            if total_ov > 0 and a_cal > 0:
                for h in range(24):
                    if hour_overlaps[h] > 0:
                        c_h = a_cal * (hour_overlaps[h] / total_ov)
                        hourly_active_cal[h] += c_h
                        act_cals_assigned += c_h

        # Če vsota treningov presega `cal_act` (ker Garmin v `act["calories"]` šteje tudi BMR med treningom),
        # skaliraj vadjske aktivne kalorije navzdol na največ `cal_act`
        if act_cals_assigned > cal_act and cal_act > 0:
            hourly_active_cal = (hourly_active_cal / act_cals_assigned) * cal_act
            act_cals_assigned = cal_act

        # Preostale ne-vadjske aktivne kalorije razporedi STROGO sorazmerno z dejanskimi koraki v posamezni uri (`hourly_steps`)
        rem_act_cal = max(0.0, cal_act - act_cals_assigned)
        total_h_steps = float(np.sum(hourly_steps))
        if rem_act_cal > 0 and total_h_steps > 0:
            hourly_active_cal += (hourly_steps / total_h_steps) * rem_act_cal

        hourly_steps = [int(round(float(x))) for x in np.nan_to_num(hourly_steps, nan=0.0, posinf=0.0, neginf=0.0)]
        hourly_active_cal = np.round(np.nan_to_num(hourly_active_cal, nan=0.0, posinf=0.0, neginf=0.0), 1)
        cum_steps = [int(round(float(x))) for x in np.cumsum(hourly_steps)]
        hourly_total_cal = np.round(hourly_active_cal + float(hourly_bmr), 1)
        hourly_active_cal = [float(x) for x in hourly_active_cal]
        cum_cal = [float(x) for x in np.round(np.cumsum(hourly_total_cal), 1)]

        st.markdown(f"#### 4. Intraday koraki in poraba kalorij čez dan ({sel_date})")
        st.caption(f"📅 24-urna dinamika po urah (00:00–23:59) | Skupaj danes: **{total_steps:,}** korakov, **{int(round(cal_bmr + cal_act)):,}** kcal (Bazalne: {int(round(cal_bmr))} kcal + Aktivne: {int(round(cal_act))} kcal)")

        if steps_diag:
            st.caption(steps_diag)

        c1, c2 = st.columns(2)
        with c1:
            f1 = go.Figure()
            f1.add_trace(go.Bar(
                x=hours_labels,
                y=hourly_steps,
                name="Koraki po urah",
                marker_color="#3498db",
                yaxis="y"
            ))
            f1.add_trace(go.Scatter(
                x=hours_labels,
                y=cum_steps,
                name="Kumulativno",
                line=dict(color="#1d4ed8", width=3),
                yaxis="y2"
            ))
            f1.add_hline(
                y=10000,
                line_dash="dash",
                line_color="#10b981",
                annotation_text="Cilj: 10.000 korakov",
                annotation_position="top left",
                yref="y2"
            )
            f1.update_layout(
                title="👣 Koraki čez dan (po urah in kumulativno)",
                height=350,
                xaxis=dict(title="Ura v dnevu"),
                yaxis=dict(title="Koraki v tej uri", showgrid=True),
                yaxis2=dict(title="Skupaj do te ure", overlaying="y", side="right", showgrid=False),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.12, x=0)
            )
            st.plotly_chart(f1, use_container_width=True, key=f"intraday_steps_{sel_date}")

        with c2:
            f2 = go.Figure()
            f2.add_trace(go.Bar(
                x=hours_labels,
                y=[hourly_bmr] * 24,
                name="Bazalni BMR (kcal)",
                marker_color="#64748b",
                yaxis="y"
            ))
            f2.add_trace(go.Bar(
                x=hours_labels,
                y=hourly_active_cal,
                name="Aktivne kalorije (kcal)",
                marker_color="#f97316",
                yaxis="y"
            ))
            f2.add_trace(go.Scatter(
                x=hours_labels,
                y=cum_cal,
                name="Kumulativno (kcal)",
                line=dict(color="#ea580c", width=3),
                yaxis="y2"
            ))
            f2.update_layout(
                barmode="stack",
                title="🔥 Poraba kalorij po urah (BMR + aktivne kcal)",
                height=350,
                xaxis=dict(title="Ura v dnevu"),
                yaxis=dict(title="kcal v tej uri", showgrid=True),
                yaxis2=dict(title="Skupaj porabljene kcal", overlaying="y", side="right", showgrid=False),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.12, x=0)
            )
            st.plotly_chart(f2, use_container_width=True, key=f"intraday_cals_{sel_date}")

    except Exception as ex_:
        st.caption(f"Grafa 24-urnih korakov/kalorij ni mogoče prikazati: {ex_}")


# ==========================================
# UNIFIED PERIOD NAVIGATOR KOMPONENTA
# ==========================================
def render_period_navigator(nav_key_prefix="analytics", default_preset="Zadnjih 30 dni"):
    """
    Enotna komponenta za časovno navigacijo s prednastavitvami in gumbi [ ◀ ] [ 🔄 ] [ ▶ ].
    Vrne (start_date, end_date, label).
    """
    today_dt = datetime.now().date()
    offset_key = f"{nav_key_prefix}_offset_days"
    preset_key = f"{nav_key_prefix}_preset"
    
    if offset_key not in st.session_state:
        st.session_state[offset_key] = 0

    c_sel, c_prev, c_today, c_next = st.columns([2, 1, 1, 1])
    with c_sel:
        preset_choice = st.selectbox(
            "Izbira obdobja:",
            [
                "Zadnjih 7 dni",
                "Zadnjih 14 dni",
                "Zadnjih 30 dni",
                "Zadnjih 90 dni",
                "Zadnjih 6 mesecev",
                "Zadnje leto (365 dni)",
                "Tekoča sezona (2025/2026)",
                "Vsi podatki (od 2021)",
                "Po meri (Od – Do koledar)"
            ],
            index=2 if default_preset == "Zadnjih 30 dni" else 0,
            key=preset_key
        )

    # Določitev koraka premika v dneh
    if "7 dni" in preset_choice:
        step_days = 7
    elif "14 dni" in preset_choice:
        step_days = 14
    elif "30 dni" in preset_choice:
        step_days = 30
    elif "90 dni" in preset_choice:
        step_days = 90
    elif "6 mesecev" in preset_choice:
        step_days = 180
    elif "leto" in preset_choice:
        step_days = 365
    else:
        step_days = 30

    with c_prev:
        if st.button("◀ Prejšnje", key=f"{nav_key_prefix}_btn_prev", use_container_width=True):
            st.session_state[offset_key] -= step_days
            st.rerun()
    with c_today:
        if st.button("🔄 Danes", key=f"{nav_key_prefix}_btn_today", use_container_width=True):
            st.session_state[offset_key] = 0
            st.rerun()
    with c_next:
        if st.button("Naslednje ▶", key=f"{nav_key_prefix}_btn_next", use_container_width=True):
            st.session_state[offset_key] = min(0, st.session_state[offset_key] + step_days)
            st.rerun()

    # Izračun končnega in začetnega datuma
    calc_end = today_dt + timedelta(days=st.session_state[offset_key])

    if preset_choice == "Zadnjih 7 dni":
        calc_start = calc_end - timedelta(days=7)
    elif preset_choice == "Zadnjih 14 dni":
        calc_start = calc_end - timedelta(days=14)
    elif preset_choice == "Zadnjih 30 dni":
        calc_start = calc_end - timedelta(days=30)
    elif preset_choice == "Zadnjih 90 dni":
        calc_start = calc_end - timedelta(days=90)
    elif preset_choice == "Zadnjih 6 mesecev":
        calc_start = calc_end - timedelta(days=180)
    elif preset_choice == "Zadnje leto (365 dni)":
        calc_start = calc_end - timedelta(days=365)
    elif preset_choice == "Tekoča sezona (2025/2026)":
        calc_start = datetime(2025, 11, 1).date()
        calc_end = today_dt
    elif preset_choice == "Vsi podatki (od 2021)":
        calc_start = datetime(2021, 1, 1).date()
        calc_end = today_dt
    else:
        custom_range = st.date_input(
            "Izberi obdobje:",
            value=(today_dt - timedelta(days=30), today_dt),
            max_value=today_dt,
            key=f"{nav_key_prefix}_custom_picker"
        )
        if isinstance(custom_range, tuple) and len(custom_range) == 2:
            calc_start, calc_end = custom_range
        else:
            calc_start = today_dt - timedelta(days=30)
            calc_end = today_dt

    st.caption(f"📅 Aktivni pogled: **{calc_start.strftime('%d. %m. %Y')}** do **{calc_end.strftime('%d. %m. %Y')}**")
    return calc_start, calc_end, preset_choice

# ==========================================
# AVTENTIKACIJA & SEJA
# ==========================================
current_user = get_current_user()
if not current_user:
    render_auth_page()
    st.stop()

# ==========================================
# SAMODEJNA OBNOVITEV OB ZAGONU (ZERO-CLICK RECOVERY PO PONASTAVITVI)
# ==========================================
_restore_session_flag = f"cloud_restore_checked_{current_user}"
if not st.session_state.get(_restore_session_flag):
    st.session_state[_restore_session_flag] = True
    try:
        _user_act_file = os.path.join(DATA_DIR, "users", current_user, "activities.json")
        if github_sync.is_github_configured() and not os.path.exists(_user_act_file):
            _gh_ok, _gh_msg, _gh_cnt = github_sync.github_restore_all_users_on_startup(force=False)
            if _gh_ok and _gh_cnt > 0:
                st.toast("🐙 Podatki so bili samodejno obnovljeni z GitHuba!", icon="🐙")
                st.cache_data.clear()
    except Exception as _re_err:
        print(f"Napaka pri avtomatski obnovi z GitHuba: {_re_err}")

# ==========================================
# STRANSKA VRSTICA (SIDEBAR) & NASTAVITVE
# ==========================================
_fresh_sync_data = st.session_state.pop("raw_data_fresh", None)
raw_data = _fresh_sync_data if isinstance(_fresh_sync_data, dict) else load_all_data(current_user)
st.session_state["raw_data"] = raw_data
user_cfg = load_user_config()
user_journal = load_user_journal()
disp_name = get_current_display_name()

with st.sidebar:
    st.title("🚴 Garmin Trener")
    st.markdown(f"**👤 Prijavljen:** `{disp_name}` (`@{current_user}`)")
    if st.button("🚪 Odjava", use_container_width=True):
        st.session_state["current_user"] = None
        st.session_state["display_name"] = None
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    sport_filter = st.selectbox("🏅 Filter športa (celoten dashboard):", SPORT_FILTERS, key="sport_filter")

    st.markdown("---")

    # Osveži podatke gumb
    def _latest_sleep_info(fresh):
        sl = (fresh or {}).get("sleep", {}) or {}
        if not sl:
            return "Ni zapisov spanca."
        last = max(sl.keys())
        so = sl[last] or {}
        ts_end = so.get("sleepEndTimestampGMT")
        end_txt = ""
        if ts_end:
            try:
                end_txt = datetime.fromtimestamp(ts_end / 1000.0, tz=timezone.utc).astimezone(ZoneInfo("Europe/Ljubljana")).strftime("%d.%m.%Y %H:%M")
            except Exception:
                end_txt = ""
        times = format_sleep_times(so)
        return f"Najnovejši zapis spanca: **{last}** ({times}" + (f", konec spanca {end_txt}" if end_txt else "") + ")"

    def _run_garmin_sync(extra_args, timeout_s, spinner_txt, ok_txt):
        pull_script = os.path.join(PROJECT_DIR, "pull.py")
        if not os.path.exists(pull_script):
            st.info("Skripta pull.py ni najdena.")
            return
        with st.spinner(spinner_txt):
            try:
                cmd = [sys.executable, pull_script, "--user", current_user] + list(extra_args)
                api_key_to_use = user_cfg.get("garmin_api_key", "").strip()
                if api_key_to_use:
                    cmd.extend(["--api-key", api_key_to_use])
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                                     encoding="utf-8", errors="replace",
                                     env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            except Exception as e:
                st.error(f"Napaka pri zagonu skripte: {e}")
                return
        if res.returncode != 0:
            st.error(f"Napaka pri osveževanju: {(res.stderr or '')[-300:] or (res.stdout or '')[-300:]}")
            return
        # Takoj počisti vse predpomnilnike in ponovno naloži JSON datoteke z diska
        st.cache_data.clear()
        st.cache_resource.clear()
        for _ck in [
            k for k in list(st.session_state.keys())
            if str(k).startswith(("_cached_", "intra_fetched_", "movement_fetched_", "details_checked_", "rec_cache_"))
        ]:
            st.session_state.pop(_ck, None)
        fresh = load_all_data(current_user)
        st.session_state["raw_data_fresh"] = fresh
        st.session_state["raw_data"] = fresh
        st.session_state["last_sync_msg"] = (
            f"{ok_txt} ({datetime.now().strftime('%d.%m.%Y %H:%M:%S')}). " + _latest_sleep_info(fresh)
        )
        today_key = datetime.now().strftime("%Y-%m-%d")
        if today_key not in (fresh.get("sleep") or {}):
            st.session_state["last_sync_msg"] += " ⚠️ Spanec za današnji dan Garmin še ni vrnil (preveri sinhronizacijo ure v aplikaciji Garmin Connect in poskusi znova)."
        try:
            github_sync.github_push_user_files(current_user, async_push=True)
        except Exception:
            pass
        st.rerun()

    sync_period_choice = st.selectbox(
        "Obdobje sinhronizacije:",
        [
            "Zadnjih 14 dni (Hitro)",
            "Zadnjih 90 dni",
            "Zadnje leto (365 dni)",
            "📚 Celotna zgodovina (Vsi treningi in podatki od začetka)"
        ],
        index=0,
        help="Izberite časovno okno za prenos podatkov. 'Celotna zgodovina' prenese vse pretekle treninge in zdravstvene metrike od začetka računa."
    )

    if st.button("🔄 Osveži podatke iz Garmina", use_container_width=True):
        if "14 dni" in sync_period_choice:
            s_args, s_timeout = ["--days", "14"], 600
        elif "90 dni" in sync_period_choice:
            s_args, s_timeout = ["--days", "90"], 900
        elif "leto" in sync_period_choice:
            s_args, s_timeout = ["--days", "365"], 1500
        else:
            s_args, s_timeout = ["--all", "--full"], 3600

        _run_garmin_sync(
            s_args,
            s_timeout,
            f"Prenašam podatke iz Garmina ({sync_period_choice}) za {disp_name}...",
            f"Podatki uspešno osveženi ({sync_period_choice})"
        )

    if st.session_state.get("last_sync_msg"):
        st.success(st.session_state["last_sync_msg"])

    if st.button("🛠️ Popravi / preveri tokove (HR, GPS, moč)", use_container_width=True,
                 help="Ponovno prenese sekundne tokove, kjer manjka srčni utrip ali GPS (zadnjih 400 dni)."):
        _run_garmin_sync(["--days", "3", "--repair-streams", "--stream-days", "400"], 1500,
                         "Preverjam in popravljam časovne tokove aktivnosti (lahko traja nekaj minut)...", "Tokovi preverjeni/popravljeni")

    if st.button(
        "📥 Prenesi podrobnosti za starejše treninge (po 20 treningov)",
        use_container_width=True,
        key="sb_batch_backfill_20_btn",
        help="Zaporedno prenese polni GPS zemljevid, 1-sekundne tokove, uradne Garmin kroge (Laps) in 10-minutne odseke za naslednjih 20 starejših treningov brez prekinitve strani.",
    ):
        run_batch_backfill_older_workouts(raw_data, user_cfg, batch_size=20, container=st.sidebar)

    st.markdown("---")
    st.subheader("⚙️ Športni profil & Nastavitve")

    garmin_key_val = st.text_input(
        "Garmin Personal API Key:",
        value=user_cfg.get("garmin_api_key", ""),
        type="password",
        help="Vnesite svoj Garmin Personal API ključ za sinhronizacijo aktivnosti in zdravstvenih metrik (ali uporabite Streamlit Secrets)."
    )
    _kh_fn = getattr(ai_coach_pro, "render_api_key_help", None)
    if callable(_kh_fn):
        _kh_fn("sb")
    gemini_key_val = st.text_input(
        "🔑 Moj osebni Gemini API ključ (opcijsko - za Pro modele):",
        value=user_cfg.get("gemini_api_key", ""),
        type="password",
        help="Če pustiš prazno, aplikacija uporablja privzeti hitri model (Gemini Flash - Privzeti). Z vnosom lastnega ključa odkleneš gemini-4-hydrogen, Gemini 3.8 / 2.5 Pro in poljubne modele."
    )
    claude_key_val = st.text_input(
        "🟣 Claude (Anthropic) API ključ (opcijsko – npr. za Claude Haiku 5.5 / Sonnet 4.6):",
        value=user_cfg.get("claude_api_key", ""),
        type="password",
        help="Z vnosom svojega Anthropic API ključa odkleneš izjemno ugodni Claude Haiku 5.5, Claude Sonnet 4.6 in Claude Opus 4.6 (z vključenim Prompt Cachingom za do 90 % nižji strošek)."
    )
    _tmp_sb_cfg = dict(user_cfg, gemini_api_key=gemini_key_val.strip(), claude_api_key=claude_key_val.strip())
    _get_avail_fn = getattr(ai_coach_pro, "get_available_model_options", None)
    _sb_model_opts = _get_avail_fn(_tmp_sb_cfg) if callable(_get_avail_fn) else getattr(ai_coach_pro, "MODEL_OPTIONS_PRO", FALLBACK_MODEL_OPTIONS_PRO)
    if gemini_key_val.strip() or claude_key_val.strip():
        _sb_m_cur = user_cfg.get("ai_model_choice", _sb_model_opts[0])
        _sb_m_idx = _sb_model_opts.index(_sb_m_cur) if _sb_m_cur in _sb_model_opts else 0
        sb_ai_model_choice = st.selectbox(
            "🧠 AI Model (Gemini / Claude odklenjen):",
            _sb_model_opts,
            index=_sb_m_idx,
            key="sb_ai_model_choice_sel"
        )
        sb_ai_custom_model = user_cfg.get("ai_custom_model", "")
        sb_ai_custom_model = st.text_input(
            "Ali vpiši točen API ID modela po meri (opcijsko):",
            value=str(sb_ai_custom_model),
            placeholder="npr. gemini-4-hydrogen",
            help="Če je izpolnjeno, se uporabi točno ta API ID modela.",
            key="sb_ai_custom_model_inp",
            )
    else:
        sb_ai_model_choice = _sb_model_opts[0]
        sb_ai_custom_model = user_cfg.get("ai_custom_model", "")
        st.caption("ℹ️ Uporabljaš privzeti hitri model (Gemini Flash - Privzeti). Za odklep Gemini Pro ali Claude Haiku/Sonnet modelov vnesi svoj osebni API ključ.")
    _cur_prof = user_cfg.get("sport_profile", SPORT_PROFILES[0])
    sport_profile_val = st.selectbox(
        "Primarni športni profil:", SPORT_PROFILES,
        index=SPORT_PROFILES.index(_cur_prof) if _cur_prof in SPORT_PROFILES else 0, key="sport_profile_sel")
    _show_cyc = sport_profile_val != SPORT_PROFILES[1]
    _show_run = sport_profile_val != SPORT_PROFILES[0]

    if _show_cyc:
        st.markdown("**🚴 Kolesarski parametri**")
        ftp_out_val = st.number_input("Zunanji FTP (W):", min_value=150, max_value=500, value=int(user_cfg.get("ftp_outdoor", 320)), step=1)
        ftp_in_val = st.number_input("Notranji / Trenažer FTP (W):", min_value=150, max_value=500, value=int(user_cfg.get("ftp_indoor", 305)), step=1)
    else:
        ftp_out_val = int(user_cfg.get("ftp_outdoor", 320))
        ftp_in_val = int(user_cfg.get("ftp_indoor", 305))
    weight_val = st.number_input("Teža (kg):", min_value=45.0, max_value=110.0, value=float(user_cfg.get("weight_kg", 65.5)), step=0.1)
    max_hr_val = st.number_input("Max HR (bpm):", min_value=140, max_value=220, value=int(user_cfg.get("max_hr", 202)), step=1)
    rhr_base_val = st.number_input("Bazalni RHR (bpm):", min_value=30, max_value=90, value=int(user_cfg.get("rhr_base", 48)), step=1)

    thr_pace_val = user_cfg.get("threshold_pace", "")
    lthr_val = int(user_cfg.get("lthr") or 0)
    vd_dist_val = user_cfg.get("vdot_ref_dist", "Brez")
    vd_time_val = user_cfg.get("vdot_ref_time", "")
    run_cad_val = int(user_cfg.get("run_cadence") or 180)
    if _show_run:
        st.markdown("**🏃 Tekaški parametri**")
        thr_pace_val = st.text_input("Pragovni tempo – Threshold Pace / T-Pace (min:sek /km):", value=str(thr_pace_val),
                                     placeholder="npr. 4:15", help="Tempo, ki ga zdržiš ~45–60 min na laktatnem pragu (tekaški ekvivalent FTP).")
        lthr_val = st.number_input("Srčni utrip na laktatnem pragu – LTHR (bpm):", min_value=0, max_value=220, value=lthr_val, step=1,
                                   help="0 = ni nastavljeno.")
        vd_dist_val = st.selectbox("Referenčna razdalja za VDOT (opcijsko):", list(VDOT_REF_DISTS.keys()),
                                   index=list(VDOT_REF_DISTS.keys()).index(vd_dist_val) if vd_dist_val in VDOT_REF_DISTS else 0)
        vd_time_val = st.text_input("Referenčni čas za VDOT (mm:ss ali h:mm:ss):", value=str(vd_time_val), placeholder="npr. 19:30")
        run_cad_val = st.number_input("Kritična / ciljna kadenca (spm):", min_value=120, max_value=220, value=run_cad_val, step=1)

    if st.button("💾 Shrani nastavitve profila", use_container_width=True):
        _bad = False
        if _show_run and str(thr_pace_val).strip() and not parse_hms(thr_pace_val):
            st.error("Neveljaven Threshold Pace – uporabi obliko m:ss (npr. 4:15).")
            _bad = True
        if _show_run and str(vd_time_val).strip() and not parse_hms(vd_time_val):
            st.error("Neveljaven referenčni čas – uporabi mm:ss ali h:mm:ss.")
            _bad = True
        if not _bad:
            user_cfg["garmin_api_key"] = garmin_key_val.strip()
            user_cfg["gemini_api_key"] = gemini_key_val.strip()
            user_cfg["claude_api_key"] = claude_key_val.strip()
            user_cfg["personal_gemini_key_set"] = bool(gemini_key_val.strip())
            user_cfg["ai_model_choice"] = sb_ai_model_choice
            user_cfg["ai_custom_model"] = str(sb_ai_custom_model).strip()
            user_cfg["sport_profile"] = sport_profile_val
            user_cfg["ftp_outdoor"] = int(ftp_out_val)
            user_cfg["ftp_indoor"] = int(ftp_in_val)
            user_cfg["weight_kg"] = round(float(weight_val), 1)
            user_cfg["max_hr"] = int(max_hr_val)
            user_cfg["rhr_base"] = int(rhr_base_val)
            if _show_run:
                _tp = parse_hms(thr_pace_val)
                user_cfg["threshold_pace"] = f"{int(_tp // 60)}:{int(_tp % 60):02d}" if _tp else ""
                user_cfg["lthr"] = int(lthr_val)
                user_cfg["vdot_ref_dist"] = vd_dist_val
                user_cfg["vdot_ref_time"] = str(vd_time_val).strip()
                user_cfg["run_cadence"] = int(run_cad_val)
            if save_user_config(user_cfg):
                st.success("Nastavitve uspešno shranjene!")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Napaka pri shranjevanju.")

    wpkg_out = round(ftp_out_val / weight_val, 2)
    wpkg_in = round(ftp_in_val / weight_val, 2)
    if _show_cyc:
        st.markdown(f"**W/kg Zunaj:** `{wpkg_out} W/kg` | **Trenažer:** `{wpkg_in} W/kg`")
    if _show_run:
        _tmp_cfg = {"threshold_pace": thr_pace_val, "vdot_ref_dist": vd_dist_val, "vdot_ref_time": vd_time_val}
        _tp_eff = effective_threshold_pace(_tmp_cfg)
        with st.expander("🏃 5 tekaških con tempa in utripa", expanded=bool(_tp_eff)):
            if _tp_eff:
                st.caption(f"T-Pace: **{fmt_pace(_tp_eff)} /km**" + (f" · LTHR: **{int(lthr_val)} bpm**" if lthr_val else ""))
                st.dataframe(pd.DataFrame(running_pace_zones(_tp_eff)), use_container_width=True, hide_index=True)
                if lthr_val:
                    st.caption("Cone utripa (% LTHR): " + " · ".join(running_hr_zones(int(lthr_val))))
            else:
                st.caption("Vnesi Threshold Pace (ali referenčni čas za VDOT), da se izračunajo cone.")

    st.markdown("---")
    with st.expander("☁️ GitHub Cloud Sinhronizacija & Varnostna kopija", expanded=True):
        st.markdown("### ☁️ GitHub Sinhronizacija & Varnostne kopije")
        st.caption(
            "Neposredna sinhronizacija posameznih JSON datotek v tvoj GitHub repozitorij (`data/users/<uporabnik>/...`) "
            "ter ročni izvoz/uvoz varnostne kopije (.JSON / .ZIP)."
        )

        gh = github_sync.get_sync_status_info(current_user)
        if gh["is_configured"]:
            st.success(
                f"**{gh['status_badge']}**\n\n"
                f"• **Repozitorij:** `{gh['repo']}` (veja `{gh['branch']}`)\n"
                f"• **Zadnji zapis:** {gh['last_sync']}"
            )
            if gh.get("last_error"):
                st.caption(f"Opomba: {gh['last_error']}")
        else:
            st.warning(
                f"**{gh['status_badge']}**\n\n"
                "Podatki se trenutno shranjujejo lokalno na disku (`data/users/`). "
                "Za samodejno sinhronizacijo z GitHubom vnesi `GITHUB_TOKEN` in `GITHUB_REPO` v Streamlit Secrets ali v nastavitvah spodaj."
            )
            if gh.get("last_error"):
                st.caption(f"Opomba: {gh['last_error']}")

        col_gh1, col_gh2 = st.columns(2)
        with col_gh1:
            if st.button(
                "📤 Shrani vse podatke na GitHub zdaj",
                use_container_width=True,
                disabled=not gh["is_configured"],
                key="gh_sync_all_now_btn",
                help="Takoj potisne vse lokalne datoteke profila (activities.json, sleep.json, hrv.json, klepete, koledar, nastavitve) v GitHub repozitorij.",
            ):
                with st.spinner("Shranjujem datoteke na GitHub..."):
                    g_ok, g_msg = github_sync.github_push_user_files(current_user, async_push=False)
                if g_ok:
                    st.toast(g_msg, icon="🟢")
                    st.success(g_msg)
                else:
                    st.error(g_msg)
        with col_gh2:
            if st.button(
                "📥 Obnovi podatke iz GitHuba",
                use_container_width=True,
                disabled=not gh["is_configured"],
                key="gh_restore_all_now_btn",
                help="Prenese najnovejše datoteke iz GitHub repozitorija na lokalni disk.",
            ):
                with st.spinner("Prenašam datoteke iz GitHuba..."):
                    r_ok, r_msg, _ = github_sync.github_restore_all_users_on_startup(force=True)
                if r_ok:
                    st.toast(r_msg, icon="🐙")
                    st.success(r_msg)
                    st.cache_data.clear()
                    for _ck in [k for k in list(st.session_state.keys()) if str(k).startswith("_cached_") or str(k).startswith("details_checked_")]:
                        st.session_state.pop(_ck, None)
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.warning(r_msg)

        st.markdown("---")
        st.markdown("##### 📤 Uvozi shranjene JSON / ZIP podatke in jih potisni na GitHub")
        st.caption(
            "Naloži rešene `.json` datoteke (npr. `garmin_backup_...json`, `activities.json`, `ai_chat_history.json`, `sleep.json` ali `.zip` arhiv). "
            "Sistem jih samodejno združi z lokalno mapo in shrani na GitHub!"
        )

        uploaded_backups = st.file_uploader(
            "Izberi obstoječe .json ali .zip datoteke:",
            type=["json", "zip"],
            accept_multiple_files=True,
            key=f"cloud_restore_upload_{current_user}",
        )

        if uploaded_backups:
            if st.button("🚀 Uvozi izbrane datoteke in shrani na GitHub", use_container_width=True, type="primary"):
                with st.spinner("Uvažam datoteke in sinhroniziram z GitHubom..."):
                    all_ok = True
                    msgs = []
                    for up_f in uploaded_backups:
                        ok_u, msg_u, _ = github_sync.restore_from_uploaded_backup(
                            current_user, up_f.getvalue(), up_f.name
                        )
                        all_ok = all_ok and ok_u
                        msgs.append(f"{up_f.name}: {msg_u}")
                    summary_msg = " | ".join(msgs)
                    if all_ok:
                        st.success(summary_msg)
                        st.toast("🎉 Podatki uspešno uvoženi in poslani na GitHub!", icon="🎉")
                        st.cache_data.clear()
                        for _ck in [k for k in list(st.session_state.keys()) if str(k).startswith("_cached_") or str(k).startswith("details_checked_")]:
                            st.session_state.pop(_ck, None)
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error(summary_msg)

        st.markdown("---")
        st.markdown("##### 💾 Ročni prenos varnostne kopije na računalnik / telefon")
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            try:
                b_zip = backend.create_user_backup_zip(current_user)
                st.download_button(
                    "⬇️ Prenesi bazo (.ZIP)",
                    data=b_zip,
                    file_name=f"garmin_backup_{current_user}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    help="Celoten arhiv z vsemi podatki, tokovi in odseki.",
                )
            except Exception as e:
                st.caption(f"ZIP: {e}")
        with c_dl2:
            try:
                b_json = backend.create_user_backup_json(current_user)
                st.download_button(
                    "⬇️ Prenesi bazo (.JSON)",
                    data=b_json,
                    file_name=f"garmin_backup_{current_user}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Enotna JSON datoteka z vsemi ključnimi evidencami.",
                )
            except Exception as e:
                st.caption(f"JSON: {e}")

        st.markdown("---")
        with st.expander("⚙️ Nastavitve GitHub povezave (GITHUB_TOKEN & GITHUB_REPO)", expanded=not gh["is_configured"]):
            st.markdown(
                "1. Na GitHubu ustvari **Personal Access Token** (Fine-grained s pravico **Contents: Read and write** ali Classic s pravico **repo**).\n"
                "2. V Streamlit **Secrets** dodaj:\n"
                "```toml\n"
                'GITHUB_TOKEN = "ghp_tvoj_zeton"\n'
                'GITHUB_REPO = "uporabnisko-ime/ime-repozitorija"\n'
                'GITHUB_BRANCH = "main"\n'
                "```\n"
                "ali pa podatke shrani neposredno v obrazcu spodaj:"
            )
            with st.form("github_direct_setup_form"):
                in_gh_tok = st.text_input("🔑 GITHUB_TOKEN:", type="password", placeholder="ghp_... ali github_pat_...")
                in_gh_repo = st.text_input("📦 GITHUB_REPO (lastnik/repozitorij):", value=gh.get("repo", ""), placeholder="uporabnik/garmin-program")
                in_gh_branch = st.text_input("🌿 Veja (GITHUB_BRANCH):", value=gh.get("branch", "main"), placeholder="main")
                if st.form_submit_button("💾 Shrani GitHub povezavo", use_container_width=True):
                    s_ok, s_msg = github_sync.save_local_github_config(in_gh_tok, in_gh_repo, in_gh_branch)
                    if s_ok:
                        st.success(s_msg)
                        time.sleep(0.6)
                        st.rerun()
                    else:
                        st.error(s_msg)

# ==========================================
# OBDELAVA VSEH PODATKOV (CELOTNA ZGODOVINA)
# ==========================================
_acts_cache_key = f"_cached_acts_sorted_{current_user}_{id(raw_data)}_{len(raw_data.get('activities') or {})}"
if _acts_cache_key in st.session_state:
    activities_list = st.session_state[_acts_cache_key]
else:
    activities_list = list(raw_data["activities"].values())
    activities_list.sort(key=lambda x: x.get("startTimeLocal", ""), reverse=True)
    st.session_state[_acts_cache_key] = activities_list

if not activities_list:
    st.info(
        f"👋 Pozdravljen, **{disp_name}**! V vašem računu še ni zabeleženih aktivnosti.\n\n"
        f"1. V stranski vrstici levo vnesite svoj **Garmin Personal API Key**.\n"
        f"2. Kliknite **'💾 Shrani nastavitve profila'**.\n"
        f"3. Kliknite gumb **'🔄 Osveži podatke iz Garmina'**, da se vaša kolesarska zgodovina, intervali in regeneracija prenesejo v vaš profil."
    )

dailies = raw_data["dailies"]
sleep = raw_data["sleep"]
hrv = raw_data["hrv"]
weight = raw_data["weight"]

_df_cache_key = (
    f"_cached_df_daily_{current_user}_{id(raw_data)}_"
    f"{len(dailies)}_{len(sleep)}_{len(hrv)}_{len(weight)}_{len(activities_list)}_{user_cfg.get('weight_kg', 65.5)}"
)
if _df_cache_key in st.session_state:
    df_daily_all = st.session_state[_df_cache_key]
else:
    all_dates = sorted(list(set(list(dailies.keys()) + list(sleep.keys()) + list(hrv.keys()))))
    daily_rows = []
    for d_str in all_dates:
        try:
            d_obj = datetime.strptime(d_str, "%Y-%m-%d").date()
        except Exception:
            continue

        daily_obj = dailies.get(d_str, {})
        sleep_obj = sleep.get(d_str, {})
        hrv_obj = hrv.get(d_str, {})
        wt_obj = weight.get(d_str, {})

        intra_obj = (raw_data.get("intraday", {}) or {}).get(d_str, {}) if isinstance(raw_data, dict) else {}

        rhr = None
        if "heartRate" in daily_obj and isinstance(daily_obj["heartRate"], dict):
            rhr = daily_obj["heartRate"].get("restingValue")
        if not rhr:
            rhr = daily_obj.get("restingHeartRate") or sleep_obj.get("restingHeartRate")

        stress = daily_obj.get("stress", {}).get("avgLevel") if isinstance(daily_obj.get("stress"), dict) else None
        bb_max = daily_obj.get("bodyBattery", {}).get("maxValue") if isinstance(daily_obj.get("bodyBattery"), dict) else None
        bb_min = daily_obj.get("bodyBattery", {}).get("minValue") if isinstance(daily_obj.get("bodyBattery"), dict) else None
        if (bb_max is None or bb_min is None) and isinstance(intra_obj, dict):
            _bb_vals = [
                int(x[1]) for x in (intra_obj.get("bodyBatteryValuesArray") or [])
                if isinstance(x, (list, tuple)) and len(x) >= 2 and x[1] is not None
            ]
            if _bb_vals:
                if bb_max is None:
                    bb_max = max(_bb_vals)
                if bb_min is None:
                    bb_min = min(_bb_vals)

        # Koraki in kalorije
        mov_obj = daily_obj.get("movement", {}) if isinstance(daily_obj.get("movement"), dict) else {}
        steps = mov_obj.get("steps", {}).get("value") if isinstance(mov_obj.get("steps"), dict) else None
        steps_goal = mov_obj.get("steps", {}).get("goal", 10000) if isinstance(mov_obj.get("steps"), dict) else 10000

        cal_obj = daily_obj.get("calories", {}) if isinstance(daily_obj.get("calories"), dict) else {}
        cal_bmr = cal_obj.get("burnedResting")
        cal_act = cal_obj.get("burnedActive")
        cal_tot = cal_obj.get("burnedTotal")

        sleep_sec = sleep_obj.get("sleepTimeSeconds")
        sleep_hours = round(sleep_sec / 3600.0, 2) if sleep_sec else None
        sleep_score = None
        if "sleepScores" in sleep_obj and isinstance(sleep_obj["sleepScores"], dict):
            sleep_score = sleep_obj["sleepScores"].get("overall", {}).get("value")
        if sleep_score is None:
            sleep_score = sleep_obj.get("sleepScore")
        deep_sec = sleep_obj.get("deepSleepSeconds")
        rem_sec = sleep_obj.get("remSleepSeconds")
        light_sec = sleep_obj.get("lightSleepSeconds")
        awake_sec = sleep_obj.get("awakeSleepSeconds")

        avg_sleep_hr = sleep_obj.get("avgHeartRate")
        avg_resp = sleep_obj.get("averageRespirationValue")
        awake_count = sleep_obj.get("awakeCount")
        bedtime_str = format_sleep_times(sleep_obj)

        # Dnevni povprečni utrip (ponderirano glede na spanje, mirovanje in stres)
        if avg_sleep_hr and rhr:
            daily_avg_hr = round(0.35 * avg_sleep_hr + 0.65 * (rhr + 14 + (stress or 25) * 0.18), 1)
        elif rhr:
            daily_avg_hr = round(rhr + 16 + (stress or 25) * 0.18, 1)
        else:
            daily_avg_hr = None

        _intra_hrv = intra_obj.get("hrvSummary") if isinstance(intra_obj, dict) and isinstance(intra_obj.get("hrvSummary"), dict) else {}
        hrv_night = hrv_obj.get("lastNightAvg") or sleep_obj.get("avgOvernightHrv") or _intra_hrv.get("lastNightAvg")
        hrv_weekly = hrv_obj.get("weeklyAvg") or _intra_hrv.get("weeklyAvg")
        hrv_status = hrv_obj.get("status") or sleep_obj.get("hrvStatus") or _intra_hrv.get("status")

        w_kg = None
        if "latestWeight" in wt_obj and isinstance(wt_obj["latestWeight"], dict):
            wg = wt_obj["latestWeight"].get("weight")
            if wg:
                w_kg = round(wg / 1000.0, 1)

        daily_rows.append({
            "date": d_str,
            "date_obj": d_obj,
            "rhr": rhr,
            "stress": stress,
            "bb_max": bb_max,
            "bb_min": bb_min,
            "steps": steps,
            "steps_goal": steps_goal,
            "cal_bmr": cal_bmr,
            "cal_act": cal_act,
            "cal_tot": cal_tot,
            "daily_avg_hr": daily_avg_hr,
            "sleep_hours": sleep_hours,
            "sleep_score": sleep_score,
            "deep_min": round(deep_sec / 60.0, 1) if deep_sec else None,
            "rem_min": round(rem_sec / 60.0, 1) if rem_sec else None,
            "light_min": round(light_sec / 60.0, 1) if light_sec else None,
            "awake_min": round(awake_sec / 60.0, 1) if awake_sec else None,
            "avg_sleep_hr": avg_sleep_hr,
            "avg_resp": avg_resp,
            "awake_count": awake_count,
            "bedtime_str": bedtime_str,
            "hrv_night": hrv_night,
            "hrv_weekly": hrv_weekly,
            "hrv_status": hrv_status,
            "weight_kg": w_kg
        })

    df_daily_all = pd.DataFrame(daily_rows)
    if not df_daily_all.empty:
        df_daily_all.sort_values("date", inplace=True)
        df_daily_all.reset_index(drop=True, inplace=True)

        daily_loads = {}
        daily_vo2 = {}
        for a in activities_list:
            st_l = a.get("startTimeLocal", "")[:10]
            if st_l:
                ld = a.get("activityTrainingLoad") or a.get("trainingStressScore") or 0
                daily_loads[st_l] = daily_loads.get(st_l, 0) + ld

                # VO2 max ocena iz aktivnosti (kolesarski MAP ali tekaški tempo)
                tkey = a.get("activityType", {}).get("typeKey", "")
                dur = a.get("movingDuration") or a.get("duration") or 0
                dist = a.get("distance") or 0
                np_w = a.get("normPower") or a.get("normalizedPower") or a.get("avgPower") or 0
                max_p = a.get("maxPower") or 0
                user_w = float(user_cfg.get("weight_kg", 65.5))

                act_vo2 = None
                if any(k in tkey for k in ["cycling", "road_biking", "virtual_ride", "indoor_cycling"]):
                    aid = str(a.get("activityId"))
                    p300 = raw_data.get("mmp", {}).get(aid, {}).get("mmp", {}).get("300")
                    if p300 and p300 > 150:
                        map_w = p300
                    elif np_w > 180 and dur >= 300:
                        map_w = max(np_w * 1.18, max_p * 0.65)
                    else:
                        map_w = None
                    if map_w:
                        act_vo2 = round(10.8 * (map_w / user_w) + 7.0, 1)
                elif "running" in tkey:
                    if dur >= 600 and dist > 1000:
                        v_m_min = dist / (dur / 60.0)
                        if 140 <= v_m_min <= 360:
                            act_vo2 = round(-4.60 + 0.182258 * v_m_min + 0.000104 * (v_m_min ** 2), 1)

                if act_vo2 and 35 <= act_vo2 <= 88:
                    if st_l not in daily_vo2 or act_vo2 > daily_vo2[st_l]:
                        daily_vo2[st_l] = act_vo2

        df_daily_all["daily_load"] = df_daily_all["date"].map(lambda x: daily_loads.get(x, 0))
        df_daily_all["acute_load_7d"] = df_daily_all["daily_load"].ewm(span=7, min_periods=1).mean()
        df_daily_all["chronic_load_28d"] = df_daily_all["daily_load"].ewm(span=28, min_periods=1).mean().clip(lower=40.0)
        df_daily_all["acwr"] = (df_daily_all["acute_load_7d"] / df_daily_all["chronic_load_28d"]).round(2)
        
        # Izračun standardnega pasu za nočni HRV (7d mean ± 0.75 * 28d std)
        df_daily_all["hrv_7d_mean"] = df_daily_all["hrv_night"].rolling(window=7, min_periods=3).mean()
        df_daily_all["hrv_28d_std"] = df_daily_all["hrv_night"].rolling(window=28, min_periods=7).std()
        df_daily_all["hrv_band_upper"] = df_daily_all["hrv_7d_mean"] + 0.75 * df_daily_all["hrv_28d_std"]
        df_daily_all["hrv_band_lower"] = df_daily_all["hrv_7d_mean"] - 0.75 * df_daily_all["hrv_28d_std"]

        # VO2 max serija in 42-dnevna kapaciteta
        df_daily_all["vo2max_daily"] = df_daily_all["date"].map(daily_vo2)
        df_daily_all["vo2max_capacity"] = df_daily_all["vo2max_daily"].rolling(window=42, min_periods=1).max().interpolate(method="linear").bfill()
    else:
        df_daily_all = pd.DataFrame()
    st.session_state[_df_cache_key] = df_daily_all

# ==========================================
# GLAVNI KPI BANNER & ODBLICEVANJE DIRK
# ==========================================
activities_all = activities_list
if sport_filter != "Vsi športi":
    activities_list = [a for a in activities_list if sport_group(a.get("activityType", {}).get("typeKey")) == sport_filter]
    st.info(f"🏅 Aktiven filter športa: **{sport_filter}** ({len(activities_list)} aktivnosti). Krivulja moči in PMC obremenitev ostaneta izračunana iz vseh podatkov.")

# ==========================================
# MOJI CILJI IN TEKME (KARTICE ODŠTEVANJA)
# ==========================================
today_date = datetime.now().date()
goals_all = get_goals(user_cfg)
_cur_weight = user_cfg.get("weight_kg", 65.5)
if not df_daily_all.empty and "weight_kg" in df_daily_all.columns and df_daily_all["weight_kg"].notna().any():
    _cur_weight = float(df_daily_all["weight_kg"].dropna().iloc[-1])
_best_all_dist, _, _ = gather_best_efforts(raw_data)
_best_5k = _best_all_dist.get("5000")

upcoming_goals = sorted([g for g in goals_all if (goal_days_left(g) is not None and goal_days_left(g) >= 0)], key=lambda g: g.get("date", ""))
PRIO_ICON = {"A": "🔴 A", "B": "🟠 B", "C": "🟢 C"}
if upcoming_goals:
    st.markdown("### 🎯 Moji cilji in odštevanje")
    gcols = st.columns(min(4, len(upcoming_goals)))
    for gi, g in enumerate(upcoming_goals[:4]):
        with gcols[gi]:
            with st.container(border=True):
                dl = goal_days_left(g)
                st.markdown(f"**{PRIO_ICON.get(str(g.get('priority', 'B'))[:1], '🟠 B')} · {g.get('name', 'Cilj')}**")
                st.metric("Število dni do cilja", f"{dl} dni", f"{g.get('date')} · {g.get('sport', '')}", delta_color="off")
                if g.get("metric"):
                    st.caption(f"🎯 {g.get('metric')}")
                if g.get("target_ftp"):
                    prog = min(1.0, user_cfg.get("ftp_outdoor", 0) / float(g["target_ftp"]))
                    st.progress(prog, text=f"FTP {user_cfg.get('ftp_outdoor')} / {g['target_ftp']} W")
                if g.get("target_weight"):
                    tw = float(g["target_weight"])
                    prog = 1.0 if _cur_weight <= tw else max(0.0, min(1.0, tw / _cur_weight))
                    st.progress(prog, text=f"Teža {_cur_weight:.1f} → {tw:.1f} kg")
                if g.get("target_pace"):
                    try:
                        mm, ss = str(g["target_pace"]).split(":")
                        tp = int(mm) * 60 + int(ss)
                        if _best_5k:
                            cp = _best_5k["sec"] / 5.0
                            st.progress(min(1.0, tp / cp), text=f"Najboljši 5 km tempo {fmt_pace(cp)} / cilj {g['target_pace']} /km")
                        else:
                            st.caption(f"Cilj tempa: {g['target_pace']} /km (rekordi se pojavijo po prenosu tokov)")
                    except Exception:
                        pass
                if g.get("race_dist_km") and g.get("target_time"):
                    render_race_goal_info(g, user_cfg, _best_all_dist)
                if g.get("target_wkg"):
                    st.caption(f"⚡ Cilj: {g['target_wkg']} W/kg")
                if g.get("climb_time"):
                    st.caption(f"⛰️ Ciljni čas na vzponu: {g['climb_time']}")

with st.expander("🎯 Moji cilji in tekme (dodaj, uredi, izbriši)"):
    st.caption("Cilji so shranjeni v vašem profilu (`config.json`) in so vidni samo vam. Nov cilj dodate spodaj (polja se prilagodijo športu), obstoječe urejate v tabeli.")
    _v = st.session_state.setdefault("ng_ver", 0)
    st.markdown("##### ➕ Dodaj nov cilj")
    ng1, ng2, ng3, ng4 = st.columns([2, 1.2, 1.2, 1.2])
    ng_name = ng1.text_input("Ime cilja / tekme", key=f"ng_name_{_v}")
    ng_date = ng2.date_input("Datum cilja", value=datetime.now().date() + timedelta(days=90), key=f"ng_date_{_v}")
    ng_sport = ng3.selectbox("Šport", ["Kolesarjenje", "Tek", "Ostalo"], key=f"ng_sport_{_v}")
    ng_prio = ng4.selectbox("Prioriteta", ["A - Glavni cilj", "B - Pomemben cilj", "C - Vmesni cilj"], index=1, key=f"ng_prio_{_v}")
    ng_extra = {}
    if ng_sport == "Kolesarjenje":
        k1, k2, k3, k4 = st.columns(4)
        _tf = k1.number_input("Ciljna moč v W (FTP)", min_value=0, max_value=600, value=0, step=1, key=f"ng_ftp_{_v}")
        _tw = k2.number_input("Ciljna teža v kg", min_value=0.0, max_value=130.0, value=0.0, step=0.1, key=f"ng_wt_{_v}")
        _twk = k3.number_input("Ciljni W/kg", min_value=0.0, max_value=8.0, value=0.0, step=0.05, key=f"ng_wkg_{_v}")
        _ct = k4.text_input("Ciljni čas na vzponu (npr. 0:45:00)", key=f"ng_climb_{_v}")
        if _tf:
            ng_extra["target_ftp"] = int(_tf)
        if _tw:
            ng_extra["target_weight"] = float(_tw)
        if _twk:
            ng_extra["target_wkg"] = float(_twk)
        if _ct.strip():
            ng_extra["climb_time"] = _ct.strip()
    elif ng_sport == "Tek":
        r1, r2, r3 = st.columns(3)
        _rd = r1.selectbox("Tekmovalna razdalja", ["5 km", "10 km", "Polmaraton - 21.1 km", "Maraton - 42.2 km", "Trail / Gorski tek", "Po meri v km"], key=f"ng_rd_{_v}")
        _rd_map = {"5 km": 5.0, "10 km": 10.0, "Polmaraton - 21.1 km": 21.0975, "Maraton - 42.2 km": 42.195}
        if _rd in _rd_map:
            _rkm = _rd_map[_rd]
        else:
            _rkm = r2.number_input("Razdalja (km)", min_value=1.0, max_value=300.0, value=15.0, step=0.1, key=f"ng_rkm_{_v}")
        _rt = r3.text_input("Ciljni čas (hh:mm:ss)", value="00:39:59", key=f"ng_rt_{_v}")
        _rsec = parse_hms(_rt)
        if _rsec:
            ng_extra.update({"race_dist_km": float(_rkm), "target_time": fmt_hms(_rsec)})
            render_race_goal_info(ng_extra, user_cfg, _best_all_dist)
        else:
            st.caption("Vnesi veljaven čas v obliki hh:mm:ss.")
    if st.button("➕ Dodaj cilj", key=f"ng_add_{_v}"):
        if not ng_name.strip():
            st.error("Vnesite ime cilja.")
        else:
            _g = {"name": ng_name.strip(), "date": ng_date.strftime("%Y-%m-%d"), "sport": ng_sport, "metric": "", "priority": ng_prio}
            if ng_extra.get("race_dist_km"):
                _g["metric"] = f"{ng_extra['race_dist_km']:.1f} km v {ng_extra['target_time']} (tempo {fmt_pace(parse_hms(ng_extra['target_time']) / ng_extra['race_dist_km'])} /km)"
            _g.update(ng_extra)
            user_cfg["goals"] = get_goals(user_cfg) + [_g]
            save_user_config(user_cfg)
            st.session_state["ng_ver"] = _v + 1
            st.rerun()
    st.markdown("##### ✏️ Uredi obstoječe cilje")
    gdf = pd.DataFrame([{
        "Ime cilja / tekme": g.get("name", ""),
        "Datum cilja": (datetime.strptime(g["date"], "%Y-%m-%d").date() if g.get("date") else None),
        "Šport": g.get("sport", "Kolesarjenje"),
        "Ciljna metrika / opomba": g.get("metric", ""),
        "Prioriteta": g.get("priority", "B - Pomemben cilj"),
        "Cilj FTP (W)": g.get("target_ftp"),
        "Cilj teža (kg)": g.get("target_weight"),
        "Cilj tempo (m:ss /km)": g.get("target_pace"),
    } for g in goals_all]) if goals_all else pd.DataFrame(columns=["Ime cilja / tekme", "Datum cilja", "Šport", "Ciljna metrika / opomba", "Prioriteta", "Cilj FTP (W)", "Cilj teža (kg)", "Cilj tempo (m:ss /km)"])
    edited_goals = st.data_editor(
        gdf, num_rows="dynamic", use_container_width=True, key="goals_editor",
        column_config={
            "Datum cilja": st.column_config.DateColumn("Datum cilja", format="YYYY-MM-DD"),
            "Šport": st.column_config.SelectboxColumn("Šport", options=["Kolesarjenje", "Tek", "Ostalo"]),
            "Prioriteta": st.column_config.SelectboxColumn("Prioriteta", options=["A - Glavni cilj", "B - Pomemben cilj", "C - Vmesni cilj"]),
            "Cilj FTP (W)": st.column_config.NumberColumn("Cilj FTP (W)", min_value=100, max_value=600, step=1),
            "Cilj teža (kg)": st.column_config.NumberColumn("Cilj teža (kg)", min_value=40.0, max_value=130.0, step=0.1),
        },
    )
    if st.button("💾 Shrani cilje", key="save_goals_btn"):
        new_goals = []
        for _, r in edited_goals.iterrows():
            nm = str(r.get("Ime cilja / tekme") or "").strip()
            dt_v = r.get("Datum cilja")
            if not nm or dt_v is None or pd.isna(dt_v):
                continue
            _old_g = dict(next((x for x in goals_all if x.get("name") == nm), {}))
            _old_g.update({
                "name": nm,
                "date": pd.to_datetime(dt_v).strftime("%Y-%m-%d"),
                "sport": r.get("Šport") or "Kolesarjenje",
                "metric": str(r.get("Ciljna metrika / opomba") or ""),
                "priority": r.get("Prioriteta") or "B - Pomemben cilj",
                "target_ftp": (int(r["Cilj FTP (W)"]) if pd.notna(r.get("Cilj FTP (W)")) else None),
                "target_weight": (float(r["Cilj teža (kg)"]) if pd.notna(r.get("Cilj teža (kg)")) else None),
                "target_pace": (str(r["Cilj tempo (m:ss /km)"]) if pd.notna(r.get("Cilj tempo (m:ss /km)")) and str(r.get("Cilj tempo (m:ss /km)")).strip() else None),
            })
            new_goals.append(_old_g)
        user_cfg["goals"] = new_goals
        save_user_config(user_cfg)
        st.success("Cilji shranjeni!")
        st.rerun()

_prof_kpi = user_cfg.get("sport_profile", SPORT_PROFILES[0])
_is_cyc_kpi = "Kolesar" in _prof_kpi or "Oboje" in _prof_kpi
_is_run_kpi = "Tekač" in _prof_kpi or "Oboje" in _prof_kpi
_t_pace_sec_kpi = effective_threshold_pace(user_cfg)
_t_pace_str_kpi = fmt_pace_sec(_t_pace_sec_kpi) if _t_pace_sec_kpi else str(user_cfg.get("threshold_pace") or "4:15")
_lthr_kpi = user_cfg.get("lthr") or "--"
if _is_cyc_kpi and _is_run_kpi:
    col_kpi3, col_kpi4, col_kpi5 = st.columns(3)
    with col_kpi3:
        st.metric("⚡ Zunanji FTP / Teža", f"{user_cfg['ftp_outdoor']} W", f"{wpkg_out} W/kg ({user_cfg.get('weight_kg')} kg)")
    with col_kpi4:
        st.metric("🏠 Trenažer FTP / Teža", f"{user_cfg['ftp_indoor']} W", f"{wpkg_in} W/kg ({user_cfg.get('weight_kg')} kg)")
    with col_kpi5:
        st.metric("🏃 Pragovni tempo (T-Pace)", f"{_t_pace_str_kpi} /km", f"LTHR: {_lthr_kpi} bpm")
elif _is_run_kpi:
    col_kpi3, col_kpi4 = st.columns(2)
    with col_kpi3:
        st.metric("🏃 Pragovni tempo (T-Pace)", f"{_t_pace_str_kpi} /km", f"LTHR: {_lthr_kpi} bpm")
    with col_kpi4:
        st.metric("👟 Ciljna kadenca / Teža", f"{user_cfg.get('run_cadence', 180)} spm", f"Teža: {user_cfg.get('weight_kg')} kg")
else:
    col_kpi3, col_kpi4 = st.columns(2)
    with col_kpi3:
        st.metric("⚡ Zunanji FTP / Teža", f"{user_cfg['ftp_outdoor']} W", f"{wpkg_out} W/kg ({user_cfg.get('weight_kg')} kg)")
    with col_kpi4:
        st.metric("🏠 Trenažer FTP / Teža", f"{user_cfg['ftp_indoor']} W", f"{wpkg_in} W/kg ({user_cfg.get('weight_kg')} kg)")


st.markdown("---")

# ==============================================================================
# GLAVNI ZAVIHKI APLIKACIJE
# ==============================================================================
tab_ai_pro, tab_coach, tab_mmp, tab_analytics, tab_recovery, tab_pace = st.tabs([
    "🤖 AI Trener & Planiranje",
    "📋 Trener povzetki (AI Coach Hub)",
    "⚡ Krivulja moči (Intervals.icu MMP)",
    "📈 Analiza treningov & PMC",
    "💤 Regeneracija & Fiziologija",
    "🏃 Tempo & rekordi"
])

with tab_ai_pro:
    _render_ai_tab_fn = getattr(ai_coach_pro, "render_ai_coach_and_calendar_tab", None)
    if callable(_render_ai_tab_fn):
        _render_ai_tab_fn(
        user_cfg=user_cfg,
        save_user_config_fn=save_user_config,
        user_journal=user_journal,
        df_daily_all=df_daily_all,
        activities_all=activities_all,
        raw_data=raw_data,
        user_dir=get_active_user_dir(),
        project_dir=PROJECT_DIR,
        display_name=disp_name,
        today_date=today_date,
        get_sport_cat_fn=get_sport_category,
        compute_fixed_blocks_fn=compute_fixed_10m_blocks,
        compute_run_blocks_fn=compute_run_time_blocks,
        compute_run_load_fn=compute_run_load,
            parse_stream_fn=None,
        )

# ==============================================================================
# TAB 1: AI TRENER POVZETKI (Z DNEVNIKOM & ISKALNIKOM ZGODOVINE)
# ==============================================================================
with tab_coach:
    st.header("📋 Generator strukturiranih poročil za AI kolesarskega trenerja")
    st.caption("Pripravljeni Markdown bloki za hiter uvoz v klepet s trenerjem. Vsi vaši komentarji in ocene se samodejno trajno shranjujejo.")

    sub_morning, sub_workout, sub_weekly = st.tabs([
        "🌅 A) Jutranji pregled stanja (Readiness)",
        "🚴 B) Analiza končanega treninga (Multi-Sport & Iskalnik)",
        "📅 C) Tedenski / Obdobni pregled"
    ])

    # ----------------------------------------------------
    # A) JUTRANJI PREGLED STANJA (AUTO-SAVE V DNEVNIK)
    # ----------------------------------------------------
    with sub_morning:
        st.subheader("Jutranji pregled pripravljenosti in regeneracije")

        avail_dates = df_daily_all["date"].tolist()[::-1] if (not df_daily_all.empty and "date" in df_daily_all.columns) else [today_date.strftime("%Y-%m-%d")]
        sel_m_date = st.selectbox("Izberi datum pregleda (privzeto danes):", avail_dates, index=0, key="morning_date_picker")

        sel_row = df_daily_all[df_daily_all["date"] == sel_m_date] if (not df_daily_all.empty and "date" in df_daily_all.columns) else pd.DataFrame()
        m_day = sel_row.iloc[0].to_dict() if not sel_row.empty else {}

        # Naloži shranjene subjektivne vnose za ta datum
        m_saved = user_journal.get("morning_entries", {}).get(sel_m_date, {})

        # 7-dnevna povprečja
        idx_match = df_daily_all[df_daily_all["date"] == sel_m_date].index if (not df_daily_all.empty and "date" in df_daily_all.columns) else []
        if len(idx_match) > 0:
            t_idx = idx_match[0]
            w_start = max(0, t_idx - 6)
            df_w = df_daily_all.iloc[w_start:t_idx+1]
            avg_sleep_7d = df_w["sleep_hours"].dropna().mean()
            avg_score_7d = df_w["sleep_score"].dropna().mean()
            avg_rhr_7d = df_w["rhr"].dropna().mean()
            avg_sleep_hr_7d = df_w["avg_sleep_hr"].dropna().mean()
            avg_hrv_7d = df_w["hrv_night"].dropna().mean()
        else:
            avg_sleep_7d, avg_score_7d, avg_rhr_7d, avg_sleep_hr_7d, avg_hrv_7d = None, None, None, None, None

        st.markdown("#### 1. Vnos subjektivnih občutkov & Jutranja teža")
        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            subj_sleep_score = st.slider(
                "Subjektivna ocena spanca (1=katastrofa, 10=odlično naspan):",
                1, 10, int(m_saved.get("sleep_feel", 8)),
                key=f"m_sleep_score_{sel_m_date}"
            )
            subj_fatigue = st.slider(
                "Občutek splošne utrujenosti (1=popolnoma spočit, 10=izčrpan):",
                1, 10, int(m_saved.get("fatigue", 3)),
                key=f"m_fatigue_{sel_m_date}"
            )
            
            leg_options = [
                "1 - Izjemno sveže in eksplozivne (komaj čakam trening)",
                "2 - Dobre / Spočite (normalno stanje)",
                "3 - Rahlo čutim včerajšnji trening (po ogrevanju bo OK)",
                "4 - Utrujene / Tope (manjka dinamike)",
                "5 - Težke / Zakisane (visoka mišična utrujenost)",
                "6 - Boleče mišice / DOMS ali zategnjenost"
            ]
            saved_leg = m_saved.get("leg_feel", leg_options[1])
            leg_idx = leg_options.index(saved_leg) if saved_leg in leg_options else 1
            subj_legs = st.selectbox("Občutek v nogah:", leg_options, index=leg_idx, key=f"m_legs_{sel_m_date}")

        with col_m2:
            st.markdown("**Jutranje tehtanje:**")
            weighed_saved = m_saved.get("weighed_today", False)
            not_weighed = st.checkbox(
                "☑️ Danes se nisem tehtal",
                value=not weighed_saved,
                key=f"m_not_weighed_{sel_m_date}"
            )
            if not_weighed:
                active_morning_weight = float(user_cfg.get("weight_kg", 65.5))
                st.caption(f"Uporabljena privzeta/zadnja teža: **{active_morning_weight:.1f} kg**")
                weight_md_line = f"- **Jutranja teža:** Danes se nisem tehtal (Zadnja znana / nastavljena teža za izračune: {active_morning_weight:.1f} kg)"
            else:
                active_morning_weight = st.number_input(
                    "Vnesi današnjo jutranjo težo (kg):",
                    min_value=45.0, max_value=110.0,
                    value=float(m_saved.get("weight_kg", user_cfg.get("weight_kg", 65.5))),
                    step=0.1,
                    key=f"m_weight_input_{sel_m_date}"
                )
                _gw = nearest_goal_with(user_cfg, "target_weight")
                if _gw:
                    _tw = float(_gw["target_weight"])
                    diff_weight = active_morning_weight - _tw
                    st.caption(f"Ciljna teža ({_gw['name']}): {_tw:.1f} kg (Razlika: {diff_weight:+.1f} kg)")
                    weight_md_line = f"- **Jutranja teža (izmerjena danes):** {active_morning_weight:.1f} kg (Cilj: {_tw:.1f} kg | Razlika: {diff_weight:+.1f} kg)"
                else:
                    weight_md_line = f"- **Jutranja teža (izmerjena danes):** {active_morning_weight:.1f} kg"

            subj_motivation = st.slider(
                "Motivacija za trening (1=brez, 10=maksimalna):",
                1, 10, int(m_saved.get("motivation", 8)),
                key=f"m_mot_{sel_m_date}"
            )
            
            stress_options = [
                "Nizek (mirno, prost dan / počitek)",
                "Zmeren (običajen dan na faksu / doma)",
                "Povišan (več obveznosti, učenje, manj časa za počitek)",
                "Visok (izpiti, dolg naporen dan, veliko tekanja okoli)",
                "Zelo visok (izjemen psihični ali fizični napor izven kolesa)"
            ]
            saved_stress = m_saved.get("stress_level", stress_options[1])
            stress_idx = stress_options.index(saved_stress) if saved_stress in stress_options else 1
            subj_stress = st.selectbox("Pričakovan stres in obveznosti v dnevu:", stress_options, index=stress_idx, key=f"m_stress_{sel_m_date}")

        with col_m3:
            daily_obligations = st.text_area(
                "📝 Opombe k spancu, počutju in današnje obveznosti / napori (opcijsko):",
                value=str(m_saved.get("notes", m_saved.get("stress_notes", ""))),
                placeholder="Npr. ponoči sem se zbujal, danes imam izpit / dolg dan na faksu / plezanje...",
                height=110,
                key=f"m_notes_{sel_m_date}"
            )
            subj_pain = st.text_input(
                "Bolečine ali znaki bolezni:",
                value=str(m_saved.get("pain", "Brez")),
                key=f"m_pain_{sel_m_date}"
            )
            _load_cal_fn = getattr(ai_coach_pro, "load_training_calendar", lambda _d: [])
            _cal_for_m = {str(w.get("date", ""))[:10]: w for w in _load_cal_fn(get_active_user_dir())}
            _cw_m = _cal_for_m.get(sel_m_date)
            _default_planned_m = str(m_saved.get("planned_workout", ""))
            if not _default_planned_m and _cw_m:
                _default_planned_m = f"{_cw_m.get('title', '')} ({_cw_m.get('duration_min', 0)} min, {_cw_m.get('target_power_w', '')})"
            subj_planned = st.text_input(
                "Planiran trening za danes (opcijsko):",
                value=_default_planned_m,
                placeholder="npr. 2h Z2 aerobna baza (200-230 W) ali 5x4min VO2 Max ali Počitek",
                key=f"m_planned_{sel_m_date}"
            )

        new_m_entry = {
            "weighed_today": not not_weighed,
            "weight_kg": round(float(active_morning_weight), 1),
            "sleep_feel": int(subj_sleep_score),
            "fatigue": int(subj_fatigue),
            "leg_feel": subj_legs,
            "motivation": int(subj_motivation),
            "stress_level": subj_stress,
            "notes": daily_obligations.strip(),
            "pain": subj_pain.strip(),
            "planned_workout": subj_planned.strip()
        }

        col_save_btn, col_save_info = st.columns([1, 3])
        with col_save_btn:
            if st.button("💾 Shrani jutranjo oceno", key=f"btn_save_m_{sel_m_date}", use_container_width=True):
                if "morning_entries" not in user_journal:
                    user_journal["morning_entries"] = {}
                user_journal["morning_entries"][sel_m_date] = new_m_entry
                save_user_journal(user_journal)
                st.toast("✅ Jutranja ocena in načrt treninga uspešno shranjena!")

        # Samodejno shranjevanje v dnevnik ob spremembi
        if user_journal.get("morning_entries", {}).get(sel_m_date) != new_m_entry:
            if "morning_entries" not in user_journal:
                user_journal["morning_entries"] = {}
            user_journal["morning_entries"][sel_m_date] = new_m_entry
            save_user_journal(user_journal)
            st.toast("✅ Jutranji vnos samodejno shranjen v dnevnik!")

        st.caption("✅ Vsi vaši vnosi za ta datum so samodejno shranjeni v `data/user_journal.json`.")

        # Kontekst zadnjih 5 treningov
        sel_d_obj = datetime.strptime(sel_m_date, "%Y-%m-%d").date()
        recent_5_acts = [a for a in activities_all if datetime.strptime(a.get("startTimeLocal","")[:10], "%Y-%m-%d").date() <= sel_d_obj][:5]
        recent_acts_lines = []
        for ra in recent_5_acts:
            r_aid = str(ra.get("activityId"))
            r_d = ra.get("startTimeLocal", "")[:10]
            r_name = ra.get("activityName", "Trening")
            r_sport = get_sport_category(ra.get("activityType", {}).get("typeKey"), r_name)
            r_dur = str(timedelta(seconds=int(ra.get("movingDuration") or ra.get("duration") or 0)))
            r_stream = raw_data["streams"].get(r_aid, {}).get("summary", {})
            r_avg_w = r_stream.get("trueAvgPower") or ra.get("avgPower") or ra.get("averagePower") or 0
            r_np = r_stream.get("np") or ra.get("normPower") or ra.get("normalizedPower") or r_avg_w
            r_hr = round(ra.get("averageHR", 0)) if ra.get("averageHR") else "--"
            r_load = round(ra.get("activityTrainingLoad", 0)) if ra.get("activityTrainingLoad") else "--"
            work_kj = r_stream.get("work_kJ") or (round(r_avg_w * (ra.get("movingDuration") or ra.get("duration") or 0) / 1000.0) if r_avg_w else "--")
            ef_str = f"{r_np/r_hr:.2f}" if (r_np and isinstance(r_hr, (int, float)) and r_hr > 0) else "--"
            whr_str = f"{r_avg_w/r_hr:.2f}" if (r_avg_w and isinstance(r_hr, (int, float)) and r_hr > 0) else "--"
            line = f"- **{r_d}**: {r_name} ({r_sport}) | Čas: {r_dur} | Moč: {int(round(r_avg_w))} W (NP: {int(round(r_np))} W) | HR: {r_hr} bpm | Delo: {work_kj} kJ (Load: {r_load}) | EF: {ef_str} | W/HR: {whr_str}"
            recent_acts_lines.append(line)
        recent_acts_text = "\n".join(recent_acts_lines) if recent_acts_lines else "Ni nedavnih aktivnosti."

        # Izračun W/kg za jutranji povzetek
        wpkg_out_m = round(user_cfg["ftp_outdoor"] / active_morning_weight, 2)
        wpkg_in_m = round(user_cfg["ftp_indoor"] / active_morning_weight, 2)
        _gf = nearest_goal_with(user_cfg, "target_ftp")
        wpkg_target_m = round(float(_gf["target_ftp"]) / active_morning_weight, 2) if _gf else None
        goal_wkg_text = f" | Cilj {_gf['name']} ({_gf['target_ftp']} W) = {wpkg_target_m} W/kg" if _gf else ""

        bedtime_display = m_day.get("bedtime_str") or "Ni zabeleženo"
        sleep_dur_str = f"{int(m_day.get('sleep_hours', 0))}h {int((m_day.get('sleep_hours', 0)%1)*60)}m" if m_day.get('sleep_hours') else "Ni podatka"
        avg_sleep_7d_str = f"{avg_sleep_7d:.1f} h" if avg_sleep_7d else "--"
        avg_score_7d_str = f"{avg_score_7d:.0f}" if avg_score_7d else "--"
        avg_rhr_7d_str = f"{avg_rhr_7d:.1f} bpm" if avg_rhr_7d else "--"
        avg_sleep_hr_7d_str = f"{avg_sleep_hr_7d:.1f} bpm" if avg_sleep_hr_7d else "--"
        avg_hrv_7d_str = m_day.get("hrv_weekly") or (f"{avg_hrv_7d:.0f}" if avg_hrv_7d else "--")

        obligations_line = f"- **Opombe (spanje, počutje, današnje obveznosti / napori):** {daily_obligations.strip()}" if daily_obligations.strip() else ""

        planned_display = subj_planned.strip() if subj_planned.strip() else "Ni vnešeno / po presoji trenerja"

        def _day_move_line(_d_str, _label):
            _dl = (raw_data.get("dailies") or {}).get(_d_str) or {}
            _slp = (raw_data.get("sleep") or {}).get(_d_str) or {}
            _intr = (raw_data.get("intraday") or {}).get(_d_str) or {}
            if not _dl and not _slp and not _intr:
                return f"- **{_label}:** Ni podatkov o dnevnih korakih in kalorijah."
            _mov = _dl.get("movement") if isinstance(_dl.get("movement"), dict) else {}
            _stp_obj = _mov.get("steps") if isinstance(_mov.get("steps"), dict) else {}
            _dst_obj = _mov.get("distance") if isinstance(_mov.get("distance"), dict) else {}
            _cal_obj = _dl.get("calories") if isinstance(_dl.get("calories"), dict) else {}
            _str_obj = _dl.get("stress") if isinstance(_dl.get("stress"), dict) else {}
            _sd_obj = _intr.get("stepsDaily") if isinstance(_intr.get("stepsDaily"), dict) else (_slp.get("stepsDaily") if isinstance(_slp.get("stepsDaily"), dict) else {})

            _stp = int(
                _dl.get("totalSteps")
                or _stp_obj.get("value")
                or _sd_obj.get("totalSteps")
                or 0
            )
            _dist_m = float(
                _dl.get("totalDistanceMeters")
                or _dst_obj.get("value")
                or _sd_obj.get("totalDistance")
                or (_stp * 0.78)
            )
            _km = _dist_m / 1000.0 if _dist_m > 50 else _dist_m
            _bmr = float(
                _dl.get("bmrKilocalories")
                or _cal_obj.get("burnedResting")
                or 0
            )
            _act = float(
                _dl.get("activeKilocalories")
                or _cal_obj.get("burnedActive")
                or 0
            )
            _tot = float(
                _dl.get("totalKilocalories")
                or _cal_obj.get("burnedTotal")
                or (_bmr + _act)
            )
            _str = _dl.get("averageStressLevel") if _dl.get("averageStressLevel") is not None else _str_obj.get("avgLevel")
            _wk = []
            for _a in (raw_data.get("activities") or {}).values():
                if str(_a.get("startTimeLocal") or "")[:10] == _d_str:
                    _sec = int(_a.get("movingDuration") or _a.get("duration") or 0)
                    _p = _a.get("avgPower") or _a.get("averagePower")
                    _kj = f", {round(_p * _sec / 1000.0)} kJ" if _p else ""
                    _wk.append(f"{_a.get('activityName', 'Trening')} ({str(timedelta(seconds=_sec))}{_kj})")
            _wk_txt = "; ".join(_wk) if _wk else "brez treninga"
            return (f"- **{_label}:** Prehojeni koraki: {_stp:,} korakov ({_km:.1f} km) | "
                    f"Skupna poraba: {int(round(_tot))} kcal (Aktivne kalorije: {int(round(_act))} kcal, BMR: {int(round(_bmr))} kcal) | "
                    f"Povprečni dnevni stres: {_str if _str is not None else '--'} | Treningi: [{_wk_txt}]")

        _yday_str = (sel_d_obj - timedelta(days=1)).strftime("%Y-%m-%d")
        yesterday_line = _day_move_line(_yday_str, f"Stres in gibanje prejšnjega dne (Včeraj - {_yday_str})").replace("| Treningi:", "| Treningi včeraj:")
        today_move_line = _day_move_line(sel_m_date, "Danes (trenutno)")

        morning_prompt = f"""### 🌅 JUTRANJI PREGLED STANJA ({sel_m_date})

**1. SPANEC IN NOČNA REGENERACIJA**
- **Čas spanja (Od – Do):** {bedtime_display} (Skupaj: {sleep_dur_str} | 7d povp: {avg_sleep_7d_str})
- **Ocena spanca:** Garmin Sleep Score: {m_day.get('sleep_score', '--')}/100 (7d povp: {avg_score_7d_str}) | Moj občutek: {subj_sleep_score}/10
- **Faze spanca:** Globoki: {m_day.get('deep_min', '--')} min | REM: {m_day.get('rem_min', '--')} min | Lahek: {m_day.get('light_min', '--')} min | Budnost: {m_day.get('awake_min', '--')} min ({m_day.get('awake_count', '--')} prebujanj)
- **Srčni utrip med spanjem:** Povprečni nočni HR: {m_day.get('avg_sleep_hr', '--')} bpm (7d povp: {avg_sleep_hr_7d_str}) | Dnevni RHR: {m_day.get('rhr', '--')} bpm (7d povp: {avg_rhr_7d_str})
- **HRV (Variabilnost srčnega utripa):** Nočno povprečje: {m_day.get('hrv_night', '--')} ms | 7d povprečje: {avg_hrv_7d_str} ms (Status: {m_day.get('hrv_status', 'BALANCED')})
- **Dihanje & Stres:** Nočno dihanje: {m_day.get('avg_resp', '--')} brpm | Body Battery: Min {m_day.get('bb_min', '--')} / Max {m_day.get('bb_max', '--')} (Dnevni stres: {m_day.get('stress', '--')})
{yesterday_line}
{today_move_line}

**2. SUBJEKTIVNO POČUTJE & DNEVNI STRES**
{weight_md_line}
- **Razmerje W/kg z današnjo težo:** Zunanji FTP ({user_cfg['ftp_outdoor']} W) = {wpkg_out_m} W/kg | Trenažer ({user_cfg['ftp_indoor']} W) = {wpkg_in_m} W/kg{goal_wkg_text}
- **Občutek utrujenosti:** {subj_fatigue}/10
- **Občutek v nogah:** {subj_legs}
- **Motivacija za trening:** {subj_motivation}/10
- **Pričakovan stres in obveznosti:** {subj_stress}
{obligations_line}
- **Bolečine ali znaki bolezni:** {subj_pain}

**3. KONTEKST ZADNJIH TRENINGOV (Zadnjih 5 enot)**
{recent_acts_text}

**4. DANAŠNJI NAČRT**
- **Planiran trening za danes:** {planned_display}
"""
        morning_prompt = show_prompt(morning_prompt, q_morning(planned_display), "Kopiraj jutranji povzetek za AI trenerja:", "morning")
        if st.button("🤖 Vprašaj Gemini AI Trenerja", key=f"gemini_btn_m_{sel_m_date}"):
            with st.spinner("Gemini AI trener pripravlja priporočilo za današnji dan..."):
                ans_m = call_gemini_coach(morning_prompt, user_cfg.get("gemini_api_key"))
                st.session_state[f"ans_gemini_m_{sel_m_date}"] = ans_m
                st.session_state[f"ans_gemini_m_{sel_m_date}_model"] = st.session_state.get("_last_coach_model_used", "")
                st.session_state[f"ans_gemini_m_{sel_m_date}_tokens"] = st.session_state.get("_last_coach_token_usage")
        if st.session_state.get(f"ans_gemini_m_{sel_m_date}"):
            st.markdown("#### 💬 Odgovor Gemini AI Trenerja:")
            st.info(st.session_state[f"ans_gemini_m_{sel_m_date}"])
            _m_used = st.session_state.get(f"ans_gemini_m_{sel_m_date}_model", "")
            if _m_used:
                st.caption(f"🤖 Odgovor ustvarjen z modelom: `{_m_used}`")
            _m_tok = st.session_state.get(f"ans_gemini_m_{sel_m_date}_tokens")
            _m_tok_cap = ai_coach_pro.format_token_usage_caption(_m_tok) if hasattr(ai_coach_pro, "format_token_usage_caption") else ""
            if _m_tok_cap:
                st.caption(_m_tok_cap)

        render_morning_wellness_charts(sel_m_date, m_day, raw_data, user_cfg)
        render_steps_calories_charts(df_daily_all, sel_m_date, raw_data=raw_data, user_cfg=user_cfg, days=14)

    # ----------------------------------------------------
    # B) ANALIZA KONČANEGA TRENINGA (CELOTNA ZGODOVINA & ISKANJE)
    # ----------------------------------------------------
    with sub_workout:
        st.subheader("Analiza posameznega treninga (Celotna zgodovina & Iskalnik)")

        c_search1, c_search2 = st.columns([2, 1])
        with c_search1:
            search_query = st.text_input(
                "🔍 Išči po imenu, športu ali datumu (npr. 'Rogla', 'Z2', 'Poreč', '2025-08'):",
                value="",
                key="workout_search_input"
            )
        with c_search2:
            search_date = st.date_input(
                "📅 Skoči na datum (opcijsko):",
                value=None,
                max_value=today_date,
                key="workout_search_date"
            )

        # Filtriranje celotne baze vseh 1,143 aktivnosti
        matched_acts = []
        q_lower = search_query.strip().lower()
        d_str_match = search_date.strftime("%Y-%m-%d") if search_date else ""

        for a in activities_list:
            st_l = a.get("startTimeLocal", "")
            name = a.get("activityName", "Aktivnost")
            sp = get_sport_category(a.get("activityType", {}).get("typeKey"), name)

            if d_str_match and not st_l.startswith(d_str_match):
                continue
            if q_lower:
                combined_txt = f"{st_l} {name} {sp}".lower()
                if q_lower not in combined_txt:
                    continue
            matched_acts.append(a)

        st.caption(f"Prikazano: **{len(matched_acts)}** od skupno **{len(activities_list)}** aktivnosti v celotni zgodovini.")

        if not matched_acts:
            st.warning("Nobena aktivnost ne ustreza iskalnemu kriteriju. Počistite iskalno polje ali datum.")

        else:
            act_options = []
            act_map = {}
            for a in matched_acts:
                aid = str(a.get("activityId"))
                d_str = a.get("startTimeLocal", "")[:10]
                name = a.get("activityName", "Aktivnost")
                sp = get_sport_category(a.get("activityType", {}).get("typeKey"), name)
                dur_m = int(round((a.get("movingDuration") or a.get("duration") or 0) / 60.0))
                dist_k = round((a.get("distance", 0) or 0) / 1000.0, 1)
                dist_str = f"{dist_k} km, " if dist_k > 0 else ""

                # Dodatni prikaz moči če je na voljo
                stream_info = raw_data["streams"].get(aid, {}).get("summary", {})
                p_val = stream_info.get("trueAvgPower") or a.get("avgPower") or a.get("averagePower")
                p_str = f"{round(p_val)} W, " if p_val else ""

                lbl = f"{d_str} | {sp} | {name} ({dist_str}{p_str}{dur_m} min)"
                act_options.append(lbl)
                act_map[lbl] = a

            if st.button(
                "📥 Prenesi podrobnosti za starejše treninge (po 20 treningov)",
                use_container_width=True,
                key="main_batch_backfill_20_btn",
                help="Zaporedno prenese polni GPS zemljevid, 1-sekundne tokove, uradne Garmin kroge (Laps) in 10-minutne odseke za naslednjih 20 starejših treningov brez ponastavitve zavihka.",
            ):
                run_batch_backfill_older_workouts(raw_data, user_cfg, batch_size=20)

            sel_act_lbl = st.selectbox("Izberi aktivnost za podrobno analizo:", act_options, index=0, key="act_dropdown_history")
            act_obj = act_map[sel_act_lbl]
            sel_aid = str(act_obj.get("activityId"))
            sel_type_key = act_obj.get("activityType", {}).get("typeKey", "")
            sel_name = act_obj.get("activityName", "")
            detected_sport = get_sport_category(sel_type_key, sel_name)

            is_cycling = any(k in sel_type_key for k in ["cycling", "road_biking", "virtual_ride", "indoor_cycling", "mountain_biking", "gravel_cycling"])
            is_running_hiking = any(k in sel_type_key for k in ["running", "trail_running", "walking", "hiking", "mountaineering"])
            is_swimming = any(k in sel_type_key for k in ["lap_swimming", "open_water_swimming", "swimming"])

            # Naloži shranjene opombe za to aktivnost
            w_saved = user_journal.get("workout_entries", {}).get(sel_aid, {})

            elapsed_sec = int(act_obj.get("elapsedDuration") or act_obj.get("duration") or 0)
            moving_sec = int(act_obj.get("movingDuration") or act_obj.get("duration") or 0)
            pause_sec = max(0, elapsed_sec - moving_sec)
            dist_km = round((act_obj.get("distance", 0) or 0) / 1000.0, 2)
            elev_m = round(act_obj.get("elevationGain", 0) or 0)
            avg_hr = round(act_obj.get("averageHR", 0)) if act_obj.get("averageHR") else None
            max_hr = round(act_obj.get("maxHR", 0)) if act_obj.get("maxHR") else None
            cal_val = round(act_obj.get("calories", 0) or 0)
            load_val = round(act_obj.get("activityTrainingLoad", 0)) if act_obj.get("activityTrainingLoad") else "--"

            # Ročni prenos ali samodejni prenos podrobnosti (1 Hz tok, GPS, krogi) za izbran (tudi večletni) trening
            force_refetch_clicked = st.button(
                "🔄 Prenesi podrobne podatke in zemljevid za ta trening",
                key=f"refetch_details_{sel_aid}",
                help="Vsili ponovni prenos podrobnega 1-sekundnega toka (/details), uradnih krogov (/splits) ali FIT datoteke iz Garmina za ta trening.",
            )
            ensure_activity_details(sel_aid, act_obj, raw_data, force=force_refetch_clicked, user_cfg=user_cfg)

            full_stream_obj = raw_data["streams"].get(sel_aid, {})
            stream_data = full_stream_obj.get("summary", {})
            split_obj_raw = raw_data["splits"].get(sel_aid, {})
            split_raw = split_obj_raw.get("laps") or split_obj_raw.get("lapDTOs") or act_obj.get("laps") or []
            _u_weight_kg = float(user_cfg.get("weight_kg", 65.5) or 65.5) if isinstance(user_cfg, dict) else 65.5
            split_data = normalize_laps(split_raw, sf=full_stream_obj, act_obj=act_obj, split_obj=split_obj_raw, weight_kg=_u_weight_kg)
            avg_temp_act, min_temp_act, max_temp_act, _ = extract_activity_temp_summary(act_obj, full_stream_obj, split_obj_raw)
            if avg_temp_act is not None:
                if min_temp_act is not None and max_temp_act is not None:
                    temp_summary_str = f"Povp: {avg_temp_act:.1f} °C (Min: {min_temp_act:.1f} °C / Max: {max_temp_act:.1f} °C)"
                else:
                    temp_summary_str = f"Povp: {avg_temp_act:.1f} °C"
            else:
                temp_summary_str = "Ni zabeleženo"

            # Vnos osebnih opomb & RPE z avtomatskim shranjevanjem
            c_nt1, c_nt2 = st.columns([1, 2])
            with c_nt1:
                workout_rpe = st.slider(
                    "Subjektivni napor (RPE 1-10):",
                    1, 10, int(w_saved.get("rpe", 5)),
                    key=f"w_rpe_{sel_aid}"
                )
            with c_nt2:
                user_workout_note = st.text_area(
                    "Osebni zapis o treningu (občutki, prehrana, težavnost, vreme/veter):",
                    value=str(w_saved.get("comment", "Noge v redu, prehrana 60g OH/h, tempo nadzorovan.")),
                    key=f"w_note_{sel_aid}"
                )

            # Samodejno shranjevanje v dnevnik
            new_w_entry = {
                "rpe": int(workout_rpe),
                "comment": user_workout_note.strip()
            }
            if user_journal.get("workout_entries", {}).get(sel_aid) != new_w_entry:
                if "workout_entries" not in user_journal:
                    user_journal["workout_entries"] = {}
                user_journal["workout_entries"][sel_aid] = new_w_entry
                save_user_journal(user_journal)
                st.toast("✅ Opomba o treningu samodejno shranjena v dnevnik!")

            # Prikaz osnovnih metrik v preglednih karticah (vključno s temperaturo)
            c_m1, c_m2, c_m3, c_m4, c_m5, c_m6 = st.columns(6)
            c_m1.metric("Čas gibanja", str(timedelta(seconds=moving_sec)))
            c_m2.metric("Razdalja", f"{dist_km:.2f} km")
            c_m3.metric("Višinski metri", f"+{elev_m} m")
            c_m4.metric("Povprečni HR", f"{avg_hr} bpm" if avg_hr else "--")
            c_m5.metric("Training Load", f"{load_val}" if load_val != "--" else f"{cal_val} kcal")
            c_m6.metric(
                "Temperatura (Povp)",
                f"{avg_temp_act:.1f} °C" if avg_temp_act is not None else "--",
                f"Min {min_temp_act:.0f}° / Max {max_temp_act:.0f}°C" if (min_temp_act is not None and max_temp_act is not None) else None,
                delta_color="off",
            )

            if is_cycling or is_running_hiking:
                c_blk1, c_blk2 = st.columns([2, 1])
                with c_blk1:
                    blk_choice = st.selectbox(
                        "Velikost časovnih odsekov za analizo:",
                        ["5 minut", "10 minut (privzeto)", "15 minut", "20 minut", "30 minut", "Po meri (vnesi minute)"],
                        index=1, key="workout_block_size_sel")
                with c_blk2:
                    if blk_choice.startswith("Po meri"):
                        block_min = int(st.number_input("Dolžina odseka (minute):", min_value=1, max_value=120, value=7, step=1, key="workout_block_custom"))
                    else:
                        block_min = int(blk_choice.split()[0])
            else:
                block_min = 10
            block_sec = block_min * 60

            # Prikaz interaktivne trase na zemljevidu (OpenStreetMap)
            render_activity_map(act_obj, raw_data["streams"].get(sel_aid, {}), sel_aid)

            # ----------------------------------------
            # KOLESARJENJE
            # ----------------------------------------
            if is_cycling:
                is_ind = is_indoor_cycling(act_obj)

                col_ftp1, col_ftp2 = st.columns([2, 1])
                with col_ftp1:
                    ftp_choice = st.radio(
                        "Uporabi FTP za analizo:",
                        [f"Avtomatsko ({'Trenažer ' + str(user_cfg['ftp_indoor']) + ' W' if is_ind else 'Zunanji ' + str(user_cfg['ftp_outdoor']) + ' W'})",
                         f"Zunanji FTP ({user_cfg['ftp_outdoor']} W)",
                         f"Notranji FTP ({user_cfg['ftp_indoor']} W)",
                         "Po meri"],
                        horizontal=True,
                        key=f"ftp_rad_{sel_aid}"
                    )
                with col_ftp2:
                    if "Po meri" in ftp_choice:
                        active_calc_ftp = st.number_input("Vnesi FTP (W):", min_value=150, max_value=500, value=user_cfg["ftp_outdoor"], step=1, key=f"custom_ftp_{sel_aid}")
                        ftp_origin = "Po meri"
                    elif "Notranji" in ftp_choice or (("Avtomatsko" in ftp_choice) and is_ind):
                        active_calc_ftp = user_cfg["ftp_indoor"]
                        ftp_origin = "Notranji / Trenažer"
                    else:
                        active_calc_ftp = user_cfg["ftp_outdoor"]
                        ftp_origin = "Zunanji"

                full_stream_obj = raw_data["streams"].get(sel_aid, {})
                p_stream = full_stream_obj.get("power", [])
                hr_stream = full_stream_obj.get("heartRate", [])
                cad_stream = full_stream_obj.get("cadence", [])
                spd_stream = full_stream_obj.get("speed", [])
                if p_stream and not any(h > 0 for h in (hr_stream or [])):
                    _hr_fb, _ = hr_fallback_series(len(p_stream), act_obj, split_data)
                    if np.any(_hr_fb > 0):
                        hr_stream = _hr_fb.tolist()

                stream_avg_p = stream_data.get("trueAvgPower")
                summary_avg_p = act_obj.get("avgPower") or act_obj.get("averagePower") or 0
                avg_pwr = round(stream_avg_p if stream_avg_p is not None else summary_avg_p)
                norm_pwr = round(stream_data.get("np") or act_obj.get("normPower") or act_obj.get("normalizedPower") or avg_pwr)
                max_pwr = round(act_obj.get("maxPower", 0) or 0)
                avg_cad = round(act_obj.get("averageBikingCadenceInRevPerMinute", 0)) if act_obj.get("averageBikingCadenceInRevPerMinute") else "--"

                if_val = round(norm_pwr / active_calc_ftp, 2) if norm_pwr and active_calc_ftp > 0 else 0.0
                tss_calc = round((moving_sec * norm_pwr * if_val) / (active_calc_ftp * 36.0)) if active_calc_ftp > 0 else (load_val if isinstance(load_val, (int, float)) else 0)
                vi_val = round(norm_pwr / avg_pwr, 2) if norm_pwr and avg_pwr and avg_pwr > 0 else 1.0

                ef_val = round(norm_pwr / avg_hr, 2) if norm_pwr and avg_hr and avg_hr > 0 else None
                whr_val = round(avg_pwr / avg_hr, 2) if avg_pwr and avg_hr and avg_hr > 0 else None
                work_kj = stream_data.get("work_kJ") or (round(avg_pwr * moving_sec / 1000.0) if avg_pwr else 0)
                carb_burn = round(work_kj / 18.5) if work_kj > 0 else 0

                coasting_sec = stream_data.get("coastingSeconds", 0)
                coasting_pct = stream_data.get("coastingPct", 0.0)
                coasting_str = f"{coasting_sec//60}m {coasting_sec%60:02d}s ({coasting_pct:.1f}% časa gibanja)" if coasting_sec > 0 else "0m 00s (0.0%)"

                drift_obj = stream_data.get("cardiacDrift", {})
                if drift_obj and "decouplingPct" in drift_obj:
                    d_pct = drift_obj["decouplingPct"]
                    drift_str = f"{d_pct:+.1f}% (1. polčas: {drift_obj['avgPower1']} W @ {drift_obj['avgHR1']} bpm [EF {drift_obj['ef1']:.2f}] | 2. polčas: {drift_obj['avgPower2']} W @ {drift_obj['avgHR2']} bpm [EF {drift_obj['ef2']:.2f}])"
                else:
                    drift_str = "Ni na voljo (krajša aktivnost ali manjkajoči podatki)"

                # Fiksni 10-minutni bloki za vsa kolesarjenja (ne glede na trajanje, s temperaturo)
                temp_stream = full_stream_obj.get("temperature", [])
                if p_stream and len(p_stream) > 0:
                    blocks_list = compute_fixed_10m_blocks(
                        p_stream, hr_stream, cad_stream, spd_stream,
                        block_sec=block_sec, temp_arr=temp_stream, fallback_temp=avg_temp_act, ele_arr=full_stream_obj.get("elevation", [])
                    )
                else:
                    blocks_list = stream_data.get("blocks", [])
                    for _bl in blocks_list:
                        if _bl.get("avgTemp") is None and avg_temp_act is not None:
                            _bl["avgTemp"] = round(float(avg_temp_act), 1)

                blocks_md = []
                if blocks_list:
                    blocks_md.append("| Odsek | Čas gibanja | Avg Moč | NP | Max W | VI (NP/AvgW) | Prosti tek | Avg HR | Min→Max HR | Kadenca | Vzpon | EF | Temp (°C) |")
                    blocks_md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
                    for bl in blocks_list:
                        t_bl_s = f"{bl['avgTemp']:.1f} °C" if bl.get("avgTemp") is not None else "--"
                        blocks_md.append(f"| {bl['label']} | {bl['movingTime']} | {bl['avgW']} W | {bl['np']} W | {bl.get('maxW', bl.get('np', '--'))} W | {bl.get('vi', 1.0):.2f} | {bl.get('coastingPct', 0.0):.1f}% | {bl['avgHR']} bpm | {bl.get('minHR', '--')}→{bl['maxHR']} bpm | {bl['avgCad']} rpm | +{bl.get('elevGain', 0)} m | {bl['ef']:.2f} | {t_bl_s} |")
                blocks_text = "\n".join(blocks_md) if blocks_md else "Razrez po 10-minutnih blokih ni na voljo."

                # Visokoločljivostni 2-minutni odseki z notranjo variabilnostjo (moč/HR znotraj 10-min blokov)
                try:
                    _ele_stream = full_stream_obj.get("elevation", [])
                    fine_segs = streamlib.compute_fine_segments(
                        p_stream, hr_stream, cad_stream, spd_stream, _ele_stream,
                        seg_sec=120, weight_kg=float(user_cfg.get("weight_kg", 65.5)))
                    fine_text = streamlib.fine_segments_markdown(fine_segs)
                except Exception as _fe:
                    fine_segs, fine_text = [], f"Visokoločljivostni odseki niso na voljo ({_fe})."

                laps_md = []
                if split_data:
                    laps_md.append("| Lap | Razdalja | Čas | Avg Moč | NP | W/kg | Max 30s W | σ moči (VI) | Prosti tek | Avg HR | Min→Max HR | Kadenca | Vzpon | EF | Temp (°C) |")
                    laps_md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
                    _lap_t0 = 0
                    _p_arr = np.nan_to_num(np.asarray(p_stream if p_stream is not None else [], dtype=float))
                    _h_arr = np.nan_to_num(np.asarray(hr_stream if hr_stream is not None else [], dtype=float))
                    for lp in split_data:
                        t_lp_s = f"{lp['avgTemp']:.1f} °C" if lp.get("avgTemp") is not None else "--"
                        _ld = int(round(float(lp.get("duration_s") or lp.get("duration") or lp.get("elapsedDuration") or lp.get("movingDuration") or 0)))
                        m30 = sd_vi = coast = hr_rng = "--"
                        if _ld > 0 and len(_p_arr) >= _lap_t0 + _ld:
                            _pw = _p_arr[_lap_t0:_lap_t0 + _ld]
                            if len(_pw):
                                _r30 = streamlib._roll_max_mean(_pw, 30)
                                m30 = f"{round(_r30)} W" if _r30 is not None else f"{round(float(np.max(_pw)))} W"
                                _am = float(np.mean(_pw))
                                sd_vi = f"{np.std(_pw):.0f} (VI {(round(float(lp.get('normPower') or _am) / _am, 2) if _am > 0 else 1.0)})"
                                coast = f"{np.mean(_pw == 0) * 100:.0f}%"
                            if len(_h_arr) >= _lap_t0 + _ld:
                                _hv = _h_arr[_lap_t0:_lap_t0 + _ld]
                                _hv = _hv[_hv > 0]
                                if len(_hv):
                                    hr_rng = f"{int(_hv.min())}→{int(_hv.max())} bpm"
                        if hr_rng == "--" and lp.get("maxHR"):
                            hr_rng = f"--→{round(lp['maxHR'])} bpm"
                        _lap_t0 += _ld
                        wkg_s = f"{lp['wkg']:.2f}" if lp.get("wkg") is not None else "--"
                        ef_s = f"{lp['ef']:.2f}" if lp.get("ef") is not None else "--"
                        asc_s = f"+{round(lp.get('ascent') or 0)} m"
                        laps_md.append(f"| {lp.get('lapIndex')} | {lp.get('distance_km', 0):.2f} km | {lp.get('duration_str', '--')} | {round(lp.get('avgPower') or 0)} W | {round(lp.get('normPower') or 0)} W | {wkg_s} | {m30} | {sd_vi} | {coast} | {round(lp.get('avgHR') or 0)} bpm | {hr_rng} | {round(lp.get('avgCadence') or 0)} rpm | {asc_s} | {ef_s} | {t_lp_s} |")
                laps_text = "\n".join(laps_md) if laps_md else "Ni zabeleženih posebnih intervalov/lapov."

                # Samodejna zaznava mikro-intervalov (30/15, 40/20) in neprekinjenih naporov / klancev iz 1s toka
                auto_intervals_cyc = streamlib.detect_stream_intervals(
                    sf=full_stream_obj,
                    act_obj=act_obj,
                    sport="Kolesarjenje",
                    ftp=float(active_calc_ftp),
                    weight_kg=float(user_cfg.get("weight_kg", 65.5)),
                )
                auto_intervals_cyc_text = auto_intervals_cyc.get("markdown_summary", "Ni samodejno zaznanih mikro-intervalov.")

                post_workout_prompt = f"""### 🚴 ANALIZA KONČANEGA TRENINGA: {act_obj.get('activityName')} ({act_obj.get('startTimeLocal', '')[:10]})
*Zvrst:* {detected_sport} | *Uporabljen FTP:* {active_calc_ftp} W ({ftp_origin}) | *Teža:* {user_cfg['weight_kg']} kg | *Subjektivni napor (RPE):* {workout_rpe}/10

**1. GLAVNE METRIKE OBREMENITVE IN MOČI**
- **Čas gibanja (Moving Time):** {timedelta(seconds=moving_sec)} (Skupni čas: {timedelta(seconds=elapsed_sec)} | Pavze: {pause_sec//60}m {pause_sec%60:02d}s)
- **Razdalja & Višinski metri:** {dist_km} km | +{elev_m} m vzpona
- **Resnična povprečna moč (z ničlami):** {avg_pwr} W ({avg_pwr/user_cfg['weight_kg']:.2f} W/kg)
- **Normalized Power (NP):** {norm_pwr} W ({norm_pwr/user_cfg['weight_kg']:.2f} W/kg)
- **Maksimalna moč:** {max_pwr} W
- **Intenzivnostni faktor (IF):** {if_val:.2f} (TSS: {tss_calc} | Variabilnostni indeks VI: {vi_val:.2f})
- **Mehansko delo & Poraba:** {work_kj} kJ dela ({cal_val} kcal | Ocenjena poraba OH: ~{carb_burn} g)
- **Prosti tek (brez poganjanja):** {coasting_str}
- **Kadenca:** Povprečna: {avg_cad} rpm
- **Temperatura okolja / naprave:** {temp_summary_str}

**2. SRČNI UTRIP, UČINKOVITOST IN RAZKLOPITEV**
- **Srčni utrip:** Povprečni: {avg_hr or '--'} bpm | Maksimalni: {max_hr or '--'} bpm
- **Aerobna učinkovitost (EF = NP / HR):** {f'{ef_val:.2f}' if ef_val else '--'} (W/HR razmerje: {f'{whr_val:.2f}' if whr_val else '--'})
- **Srčni drift (Pa:Hr Decoupling):** {drift_str}

**3. RAZREZ VOŽNJE PO {block_min}-MINUTNIH BLOKIH (Potek po {block_min}-minutnih odsekih)**
{blocks_text}

**3b. VISOKOLOČLJIVOSTNI 2-MINUTNI ODSEKI (notranja variabilnost moči in HR, špice 30s/1min, prosti tek)**
{fine_text}

**4. INTERVALI / LAPI (GARMIN)**
{laps_text}

**4b. ⚡ SAMODEJNO ZAZNANI INTERVALI IN MIKRO-INTERVALI (IZ SEKUNDNEGA TOKA)**
{auto_intervals_cyc_text}

**5. SUBJEKTIVNI VTIS IN PREHRANA MED VOŽNJO**
- {user_workout_note}
"""
                post_workout_prompt = show_prompt(post_workout_prompt, Q_CYCLING, "Kopiraj analizo kolesarskega treninga za AI trenerja:", "workout")

                # ---------------------------------------------
                # GRAFI ZA KOLESARJENJE
                # ---------------------------------------------
                render_cycling_chart(sel_aid, full_stream_obj, act_obj, split_data, blocks_list, active_calc_ftp)

                # Plotly Graf 3: Stolpčni graf 10-minutnih blokov (NP & EF)
                if blocks_list:
                    b_labels = [bl["label"] for bl in blocks_list]
                    b_nps = [bl["np"] for bl in blocks_list]
                    b_efs = [float(bl["ef"]) if isinstance(bl.get("ef"), (int, float)) else 0.0 for bl in blocks_list]
                    hover_texts = [
                        f"<b>{bl['label']}</b><br>NP: {bl['np']} W<br>Avg Moč: {bl['avgW']} W<br>Avg HR: {bl['avgHR']} bpm<br>EF: {bl['ef']}<br>Kadenca: {bl['avgCad']} rpm<br>Temp: {str(bl.get('avgTemp')) + ' °C' if bl.get('avgTemp') is not None else '--'}<br>Čas gibanja: {bl['movingTime']}"
                        for bl in blocks_list
                    ]
                    fig_blocks = go.Figure(go.Bar(
                        x=b_labels,
                        y=b_nps,
                        marker=dict(
                            color=b_efs,
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="EF (W/bpm)")
                        ),
                        text=[f"{val} W" for val in b_nps],
                        textposition="outside",
                        hovertext=hover_texts,
                        hoverinfo="text"
                    ))
                    fig_blocks.add_hline(y=active_calc_ftp, line_dash="dash", line_color="#f59e0b",
                                        annotation_text=f"FTP ({active_calc_ftp} W)", annotation_position="top right")
                    fig_blocks.update_layout(
                        title=f"🧱 Razporeditev {block_min}-minutnih blokov (Višina: NP, Barva: Aerobni izkoristek EF)",
                        xaxis=dict(title=f"{block_min}-minutni odsek"),
                        yaxis=dict(title="Normalized Power (W)"),
                        height=360,
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig_blocks, use_container_width=True)
                    with st.expander(f"📋 Tabelarični prikaz {block_min}-minutnih blokov in Garmin lapov (z močjo, utripom in temperaturo)", expanded=True):
                        st.markdown(f"**Potek po {block_min}-minutnih odsekih:**")
                        st.dataframe(pd.DataFrame([{
                            "Odsek": bl["label"],
                            "Čas gibanja": bl["movingTime"],
                            "Avg Moč": f"{bl['avgW']} W",
                            "NP": f"{bl['np']} W",
                            "Avg HR": f"{bl['avgHR']} bpm",
                            "Max HR": f"{bl['maxHR']} bpm",
                            "Kadenca": f"{bl['avgCad']} rpm",
                            "EF": f"{bl['ef']:.2f}",
                            "Temp (°C)": f"{bl['avgTemp']:.1f} °C" if bl.get("avgTemp") is not None else "--",
                        } for bl in blocks_list]), use_container_width=True, hide_index=True)
                        if split_data:
                            st.markdown("**Zabeleženi intervali / Krogi (Garmin Laps):**")
                            st.dataframe(pd.DataFrame([{
                                "Lap": lp.get("lapIndex"),
                                "Razdalja": f"{lp.get('distance_km', 0):.2f} km",
                                "Čas": lp.get("duration_str", "--"),
                                "Avg Moč": f"{round(lp.get('avgPower') or 0)} W",
                                "NP": f"{round(lp.get('normPower') or 0)} W",
                                "W/kg": f"{lp['wkg']:.2f}" if lp.get("wkg") is not None else "--",
                                "Avg HR": f"{round(lp.get('avgHR') or 0)} bpm",
                                "Max HR": f"{round(lp.get('maxHR') or 0)} bpm",
                                "Kadenca": f"{round(lp.get('avgCadence') or 0)} rpm",
                                "Vzpon": f"+{round(lp.get('ascent') or 0)} m",
                                "EF": f"{lp['ef']:.2f}" if lp.get("ef") is not None else "--",
                                "Temp (°C)": f"{lp['avgTemp']:.1f} °C" if lp.get("avgTemp") is not None else "--",
                            } for lp in split_data]), use_container_width=True, hide_index=True)
                        st.markdown("**⚡ Samodejno zaznani intervali in mikro-intervali (iz sekundnega toka):**")
                        st.markdown(auto_intervals_cyc_text)
                elif split_data:
                    with st.expander("📋 Zabeleženi intervali / Krogi (Garmin Laps)", expanded=True):
                        st.dataframe(pd.DataFrame([{
                            "Lap": lp.get("lapIndex"),
                            "Razdalja": f"{lp.get('distance_km', 0):.2f} km",
                            "Čas": lp.get("duration_str", "--"),
                            "Avg Moč": f"{round(lp.get('avgPower') or 0)} W",
                            "NP": f"{round(lp.get('normPower') or 0)} W",
                            "W/kg": f"{lp['wkg']:.2f}" if lp.get("wkg") is not None else "--",
                            "Avg HR": f"{round(lp.get('avgHR') or 0)} bpm",
                            "Max HR": f"{round(lp.get('maxHR') or 0)} bpm",
                            "Kadenca": f"{round(lp.get('avgCadence') or 0)} rpm",
                            "Vzpon": f"+{round(lp.get('ascent') or 0)} m",
                            "EF": f"{lp['ef']:.2f}" if lp.get("ef") is not None else "--",
                            "Temp (°C)": f"{lp['avgTemp']:.1f} °C" if lp.get("avgTemp") is not None else "--",
                        } for lp in split_data]), use_container_width=True, hide_index=True)

            # ----------------------------------------
            # TEK / HOJA / POHODNIŠTVO
            # ----------------------------------------
            elif is_running_hiking:
                pace_sec_km = (moving_sec / dist_km) if dist_km > 0 else 0
                pace_str = f"{fmt_pace(pace_sec_km)} /km"

                grade_pct = (elev_m / (dist_km * 1000.0) * 100.0) if dist_km > 0 else 0.0
                gap_factor = 1.0 + (0.035 * max(0.0, grade_pct))
                gap_sec_km = pace_sec_km / gap_factor if gap_factor > 0 else pace_sec_km
                gap_str = f"{fmt_pace(gap_sec_km)} /km"

                cad_run = round(act_obj.get("averageRunningCadenceInStepsPerMinute", 0)) if act_obj.get("averageRunningCadenceInStepsPerMinute") else "--"
                stride_cm = round(act_obj.get("avgStrideLength", 0)) if act_obj.get("avgStrideLength") else "--"

                speed_m_min = (dist_km * 1000.0) / (moving_sec / 60.0) if moving_sec > 0 else 0
                pace_hr_ef = round(speed_m_min / avg_hr, 2) if (avg_hr and avg_hr > 0 and speed_m_min > 0) else "--"

                splits_run = get_km_splits(act_obj, full_stream_obj, split_raw, split_obj=split_obj_raw)
                splits_md = []
                if splits_run:
                    splits_md.append("| Kilometer | Razdalja | Čas | Tempo | Avg HR | Kadenca | Vzpon | Temp (°C) |")
                    splits_md.append("|---|---|---|---|---|---|---|---|")
                    for s in splits_run:
                        t_km_s = f"{s['avgTemp']:.1f} °C" if s.get("avgTemp") is not None else "--"
                        splits_md.append(f"| {s['km']}. km | {s['distance_km']:.2f} km | {s['duration_str']} | {s['pace_str']} /km | {str(s['avgHR']) + ' bpm' if s['avgHR'] else '--'} | {str(s['avgCad']) + ' spm' if s['avgCad'] else '--'} | +{s['elev']} m | {t_km_s} |")
                splits_text = "\n".join(splits_md) if splits_md else "Razrez po 1 km ni na voljo (časovni tok še ni prenesen)."

                run_laps_md = []
                if split_data:
                    run_laps_md.append("| Lap | Razdalja | Čas | Tempo | Avg HR | Max HR | Kadenca | Temp (°C) |")
                    run_laps_md.append("|---|---|---|---|---|---|---|---|")
                    for lp in split_data:
                        lcad = lp.get("avgCadence") or 0
                        lcad = lcad * 2 if (0 < lcad < 120 and "running" in sel_type_key) else lcad
                        t_lp_s = f"{lp['avgTemp']:.1f} °C" if lp.get("avgTemp") is not None else "--"
                        run_laps_md.append(f"| {lp.get('lapIndex')} | {lp.get('distance_km', 0):.2f} km | {lp.get('duration_str', '--')} | {lp.get('pace_str', '--')} /km | {round(lp.get('avgHR') or 0) or '--'} bpm | {round(lp.get('maxHR') or 0) or '--'} bpm | {round(lcad) or '--'} spm | {t_lp_s} |")
                run_laps_text = "\n".join(run_laps_md) if run_laps_md else "Ni zabeleženih Garmin lapov."

                run_blocks = compute_run_time_blocks(full_stream_obj, block_sec, avg_temp_act)
                _rb = []
                if run_blocks:
                    _rb.append("| Odsek | Čas gibanja | Razdalja | Tempo | Avg HR | Max HR | Kadenca | Vzpon | Temp (°C) |")
                    _rb.append("|---|---|---|---|---|---|---|---|---|")
                    for rb_ in run_blocks:
                        _rb.append(f"| {rb_['label']} | {rb_['movingTime']} | {rb_['dist_km']:.2f} km | {rb_['pace_str']} /km | "
                                   f"{str(rb_['avgHR']) + ' bpm' if rb_['avgHR'] else '--'} | {str(rb_['maxHR']) + ' bpm' if rb_['maxHR'] else '--'} | "
                                   f"{str(rb_['avgCad']) + ' spm' if rb_['avgCad'] else '--'} | +{rb_['ascent']} m | "
                                   f"{str(rb_['avgTemp']) + ' °C' if rb_['avgTemp'] is not None else '--'} |")
                run_blocks_text = "\n".join(_rb) if _rb else "Razrez po časovnih odsekih ni na voljo (časovni tok še ni prenesen)."

                run_load = None
                run_load_md = ""
                _t_pace = effective_threshold_pace(user_cfg, _best_all_dist)
                _lthr_cfg = int(user_cfg.get("lthr") or 0)
                if "running" in sel_type_key and _t_pace:
                    run_load = compute_run_load(full_stream_obj, act_obj, moving_sec, dist_km, elev_m, _t_pace, _lthr_cfg)
                if run_load:
                    _pz = run_load["pace_zones"]
                    _ptot = max(1.0, sum(_pz))
                    _pz_txt = " | ".join(f"{ZONE_LBL[i]}: {int(round(_pz[i] / 60))} min ({_pz[i] / _ptot * 100:.0f} %)" for i in range(5))
                    run_load_md = (f"\n- **rTSS (Running Training Stress Score):** {run_load['rtss']:.0f} (Threshold Pace {fmt_pace(_t_pace)} /km)"
                                   f"\n- **NGP (normiran tempo):** {fmt_pace(run_load['ngp'])} /km | **Intensity Factor (IF):** {run_load['IF']:.2f} [{run_load['src']}]"
                                   f"\n- **Čas v conah tempa:** {_pz_txt}")
                    if run_load["hr_zones"]:
                        _hz = run_load["hr_zones"]
                        _htot = max(1.0, sum(_hz))
                        run_load_md += f"\n- **Čas v conah utripa (LTHR {_lthr_cfg} bpm):** " + " | ".join(
                            f"{ZONE_LBL[i]}: {int(round(_hz[i] / 60))} min ({_hz[i] / _htot * 100:.0f} %)" for i in range(5))
                elif "running" in sel_type_key:
                    run_load_md = "\n- **rTSS / IF:** Ni izračunano (v profilu nastavi Threshold Pace ali referenčni čas za VDOT)."

                auto_intervals_run = streamlib.detect_stream_intervals(
                    sf=full_stream_obj,
                    act_obj=act_obj,
                    sport="Tek",
                    weight_kg=float(user_cfg.get("weight_kg", 65.5)),
                    threshold_pace_sec=float(_t_pace or 255.0),
                )
                auto_intervals_run_text = auto_intervals_run.get("markdown_summary", "Ni samodejno zaznanih mikro-intervalov.")

                post_workout_prompt = f"""### 🏃 ANALIZA TRENINGA: {act_obj.get('activityName')} ({act_obj.get('startTimeLocal', '')[:10]})
*Zvrst:* {detected_sport} | *Razdalja:* {dist_km} km | *Vzpon:* +{elev_m} m | *RPE:* {workout_rpe}/10

**1. GLAVNE TEKAŠKE METRIKE**
- **Čas gibanja:** {timedelta(seconds=moving_sec)} (Skupni čas: {timedelta(seconds=elapsed_sec)} | Pavze: {pause_sec//60}m {pause_sec%60:02d}s)
- **Povprečni tempo:** {pace_str}
- **Ocenjen GAP (Grade Adjusted Pace):** {gap_str} (Povprečni naklon: {grade_pct:.1f}%)
- **Tekaška kadenca:** {cad_run} spm | Dolžina koraka: {stride_cm} cm
- **Poraba:** {cal_val} kcal | Training Load: {load_val}
- **Temperatura okolja / naprave:** {temp_summary_str}

**2. SRČNI UTRIP IN TEKAŠKA UČINKOVITOST**
- **Srčni utrip:** Povprečni: {avg_hr or '--'} bpm | Maksimalni: {max_hr or '--'} bpm
- **Pace:HR Učinkovitost:** {pace_hr_ef} m/utrip{run_load_md}

**3. RAZREZ TEMPA PO 1 KM (KILOMETER SPLITS)**
{splits_text}

**3b. POTEK PO {block_min}-MINUTNIH ODSEKIH**
{run_blocks_text}

**4. GARMIN LAPI / INTERVALI**
{run_laps_text}

**4b. ⚡ SAMODEJNO ZAZNANI INTERVALI IN MIKRO-INTERVALI (IZ SEKUNDNEGA TOKA)**
{auto_intervals_run_text}

**5. SUBJEKTIVNI VTIS**
- {user_workout_note}
"""
                post_workout_prompt = show_prompt(post_workout_prompt, Q_RUN, "Kopiraj analizo teka/hoje za AI trenerja:", "workout")

                has_run_stream = render_running_chart(sel_aid, full_stream_obj, act_obj, split_data)

                if splits_run:
                    st.markdown("#### ⏱️ Tabela tempa po 1 km (Kilometer Splits)")
                    st.dataframe(pd.DataFrame([{
                        "Kilometer": f"{s['km']}. km",
                        "Razdalja": f"{s['distance_km']:.2f} km",
                        "Čas": s["duration_str"],
                        "Tempo": f"{s['pace_str']} /km",
                        "Avg HR": f"{s['avgHR']} bpm" if s["avgHR"] else "--",
                        "Kadenca": f"{s['avgCad']} spm" if s["avgCad"] else "--",
                        "Vzpon": f"+{s['elev']} m",
                        "Temp (°C)": f"{s['avgTemp']:.1f} °C" if s.get("avgTemp") is not None else "--",
                    } for s in splits_run]), use_container_width=True, hide_index=True)
                    if not has_run_stream:
                        fig_km = go.Figure()
                        fig_km.add_trace(go.Scatter(x=[s["km"] for s in splits_run], y=[s["pace_sec"] for s in splits_run], mode="lines+markers",
                                                    name="Tempo (min/km)", line=dict(color="#ea580c", width=3)))
                        hrs_km = [s["avgHR"] for s in splits_run]
                        if any(hrs_km):
                            fig_km.add_trace(go.Scatter(x=[s["km"] for s in splits_run], y=hrs_km, mode="lines+markers", name="Srčni utrip (bpm)",
                                                        yaxis="y2", line=dict(color="#ef4444", width=2.5, dash="dot")))
                        fig_km.update_layout(height=360, hovermode="x unified", xaxis=dict(title="Kilometer", dtick=1),
                                             yaxis=dict(title="Tempo (sek/km) – hitreje zgoraj", autorange="reversed"),
                                             yaxis2=dict(title="Srčni utrip (bpm)", overlaying="y", side="right", showgrid=False))
                        st.plotly_chart(fig_km, use_container_width=True, key=f"km_chart_{sel_aid}")

                if run_load:
                    st.markdown("#### 💪 Tekaška obremenitev (rTSS, NGP, IF)")
                    cl1, cl2, cl3 = st.columns(3)
                    cl1.metric("rTSS", f"{run_load['rtss']:.0f}")
                    cl2.metric("NGP", f"{fmt_pace(run_load['ngp'])} /km")
                    cl3.metric("IF", f"{run_load['IF']:.2f}")
                    _pz = run_load["pace_zones"]
                    _ptot = max(1.0, sum(_pz))
                    st.dataframe(pd.DataFrame([{"Cona tempa": z["Cona"], "Razpon": z["Tempo"], "Čas": f"{int(round(_pz[i] / 60))} min",
                                                "Delež": f"{_pz[i] / _ptot * 100:.0f} %"} for i, z in enumerate(running_pace_zones(_t_pace))]),
                                 use_container_width=True, hide_index=True)
                if run_blocks:
                    st.markdown(f"#### ⏱️ Potek po {block_min}-minutnih odsekih")
                    st.dataframe(pd.DataFrame([{
                        "Odsek": rb_["label"], "Čas gibanja": rb_["movingTime"], "Razdalja": f"{rb_['dist_km']:.2f} km",
                        "Tempo": f"{rb_['pace_str']} /km", "Avg HR": f"{rb_['avgHR']} bpm" if rb_["avgHR"] else "--",
                        "Max HR": f"{rb_['maxHR']} bpm" if rb_["maxHR"] else "--",
                        "Kadenca": f"{rb_['avgCad']} spm" if rb_["avgCad"] else "--", "Vzpon": f"+{rb_['ascent']} m",
                        "Temp (°C)": f"{rb_['avgTemp']} °C" if rb_["avgTemp"] is not None else "--",
                    } for rb_ in run_blocks]), use_container_width=True, hide_index=True)
                    _pv = [rb_["pace_sec"] if rb_["pace_sec"] > 0 else None for rb_ in run_blocks]
                    fig_rb = go.Figure(go.Bar(x=[rb_["label"] for rb_ in run_blocks], y=_pv,
                                              text=[rb_["pace_str"] for rb_ in run_blocks], textposition="outside", marker_color="#ea580c"))
                    fig_rb.update_layout(title=f"Tempo po {block_min}-minutnih odsekih (nižji stolpec = hitrejši tempo)",
                                         yaxis=dict(title="Tempo (sek/km)"), xaxis=dict(title=f"{block_min}-minutni odsek"), height=340)
                    st.plotly_chart(fig_rb, use_container_width=True, key=f"run_blocks_{sel_aid}_{block_min}")
                if split_data:
                    st.markdown("#### 🏁 Garmin lapi / intervali")
                    st.dataframe(pd.DataFrame([{
                        "Lap": lp.get("lapIndex"),
                        "Razdalja": f"{lp.get('distance_km', 0):.2f} km",
                        "Čas": lp.get("duration_str", "--"),
                        "Tempo": f"{lp.get('pace_str', '--')} /km",
                        "Avg HR": f"{round(lp.get('avgHR'))} bpm" if lp.get("avgHR") else "--",
                        "Max HR": f"{round(lp.get('maxHR'))} bpm" if lp.get("maxHR") else "--",
                        "Vzpon": f"+{round(lp.get('ascent') or 0)} m",
                        "Temp (°C)": f"{lp['avgTemp']:.1f} °C" if lp.get("avgTemp") is not None else "--",
                    } for lp in split_data]), use_container_width=True, hide_index=True)
                st.markdown("#### ⚡ Samodejno zaznani intervali in mikro-intervali (iz sekundnega toka)")
                st.markdown(auto_intervals_run_text)

            # ----------------------------------------
            # PLAVANJE
            # ----------------------------------------
            elif is_swimming:
                dist_m = int(round(act_obj.get("distance", 0) or 0))
                pace_100m_sec = (moving_sec / (dist_m / 100.0)) if dist_m > 0 else 0
                pace_100m_str = f"{int(pace_100m_sec // 60)}:{int(pace_100m_sec % 60):02d} /100m"
                swolf = act_obj.get("averageSwolf", "--")
                strokes = act_obj.get("strokes", "--")

                post_workout_prompt = f"""### 🏊 ANALIZA PLAVALNEGA TRENINGA: {act_obj.get('activityName')} ({act_obj.get('startTimeLocal', '')[:10]})
*Zvrst:* Plavanje | *Razdalja:* {dist_m} m | *RPE:* {workout_rpe}/10

**1. GLAVNE PLAVALNE METRIKE**
- **Aktivni čas plavanja:** {timedelta(seconds=moving_sec)} (Skupni čas: {timedelta(seconds=elapsed_sec)} | Počitek ob robu: {pause_sec//60}m {pause_sec%60:02d}s)
- **Povprečni tempo:** {pace_100m_str}
- **SWOLF indeks:** {swolf}
- **Zamahi (Strokes):** Skupaj: {strokes}
- **Srčni utrip:** Povprečni: {avg_hr or '--'} bpm | Max: {max_hr or '--'} bpm
- **Poraba & Obremenitev:** {cal_val} kcal | Training Load: {load_val}

**2. SUBJEKTIVNI VTIS**
- {user_workout_note}
"""
                post_workout_prompt = show_prompt(post_workout_prompt, Q_SWIM, "Kopiraj analizo plavanja za AI trenerja:", "workout")

            # ----------------------------------------
            # FITNES / MOČ / DRUGO
            # ----------------------------------------
            else:
                post_workout_prompt = f"""### 🏋️ ANALIZA TRENINGA MOČI / FITNESA: {act_obj.get('activityName')} ({act_obj.get('startTimeLocal', '')[:10]})
*Zvrst:* {detected_sport} | *RPE:* {workout_rpe}/10

**1. TRAJANJE IN INTENZIVNOST**
- **Aktivni čas:** {timedelta(seconds=moving_sec)} (Skupni čas: {timedelta(seconds=elapsed_sec)})
- **Srčni utrip:** Povprečni: {avg_hr or '--'} bpm | Maksimalni: {max_hr or '--'} bpm
- **Obremenitev:** Training Load: {load_val} | Kalorije: {cal_val} kcal
- **Aerobni TE:** {act_obj.get('aerobicTrainingEffect', '--')} | Anaerobni TE: {act_obj.get('anaerobicTrainingEffect', '--')}

**2. SUBJEKTIVNI VTIS**
- {user_workout_note}
"""
                post_workout_prompt = show_prompt(post_workout_prompt, Q_OTHER, "Kopiraj analizo vadbe za AI trenerja:", "workout")

            if "post_workout_prompt" in locals() and post_workout_prompt:
                if st.button("🤖 Vprašaj Gemini AI Trenerja o tem treningu", key=f"gemini_btn_w_{sel_aid}"):
                    with st.spinner("Gemini AI trener analizira opravljen trening..."):
                        ans_w = call_gemini_coach(post_workout_prompt, user_cfg.get("gemini_api_key"))
                        st.session_state[f"ans_gemini_w_{sel_aid}"] = ans_w
                        st.session_state[f"ans_gemini_w_{sel_aid}_model"] = st.session_state.get("_last_coach_model_used", "")
                        st.session_state[f"ans_gemini_w_{sel_aid}_tokens"] = st.session_state.get("_last_coach_token_usage")
                if st.session_state.get(f"ans_gemini_w_{sel_aid}"):
                    st.markdown("#### 💬 Odgovor Gemini AI Trenerja:")
                    st.info(st.session_state[f"ans_gemini_w_{sel_aid}"])
                    _w_used = st.session_state.get(f"ans_gemini_w_{sel_aid}_model", "")
                    if _w_used:
                        st.caption(f"🤖 Odgovor ustvarjen z modelom: `{_w_used}`")
                    _w_tok = st.session_state.get(f"ans_gemini_w_{sel_aid}_tokens")
                    _w_tok_cap = ai_coach_pro.format_token_usage_caption(_w_tok) if hasattr(ai_coach_pro, "format_token_usage_caption") else ""
                    if _w_tok_cap:
                        st.caption(_w_tok_cap)

    # ----------------------------------------------------
    # C) TEDENSKI / OBDOBNI PREGLED (PRIMERJALNI POVZETEK ZA AI TRENERJA)
    # ----------------------------------------------------
    with sub_weekly:
        st.subheader("📊 Primerjalni povzetek tedna / bloka (Za AI Trenerja)")
        st.caption("Samodejna primerjava izbranega obdobja (npr. zadnjih 7 dni) z enako dolgim predhodnim obdobjem (prejšnjih 7 dni) z natančnimi razlikami (+/- % in absolutno).")
        w_start, w_end, w_label = render_period_navigator(nav_key_prefix="coach_weekly", default_preset="Zadnjih 7 dni")

        span_days = max(1, (w_end - w_start).days + 1)
        prev_end = w_start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=span_days - 1)

        def _collect_period_stats(p_start, p_end):
            p_acts = []
            for a in activities_list:
                st_l = a.get("startTimeLocal", "")[:10]
                if st_l:
                    try:
                        d_act = datetime.strptime(st_l, "%Y-%m-%d").date()
                        if p_start <= d_act <= p_end:
                            p_acts.append(a)
                    except Exception:
                        pass
            p_acts.sort(key=lambda x: x.get("startTimeLocal", ""))

            hours = sum((a.get("movingDuration") or a.get("duration") or 0) / 3600.0 for a in p_acts)
            km = sum((a.get("distance") or 0) / 1000.0 for a in p_acts)
            elev = sum((a.get("elevationGain") or 0) for a in p_acts)
            load = sum((a.get("activityTrainingLoad") or a.get("trainingStressScore") or 0) for a in p_acts)

            work_kj_tot = 0.0
            z_sec = {"z12": 0.0, "z3": 0.0, "z4": 0.0, "z5": 0.0}
            ftp_out_ref = float(user_cfg.get("ftp_outdoor", 320))
            ftp_in_ref = float(user_cfg.get("ftp_indoor", 305))
            max_hr_ref = float(user_cfg.get("max_hr", 202))

            sports_cnt = {}
            sports_hrs = {}

            for a in p_acts:
                aid = str(a.get("activityId"))
                dur_s = float(a.get("movingDuration") or a.get("duration") or 0)
                sp = get_sport_category(a.get("activityType", {}).get("typeKey"), a.get("activityName", ""))
                sports_cnt[sp] = sports_cnt.get(sp, 0) + 1
                sports_hrs[sp] = sports_hrs.get(sp, 0.0) + dur_s / 3600.0

                sf_p = raw_data["streams"].get(aid, {})
                summ_p = sf_p.get("summary") or {}
                avg_w = summ_p.get("trueAvgPower") or a.get("avgPower") or a.get("averagePower") or 0
                wk = summ_p.get("work_kJ") or (avg_w * dur_s / 1000.0 if avg_w else 0.0)
                work_kj_tot += float(wk or 0.0)

                # Razdelitev časa po conah intenzivnosti (Z1/Z2 Aerobna baza vs Z3/SST vs Z4 Prag vs Z5+ VO2max)
                p_arr = sf_p.get("power") or []
                h_arr = sf_p.get("heartRate") or []
                ftp_ref = ftp_in_ref if is_indoor_cycling(a) else ftp_out_ref

                if p_arr and any(x > 0 for x in p_arr) and ftp_ref > 0:
                    pa = np.nan_to_num(np.array(p_arr, dtype=float))
                    scale_s = dur_s / max(len(pa), 1)
                    z_sec["z12"] += float(np.sum(pa < 0.75 * ftp_ref)) * scale_s
                    z_sec["z3"] += float(np.sum((pa >= 0.75 * ftp_ref) & (pa < 0.90 * ftp_ref))) * scale_s
                    z_sec["z4"] += float(np.sum((pa >= 0.90 * ftp_ref) & (pa < 1.05 * ftp_ref))) * scale_s
                    z_sec["z5"] += float(np.sum(pa >= 1.05 * ftp_ref)) * scale_s
                elif h_arr and any(x > 0 for x in h_arr) and max_hr_ref > 0:
                    ha = np.nan_to_num(np.array(h_arr, dtype=float))
                    valid_h = ha[ha > 40]
                    if len(valid_h) > 0:
                        scale_s = dur_s / len(valid_h)
                        z_sec["z12"] += float(np.sum(valid_h < 0.78 * max_hr_ref)) * scale_s
                        z_sec["z3"] += float(np.sum((valid_h >= 0.78 * max_hr_ref) & (valid_h < 0.87 * max_hr_ref))) * scale_s
                        z_sec["z4"] += float(np.sum((valid_h >= 0.87 * max_hr_ref) & (valid_h < 0.93 * max_hr_ref))) * scale_s
                        z_sec["z5"] += float(np.sum(valid_h >= 0.93 * max_hr_ref)) * scale_s
                else:
                    # Ocena iz povprečne moči ali HR
                    np_w = summ_p.get("np") or a.get("normPower") or a.get("normalizedPower") or avg_w
                    hr_a = a.get("averageHR") or 0
                    if np_w and ftp_ref > 0:
                        if_r = float(np_w) / ftp_ref
                        if if_r < 0.76:
                            z_sec["z12"] += dur_s * 0.85
                            z_sec["z3"] += dur_s * 0.15
                        elif if_r < 0.88:
                            z_sec["z12"] += dur_s * 0.45
                            z_sec["z3"] += dur_s * 0.45
                            z_sec["z4"] += dur_s * 0.10
                        elif if_r < 1.02:
                            z_sec["z12"] += dur_s * 0.30
                            z_sec["z3"] += dur_s * 0.30
                            z_sec["z4"] += dur_s * 0.35
                            z_sec["z5"] += dur_s * 0.05
                        else:
                            z_sec["z12"] += dur_s * 0.25
                            z_sec["z4"] += dur_s * 0.45
                            z_sec["z5"] += dur_s * 0.30
                    elif hr_a and max_hr_ref > 0:
                        hr_r = float(hr_a) / max_hr_ref
                        if hr_r < 0.78:
                            z_sec["z12"] += dur_s
                        elif hr_r < 0.87:
                            z_sec["z12"] += dur_s * 0.4
                            z_sec["z3"] += dur_s * 0.6
                        elif hr_r < 0.93:
                            z_sec["z3"] += dur_s * 0.4
                            z_sec["z4"] += dur_s * 0.6
                        else:
                            z_sec["z4"] += dur_s * 0.4
                            z_sec["z5"] += dur_s * 0.6
                    else:
                        z_sec["z12"] += dur_s

            df_d = df_daily_all[(df_daily_all["date_obj"] >= p_start) & (df_daily_all["date_obj"] <= p_end)] if not df_daily_all.empty else pd.DataFrame()

            def _mean_col(col_n):
                if df_d.empty or col_n not in df_d.columns:
                    return None
                s = pd.to_numeric(df_d[col_n], errors="coerce").dropna()
                return float(s.mean()) if not s.empty else None

            # Če teža v Garminu ni zabeležena, preveri še jutranji dnevnik
            w_mean = _mean_col("weight_kg")
            if w_mean is None:
                j_weights = []
                for d_k, j_e in (user_journal.get("morning_entries") or {}).items():
                    try:
                        d_jo = datetime.strptime(d_k, "%Y-%m-%d").date()
                        if p_start <= d_jo <= p_end and j_e.get("weighed_today") and j_e.get("weight_kg"):
                            j_weights.append(float(j_e["weight_kg"]))
                    except Exception:
                        pass
                if j_weights:
                    w_mean = float(np.mean(j_weights))

            return {
                "acts": p_acts,
                "count": len(p_acts),
                "hours": hours,
                "km": km,
                "elev": elev,
                "work_kj": work_kj_tot,
                "load": load,
                "z_sec": z_sec,
                "sports_cnt": sports_cnt,
                "sports_hrs": sports_hrs,
                "sleep_h": _mean_col("sleep_hours"),
                "sleep_score": _mean_col("sleep_score"),
                "hrv": _mean_col("hrv_night"),
                "rhr": _mean_col("rhr"),
                "sleep_hr": _mean_col("avg_sleep_hr"),
                "steps": _mean_col("steps"),
                "weight": w_mean,
            }

        cur_st = _collect_period_stats(w_start, w_end)
        prv_st = _collect_period_stats(prev_start, prev_end)

        def _fmt_delta(cur_v, prv_v, unit="", decimals=1, is_int=False):
            if cur_v is None:
                return "--", "--", "--"
            c_s = f"{int(round(cur_v))}{unit}" if is_int else f"{cur_v:.{decimals}f}{unit}"
            if prv_v is None or prv_v == 0:
                p_s = f"{int(round(prv_v))}{unit}" if (prv_v is not None and is_int) else (f"{prv_v:.{decimals}f}{unit}" if prv_v is not None else "--")
                return c_s, p_s, "--"
            p_s = f"{int(round(prv_v))}{unit}" if is_int else f"{prv_v:.{decimals}f}{unit}"
            diff = cur_v - prv_v
            pct = (diff / abs(prv_v)) * 100.0
            d_abs = f"{int(round(diff)):+d}{unit}" if is_int else f"{diff:+.{decimals}f}{unit}"
            return c_s, p_s, f"{d_abs} ({pct:+.1f}%)"

        # KPI kartice z neposredno primerjavo proti prejšnjemu obdobju
        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        c_h, p_h, d_h = _fmt_delta(cur_st["hours"], prv_st["hours"], " h", 1)
        c_k, p_k, d_k = _fmt_delta(cur_st["km"], prv_st["km"], " km", 1)
        c_e, p_e, d_e = _fmt_delta(cur_st["elev"], prv_st["elev"], " m", 0, True)
        c_w, p_w, d_w = _fmt_delta(cur_st["work_kj"], prv_st["work_kj"], " kJ", 0, True)
        c_l, p_l, d_l = _fmt_delta(cur_st["load"], prv_st["load"], "", 0, True)

        kc1.metric("Čas treninga", c_h, d_h if d_h != "--" else f"Prej: {p_h}")
        kc2.metric("Razdalja", c_k, d_k if d_k != "--" else f"Prej: {p_k}")
        kc3.metric("Višinski metri", f"+{c_e}", d_e if d_e != "--" else f"Prej: +{p_e}")
        kc4.metric("Mehansko delo", c_w, d_w if d_w != "--" else f"Prej: {p_w}")
        kc5.metric("TSS / Load", c_l, d_l if d_l != "--" else f"Prej: {p_l}")

        # 1. Tabela volumna in obremenitve
        vol_rows = [
            ("Skupni čas treninga", *_fmt_delta(cur_st["hours"], prv_st["hours"], " h", 1)),
            ("Skupna razdalja", *_fmt_delta(cur_st["km"], prv_st["km"], " km", 1)),
            ("Višinski metri (vzpon)", *_fmt_delta(cur_st["elev"], prv_st["elev"], " m", 0, True)),
            ("Opravljeno mehansko delo", *_fmt_delta(cur_st["work_kj"], prv_st["work_kj"], " kJ", 0, True)),
            ("Trenažna obremenitev (TSS / Load)", *_fmt_delta(cur_st["load"], prv_st["load"], "", 0, True)),
            ("Število opravljenih treningov", *_fmt_delta(cur_st["count"], prv_st["count"], "x", 0, True)),
        ]

        # 2. Razdelitev časa po conah intenzivnosti
        tot_z_cur = max(1.0, sum(cur_st["z_sec"].values()))
        tot_z_prv = max(1.0, sum(prv_st["z_sec"].values()))

        def _fmt_zone_row(z_key, z_label):
            sc = cur_st["z_sec"][z_key]
            sp = prv_st["z_sec"][z_key]
            hc, hp = sc / 3600.0, sp / 3600.0
            pct_c = (sc / tot_z_cur) * 100.0 if sum(cur_st["z_sec"].values()) > 0 else 0.0
            pct_p = (sp / tot_z_prv) * 100.0 if sum(prv_st["z_sec"].values()) > 0 else 0.0
            cur_txt = f"{hc:.1f} h ({int(round(sc/60))} min | {pct_c:.0f}%)"
            prv_txt = f"{hp:.1f} h ({int(round(sp/60))} min | {pct_p:.0f}%)"
            diff_m = int(round((sc - sp) / 60.0))
            diff_pct_pts = pct_c - pct_p
            delta_txt = f"{diff_m:+d} min ({diff_pct_pts:+.1f} odst. t.)"
            return (z_label, cur_txt, prv_txt, delta_txt)

        zone_rows = [
            _fmt_zone_row("z12", "Z1/Z2 Aerobna baza (< 75% FTP / nizka intenzivnost)"),
            _fmt_zone_row("z3", "Z3 Tempo / Sweet Spot (75–90% FTP)"),
            _fmt_zone_row("z4", "Z4 Laktatni prag / Threshold (90–105% FTP)"),
            _fmt_zone_row("z5", "Z5+ VO2max & Anaerobno (> 105% FTP)"),
        ]

        # 3. Povprečje regeneracije v tednu vs. prejšnji teden
        reg_rows = [
            ("Spanje – trajanje (povp./noč)", *_fmt_delta(cur_st["sleep_h"], prv_st["sleep_h"], " h", 2)),
            ("Garmin Sleep Score (0–100)", *_fmt_delta(cur_st["sleep_score"], prv_st["sleep_score"], "", 0, True)),
            ("Nočni HRV (povp.)", *_fmt_delta(cur_st["hrv"], prv_st["hrv"], " ms", 1)),
            ("Jutranji RHR v mirovanju", *_fmt_delta(cur_st["rhr"], prv_st["rhr"], " bpm", 1)),
            ("Povprečni nočni HR med spanjem", *_fmt_delta(cur_st["sleep_hr"], prv_st["sleep_hr"], " bpm", 1)),
            ("Dnevni koraki (povp./dan)", *_fmt_delta(cur_st["steps"], prv_st["steps"], "", 0, True)),
            ("Povprečna telesna teža", *_fmt_delta(cur_st["weight"], prv_st["weight"], " kg", 1)),
        ]

        # 4. Seznam vseh treningov v izbranem obdobju
        act_md_lines = [
            "| Datum | Šport | Trening | Čas | Razdalja | Vzpon | Moč / Tempo | Avg HR | Delo / Load | Temp | RPE |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        act_ui_rows = []
        for a in cur_st["acts"]:
            aid_w = str(a.get("activityId"))
            d_w = a.get("startTimeLocal", "")[:10]
            nm_w = a.get("activityName", "Trening")
            sp_w = get_sport_category(a.get("activityType", {}).get("typeKey"), nm_w)
            dur_w_s = int(a.get("movingDuration") or a.get("duration") or 0)
            dur_w_str = str(timedelta(seconds=dur_w_s))
            dist_w_km = (a.get("distance") or 0) / 1000.0
            elev_w_m = int(round(a.get("elevationGain") or 0))
            hr_w = int(round(a.get("averageHR"))) if a.get("averageHR") else "--"
            ld_w = int(round(a.get("activityTrainingLoad") or a.get("trainingStressScore") or 0))
            sf_w = raw_data["streams"].get(aid_w, {})
            sm_w = sf_w.get("summary") or {}
            sp_obj_w = raw_data["splits"].get(aid_w, {})
            t_avg_w, _, _, _ = extract_activity_temp_summary(a, sf_w, sp_obj_w)
            t_w_str = f"{t_avg_w:.1f} °C" if t_avg_w is not None else "--"
            rpe_w = (user_journal.get("workout_entries") or {}).get(aid_w, {}).get("rpe", "--")

            avg_p_w = sm_w.get("trueAvgPower") or a.get("avgPower") or a.get("averagePower")
            np_p_w = sm_w.get("np") or a.get("normPower") or a.get("normalizedPower") or avg_p_w
            wk_w = sm_w.get("work_kJ") or (round(avg_p_w * dur_w_s / 1000.0) if avg_p_w else None)

            if avg_p_w:
                pwr_pace_str = f"{int(round(avg_p_w))} W (NP {int(round(np_p_w))} W)"
            elif dist_w_km > 0 and dur_w_s > 0:
                pwr_pace_str = f"{fmt_pace(dur_w_s / dist_w_km)} /km"
            else:
                pwr_pace_str = "--"

            wk_load_str = f"{int(round(wk_w))} kJ / {ld_w}" if wk_w else f"Load {ld_w}"
            act_md_lines.append(
                f"| {d_w} | {sp_w} | {nm_w} | {dur_w_str} | {dist_w_km:.1f} km | +{elev_w_m} m | {pwr_pace_str} | {hr_w} bpm | {wk_load_str} | {t_w_str} | {rpe_w} |"
            )
            act_ui_rows.append({
                "Datum": d_w,
                "Šport": sp_w,
                "Trening": nm_w,
                "Čas": dur_w_str,
                "Razdalja": f"{dist_w_km:.1f} km",
                "Vzpon": f"+{elev_w_m} m",
                "Moč / Tempo": pwr_pace_str,
                "Avg HR": f"{hr_w} bpm" if hr_w != "--" else "--",
                "Delo / Load": wk_load_str,
                "Temp (°C)": t_w_str,
                "RPE": rpe_w,
            })

        acts_table_md = "\n".join(act_md_lines) if cur_st["acts"] else "V izbranem obdobju ni zabeleženih treningov."

        # Vizualni prikaz primerjalnih tabel v UI
        col_wk_t1, col_wk_t2 = st.columns(2)
        with col_wk_t1:
            st.markdown(f"##### 1. Volumen in obremenitev ({w_start.strftime('%d.%m.')}–{w_end.strftime('%d.%m.')} vs. {prev_start.strftime('%d.%m.')}–{prev_end.strftime('%d.%m.')})")
            st.dataframe(pd.DataFrame(vol_rows, columns=["Metrika", "Izbrano obdobje", "Prejšnje obdobje", "Razlika (Δ)"]), use_container_width=True, hide_index=True)
            st.markdown("##### 2. Razdelitev časa po conah intenzivnosti")
            st.dataframe(pd.DataFrame(zone_rows, columns=["Cona intenzivnosti", "Izbrano obdobje", "Prejšnje obdobje", "Razlika (Δ)"]), use_container_width=True, hide_index=True)
        with col_wk_t2:
            st.markdown("##### 3. Povprečje regeneracije in fiziologije")
            st.dataframe(pd.DataFrame(reg_rows, columns=["Parameter regeneracije", "Izbrano obdobje", "Prejšnje obdobje", "Razlika (Δ)"]), use_container_width=True, hide_index=True)

        if act_ui_rows:
            st.markdown(f"##### 4. Seznam opravljenih treningov v obdobju ({len(act_ui_rows)} enot)")
            st.dataframe(pd.DataFrame(act_ui_rows), use_container_width=True, hide_index=True)

        vol_md_table = "| Metrika | Izbrano obdobje | Prejšnje obdobje | Razlika (Δ) |\n|---|---|---|---|\n" + "\n".join(
            [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in vol_rows]
        )
        zone_md_table = "| Cona intenzivnosti | Izbrano obdobje | Prejšnje obdobje | Razlika (Δ) |\n|---|---|---|---|\n" + "\n".join(
            [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in zone_rows]
        )
        reg_md_table = "| Parameter regeneracije | Izbrano obdobje | Prejšnje obdobje | Razlika (Δ) |\n|---|---|---|---|\n" + "\n".join(
            [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in reg_rows]
        )
        sports_text = "\n".join([f"- **{k}:** {v:.1f} h ({cur_st['sports_cnt'][k]}x)" for k, v in cur_st["sports_hrs"].items()]) if cur_st["sports_hrs"] else "- Ni zabeleženih aktivnosti."

        weekly_summary_prompt = f"""### 📊 PRIMERJALNI POVZETEK TEDNA / BLOKA ({w_start.strftime('%d. %m. %Y')} – {w_end.strftime('%d. %m. %Y')} vs. {prev_start.strftime('%d. %m. %Y')} – {prev_end.strftime('%d. %m. %Y')})
*Aktivni profil:* Zunanji FTP = {user_cfg['ftp_outdoor']} W | Trenažer FTP = {user_cfg['ftp_indoor']} W | Teža = {user_cfg['weight_kg']} kg | Cilji: {goals_inline(user_cfg)}

**1. PRIMERJAVA VOLUMNA, DELA IN OBREMENITVE ({span_days} dni vs. prejšnjih {span_days} dni)**
{vol_md_table}

**2. RAZDELITEV ČASA PO CONAH INTENZIVNOSTI (Z1/Z2 Baza vs. Z3/SST vs. Z4 Prag vs. Z5 VO2max)**
{zone_md_table}

**3. RAZDELITEV PO ŠPORTIH V IZBRANEM OBDOBJU**
{sports_text}

**4. POVPREČJE REGENERACIJE IN BIOMETRIJE (Izbrano obdobje vs. prejšnje obdobje)**
{reg_md_table}

**5. SEZNAM VSEH OPRAVLJENIH TRENINGOV V OBDOBJU**
{acts_table_md}
"""
        weekly_summary_prompt = show_prompt(weekly_summary_prompt, Q_WEEKLY, "Kopiraj primerjalni tedenski povzetek za AI trenerja:", "weekly")
        if st.button("🤖 Vprašaj Gemini AI Trenerja o obdobju", key=f"gemini_btn_wk_{w_label}"):
            with st.spinner("Gemini AI trener analizira celotno trenažno obdobje in primerjavo..."):
                ans_wk = call_gemini_coach(weekly_summary_prompt, user_cfg.get("gemini_api_key"))
                st.session_state[f"ans_gemini_wk_{w_label}"] = ans_wk
                st.session_state[f"ans_gemini_wk_{w_label}_model"] = st.session_state.get("_last_coach_model_used", "")
                st.session_state[f"ans_gemini_wk_{w_label}_tokens"] = st.session_state.get("_last_coach_token_usage")
        if st.session_state.get(f"ans_gemini_wk_{w_label}"):
            st.markdown("#### 💬 Odgovor Gemini AI Trenerja:")
            st.info(st.session_state[f"ans_gemini_wk_{w_label}"])
            _wk_used = st.session_state.get(f"ans_gemini_wk_{w_label}_model", "")
            if _wk_used:
                st.caption(f"🤖 Odgovor ustvarjen z modelom: `{_wk_used}`")
            _wk_tok = st.session_state.get(f"ans_gemini_wk_{w_label}_tokens")
            _wk_tok_cap = ai_coach_pro.format_token_usage_caption(_wk_tok) if hasattr(ai_coach_pro, "format_token_usage_caption") else ""
            if _wk_tok_cap:
                st.caption(_wk_tok_cap)

# ==============================================================================
# TAB 2: KRIVULJA MOČI PO ČASU & UPRAVLJANJE SEZON (INTERVALS.ICU LAYOUT)
# ==============================================================================
with tab_mmp:
    st.header("⚡ Krivulja moči po času (Mean Maximal Power Curve – Intervals.icu)")
    st.caption("Side-by-side primerjava sezon in obdobij z natančnim izračunom eFTP, Critical Power (W'), MAP in VO2max.")

    # ----------------------------------------------------
    # UPRAVLJALEC SEZON (EXPANDER)
    # ----------------------------------------------------
    with st.expander("➕ Dodaj novo sezono ali obdobje", expanded=False):
        with st.form("add_season_form", clear_on_submit=True):
            n_name = st.text_input("Ime sezone / obdobja", placeholder="npr. Zimska baza 2026/27, Priprave Maj 2027, Vrhunec Rogla 2026")
            fc1, fc2 = st.columns(2)
            n_from = fc1.date_input("Datum od", value=today_date - timedelta(days=90), format="DD.MM.YYYY")
            n_to = fc2.date_input("Datum do", value=today_date, format="DD.MM.YYYY")
            fc3, fc4 = st.columns(2)
            n_wt = fc3.number_input("Referenčna teža v tem obdobju (kg)", min_value=30.0, max_value=200.0, value=64.0, step=0.1)
            n_col = fc4.color_picker("Barva krivulje", value="#0ea5e9")
            n_submit = st.form_submit_button("➕ Dodaj in shrani sezono")
        if n_submit:
            _nm = (n_name or "").strip()
            _existing_names = [s.get("name") for s in user_cfg.get("seasons", {}).values()]
            if not _nm:
                st.error("Vnesite ime sezone / obdobja.")
            elif n_to < n_from:
                st.error("Datum 'do' mora biti za datumom 'od'.")
            elif _nm in _existing_names:
                st.error("Sezona s tem imenom že obstaja.")
            else:
                _slug = "".join(ch if ch.isalnum() else "_" for ch in _nm.lower())[:30]
                _key = f"custom_{_slug}_{int(time.time())}"
                user_cfg.setdefault("seasons", {})[_key] = {
                    "name": _nm, "type": "fixed",
                    "start": n_from.strftime("%Y-%m-%d"), "end": n_to.strftime("%Y-%m-%d"),
                    "color": n_col, "weight_kg": float(n_wt)}
                save_user_config(user_cfg)
                _cur = list(st.session_state.get("mmp_active_seasons_ms", []))
                st.session_state["mmp_active_seasons_ms"] = _cur + [_nm]
                st.success(f"Sezona »{_nm}« dodana in izbrana.")
                st.rerun()

    with st.expander("📅 Nastavi in uredi sezone (Upravljanje datumov, barv in teže)"):
        st.write("Tukaj lahko prilagodite datume, imena, barve in referenčne teže svojih sezon. Spremembe se trajno shranijo v `config.json`.")
        seasons_dict = user_cfg.get("seasons", {})
        
        # Prikaz seznama sezon v vrsticah
        updated_seasons = {}
        for s_key, s_data in seasons_dict.items():
            sc1, sc2, sc3, sc4, sc5, sc6 = st.columns([1.5, 1.2, 1.2, 1, 1, 0.8])
            with sc1:
                s_name = st.text_input(f"Ime ({s_key}):", value=s_data.get("name", s_key), key=f"s_name_{s_key}")
            with sc2:
                s_start = st.text_input(f"Začetek (YYYY-MM-DD):", value=s_data.get("start", ""), key=f"s_start_{s_key}") if s_data.get("type") != "rolling_42" else "Zadnjih 42 dni"
            with sc3:
                s_end = st.text_input(f"Konec (YYYY-MM-DD):", value=s_data.get("end", ""), key=f"s_end_{s_key}") if s_data.get("type") != "rolling_42" else "Danes"
            with sc4:
                s_col = st.color_picker(f"Barva:", value=s_data.get("color", "#7e57c2"), key=f"s_col_{s_key}")
            with sc5:
                s_wt = st.number_input(f"Teža (kg):", value=float(s_data.get("weight_kg", user_cfg.get("weight_kg", 65.5))), step=0.1, key=f"s_wt_{s_key}")

            with sc6:
                s_del = st.checkbox("🗑️ Izbriši", key=f"s_del_{s_key}")
            if s_del:
                continue
            updated_seasons[s_key] = {
                "name": s_name,
                "type": s_data.get("type", "fixed"),
                "start": s_start if s_data.get("type") != "rolling_42" else "",
                "end": s_end if s_data.get("type") != "rolling_42" else "",
                "color": s_col,
                "weight_kg": s_wt
            }

        if st.button("💾 Shrani posodobljene sezone v nastavitve"):
            user_cfg["seasons"] = updated_seasons
            save_user_config(user_cfg)
            st.success("Sezone uspešno posodobljene!")
            st.rerun()

    # ----------------------------------------------------
    # KONTROLE NAD GRAFOM
    # ----------------------------------------------------
    seasons_available = user_cfg.get("seasons", {})
    ctl_mmp1, ctl_mmp2, ctl_mmp3 = st.columns([2, 1, 1])
    with ctl_mmp1:
        season_names_list = [s["name"] for s in seasons_available.values()]
        _cur_sel = st.session_state.get("mmp_active_seasons_ms")
        if _cur_sel is None:
            _cur_sel = season_names_list[:4] if len(season_names_list) >= 4 else season_names_list
        st.session_state["mmp_active_seasons_ms"] = [x for x in _cur_sel if x in season_names_list]
        active_seasons_chosen = st.multiselect(
            "Izberi sezone za prikaz in primerjavo:",
            season_names_list,
            key="mmp_active_seasons_ms"
        )
    with ctl_mmp2:
        y_axis_mode = st.radio("Enote Y-osi:", ["W (Vati)", "W/kg (Vati na kg)"], horizontal=True, key="mmp_y_mode")
    with ctl_mmp3:
        env_filter = st.selectbox("Filter okolja:", ["Vse vožnje", "Samo zunaj (Cesta)", "Samo trenažer (Indoor)"], key="mmp_env_f")

    x_scale_choice = st.radio(
        "Skala časovne osi X:",
        ["Poudarek na klancih in daljših intervalih (Korenska / Prilagojena skala - Priporočeno)",
         "Samo ključni vzponi (1 min – 90 min, Linearna skala v minutah)",
         "Klasična logaritemska (1s – 4h)"],
        index=0, key="mmp_x_scale")
    x_mode = 1 if x_scale_choice.startswith("Poudarek") else (2 if x_scale_choice.startswith("Samo") else 3)

    # ----------------------------------------------------
    # IZRAČUN KRIVULJ MOČI (ENVELOPE) ZA IZBRANE SEZONE
    # ----------------------------------------------------
    mmp_all = raw_data.get("mmp", {})

    def calculate_season_envelope(s_info):
        s_type = s_info.get("type", "fixed")
        if s_type == "rolling_42":
            s_dt = today_date - timedelta(days=42)
            e_dt = today_date
        else:
            try:
                s_dt = datetime.strptime(s_info.get("start", "2020-01-01"), "%Y-%m-%d").date()
                e_dt = datetime.strptime(s_info.get("end", "2030-01-01"), "%Y-%m-%d").date()
            except Exception:
                s_dt, e_dt = today_date - timedelta(days=42), today_date

        envelope = {d: {"w": 0, "act_name": "", "date": "", "aid": ""} for d in STANDARD_MMP_DURATIONS}

        # 1. Natančni FIT streams iz data/mmp/
        for aid, m_obj in mmp_all.items():
            st_l = m_obj.get("startTimeLocal", "")[:10]
            try:
                act_dt = datetime.strptime(st_l, "%Y-%m-%d").date()
            except Exception:
                continue

            if not (s_dt <= act_dt <= e_dt):
                continue

            is_ind = m_obj.get("is_indoor", False)
            if env_filter == "Samo zunaj (Cesta)" and is_ind:
                continue
            if env_filter == "Samo trenažer (Indoor)" and not is_ind:
                continue

            act_mmp = m_obj.get("mmp", {})
            known_d = sorted([int(k) for k, v in act_mmp.items() if v and float(v) > 0])
            for d in STANDARD_MMP_DURATIONS:
                w = act_mmp.get(str(d)) or 0
                if not w and len(known_d) >= 2 and known_d[0] <= d <= known_d[-1]:
                    w = int(round(math.exp(float(np.interp(
                        math.log(d),
                        [math.log(k) for k in known_d],
                        [math.log(float(act_mmp[str(k)])) for k in known_d]
                    )))))
                if w > envelope[d]["w"]:
                    envelope[d] = {
                        "w": int(round(w)),
                        "act_name": m_obj.get("activityName", "Kolesarjenje"),
                        "date": st_l,
                        "aid": aid
                    }

        # 2. Za vožnje brez FIT stream-a oceni moč iz summary podatkov (activities.json)
        for act in activities_list:
            aid = str(act.get("activityId"))
            if aid in mmp_all:
                continue # Že obdelano iz 1-sekundnih podatkov

            st_l = act.get("startTimeLocal", "")[:10]
            try:
                act_dt = datetime.strptime(st_l, "%Y-%m-%d").date()
            except Exception:
                continue

            if not (s_dt <= act_dt <= e_dt):
                continue

            sp_cat = get_sport_category(act.get("activityType", {}).get("typeKey"), act.get("activityName", ""))
            if sp_cat not in ["Cestno kolo", "Trenažer / Indoor"]:
                continue

            is_ind = is_indoor_cycling(act)
            if env_filter == "Samo zunaj (Cesta)" and is_ind:
                continue
            if env_filter == "Samo trenažer (Indoor)" and not is_ind:
                continue

            max_p = act.get("maxPower") or 0
            norm_p = act.get("normPower") or act.get("normalizedPower") or act.get("avgPower") or 0
            avg_p = act.get("avgPower") or act.get("averagePower") or 0
            dur_sec = act.get("movingDuration") or act.get("duration") or 0

            if max_p <= 0 and norm_p <= 0:
                continue

            for d in STANDARD_MMP_DURATIONS:
                if d > dur_sec and dur_sec > 0:
                    continue
                if d <= 5:
                    est_w = max_p if max_p > 0 else norm_p * 1.3
                elif d <= 180:
                    est_w = norm_p + (max_p - norm_p) * np.exp(-(d - 5) / 60.0) if max_p > norm_p else norm_p * 1.15
                elif d <= 1200:
                    est_w = norm_p * ((1200.0 / d) ** 0.04)
                else:
                    est_w = avg_p + (norm_p - avg_p) * np.exp(-(d - 1200) / 1800.0) if norm_p > avg_p else avg_p

                est_w = int(round(est_w))
                if est_w > envelope[d]["w"]:
                    envelope[d] = {
                        "w": est_w,
                        "act_name": act.get("activityName", "Kolesarjenje"),
                        "date": st_l,
                        "aid": aid
                    }

        return envelope

    # Zgradi modele za izbrane sezone (s hitrim predpomnjenjem v st.session_state)
    _mmp_env_cache_key = (
        f"_cached_mmp_seasons_{current_user}_{id(raw_data)}_{len(mmp_all)}_{len(activities_list)}_"
        f"{env_filter}_{tuple(active_seasons_chosen)}_{json.dumps(seasons_available, sort_keys=True)}"
    )
    if _mmp_env_cache_key in st.session_state:
        active_seasons_data = st.session_state[_mmp_env_cache_key]
    else:
        active_seasons_data = {}
        for s_key, s_info in seasons_available.items():
            if s_info["name"] in active_seasons_chosen:
                env = calculate_season_envelope(s_info)
                s_weight = float(s_info.get("weight_kg", user_cfg.get("weight_kg", 65.5)))

                # Izračun CP in W' (Critical Power linearni model)
                # Uporabimo 3m (180s), 5m (300s), 10m (600s), 20m (1200s)
                t_pts = []
                p_pts = []
                for dur_pt in [180, 300, 600, 1200]:
                    if env[dur_pt]["w"] > 0:
                        t_pts.append(dur_pt)
                        p_pts.append(env[dur_pt]["w"])

                if len(t_pts) >= 2:
                    t_arr = np.array(t_pts, dtype=float)
                    w_arr = np.array(p_pts, dtype=float) * t_arr
                    slope, intercept = np.polyfit(t_arr, w_arr, 1)
                    cp_calc = max(150, round(slope))
                    w_prime_calc = max(5000, round(intercept))
                    p20_est = env[1200]["w"]
                    eftp_calc = max(cp_calc, round(p20_est * 0.95)) if p20_est > 0 else cp_calc
                else:
                    p20 = env[1200]["w"]
                    eftp_calc = round(p20 * 0.95) if p20 > 0 else int(user_cfg["ftp_outdoor"])
                    w_prime_calc = 18000

                map_w = max(env[180]["w"], env[300]["w"])
                if map_w <= 0:
                    map_w = round(eftp_calc * 1.18)

                tte_min = min(65, max(35, round(40 + (w_prime_calc / 1000.0 - 15) * 0.8)))
                vo2max_est = round(10.8 * (map_w / s_weight) + 7.0, 1)

                active_seasons_data[s_info["name"]] = {
                    "key": s_key,
                    "info": s_info,
                    "color": s_info.get("color", "#3b82f6"),
                    "weight_kg": s_weight,
                    "envelope": env,
                    "eftp": eftp_calc,
                    "w_prime": w_prime_calc,
                    "map": map_w,
                    "tte": tte_min,
                    "vo2max": vo2max_est
                }
        st.session_state[_mmp_env_cache_key] = active_seasons_data

    # ----------------------------------------------------
    # SIDE-BY-SIDE POSTAVITEV (60% LEVI GRAF | 40% DESNA TABELA)
    # ----------------------------------------------------
    col_chart_left, col_table_right = st.columns([6, 4])

    with col_chart_left:
        fig_mmp = go.Figure()

        # X-os: tri izbirne skale (glej radio gumb nad grafom)
        _bp_s = np.log([1, 60, 600, 3600, 14400, 28800])
        _bp_x = [0.0, 0.18, 0.50, 0.85, 1.0, 1.10]

        def _xmap(d):
            if x_mode == 1:
                return float(np.interp(np.log(max(d, 1)), _bp_s, _bp_x))
            if x_mode == 2:
                return d / 60.0
            return d

        if x_mode == 1:
            tick_secs = [5, 15, 30, 60, 120, 180, 300, 480, 600, 900, 1200, 1800, 2400, 3600, 5400, 7200, 10800]
            tick_vals = [_xmap(s_) for s_ in tick_secs]
            tick_texts = [DURATION_LABELS.get(s_, f"{s_}s") for s_ in tick_secs]
            x_range = [_xmap(5) - 0.01, _xmap(10800) + 0.01]
            x_title = "Trajanje napora (prilagojena skala: klanci 1–60 min poudarjeni)"
            x_type = "linear"
        elif x_mode == 2:
            tick_mins = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 90]
            tick_vals = tick_mins
            tick_texts = [f"{m_} min" for m_ in tick_mins]
            x_range = [1, 90]
            x_title = "Trajanje napora (minute, linearna skala)"
            x_type = "linear"
        else:
            tick_vals = [5, 15, 30, 60, 120, 180, 300, 600, 1200, 1800, 2400, 3600, 5400, 7200, 10800, 14400]
            tick_texts = [DURATION_LABELS.get(t, f"{t}s") for t in tick_vals]
            x_range = [np.log10(5), np.log10(14400)]
            x_title = "Trajanje napora (logaritemska os)"
            x_type = "log"

        for s_name, s_calc in active_seasons_data.items():
            env = s_calc["envelope"]
            s_col = s_calc["color"]
            s_wt = s_calc["weight_kg"]
            s_scale = (1.0 / s_wt) if "W/kg" in y_axis_mode else 1.0

            x_pts, y_pts, hover_texts = [], [], []
            for d in STANDARD_MMP_DURATIONS:
                info = env[d]
                if x_mode == 2 and (d < 60 or d > 5400):
                    continue
                if x_mode == 1 and (d < 5 or d > 14400):
                    continue
                if x_mode == 3 and d < 5:
                    continue
                if info["w"] > 0:
                    x_pts.append(_xmap(d))
                    y_val = round(info["w"] * s_scale, 2 if "W/kg" in y_axis_mode else 0)
                    y_pts.append(y_val)
                    wpk = round(info["w"] / s_wt, 2)
                    lbl = DURATION_LABELS.get(d, f"{d}s")
                    hover_texts.append(f"<b>{s_name}</b><br>{lbl}: <b>{info['w']} W</b> ({wpk} W/kg)<br>Datum: {info['date']}<br>{info['act_name']}")

            if x_pts:
                fig_mmp.add_trace(go.Scatter(
                    x=x_pts,
                    y=y_pts,
                    mode="lines+markers",
                    marker=dict(size=4),
                    name=s_name,
                    line=dict(color=s_col, width=2.8),
                    hovertext=hover_texts,
                    hoverinfo="text"
                ))

                # eFTP referenčna črta za to sezono
                eftp_y = s_calc["eftp"] * s_scale
                fig_mmp.add_hline(
                    y=eftp_y,
                    line_dash="dot",
                    line_color=s_col,
                    opacity=0.6,
                    annotation_text=f"{s_name} eFTP ({s_calc['eftp']} W)",
                    annotation_position="bottom right"
                )

        y_title = "Moč (W/kg)" if "W/kg" in y_axis_mode else "Moč (W)"
        fig_mmp.update_layout(
            title=f"Krivulja moči po času (MMP) – {y_axis_mode} [Filter: {env_filter}]",
            xaxis=dict(
                title=x_title,
                type=x_type,
                tickvals=tick_vals,
                ticktext=tick_texts,
                range=x_range,
                gridcolor="#334155"
            ),
            yaxis=dict(
                title=y_title,
                gridcolor="#334155"
            ),
            dragmode="zoom",
            hovermode="closest",
            height=540,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_mmp, use_container_width=True, config={"scrollZoom": True}, key="mmp_main_chart")

        # Season Legend Summary spodaj (točno po Intervals.icu vzoru)
        st.markdown("#### 📌 Povzetek sezone (eFTP & Critical Power model)")
        legend_cols = st.columns(len(active_seasons_data)) if active_seasons_data else [st.container()]
        for idx, (s_name, s_calc) in enumerate(active_seasons_data.items()):
            col_target = legend_cols[idx] if idx < len(legend_cols) else st
            with col_target:
                wpk_eftp = round(s_calc['eftp'] / s_calc['weight_kg'], 2)
                st.metric(
                    label=f"{s_name} – eFTP",
                    value=f"{s_calc['eftp']} W",
                    delta=f"{wpk_eftp} W/kg"
                )
                st.caption(f"**W':** {round(s_calc['w_prime']/1000, 1)} kJ | **MAP:** {s_calc['map']} W | **VO2max:** {s_calc['vo2max']}")

    # ----------------------------------------------------
    # DESNA TABELA: PRIMERJAVA NAJVIŠJIH MOČI (INTERVALS.ICU STYLE)
    # ----------------------------------------------------
    with col_table_right:
        st.subheader("📊 Primerjava naporov po sezonah")

        compare_durations = [1, 5, 15, 30, 60, 90, 120, 180, 240, 300, 360, 480, 600, 720, 900, 1200, 1800, 2400, 2700, 3600, 5400, 7200, 10800]

        csv_rows = []
        md_table_lines = []
        md_header = "| Čas | " + " | ".join([f"{s} (W) | W/kg" for s in active_seasons_data.keys()]) + " |"
        md_sep = "|---|" + "|".join(["---|---" for _ in active_seasons_data]) + "|"
        md_table_lines.extend([md_header, md_sep])

        df_table_rows = []
        for d in compare_durations:
            d_lbl = DURATION_LABELS.get(d, f"{d}s")
            row_powers = {s_name: s_calc["envelope"][d]["w"] for s_name, s_calc in active_seasons_data.items()}
            max_row_w = max(row_powers.values()) if row_powers and max(row_powers.values()) > 0 else 0

            csv_row = {"Trajanje": d_lbl}
            display_row = {"Čas": d_lbl}
            md_row_parts = [d_lbl]

            for s_name, s_calc in active_seasons_data.items():
                w_val = s_calc["envelope"][d]["w"]
                wpk_val = round(w_val / s_calc["weight_kg"], 2) if w_val > 0 else 0

                is_best = (w_val == max_row_w and w_val > 0)
                if is_best:
                    md_w_str = f"**{w_val} W**"
                    md_wpk_str = f"**{wpk_val:.2f}**"
                else:
                    md_w_str = f"{w_val} W" if w_val > 0 else "--"
                    md_wpk_str = f"{wpk_val:.2f}" if wpk_val > 0 else "--"

                w_display = f"{w_val} W" if w_val > 0 else "--"
                wpk_display = f"{wpk_val:.2f}" if wpk_val > 0 else "--"

                display_row[f"{s_name} (W)"] = w_display
                display_row[f"{s_name} (W/kg)"] = wpk_display

                csv_row[f"{s_name}_W"] = w_val
                csv_row[f"{s_name}_Wpkg"] = wpk_val
                md_row_parts.extend([md_w_str, md_wpk_str])

            df_table_rows.append(display_row)
            csv_rows.append(csv_row)
            md_table_lines.append("| " + " | ".join(md_row_parts) + " |")

        # Prikaz glavne tabele z native st.dataframe
        st.dataframe(pd.DataFrame(df_table_rows), use_container_width=True, hide_index=True, height=440)
        with st.expander("🔍 Celotna gostota vseh trajanj (1 s – 240 min)"):
            dense_mmp_rows = []
            for d_all in STANDARD_MMP_DURATIONS:
                r_all = {"Čas": DURATION_LABELS.get(d_all, f"{d_all}s")}
                for s_name, s_calc in active_seasons_data.items():
                    wv = s_calc["envelope"][d_all]["w"]
                    wpkv = round(wv / s_calc["weight_kg"], 2) if wv > 0 else 0
                    r_all[f"{s_name} (W)"] = f"{wv} W" if wv > 0 else "--"
                    r_all[f"{s_name} (W/kg)"] = f"{wpkv:.2f}" if wpkv > 0 else "--"
                dense_mmp_rows.append(r_all)
            st.dataframe(pd.DataFrame(dense_mmp_rows), use_container_width=True, hide_index=True)

        # Fiziološki model spodaj
        st.markdown("##### Fiziološki parametri (Intervals.icu model)")
        phys_rows = []
        md_eftp = ["eFTP"]
        md_wprime = ["W' (kJ)"]
        md_map = ["MAP"]
        md_tte = ["TTE (min)"]
        md_vo2 = ["VO2max"]

        row_eftp = {"Metrika": "eFTP"}
        row_wprime = {"Metrika": "W' (kJ)"}
        row_map = {"Metrika": "MAP"}
        row_tte = {"Metrika": "TTE"}
        row_vo2 = {"Metrika": "VO2max"}

        for s_name, s_calc in active_seasons_data.items():
            wpk_e = round(s_calc["eftp"] / s_calc["weight_kg"], 2)
            row_eftp[s_name] = f"{s_calc['eftp']} W ({wpk_e:.2f} W/kg)"
            md_eftp.extend([f"{s_calc['eftp']} W", f"{wpk_e:.2f}"])

            w_kj = round(s_calc["w_prime"] / 1000.0, 1)
            row_wprime[s_name] = f"{w_kj} kJ"
            md_wprime.extend([f"{w_kj} kJ", "--"])

            wpk_m = round(s_calc["map"] / s_calc["weight_kg"], 2)
            row_map[s_name] = f"{s_calc['map']} W ({wpk_m:.2f})"
            md_map.extend([f"{s_calc['map']} W", f"{wpk_m:.2f}"])

            row_tte[s_name] = f"{s_calc['tte']} min"
            md_tte.extend([f"{s_calc['tte']} min", "--"])

            row_vo2[s_name] = f"{s_calc['vo2max']} ml/kg"
            md_vo2.extend([f"{s_calc['vo2max']} ml/kg", "--"])

        phys_rows.extend([row_eftp, row_wprime, row_map, row_tte, row_vo2])
        st.dataframe(pd.DataFrame(phys_rows), use_container_width=True, hide_index=True)

        md_table_lines.append("| " + " | ".join(md_eftp) + " |")
        md_table_lines.append("| " + " | ".join(md_wprime) + " |")
        md_table_lines.append("| " + " | ".join(md_map) + " |")
        md_table_lines.append("| " + " | ".join(md_tte) + " |")
        md_table_lines.append("| " + " | ".join(md_vo2) + " |")

    # Spodnji vsebnik čez polno širino za izvoz podatkov in Markdown za AI trenerja
    st.markdown("---")
    c_exp_full1, c_exp_full2 = st.columns([1, 3])
    df_csv = pd.DataFrame(csv_rows)
    csv_bytes = df_csv.to_csv(index=False).encode("utf-8")
    with c_exp_full1:
        st.markdown("##### 📥 Izvoz podatkov")
        st.download_button(
            "📥 Prenesi tabelo sezon (CSV)",
            data=csv_bytes,
            file_name="kolesarske_sezone_mmp.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c_exp_full2:
        st.markdown("##### 📋 Markdown tabela za AI trenerja")
        copy_md_text = "### ⚡ INTERVALS.ICU PRIMERJAVA SEZON IN NAJVIŠJIH MOČI\n" + "\n".join(md_table_lines)
        st.code(copy_md_text, language="markdown")

# ==============================================================================
# TAB 3: ANALIZA TRENINGOV & PMC (FITNESS, FATIGUE, FORM & CELOTNI NADZOR)
# ==============================================================================
with tab_analytics:
    st.header("📈 Analiza Treningov, PMC in Zmogljivosti")
    
    # Enotna navigacija za celoten zavihek
    ana_start, ana_end, ana_label = render_period_navigator(nav_key_prefix="analytics_tab", default_preset="Zadnjih 90 dni")

    ana_acts = []
    for a in activities_list:
        st_l = a.get("startTimeLocal", "")[:10]
        if st_l:
            try:
                d_a = datetime.strptime(st_l, "%Y-%m-%d").date()
                if ana_start <= d_a <= ana_end:
                    ana_acts.append(a)
            except Exception:
                pass

    st.markdown("---")

    # ----------------------------------------------------
    # 1. PMC: PERFORMANCE MANAGEMENT CHART (CTL, ATL, TSB)
    # ----------------------------------------------------
    st.subheader("1. PMC Graf: Forma, Utrujenost in Svežina (CTL, ATL, TSB)")
    st.caption("Fitness (CTL - 42d drseče povprečje), Fatigue (ATL - 7d drseče povprečje), Form (TSB = CTL - ATL).")

    all_acts_sorted = sorted(activities_list, key=lambda x: x.get("startTimeLocal", ""))
    if all_acts_sorted:
        first_act_date = datetime.strptime(all_acts_sorted[0].get("startTimeLocal", "")[:10], "%Y-%m-%d").date()
        date_span = (today_date - first_act_date).days + 1
        daily_tss_map = {}

        for a in all_acts_sorted:
            d_k = a.get("startTimeLocal", "")[:10]
            tss_val = a.get("trainingStressScore") or a.get("activityTrainingLoad")
            if not tss_val:
                m_sec = a.get("movingDuration") or a.get("duration") or 0
                np_val = a.get("normPower") or a.get("avgPower") or 0
                if np_val and user_cfg["ftp_outdoor"] > 0:
                    int_f = np_val / user_cfg["ftp_outdoor"]
                    tss_val = (m_sec * np_val * int_f) / (user_cfg["ftp_outdoor"] * 36.0)
                elif get_sport_category(a.get("activityType", {}).get("typeKey"), a.get("activityName", "")) == "Tek":
                    _rl_pmc = compute_run_load(a, None, user_cfg)
                    tss_val = _rl_pmc.get("rtss") or 0
            daily_tss_map[d_k] = daily_tss_map.get(d_k, 0) + (tss_val or 0)

        pmc_dates = [first_act_date + timedelta(days=i) for i in range(date_span)]
        ctl_list, atl_list, tsb_list = [], [], []
        curr_ctl = 0.0
        curr_atl = 0.0

        for d in pmc_dates:
            d_str = d.strftime("%Y-%m-%d")
            tss_today = daily_tss_map.get(d_str, 0.0)
            curr_ctl = curr_ctl + (tss_today - curr_ctl) / 42.0
            curr_atl = curr_atl + (tss_today - curr_atl) / 7.0
            tsb = curr_ctl - curr_atl
            ctl_list.append(round(curr_ctl, 1))
            atl_list.append(round(curr_atl, 1))
            tsb_list.append(round(tsb, 1))

        df_pmc = pd.DataFrame({
            "Datum": [d.strftime("%Y-%m-%d") for d in pmc_dates],
            "date_obj": pmc_dates,
            "CTL": ctl_list,
            "ATL": atl_list,
            "TSB": tsb_list
        })

        df_pmc_view = df_pmc[(df_pmc["date_obj"] >= ana_start) & (df_pmc["date_obj"] <= ana_end)]

        fig_pmc = go.Figure()
        fig_pmc.add_hrect(y0=10, y1=35, fillcolor="#16a34a", opacity=0.10, line_width=0, annotation_text="Visoka svežina / Tekma (>+10)", annotation_position="top left")
        fig_pmc.add_hrect(y0=-10, y1=10, fillcolor="#94a3b8", opacity=0.10, line_width=0, annotation_text="Nevtralno / Ohranjanje (-10 do +10)", annotation_position="top left")
        fig_pmc.add_hrect(y0=-30, y1=-10, fillcolor="#f59e0b", opacity=0.12, line_width=0, annotation_text="Optimalni trenažni pas (-30 do -10)", annotation_position="top left")
        fig_pmc.add_hrect(y0=-60, y1=-30, fillcolor="#dc2626", opacity=0.12, line_width=0, annotation_text="Tveganje preutrujenosti (<-30)", annotation_position="top left")

        fig_pmc.add_trace(go.Bar(
            x=df_pmc_view["Datum"], y=df_pmc_view["TSB"],
            name="Forma (TSB)",
            marker_color=df_pmc_view["TSB"].apply(lambda x: "#16a34a" if x > 10 else ("#f59e0b" if x > -30 else "#dc2626")),
            opacity=0.6
        ))
        fig_pmc.add_trace(go.Scatter(
            x=df_pmc_view["Datum"], y=df_pmc_view["CTL"],
            mode="lines", name="Fitness (CTL - 42d)",
            line=dict(color="#2563eb", width=3)
        ))
        fig_pmc.add_trace(go.Scatter(
            x=df_pmc_view["Datum"], y=df_pmc_view["ATL"],
            mode="lines", name="Utrujenost (ATL - 7d)",
            line=dict(color="#d946ef", width=2, dash="dash")
        ))

        fig_pmc.update_layout(
            title=f"PMC Profil v obdobju {ana_start.strftime('%d.%m.')} – {ana_end.strftime('%d.%m.')}",
            xaxis=dict(title="Datum"),
            yaxis=dict(title="Obremenitev / Točke"),
            hovermode="x unified",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_pmc, use_container_width=True)

    st.markdown("---")

    # ----------------------------------------------------
    # 2. TEDENSKI VOLUMEN & DELO V OBDOBJU
    # ----------------------------------------------------
    st.subheader("2. Tedenski volumen, ure in delo po športih (Cilj 10–12 h/teden)")
    week_records = []
    for a in ana_acts:
        st_l = a.get("startTimeLocal", "")
        if st_l:
            d_obj = datetime.strptime(st_l[:10], "%Y-%m-%d").date()
            y, w, _ = d_obj.isocalendar()
            mon = d_obj - timedelta(days=d_obj.weekday())
            week_key = f"{y}-W{w:02d} ({mon.strftime('%d.%m.')})"
            sp = get_sport_category(a.get("activityType", {}).get("typeKey"), a.get("activityName", ""))
            
            # Mehansko delo v kJ
            aid = str(a.get("activityId"))
            stream_w = raw_data["streams"].get(aid, {}).get("summary", {}).get("work_kJ")
            if not stream_w:
                avg_p = a.get("avgPower") or a.get("averagePower") or 0
                m_sec = a.get("movingDuration") or a.get("duration") or 0
                stream_w = round(avg_p * m_sec / 1000.0) if avg_p > 0 else 0

            week_records.append({
                "week": week_key,
                "monday": mon,
                "sport": sp,
                "hours": round((a.get("duration", 0) or 0) / 3600.0, 2),
                "km": round((a.get("distance", 0) or 0) / 1000.0, 1),
                "elev": round(a.get("elevationGain", 0) or 0),
                "work_kJ": stream_w
            })

    if week_records:
        df_weeks = pd.DataFrame(week_records)
        df_weekly_agg = df_weeks.groupby(["week", "monday", "sport"]).sum().reset_index()
        df_weekly_agg.sort_values("monday", inplace=True)

        fig_vol = px.bar(
            df_weekly_agg,
            x="week",
            y="hours",
            color="sport",
            title="Tedenske ure treninga po športih v izbranem obdobju",
            labels={"hours": "Ure", "week": "Teden", "sport": "Šport"},
            color_discrete_map={
                "Cestno kolo": "#2563eb",
                "Trenažer / Indoor": "#38bdf8",
                "Tek": "#ea580c",
                "Hoja / Pohodništvo": "#10b981",
                "Plavanje": "#06b6d4",
                "Moč / Fitnes / Plezanje": "#8b5cf6",
                "Drugo": "#94a3b8"
            }
        )
        fig_vol.add_hrect(
            y0=10, y1=12,
            fillcolor="#16a34a", opacity=0.15,
            line_width=1, line_dash="dash", line_color="#16a34a",
            annotation_text="Optimalni volumen (10–12 h)", annotation_position="top left"
        )
        fig_vol.update_layout(barmode="stack", hovermode="x unified", height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.info("V izbranem obdobju ni zabeleženih aktivnosti.")

    st.markdown("---")

    # ----------------------------------------------------
    # 3. PORAZDELITEV INTENZIVNOSTI (POWER & HR ZONES)
    # ----------------------------------------------------
    st.subheader("3. Porazdelitev intenzivnosti v obdobju (Cone moči Z1–Z7 & Srčni utrip Z1–Z5)")
    c_pz, c_hrz = st.columns(2)

    with c_pz:
        pz_tot = [0.0]*7
        for a in ana_acts:
            for i in range(1, 8):
                pz_tot[i-1] += (a.get(f"powerTimeInZone_{i}", 0) or 0) / 3600.0

        if sum(pz_tot) > 0:
            df_pz = pd.DataFrame({
                "Cona": ["Z1 (<55%)", "Z2 (55–75%)", "Z3 (75–90%)", "Z4 Threshold", "Z5 VO2 Max", "Z6 Anaerobno", "Z7 Nevromuskularno"],
                "Ure": [round(x, 2) for x in pz_tot]
            })
            df_pz["Delež (%)"] = ((df_pz["Ure"] / df_pz["Ure"].sum()) * 100).round(1)
            fig_pz = px.bar(
                df_pz, x="Cona", y="Ure",
                color="Cona", text="Delež (%)",
                title="Čas v conah moči (Power Zones)",
                color_discrete_sequence=["#94a3b8", "#3b82f6", "#06b6d4", "#f59e0b", "#ef4444", "#a855f7", "#ec4899"]
            )
            fig_pz.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_pz.update_layout(showlegend=False, height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pz, use_container_width=True)
        else:
            st.info("V izbranem obdobju ni podatkov o močnostnih conah.")

    with c_hrz:
        hrz_tot = [0.0]*5
        for a in ana_acts:
            for i in range(1, 6):
                hrz_tot[i-1] += (a.get(f"hrTimeInZone_{i}", 0) or 0) / 3600.0

        if sum(hrz_tot) > 0:
            df_hrz = pd.DataFrame({
                "Cona": ["Z1 Ogrevanje", "Z2 Lahko / Baza", "Z3 Aerobno", "Z4 Prag", "Z5 Maksimalno"],
                "Ure": [round(x, 2) for x in hrz_tot]
            })
            df_hrz["Delež (%)"] = ((df_hrz["Ure"] / df_hrz["Ure"].sum()) * 100).round(1)
            fig_hrz = px.bar(
                df_hrz, x="Cona", y="Ure",
                color="Cona", text="Delež (%)",
                title="Čas v conah srčnega utripa (HR Zones)",
                color_discrete_sequence=["#94a3b8", "#10b981", "#3b82f6", "#f59e0b", "#ef4444"]
            )
            fig_hrz.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_hrz.update_layout(showlegend=False, height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hrz, use_container_width=True)
        else:
            st.info("V izbranem obdobju ni podatkov o conah srčnega utripa.")

    st.markdown("---")

    # ----------------------------------------------------
    # 4. AEROBNA UČINKOVITOST & TOPLOTNA TOLERANCA
    # ----------------------------------------------------
    st.subheader("4. Aerobna učinkovitost (EF = NP / HR) in toplotna toleranca")
    col_ef, col_temp = st.columns(2)

    with col_ef:
        ef_records = []
        for a in ana_acts:
            sp = get_sport_category(a.get("activityType", {}).get("typeKey"), a.get("activityName", ""))
            if sp in ["Cestno kolo", "Trenažer / Indoor"]:
                aid = str(a.get("activityId"))
                st_data = raw_data["streams"].get(aid, {}).get("summary", {})
                np_val = st_data.get("np") or a.get("normPower") or a.get("normalizedPower") or a.get("avgPower")
                hr_val = a.get("averageHR")
                dur_m = (a.get("duration", 0) or 0) / 60.0
                st_l = a.get("startTimeLocal", "")[:10]
                if np_val and hr_val and dur_m >= 30:
                    ef = round(np_val / hr_val, 2)
                    ef_records.append({
                        "Datum": st_l,
                        "Ime": a.get("activityName", "Kolesarjenje"),
                        "EF": ef,
                        "NP": np_val,
                        "HR": hr_val,
                        "Zvrst": sp
                    })

        if ef_records:
            df_ef = pd.DataFrame(ef_records)
            df_ef.sort_values("Datum", inplace=True)
            fig_ef = px.scatter(
                df_ef, x="Datum", y="EF",
                color="Zvrst", size="NP", hover_data=["Ime", "NP", "HR"],
                title="Efficiency Factor (NP / Avg HR) skozi čas"
            )
            fig_ef.update_traces(mode="lines+markers")
            fig_ef.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_ef, use_container_width=True)
        else:
            st.info("Ni kolesarskih voženj z NP in HR daljših od 30 min.")

    with col_temp:
        temp_records = []
        for a in ana_acts:
            sp = get_sport_category(a.get("activityType", {}).get("typeKey"), a.get("activityName", ""))
            if sp in ["Cestno kolo"]:
                aid = str(a.get("activityId"))
                st_data = raw_data["streams"].get(aid, {}).get("summary", {})
                np_val = st_data.get("np") or a.get("normPower") or a.get("normalizedPower") or a.get("avgPower")
                hr_val = a.get("averageHR")
                t_max = a.get("maxTemperature") or a.get("minTemperature")
                if np_val and hr_val and t_max is not None:
                    ef = round(np_val / hr_val, 2)
                    temp_records.append({
                        "Temperatura (°C)": t_max,
                        "EF (Moč/HR)": ef,
                        "Normalized Power (W)": np_val,
                        "Povprečni HR (bpm)": hr_val,
                        "Aktivnost": a.get("activityName", "Vožnja"),
                        "Datum": a.get("startTimeLocal", "")[:10]
                    })

        if temp_records:
            df_temp = pd.DataFrame(temp_records)
            fig_temp = px.scatter(
                df_temp, x="Temperatura (°C)", y="EF (Moč/HR)",
                color="Temperatura (°C)", size="Normalized Power (W)",
                hover_data=["Aktivnost", "Datum", "Povprečni HR (bpm)"],
                title="Toplotna toleranca: Temperatura proti EF",
                color_continuous_scale="Turbo"
            )
            fig_temp.add_vline(x=30, line_dash="dash", line_color="#dc2626", annotation_text="Kritična meja (>30°C)")
            fig_temp.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("Za izbrano obdobje ni zunanjih kolesarskih voženj s temperaturo.")

    st.markdown("---")

    # ----------------------------------------------------
    # 5. NAPREDEK W/kg NA KLJUČNIH TRAJANJIH (PROGRESSION)
    # ----------------------------------------------------
    st.subheader("5. Napredek maksimalne moči / tempa skozi čas (Progression Chart)")
    PROG_PRESETS = [5, 15, 30, 60, 120, 180, 300, 360, 480, 600, 720, 900, 1200, 1800, 2400, 2700, 3600, 5400, 7200]
    prog_kind_lbl = st.radio("Vir podatkov:", ["Moč (kolesarjenje)", "Tempo (tek)"], horizontal=True, key="prog_kind")
    prog_kind = "power" if prog_kind_lbl.startswith("Moč") else "pace"
    pg1, pg2 = st.columns([3, 2])
    with pg1:
        prog_sel = st.multiselect("Izberi trajanja naporov (primerjava na istem grafu):", PROG_PRESETS, default=[300, 1200],
                                  format_func=lambda x: DURATION_LABELS.get(x, f"{x}s"), key="prog_dur_ms")
    with pg2:
        use_custom = st.checkbox("⏱️ Poljuben čas (Custom Duration)", key="prog_custom_on")
        custom_secs = []
        if use_custom:
            cu1, cu2 = st.columns([2, 1])
            cval = cu1.number_input("Trajanje:", min_value=1.0, max_value=480.0 if True else 0, value=43.0, step=1.0, key="prog_custom_val")
            cunit = cu2.radio("Enota:", ["min", "s"], horizontal=True, key="prog_custom_unit")
            custom_secs = [int(round(cval * 60)) if cunit == "min" else int(round(cval))]
    prog_y_mode = "W/kg"
    if prog_kind == "power":
        prog_y_mode = st.radio("Enote:", ["W/kg", "W"], horizontal=True, key="prog_ymode")

    dur_list = [(d, DURATION_LABELS.get(d, f"{d}s")) for d in prog_sel] + [(d, f"{d // 60} min {d % 60} s (poljubno)" if d % 60 else f"{d // 60} min (poljubno)") for d in custom_secs]
    prog_records, any_interp = [], False
    if prog_kind == "power":
        sources = list(mmp_all.items())
    else:
        sources = [(aid, {"startTimeLocal": s.get("startTimeLocal", ""), "activityName": s.get("activityName", "Tek")})
                   for aid, s in raw_data["streams"].items()
                   if (s.get("summary") or {}).get("bestPace") and sport_group(raw_data["activities"].get(aid, {}).get("activityType", {}).get("typeKey")) in ("Tek", "Hoja / Pohodništvo")]
    for aid, m_obj in sources:
        st_l = m_obj.get("startTimeLocal", "")[:10]
        try:
            d_act = datetime.strptime(st_l, "%Y-%m-%d").date()
        except Exception:
            continue
        if not (ana_start <= d_act <= ana_end):
            continue
        for d, lbl in dur_list:
            if prog_kind == "power":
                if str(d) in (m_obj.get("mmp") or {}) and d not in custom_secs:
                    val, interp = float(m_obj["mmp"][str(d)]), False
                else:
                    val, interp = duration_value(raw_data, aid, m_obj, d, "power")
                if not val:
                    continue
                prog_records.append({"Datum": st_l, "Trajanje": lbl, "Vrednost": round(val / user_cfg["weight_kg"], 2) if prog_y_mode == "W/kg" else round(val),
                                     "Ime": m_obj.get("activityName", "Vožnja"), "Okolje": "Trenažer" if m_obj.get("is_indoor") else "Zunaj", "Približek": interp})
            else:
                bp = (raw_data["streams"][aid].get("summary") or {}).get("bestPace") or {}
                if str(d) in bp and d not in custom_secs:
                    val, interp = float(bp[str(d)]), False
                else:
                    val, interp = duration_value(raw_data, aid, m_obj, d, "pace")
                if not val:
                    continue
                prog_records.append({"Datum": st_l, "Trajanje": lbl, "Vrednost": round(val, 1), "Ime": m_obj.get("activityName", "Tek"),
                                     "Okolje": "", "Približek": interp})
            any_interp = any_interp or interp

    if prog_records:
        df_prog = pd.DataFrame(prog_records).sort_values("Datum")
        fig_prog = px.line(df_prog, x="Datum", y="Vrednost", color="Trajanje", markers=True, hover_data=["Ime", "Okolje"],
                           title=f"Napredek: {prog_kind_lbl} v obdobju {ana_label}")
        if prog_kind == "pace":
            tv_ = sorted(set(int(v // 15) * 15 for v in df_prog["Vrednost"]))
            fig_prog.update_yaxes(title="Tempo (min/km) – hitreje zgoraj", autorange="reversed", tickmode="array", tickvals=tv_, ticktext=[fmt_pace(v) for v in tv_])
        else:
            fig_prog.update_yaxes(title=prog_y_mode)
        fig_prog.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20), hovermode="closest", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_prog, use_container_width=True, key="prog_chart")
        if any_interp:
            st.caption("ℹ️ Pri vožnjah/tekih brez sekundnega toka je poljubno trajanje ocenjeno z interpolacijo med shranjenimi točkami krivulje.")
    else:
        st.info("V izbranem obdobju ni zabeleženih naporov za izbrana trajanja.")


    # ----------------------------------------------------
    # 6. MOČ PROTI SRČNEMU UTRIPU
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader("6. Krivulja moči proti srčnemu utripu (Power vs. HR – aerobna učinkovitost)")
    st.caption("Povprečna moč proti povprečnemu srčnemu utripu (10-min bloki, kjer so na voljo, sicer cele vožnje). Regresija po mesecih prikazuje, koliko W zmoreš pri 135, 150 in 165 bpm.")
    try:
        _ride_acts = [a for a in ana_acts if sport_group((a.get("activityType") or {}).get("typeKey")) == "Kolesarjenje"]
        _pts = []
        for a in _ride_acts:
            aid_ = str(a.get("activityId"))
            d_ = (a.get("startTimeLocal") or "")[:10]
            blocks_ = (((raw_data.get("streams", {}) or {}).get(aid_) or {}).get("summary") or {}).get("blocks") or []
            used_ = False
            for b_ in blocks_:
                if _mmss_to_sec(b_.get("movingTime")) >= 300 and (b_.get("avgW") or 0) >= 50 and (b_.get("avgHR") or 0) >= 80:
                    _pts.append({"Datum": d_, "Mesec": d_[:7], "Moč (W)": float(b_["avgW"]), "HR (bpm)": float(b_["avgHR"]),
                                 "Vir": "10-min blok", "Ime": a.get("activityName", "")})
                    used_ = True
            if not used_ and (a.get("avgPower") or 0) >= 50 and (a.get("averageHR") or 0) >= 80:
                _pts.append({"Datum": d_, "Mesec": d_[:7], "Moč (W)": float(a["avgPower"]), "HR (bpm)": float(a["averageHR"]),
                             "Vir": "cela vožnja", "Ime": a.get("activityName", "")})
        if len(_pts) < 3:
            st.info("V izbranem obdobju ni dovolj voženj s podatki o moči in srčnem utripu.")
        else:
            df_ph = pd.DataFrame(_pts)
            _pal = ["#2563eb", "#16a34a", "#ea580c", "#d81b60", "#7e57c2", "#0891b2", "#f59e0b", "#64748b"]
            fig_ph = go.Figure()
            months_ = sorted(df_ph["Mesec"].unique())
            reg_models = {}
            for mi_, mo_ in enumerate(months_):
                sub_ = df_ph[df_ph["Mesec"] == mo_]
                col_ = _pal[mi_ % len(_pal)]
                fig_ph.add_trace(go.Scatter(x=sub_["HR (bpm)"], y=sub_["Moč (W)"], mode="markers", name=mo_,
                                            marker=dict(color=col_, size=7, opacity=0.75),
                                            customdata=sub_[["Datum", "Ime", "Vir"]].values,
                                            hovertemplate="%{x:.0f} bpm · %{y:.0f} W<br>%{customdata[0]} · %{customdata[1]}<br>%{customdata[2]}<extra>" + mo_ + "</extra>"))
                if len(sub_) >= 4 and sub_["HR (bpm)"].nunique() >= 3:
                    k_, n_ = np.polyfit(sub_["HR (bpm)"], sub_["Moč (W)"], 1)
                    xs_ = np.array([sub_["HR (bpm)"].min(), sub_["HR (bpm)"].max()])
                    fig_ph.add_trace(go.Scatter(x=xs_, y=k_ * xs_ + n_, mode="lines", name=f"{mo_} trend", showlegend=False,
                                                line=dict(color=col_, dash="dash", width=2), hoverinfo="skip"))
                    reg_models[mo_] = (k_, n_, float(sub_["HR (bpm)"].min()), float(sub_["HR (bpm)"].max()))
            if df_ph["HR (bpm)"].nunique() >= 3:
                k_a, n_a = np.polyfit(df_ph["HR (bpm)"], df_ph["Moč (W)"], 1)
                xa_ = np.array([df_ph["HR (bpm)"].min(), df_ph["HR (bpm)"].max()])
                fig_ph.add_trace(go.Scatter(x=xa_, y=k_a * xa_ + n_a, mode="lines", name="Vse (regresija)", line=dict(color="#111827", width=3)))
                reg_models["Vse"] = (k_a, n_a, float(xa_[0]), float(xa_[1]))
            fig_ph.update_layout(height=440, xaxis=dict(title="Povprečni srčni utrip (bpm)"), yaxis=dict(title="Povprečna moč (W)"),
                                 margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", y=-0.2), hovermode="closest")
            st.plotly_chart(fig_ph, use_container_width=True, key="ana_power_hr_chart")
            _w_ref = float(user_cfg.get("weight_kg", 65.5))
            rows_ph = []
            for hr_t in (135, 150, 165):
                row_ = {"Srčni utrip": f"{hr_t} bpm"}
                for nm_, (k_, n_, lo_, hi_) in reg_models.items():
                    if lo_ - 5 <= hr_t <= hi_ + 5:
                        w_ = k_ * hr_t + n_
                        row_[nm_] = f"{w_:.0f} W ({w_ / _w_ref:.2f} W/kg)"
                    else:
                        row_[nm_] = "--"
                rows_ph.append(row_)
            st.markdown("**Ocenjena moč pri izbranem srčnem utripu** (znotraj izmerjenega razpona ±5 bpm):")
            st.dataframe(pd.DataFrame(rows_ph), use_container_width=True, hide_index=True)
    except Exception as ex_:
        st.error(f"Napaka pri grafu moč/HR: {ex_}")

    # ----------------------------------------------------
    # 7. TEDENSKA OBREMENITEV IN VOLUMEN
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader("7. Razmerje obremenitve in volumna (delo v kJ, TSS, ure in višinski metri na uro)")
    try:
        wk_ = {}
        for a in ana_acts:
            if sport_group((a.get("activityType") or {}).get("typeKey")) != "Kolesarjenje":
                continue
            d_s = (a.get("startTimeLocal") or "")[:10]
            try:
                d_o = datetime.strptime(d_s, "%Y-%m-%d").date()
            except Exception:
                continue
            wk_start = d_o - timedelta(days=d_o.weekday())
            mv = float(a.get("movingDuration") or a.get("duration") or 0)
            pw = float(a.get("avgPower") or 0)
            tss_ = a.get("trainingStressScore") or a.get("activityTrainingLoad")
            if not tss_ and pw and mv and user_cfg["ftp_outdoor"] > 0:
                npv = float(a.get("normPower") or pw)
                tss_ = mv * npv * (npv / user_cfg["ftp_outdoor"]) / (user_cfg["ftp_outdoor"] * 36.0)
            e_ = wk_.setdefault(wk_start, {"h": 0.0, "kj": 0.0, "tss": 0.0, "up": 0.0})
            e_["h"] += mv / 3600.0
            e_["kj"] += pw * mv / 1000.0
            e_["tss"] += float(tss_ or 0)
            e_["up"] += float(a.get("elevationGain") or 0)
        if not wk_:
            st.info("V izbranem obdobju ni kolesarskih voženj.")
        else:
            wdf = pd.DataFrame([{"Teden": k.strftime("%d.%m."), "start": k, "Ure": v["h"], "kJ": v["kj"], "TSS": v["tss"],
                                 "Vzpon (m)": v["up"], "m/h": (v["up"] / v["h"] if v["h"] > 0.5 else None)} for k, v in sorted(wk_.items())])
            c_w1, c_w2 = st.columns(2)
            with c_w1:
                fw1 = go.Figure()
                fw1.add_trace(go.Bar(x=wdf["Teden"], y=wdf["kJ"].round(0), name="Mehansko delo (kJ)", marker_color="#2563eb", opacity=0.8))
                fw1.add_trace(go.Scatter(x=wdf["Teden"], y=wdf["TSS"].round(0), name="TSS", yaxis="y2", mode="lines+markers", line=dict(color="#ea580c", width=3)))
                fw1.update_layout(title="Tedensko delo (kJ) in TSS", height=380, hovermode="x unified",
                                  yaxis=dict(title="kJ"), yaxis2=dict(title="TSS", overlaying="y", side="right", showgrid=False),
                                  legend=dict(orientation="h", y=-0.2), margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fw1, use_container_width=True, key="ana_week_kj_tss")
            with c_w2:
                fw2 = go.Figure()
                fw2.add_trace(go.Bar(x=wdf["Teden"], y=wdf["m/h"].round(0), name="Višinski metri na uro (m/h)", marker_color="#16a34a", opacity=0.8))
                fw2.add_trace(go.Scatter(x=wdf["Teden"], y=wdf["Ure"].round(1), name="Ure treninga", yaxis="y2", mode="lines+markers", line=dict(color="#7e57c2", width=3)))
                fw2.update_layout(title="Gostota vzpona (m/h) in ure treninga", height=380, hovermode="x unified",
                                  yaxis=dict(title="m/h"), yaxis2=dict(title="Ure", overlaying="y", side="right", showgrid=False),
                                  legend=dict(orientation="h", y=-0.2), margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fw2, use_container_width=True, key="ana_week_mh")
            wshow = wdf.drop(columns=["start"]).copy()
            wshow["Ure"] = wshow["Ure"].round(1)
            wshow["kJ"] = wshow["kJ"].round(0)
            wshow["TSS"] = wshow["TSS"].round(0)
            wshow["Vzpon (m)"] = wshow["Vzpon (m)"].round(0)
            wshow["m/h"] = wshow["m/h"].round(0)
            st.dataframe(wshow, use_container_width=True, hide_index=True)
    except Exception as ex_:
        st.error(f"Napaka pri tedenski obremenitvi: {ex_}")

    # ----------------------------------------------------
    # 8. KADENCA PO MOČNOSTNIH CONAH
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader("8. Analiza kadence in navora po močnostnih conah")
    try:
        _ftp_c = float(user_cfg.get("ftp_outdoor", 250)) or 250.0
        zone_defs = [("Z1 Regeneracija", 0.0, 0.55), ("Z2 Baza", 0.55, 0.75), ("Z3 Tempo", 0.75, 0.90),
                     ("Z4 Threshold", 0.90, 1.05), ("Z5 VO2max", 1.05, 1.20), ("Z6 Anaerobno", 1.20, 9.9)]
        cad_pts = []
        for a in ana_acts:
            if sport_group((a.get("activityType") or {}).get("typeKey")) != "Kolesarjenje":
                continue
            aid_ = str(a.get("activityId"))
            nm_ = a.get("activityName", "")
            d_ = (a.get("startTimeLocal") or "")[:10]
            got_ = False
            for b_ in ((((raw_data.get("streams", {}) or {}).get(aid_) or {}).get("summary") or {}).get("blocks") or []):
                if _mmss_to_sec(b_.get("movingTime")) >= 300 and (b_.get("avgW") or 0) > 40 and (b_.get("avgCad") or 0) > 30:
                    cad_pts.append((float(b_["avgW"]), float(b_["avgCad"]), d_, nm_))
                    got_ = True
            if not got_:
                for lp in ((((raw_data.get("splits", {}) or {}).get(aid_) or {}).get("laps")) or []):
                    if (lp.get("duration") or 0) >= 300 and (lp.get("avgPower") or 0) > 40 and (lp.get("avgCadence") or 0) > 30:
                        cad_pts.append((float(lp["avgPower"]), float(lp["avgCadence"]), d_, nm_))
                        got_ = True
            if not got_:
                cd_ = a.get("averageBikingCadenceInRevPerMinute")
                if (a.get("avgPower") or 0) > 40 and (cd_ or 0) > 30:
                    cad_pts.append((float(a["avgPower"]), float(cd_), d_, nm_))
        if len(cad_pts) < 3:
            st.info("V izbranem obdobju ni dovolj podatkov o kadenci in moči.")
        else:
            cdf = pd.DataFrame(cad_pts, columns=["W", "Kadenca", "Datum", "Ime"])
            cdf["Cona"] = cdf["W"].apply(lambda w: next((z[0] for z in zone_defs if z[1] <= w / _ftp_c < z[2]), zone_defs[-1][0]))
            cdf["Navor (Nm)"] = cdf["W"] / (2 * math.pi * cdf["Kadenca"] / 60.0)
            order_ = [z[0] for z in zone_defs if z[0] in set(cdf["Cona"])]
            fcb = px.box(cdf, x="Cona", y="Kadenca", points="all", category_orders={"Cona": order_}, hover_data=["Datum", "Ime", "W"],
                         color="Cona", title=f"Kadenca po conah (FTP {_ftp_c:.0f} W)")
            fcb.update_layout(height=420, showlegend=False, yaxis_title="Kadenca (rpm)", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fcb, use_container_width=True, key="ana_cad_box")
            tbl_ = cdf.groupby("Cona").agg(Vzorcev=("W", "count"), Povp_moč=("W", "mean"), Povp_kadenca=("Kadenca", "mean"),
                                           Povp_navor=("Navor (Nm)", "mean")).reindex(order_).reset_index()
            tbl_.columns = ["Cona", "Št. vzorcev", "Povp. moč (W)", "Povp. kadenca (rpm)", "Povp. navor (Nm)"]
            for c_ in tbl_.columns[2:]:
                tbl_[c_] = tbl_[c_].round(1)
            st.dataframe(tbl_, use_container_width=True, hide_index=True)
            st.caption("Navor = moč / (2π · kadenca / 60). Vzorci so 10-min bloki, krogi ali cele vožnje, odvisno od razpoložljivih podatkov.")
    except Exception as ex_:
        st.error(f"Napaka pri analizi kadence: {ex_}")

# ==============================================================================
# TAB 4: REGENERACIJA & FIZIOLOGIJA (Z ENOTNIM NAVIGATORJEM & 4 GRAFI)
# ==============================================================================
with tab_recovery:
    st.header("💤 Regeneracija, Spanje in Fiziologija")
    st.caption("Podroben vpogled v avtonomni živčni sistem, arhitekturo spanja, cirkadiani ritem in korelacije z obremenitvijo.")

    rec_start, rec_end, rec_label = render_period_navigator(nav_key_prefix="recovery_tab", default_preset="Zadnjih 30 dni")
    df_rec = df_daily_all[(df_daily_all["date_obj"] >= rec_start) & (df_daily_all["date_obj"] <= rec_end)].copy() if not df_daily_all.empty else pd.DataFrame()

    if df_rec.empty:
        st.info("V izbranem obdobju ni dnevnih regeneracijskih podatkov.")
    else:
        st.markdown("---")

        # ----------------------------------------------------
        # 1. HRV & RHR BASELINE BAND CHART
        # ----------------------------------------------------
        st.subheader("1. HRV Baseline Pas & RHR vs. Nočni srčni utrip")
        st.caption("Prikaz nočnega HRV s 7-dnevnim povprečjem in zelenim referenčnim pasom normalnega stanja (mean ± 0.75 std), skupaj z najnižjim dnevnim RHR in povprečnim nočnim utripom.")

        fig_hrv_band = go.Figure()

        # Zelen referenčni pas normalnega HRV
        fig_hrv_band.add_trace(go.Scatter(
            x=df_rec["date"], y=df_rec["hrv_band_upper"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig_hrv_band.add_trace(go.Scatter(
            x=df_rec["date"], y=df_rec["hrv_band_lower"],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(16, 185, 129, 0.15)", name="HRV Normalni Pas", hoverinfo="skip"
        ))

        # Nočni HRV in 7-dnevno povprečje
        fig_hrv_band.add_trace(go.Scatter(
            x=df_rec["date"], y=df_rec["hrv_night"],
            mode="markers+lines", name="Nočni HRV (ms)",
            line=dict(color="#a855f7", width=1.5), marker=dict(size=5)
        ))
        fig_hrv_band.add_trace(go.Scatter(
            x=df_rec["date"], y=df_rec["hrv_7d_mean"],
            mode="lines", name="7d HRV povprečje",
            line=dict(color="#7c3aed", width=3)
        ))

        # Dnevni RHR in Nočni utrip na desni Y-osi
        fig_hrv_band.add_trace(go.Scatter(
            x=df_rec["date"], y=df_rec["rhr"],
            mode="lines+markers", name="Najnižji RHR (bpm)",
            yaxis="y2", line=dict(color="#ef4444", width=1.5, dash="dot"), marker=dict(size=4)
        ))
        fig_hrv_band.add_trace(go.Scatter(
            x=df_rec["date"], y=df_rec["avg_sleep_hr"],
            mode="lines", name="Povp. nočni HR (bpm)",
            yaxis="y2", line=dict(color="#f97316", width=2)
        ))

        fig_hrv_band.update_layout(
            title="HRV Baseline Band & Utrip v mirovanju",
            xaxis=dict(title="Datum"),
            yaxis=dict(title="HRV (ms)", side="left"),
            yaxis2=dict(title="Srčni utrip (bpm)", side="right", overlaying="y", showgrid=False),
            hovermode="x unified",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_hrv_band, use_container_width=True)

        st.markdown("---")

        # ----------------------------------------------------
        # 2. ARHITEKTURA SPANJA & CIRKADIANI RITEM
        # ----------------------------------------------------
        st.subheader("2. Arhitektura spanja in doslednost cirkadianega ritma")
        c_slp1, c_slp2 = st.columns(2)

        with c_slp1:
            fig_stages = go.Figure()
            fig_stages.add_trace(go.Bar(x=df_rec["date"], y=df_rec["deep_min"], name="Globoki spanec", marker_color="#1e3a8a"))
            fig_stages.add_trace(go.Bar(x=df_rec["date"], y=df_rec["rem_min"], name="REM faza", marker_color="#0284c7"))
            fig_stages.add_trace(go.Bar(x=df_rec["date"], y=df_rec["light_min"], name="Lahek spanec", marker_color="#38bdf8"))
            fig_stages.add_trace(go.Bar(x=df_rec["date"], y=df_rec["awake_min"], name="Budnost", marker_color="#f43f5e"))
            fig_stages.update_layout(
                barmode="stack", title="Faze spanja po nočeh (minute)",
                xaxis=dict(title="Datum"), yaxis=dict(title="Minute"),
                hovermode="x unified", height=380, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_stages, use_container_width=True)

        with c_slp2:
            # Izračun ure bujenja in ure zaspanja iz bedtime_str
            bed_hours = []
            wake_hours = []
            for b_str in df_rec["bedtime_str"]:
                if b_str and "–" in b_str:
                    try:
                        pts = b_str.split("–")
                        s_h, s_m = map(int, pts[0].strip().split(":"))
                        e_h, e_m = map(int, pts[1].strip().split(":"))
                        bed_h = s_h + s_m / 60.0
                        if bed_h < 12: bed_h += 24.0 # Za ure po polnoči
                        wake_h = e_h + e_m / 60.0
                        bed_hours.append(round(bed_h, 2))
                        wake_hours.append(round(wake_h, 2))
                    except Exception:
                        bed_hours.append(None)
                        wake_hours.append(None)
                else:
                    bed_hours.append(None)
                    wake_hours.append(None)

            fig_circ = go.Figure()
            fig_circ.add_trace(go.Scatter(
                x=df_rec["date"], y=bed_hours,
                mode="lines+markers", name="Ura zaspanja",
                line=dict(color="#6366f1", width=2)
            ))
            fig_circ.add_trace(go.Scatter(
                x=df_rec["date"], y=wake_hours,
                mode="lines+markers", name="Ura bujenja",
                line=dict(color="#eab308", width=2)
            ))
            fig_circ.update_layout(
                title="Cirkadiana doslednost (Čas zaspanja in bujenja)",
                xaxis=dict(title="Datum"),
                yaxis=dict(title="Ura dneva (decimalno)", tickvals=[21, 22, 23, 24, 25, 26, 6, 7, 8, 9], ticktext=["21:00", "22:00", "23:00", "00:00", "01:00", "02:00", "06:00", "07:00", "08:00", "09:00"]),
                hovermode="x unified", height=380, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_circ, use_container_width=True)

        st.markdown("---")

        # ----------------------------------------------------
        # 3. ZGODNJI ZNAKI PREUTRUJENOSTI & BOLEZNI
        # ----------------------------------------------------
        st.subheader("3. Zgodnji indikatorji stresa, nemira in preutrujenosti")
        c_ind1, c_ind2 = st.columns(2)

        with c_ind1:
            fig_stress_rec = go.Figure()
            fig_stress_rec.add_trace(go.Scatter(
                x=df_rec["date"], y=df_rec["avg_resp"],
                mode="lines+markers", name="Nočno dihanje (brpm)",
                line=dict(color="#06b6d4", width=2)
            ))
            fig_stress_rec.add_trace(go.Bar(
                x=df_rec["date"], y=df_rec["awake_count"],
                name="Število prebujanj",
                yaxis="y2", marker_color="#f59e0b", opacity=0.5
            ))
            fig_stress_rec.update_layout(
                title="Frekvenca dihanja & Nočni nemir",
                xaxis=dict(title="Datum"),
                yaxis=dict(title="Dihanje (brpm)", side="left"),
                yaxis2=dict(title="Prebujanja", side="right", overlaying="y", showgrid=False),
                hovermode="x unified", height=360, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_stress_rec, use_container_width=True)

        with c_ind2:
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(
                x=df_rec["date"], y=df_rec["bb_max"],
                mode="lines", name="Body Battery Max",
                line=dict(color="#10b981", width=2.5)
            ))
            fig_bb.add_trace(go.Scatter(
                x=df_rec["date"], y=df_rec["stress"],
                mode="lines", name="Dnevni stres (Garmin)",
                line=dict(color="#f43f5e", width=1.8, dash="dot")
            ))
            fig_bb.update_layout(
                title="Regeneracija Body Battery vs. Dnevni stres",
                xaxis=dict(title="Datum"),
                yaxis=dict(title="Raven (0–100)"),
                hovermode="x unified", height=360, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_bb, use_container_width=True)

        st.markdown("---")

        # ----------------------------------------------------
        # 4. KORELACIJA: TRENING OBREMENITEV VS. NOČNI UTRIP & HRV
        # ----------------------------------------------------
        st.subheader("4. Vpliv trenažne obremenitve na nočni srčni utrip in HRV")
        st.caption("Prikaz odnosa med dnevno obremenitvijo (Training Load) in povprečnim srčnim utripom tiste noči ter naslednjim jutranjim HRV.")

        fig_corr = go.Figure()
        fig_corr.add_trace(go.Bar(
            x=df_rec["date"], y=df_rec["daily_load"],
            name="Dnevni Training Load",
            marker_color="#94a3b8", opacity=0.4
        ))
        fig_corr.add_trace(go.Scatter(
            x=df_rec["date"], y=df_rec["avg_sleep_hr"],
            mode="lines+markers", name="Nočni utrip med spanjem (bpm)",
            yaxis="y2", line=dict(color="#ef4444", width=2.5)
        ))
        fig_corr.add_trace(go.Scatter(
            x=df_rec["date"], y=df_rec["hrv_night"],
            mode="lines+markers", name="Nočni HRV (ms)",
            yaxis="y2", line=dict(color="#a855f7", width=2, dash="dash")
        ))
        fig_corr.update_layout(
            title="Dnevna obremenitev (Load) proti nočnemu utripu & HRV",
            xaxis=dict(title="Datum"),
            yaxis=dict(title="Training Load", side="left"),
            yaxis2=dict(title="Nočni HR (bpm) / HRV (ms)", side="right", overlaying="y", showgrid=False),
            hovermode="x unified", height=390, margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # ----------------------------------------------------
        # 5. DNEVNA AKTIVNOST & PORABA KALORIJ
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("5. Dnevna gibalna aktivnost in struktura porabe kalorij")
        c_act1, c_act2 = st.columns(2)

        with c_act1:
            fig_steps = go.Figure()
            fig_steps.add_trace(go.Bar(
                x=df_rec["date"], y=df_rec["steps"],
                name="Dnevni koraki",
                marker_color="#3b82f6"
            ))
            fig_steps.add_hline(
                y=10000, line_dash="dash", line_color="#10b981",
                annotation_text="Cilj: 10.000 korakov", annotation_position="top right"
            )
            fig_steps.update_layout(
                title="Dnevni koraki (s ciljno črto 10.000)",
                xaxis=dict(title="Datum"),
                yaxis=dict(title="Število korakov"),
                hovermode="x unified", height=380, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_steps, use_container_width=True)

        with c_act2:
            fig_cals = go.Figure()
            fig_cals.add_trace(go.Bar(
                x=df_rec["date"], y=df_rec["cal_bmr"],
                name="Bazalni metabolizem (BMR / Mirovanje)",
                marker_color="#64748b"
            ))
            fig_cals.add_trace(go.Bar(
                x=df_rec["date"], y=df_rec["cal_act"],
                name="Aktivne kalorije (Vadba & Gibanje)",
                marker_color="#f97316"
            ))
            fig_cals.update_layout(
                barmode="stack",
                title="Dnevna poraba kalorij (BMR + Aktivne kcal)",
                xaxis=dict(title="Datum"),
                yaxis=dict(title="kcal"),
                hovermode="x unified", height=380, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_cals, use_container_width=True)

        # ----------------------------------------------------
        # 6. DNEVNI SRČNI UTRIP & VO2 MAX KAPACITETA
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("6. Fiziološki trendi: Dnevni povprečni srčni utrip in VO2 Max")
        c_phys1, c_phys2 = st.columns(2)

        with c_phys1:
            fig_avg_hr = go.Figure()
            fig_avg_hr.add_trace(go.Scatter(
                x=df_rec["date"], y=df_rec["daily_avg_hr"],
                mode="lines+markers", name="Dnevni povprečni HR (cel dan)",
                line=dict(color="#ec4899", width=2.2), marker=dict(size=4)
            ))
            fig_avg_hr.add_trace(go.Scatter(
                x=df_rec["date"], y=df_rec["avg_sleep_hr"],
                mode="lines", name="Nočni utrip med spanjem",
                line=dict(color="#f97316", width=1.8)
            ))
            fig_avg_hr.add_trace(go.Scatter(
                x=df_rec["date"], y=df_rec["rhr"],
                mode="lines", name="Najnižji dnevni RHR",
                line=dict(color="#ef4444", width=1.5, dash="dot")
            ))
            fig_avg_hr.update_layout(
                title="Dnevni povprečni srčni utrip (dnevni stres & regeneracija)",
                xaxis=dict(title="Datum"),
                yaxis=dict(title="Srčni utrip (bpm)"),
                hovermode="x unified", height=380, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_avg_hr, use_container_width=True)

        with c_phys2:
            fig_vo2 = go.Figure()
            # Točke doseženih naporov
            fig_vo2.add_trace(go.Scatter(
                x=df_rec["date"], y=df_rec["vo2max_daily"],
                mode="markers", name="Dosežen VO2max (trening)",
                marker=dict(color="#8b5cf6", size=7, symbol="diamond")
            ))
            # Kapaciteta / trend linija
            fig_vo2.add_trace(go.Scatter(
                x=df_rec["date"], y=df_rec["vo2max_capacity"],
                mode="lines", name="Kapaciteta VO2max (42d kotalni vrh)",
                line=dict(color="#7c3aed", width=3)
            ))
            fig_vo2.update_layout(
                title="VO2 Max trend (iz kolesarskega MAP in tekaškega tempa)",
                xaxis=dict(title="Datum"),
                yaxis=dict(title="VO2max (ml/kg/min)"),
                hovermode="x unified", height=380, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_vo2, use_container_width=True)

        # ----------------------------------------------------
        # 5. Z-SCORE REGENERACIJSKI DASHBOARD
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("5. 7-dnevni trend odstopanja (Z-score regeneracijski dashboard)")
        st.caption("Odstopanje HRV, RHR, nočnega povprečnega HR in dolžine spanja od tvojega 30-dnevnega izhodišča. Pri RHR in nočnem HR je predznak obrnjen, tako da je pozitivna vrednost vedno 'boljša regeneracija'.")
        try:
            zb = df_daily_all.copy()
            zmetrics = [("hrv_night", "HRV (ms)", 1), ("rhr", "RHR (bpm)", -1), ("avg_sleep_hr", "Nočni povp. HR (bpm)", -1), ("sleep_hours", "Dolžina spanja (h)", 1)]
            for col_, _, sg_ in zmetrics:
                s_ = pd.to_numeric(zb[col_], errors="coerce")
                m_ = s_.shift(1).rolling(30, min_periods=10).mean()
                sd_ = s_.shift(1).rolling(30, min_periods=10).std().replace(0, np.nan)
                zb[col_ + "_z"] = ((s_ - m_) / sd_) * sg_
                zb[col_ + "_pct"] = (s_ - m_) / m_ * 100.0
                zb[col_ + "_base"] = m_
            zb["rec_idx"] = zb[[c[0] + "_z" for c in zmetrics]].mean(axis=1, skipna=True)
            win_ = zb[zb["date_obj"] <= rec_end].tail(7)
            if win_.empty or win_["rec_idx"].dropna().empty:
                st.info("Za Z-score je potrebnih vsaj 10 dni podatkov o spanju/HRV.")
            else:
                fz = go.Figure()
                fz.add_hrect(y0=-1, y1=1, fillcolor="#94a3b8", opacity=0.10, line_width=0)
                fz.add_hrect(y0=1, y1=3, fillcolor="#16a34a", opacity=0.08, line_width=0)
                fz.add_hrect(y0=-3, y1=-1.5, fillcolor="#dc2626", opacity=0.08, line_width=0)
                zcols = ["#a855f7", "#ef4444", "#f97316", "#0284c7"]
                for (col_, lbl_, _), c_ in zip(zmetrics, zcols):
                    fz.add_trace(go.Bar(x=win_["date"], y=win_[col_ + "_z"].round(2), name=lbl_, marker_color=c_, opacity=0.8))
                fz.add_trace(go.Scatter(x=win_["date"], y=win_["rec_idx"].round(2), name="Skupni indeks regeneracije", mode="lines+markers",
                                        line=dict(color="#111827", width=3)))
                fz.update_layout(barmode="group", height=420, hovermode="x unified", yaxis=dict(title="Z-score (+ = boljše)"),
                                 legend=dict(orientation="h", y=-0.2), margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fz, use_container_width=True, key="rec_zscore_chart")
                last_ = win_.iloc[-1]
                zrows = []
                for col_, lbl_, sg_ in zmetrics:
                    v_ = last_.get(col_)
                    zrows.append({"Metrika": lbl_, "Zadnji dan": f"{v_:.1f}" if pd.notna(v_) else "--",
                                  "30d izhodišče": f"{last_[col_ + '_base']:.1f}" if pd.notna(last_[col_ + '_base']) else "--",
                                  "Odstopanje (%)": f"{last_[col_ + '_pct']:+.1f} %" if pd.notna(last_[col_ + '_pct']) else "--",
                                  "Z-score (+ = boljše)": f"{last_[col_ + '_z']:+.2f}" if pd.notna(last_[col_ + '_z']) else "--"})
                st.markdown(f"**Stanje na dan {last_['date']}:**")
                st.dataframe(pd.DataFrame(zrows), use_container_width=True, hide_index=True)
                idx_ = last_["rec_idx"]
                if pd.notna(idx_):
                    if idx_ >= 0.7:
                        st.success(f"🟢 Skupni indeks {idx_:+.2f}: avtonomni živčni sistem je nad izhodiščem (dobra regeneracija / možna superkompenzacija).")
                    elif idx_ <= -1.0:
                        st.error(f"🔴 Skupni indeks {idx_:+.2f}: izrazito pod izhodiščem – možna utrujenost, bolezen ali stres. Razmisli o lažjem dnevu.")
                    elif idx_ <= -0.4:
                        st.warning(f"🟡 Skupni indeks {idx_:+.2f}: rahlo pod izhodiščem – spremljaj trend.")
                    else:
                        st.info(f"⚪ Skupni indeks {idx_:+.2f}: v mejah običajnega.")
        except Exception as ex_:
            st.error(f"Napaka pri Z-score dashboardu: {ex_}")

        # ----------------------------------------------------
        # 6. ENERGIJSKO RAVNOVESJE IN REGENERACIJA PO TEŽKIH TRENINGIH
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("6. Obremenitev dneva proti nočnemu srčnemu utripu, globokemu spancu in HRV")
        st.caption("Za vsak dan D je prikazana noč PO treningu (Garmin zapis spanca z datumom D+1).")
        try:
            eb = df_daily_all.copy()
            eb["next_sleep_hr"] = pd.to_numeric(eb["avg_sleep_hr"], errors="coerce").shift(-1)
            eb["next_deep"] = pd.to_numeric(eb["deep_min"], errors="coerce").shift(-1)
            eb["next_hrv"] = pd.to_numeric(eb["hrv_night"], errors="coerce").shift(-1)
            eb = eb[(eb["date_obj"] >= rec_start) & (eb["date_obj"] <= rec_end)]
            load_mode = st.radio("Stolpci (obremenitev dneva):", ["Trening obremenitev (load/TSS)", "Aktivne kalorije (kcal)"], horizontal=True, key="rec_energy_mode")
            ycol = "daily_load" if load_mode.startswith("Trening") else "cal_act"
            ylbl = "Obremenitev" if ycol == "daily_load" else "Aktivne kcal"
            from plotly.subplots import make_subplots
            fe = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10, specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                               subplot_titles=(f"{ylbl} dneva vs. povprečni nočni HR po treningu", "Globoki spanec in nočni HRV po treningu"))
            fe.add_trace(go.Bar(x=eb["date"], y=pd.to_numeric(eb[ycol], errors="coerce"), name=ylbl, marker_color="#2563eb", opacity=0.55), row=1, col=1, secondary_y=False)
            fe.add_trace(go.Scatter(x=eb["date"], y=eb["next_sleep_hr"], name="Povp. nočni HR (bpm)", mode="lines+markers", line=dict(color="#f97316", width=3)), row=1, col=1, secondary_y=True)
            fe.add_trace(go.Bar(x=eb["date"], y=eb["next_deep"], name="Globoki spanec (min)", marker_color="#1e3a8a", opacity=0.6), row=2, col=1, secondary_y=False)
            fe.add_trace(go.Scatter(x=eb["date"], y=eb["next_hrv"], name="Nočni HRV (ms)", mode="lines+markers", line=dict(color="#a855f7", width=3)), row=2, col=1, secondary_y=True)
            fe.update_layout(height=640, hovermode="x unified", legend=dict(orientation="h", y=-0.12), margin=dict(l=20, r=20, t=50, b=20))
            fe.update_yaxes(title_text=ylbl, row=1, col=1, secondary_y=False)
            fe.update_yaxes(title_text="bpm", row=1, col=1, secondary_y=True, showgrid=False)
            fe.update_yaxes(title_text="min", row=2, col=1, secondary_y=False)
            fe.update_yaxes(title_text="ms", row=2, col=1, secondary_y=True, showgrid=False)
            st.plotly_chart(fe, use_container_width=True, key="rec_energy_chart")
            cc_ = eb[[ycol, "next_sleep_hr"]].apply(pd.to_numeric, errors="coerce").dropna()
            cc_ = cc_[cc_[ycol] > 0]
            if len(cc_) >= 5 and cc_[ycol].std() > 0 and cc_["next_sleep_hr"].std() > 0:
                r_ = float(cc_[ycol].corr(cc_["next_sleep_hr"]))
                kslope = float(np.polyfit(cc_[ycol], cc_["next_sleep_hr"], 1)[0])
                st.info(f"Korelacija obremenitve in nočnega HR: **r = {r_:+.2f}** (n = {len(cc_)}). Vsakih +100 enot obremenitve ≈ **{kslope * 100:+.1f} bpm** nočnega utripa.")
            else:
                st.caption("Za izračun korelacije je potrebnih vsaj 5 dni z obremenitvijo in nočnim HR.")
        except Exception as ex_:
            st.error(f"Napaka pri energijskem ravnovesju: {ex_}")

st.markdown("---")
st.caption("Garmin Kolesarski Trener &bull; Prilagojeno za pripravo na vzpon na Roglo 2027 &bull; Izdelano s Streamlit & Plotly")


# ==============================================================================
# TAB 5: TEMPO, REKORDI IN NAJBOLJŠI ČASI (TEK / HOJA / KOLO)
# ==============================================================================
with tab_pace:
    st.header("🏃 Tempo in rekordi (krivulja tempa, najboljši časi, VDOT in kolesarski vzponi)")
    c_ph1, c_ph2 = st.columns([3, 1])
    with c_ph1:
        st.caption("Rekordi se računajo iz sekundnih tokov (1 Hz), kjer so na voljo; sicer iz Garmin povzetkov (najhitrejši odseki), krogov in interpolacije (označeno po viru).")
    with c_ph2:
        if st.button("🔄 Osveži in preračunaj rekorde", key="pace_refresh_btn", use_container_width=True, type="primary"):
            st.cache_data.clear()
            for _k in [k for k in list(st.session_state.keys()) if str(k).startswith("rec_cache_")]:
                del st.session_state[_k]
            st.session_state["pace_refreshed_at"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            st.rerun()
    if st.session_state.get("pace_refreshed_at"):
        st.caption(f"Rekordi zadnjič preračunani: {st.session_state['pace_refreshed_at']}")

    try:
        pace_sport = st.radio("Šport:", ["Tek", "Hoja / Pohodništvo", "Tek + hoja", "Kolesarjenje"], horizontal=True, key="pace_sport_radio")
        _groups = {"Tek": ["Tek"], "Hoja / Pohodništvo": ["Hoja / Pohodništvo"], "Tek + hoja": ["Tek", "Hoja / Pohodništvo"], "Kolesarjenje": ["Kolesarjenje"]}[pace_sport]
        pc_start, pc_end, pc_label = render_period_navigator(nav_key_prefix="pace_tab", default_preset="Zadnjih 90 dni")

        _all_acts = raw_data.get("activities", {}) or {}
        _n_sport = sum(1 for a in _all_acts.values() if sport_group((a.get("activityType") or {}).get("typeKey")) in _groups)
        _n_streams = sum(1 for aid_, a in _all_acts.items() if sport_group((a.get("activityType") or {}).get("typeKey")) in _groups
                         and ((raw_data.get("streams", {}).get(aid_) or {}).get("summary") or {}).get("bestDist"))
        st.caption(f"Pregledanih aktivnosti: **{_n_sport}**, od tega s sekundnim tokom (1 Hz): **{_n_streams}**. Ostale so izračunane iz povzetkov, krogov in Riegel modela.")

        bd_period, pace_period, asc_period = gather_best_efforts_full(raw_data, _groups, pc_start, pc_end)
        bd_all, pace_all, asc_all = gather_best_efforts_full(raw_data, _groups)

        if not bd_all and not pace_all and not asc_all:
            st.info("Za izbrani šport v podatkih ni aktivnosti z razdaljo in časom. Po sinhronizaciji z Garminom se tukaj prikažejo rekordi.")
        elif pace_sport != "Kolesarjenje":
            # ---------------- 1. KRIVULJA TEMPA PO ČASU IN PO RAZDALJI ----------------
            st.subheader("1. Krivulja tempa (po času 10 s – 180 min in po razdalji 100 m – 30 km)")
            seasons_p = user_cfg.get("seasons", {})
            season_names = [s["name"] for s in seasons_p.values()]
            c_pc_ctl1, c_pc_ctl2 = st.columns([2, 1])
            with c_pc_ctl1:
                chosen = st.multiselect(
                    "Primerjava obdobij / sezon:", ["Izbrano obdobje", "Vsi časi"] + season_names,
                    default=["Izbrano obdobje", "Vsi časi"], key="pace_curve_ms")
            with c_pc_ctl2:
                curve_axis_mode = st.radio(
                    "Prikaz krivulje tempa:",
                    ["Po času (10 s – 180 min)", "Po razdalji (100 m – 30 km)"],
                    horizontal=True, key="pace_curve_axis_mode"
                )

            palette = ["#ea580c", "#2563eb", "#16a34a", "#d81b60", "#7e57c2", "#f59e0b", "#0891b2"]
            fig_pc = go.Figure()
            all_ticks = []

            for ci, nm in enumerate(chosen):
                if nm == "Izbrano obdobje":
                    env_t, env_d = pace_period, bd_period
                elif nm == "Vsi časi":
                    env_t, env_d = pace_all, bd_all
                else:
                    s_info = next((s for s in seasons_p.values() if s["name"] == nm), None)
                    try:
                        if s_info and s_info.get("type") == "rolling_42":
                            s_dt, e_dt = today_date - timedelta(days=42), today_date
                        else:
                            s_dt = datetime.strptime(s_info.get("start"), "%Y-%m-%d").date()
                            e_dt = datetime.strptime(s_info.get("end"), "%Y-%m-%d").date()
                    except Exception:
                        continue
                    env_d, env_t, _ = gather_best_efforts_full(raw_data, _groups, s_dt, e_dt)

                if curve_axis_mode.startswith("Po času"):
                    durs = sorted(int(k) for k in env_t if int(k) in PACE_CURVE_DURS or int(k) >= 10)
                    if not durs:
                        continue
                    paces = [env_t[str(d)]["pace"] for d in durs]
                    all_ticks += paces
                    fig_pc.add_trace(go.Scatter(
                        x=durs, y=paces, mode="lines+markers", marker=dict(size=5), name=nm,
                        line=dict(color=palette[ci % len(palette)], width=2.8),
                        customdata=[[DURATION_LABELS.get(d, f"{d}s"), fmt_pace(env_t[str(d)]["pace"]), env_t[str(d)]["date"], env_t[str(d)]["name"], env_t[str(d)].get("src", "")] for d in durs],
                        hovertemplate="<b>%{customdata[0]}: %{customdata[1]} /km</b><br>%{customdata[2]} · %{customdata[3]}<br>Vir: %{customdata[4]}<extra>" + nm + "</extra>"
                    ))
                else:
                    dists_m = sorted(int(k) for k in env_d if int(k) in RUN_DISTS)
                    if not dists_m:
                        continue
                    paces = [env_d[str(dm)]["sec"] / (dm / 1000.0) for dm in dists_m]
                    all_ticks += paces
                    fig_pc.add_trace(go.Scatter(
                        x=dists_m, y=paces, mode="lines+markers", marker=dict(size=5), name=nm,
                        line=dict(color=palette[ci % len(palette)], width=2.8),
                        customdata=[[DIST_LABELS.get(str(dm), f"{dm} m"), fmt_dur(env_d[str(dm)]["sec"]), fmt_pace(env_d[str(dm)]["sec"] / (dm / 1000.0)), env_d[str(dm)]["date"], env_d[str(dm)]["name"], env_d[str(dm)].get("src", "")] for dm in dists_m],
                        hovertemplate="<b>%{customdata[0]}: %{customdata[1]} (%{customdata[2]} /km)</b><br>%{customdata[3]} · %{customdata[4]}<br>Vir: %{customdata[5]}<extra>" + nm + "</extra>"
                    ))

            if all_ticks:
                lo, hi = min(all_ticks), max(all_ticks)
                step_y = 15 if (hi - lo) < 180 else 30
                tv = list(range(int(lo // step_y) * step_y, int(hi // step_y) * step_y + step_y * 2 + 1, step_y))
                if curve_axis_mode.startswith("Po času"):
                    xt = [10, 20, 30, 45, 60, 90, 120, 180, 240, 300, 360, 480, 600, 720, 900, 1200, 1500, 1800, 2400, 2700, 3600, 4500, 5400, 7200, 10800]
                    fig_pc.update_layout(
                        height=480, hovermode="x unified", margin=dict(l=40, r=20, t=30, b=40),
                        xaxis=dict(type="log", title="Trajanje napora (10 s – 180 min)", tickvals=xt, ticktext=[DURATION_LABELS.get(d, f"{d}s") for d in xt],
                                   range=[math.log10(10), math.log10(10800)], gridcolor="#334155"),
                        yaxis=dict(title="Tempo (min/km) – hitreje zgoraj", autorange="reversed", tickmode="array", tickvals=tv, ticktext=[fmt_pace(v) for v in tv], gridcolor="#334155"),
                        legend=dict(orientation="h", y=1.05))
                else:
                    xt_d = [100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 21097, 30000]
                    fig_pc.update_layout(
                        height=480, hovermode="x unified", margin=dict(l=40, r=20, t=30, b=40),
                        xaxis=dict(type="log", title="Razdalja (100 m – 30 km)", tickvals=xt_d, ticktext=[DIST_LABELS.get(str(d), f"{d}m") for d in xt_d],
                                   range=[math.log10(100), math.log10(30000)], gridcolor="#334155"),
                        yaxis=dict(title="Tempo (min/km) – hitreje zgoraj", autorange="reversed", tickmode="array", tickvals=tv, ticktext=[fmt_pace(v) for v in tv], gridcolor="#334155"),
                        legend=dict(orientation="h", y=1.05))
                st.plotly_chart(fig_pc, use_container_width=True, config={"scrollZoom": True}, key="pace_curve_chart")
            else:
                st.info("V izbranih obdobjih ni naporov s tempom.")

            # ---------------- 2. NAJBOLJŠI ČASI NA VSEH RAZDALJAH (100 m – 30 km) IN PO ČASU ----------------
            st.subheader("2. Najboljši časi na vseh vmesnih in standardnih razdaljah (100 m – 30 km)")
            scope = st.radio("Obseg tabele rekordov:", ["Izbrano obdobje", "Vsi časi"], horizontal=True, key="pace_scope_radio")
            src_bd = bd_period if scope == "Izbrano obdobje" else bd_all
            src_pe = pace_period if scope == "Izbrano obdobje" else pace_all

            rows, bar_x, bar_y, bar_t, bar_y_all = [], [], [], [], []
            for D in [str(x) for x in RUN_DISTS]:
                e = src_bd.get(D)
                ea = bd_all.get(D)
                if not e:
                    continue
                pace = e["sec"] / (int(D) / 1000.0)
                hr_v = e.get("hr")
                rows.append({
                    "Razdalja": DIST_LABELS.get(D, f"{D} m"),
                    "Čas": fmt_dur(e["sec"]),
                    "Tempo (min/km)": f"{fmt_pace(pace)} /km",
                    "Hitrost (km/h)": f"{(int(D) / 1000.0) / (e['sec'] / 3600.0):.1f} km/h",
                    "GAP (ocena)": f"{fmt_pace(gap_pace(pace, e.get('asc'), int(D)))} /km",
                    "Povp. HR": f"{int(round(hr_v))} bpm" if hr_v else "--",
                    "Datum": e["date"],
                    "Aktivnost": e["name"],
                    "Vir": e.get("src", ""),
                })
                bar_x.append(DIST_LABELS.get(D, D))
                bar_y.append(pace)
                bar_t.append(f"{fmt_pace(pace)}")
                bar_y_all.append(ea["sec"] / (int(D) / 1000.0) if ea else None)

            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(x=bar_x, y=bar_y, text=bar_t, textposition="outside", name=scope, marker_color="#ea580c"))
                if scope == "Izbrano obdobje":
                    fig_bar.add_trace(go.Bar(x=bar_x, y=bar_y_all, text=[fmt_pace(v) if v else "" for v in bar_y_all], textposition="outside",
                                             name="Vsi časi", marker_color="#2563eb"))
                _tvb = list(range(0, int(max([v for v in bar_y + [v for v in bar_y_all if v]]) // 60 + 2) * 60, 60))
                fig_bar.update_layout(barmode="group", height=420, title="Primerjava tempa po vseh razdaljah (krajši stolpec = hitrejši tempo)",
                                      yaxis=dict(title="Tempo (min/km)", tickmode="array", tickvals=_tvb, ticktext=[fmt_pace(v) for v in _tvb]),
                                      legend=dict(orientation="h", y=-0.25), margin=dict(l=20, r=20, t=50, b=40))
                st.plotly_chart(fig_bar, use_container_width=True, key="pace_best_bar")
            else:
                st.info("Za standardne razdalje ni dovolj dolgih naporov v izbranem obsegu.")

            with st.expander("⏱️ Tabela najboljših tempov po trajanju (10 s – 180 min)"):
                dur_rows = []
                for d_sec in PACE_CURVE_DURS:
                    ep = src_pe.get(str(d_sec))
                    if not ep:
                        continue
                    p_s = ep["pace"]
                    dist_est_m = int(round((d_sec / p_s) * 1000.0)) if p_s > 0 else 0
                    dur_rows.append({
                        "Trajanje": DURATION_LABELS.get(d_sec, f"{d_sec}s"),
                        "Najboljši tempo": f"{fmt_pace(p_s)} /km",
                        "Hitrost": f"{3600.0 / p_s:.1f} km/h" if p_s > 0 else "--",
                        "Pretečena razdalja": f"{dist_est_m} m" if dist_est_m < 1000 else f"{dist_est_m / 1000.0:.2f} km",
                        "Datum": ep["date"],
                        "Aktivnost": ep["name"],
                        "Vir": ep.get("src", ""),
                    })
                if dur_rows:
                    st.dataframe(pd.DataFrame(dur_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("Ni podatkov po trajanju za izbrani obseg.")

            # ---------------- 3. VDOT IN NAPOVED TEKMOVALNIH ČASOV + DANIELS CONE ----------------
            st.markdown("---")
            st.subheader("3. VDOT izračun, napoved tekmovalnih časov in Daniels trenažni tempi")
            st.caption("Izračun po uradni formuli Jacka Danielsa (Resting & Exercise VO2 + %VO2max glede na trajanje). Upošteva najboljši veljaven tekaški napor od 1 km do polmaratona (z možnostjo ročnega vnosa).")

            # Zberi kandidate (1000 m do 21097 m): najprej iz izbranega obdobja, sicer iz vseh časov
            vdot_dists = [1000, 1500, 1609, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 18000, 20000, 21097]
            cands_period = []
            for D_i in vdot_dists:
                e_v = bd_period.get(str(D_i))
                if e_v and e_v.get("sec", 0) >= 150:
                    vd_val = vdot_from(D_i, e_v["sec"])
                    if 20.0 <= vd_val <= 85.0:
                        cands_period.append((vd_val, str(D_i), e_v, "Izbrano obdobje"))

            cands_all = []
            for D_i in vdot_dists:
                e_v = bd_all.get(str(D_i))
                if e_v and e_v.get("sec", 0) >= 150:
                    vd_val = vdot_from(D_i, e_v["sec"])
                    if 20.0 <= vd_val <= 85.0:
                        cands_all.append((vd_val, str(D_i), e_v, "Vsi časi"))

            c_vd_mode, c_vd_man1, c_vd_man2 = st.columns([2, 1, 1])
            with c_vd_mode:
                vdot_src_choice = st.radio(
                    "Vir za izračun VDOT:",
                    ["Avtomatsko (najboljši dosežek v obdobju / zgodovini)", "Vsi časi (absolutni rekord)", "Ročni vnos rezultata"],
                    horizontal=True, key="vdot_src_mode"
                )
            manual_vd = None
            manual_desc = ""
            if vdot_src_choice == "Ročni vnos rezultata":
                with c_vd_man1:
                    man_dist_lbl = st.selectbox(
                        "Razdalja:",
                        ["1500 m", "3 km", "5 km", "10 km", "Polmaraton (21,1 km)", "Maraton (42,2 km)"],
                        index=2, key="vdot_man_dist"
                    )
                with c_vd_man2:
                    man_time_str = st.text_input("Čas (mm:ss ali hh:mm:ss):", value="19:30", key="vdot_man_time")
                man_d_map = {"1500 m": 1500, "3 km": 3000, "5 km": 5000, "10 km": 10000, "Polmaraton (21,1 km)": 21097, "Maraton (42,2 km)": 42195}
                man_sec = _mmss_to_sec(man_time_str)
                if man_sec > 60:
                    manual_vd = vdot_from(man_d_map[man_dist_lbl], man_sec)
                    manual_desc = f"Ročni vnos: {man_dist_lbl} v času {man_time_str}"

            active_vd, active_vD, active_e, active_scope_lbl = None, None, None, ""
            if manual_vd and manual_vd > 15:
                active_vd = manual_vd
                active_scope_lbl = manual_desc
            elif vdot_src_choice == "Vsi časi (absolutni rekord)" and cands_all:
                active_vd, active_vD, active_e, active_scope_lbl = max(cands_all, key=lambda x: x[0])
            elif cands_period:
                active_vd, active_vD, active_e, active_scope_lbl = max(cands_period, key=lambda x: x[0])
            elif cands_all:
                active_vd, active_vD, active_e, active_scope_lbl = max(cands_all, key=lambda x: x[0])

            if active_vd and active_vd > 15:
                c_v1, c_v2 = st.columns([1.1, 2.2])
                with c_v1:
                    if active_e and active_vD:
                        ref_pace = fmt_pace(active_e["sec"] / (int(active_vD) / 1000.0))
                        st.metric(
                            "Ocenjen VDOT (Jack Daniels)",
                            f"{active_vd:.1f} ml/kg/min",
                            f"{DIST_LABELS.get(active_vD, active_vD)}: {fmt_dur(active_e['sec'])} ({ref_pace} /km)",
                            delta_color="off",
                        )
                        st.caption(f"📅 Doseženo: **{active_e['date']}** · *{active_e['name']}* ({active_scope_lbl})")
                    else:
                        st.metric("Ocenjen VDOT (Jack Daniels)", f"{active_vd:.1f} ml/kg/min", active_scope_lbl, delta_color="off")

                    if cands_all and active_scope_lbl == "Izbrano obdobje":
                        best_all_vd, best_all_d, best_all_e, _ = max(cands_all, key=lambda x: x[0])
                        if best_all_vd > active_vd + 0.2:
                            st.caption(f"🏆 Rekord vseh časov: **VDOT {best_all_vd:.1f}** ({DIST_LABELS.get(best_all_d, best_all_d)}: {fmt_dur(best_all_e['sec'])}, {best_all_e['date']})")

                with c_v2:
                    st.markdown("**🏁 Napoved tekmovalnih časov in ciljnega tempa (Daniels Race Predictor):**")
                    race_targets = [
                        ("1500 m", 1500),
                        ("3 km", 3000),
                        ("5 km", 5000),
                        ("10 km", 10000),
                        ("15 km", 15000),
                        ("Polmaraton (21,1 km)", 21097),
                        ("Maraton (42,2 km)", 42195),
                    ]
                    pred_rows = []
                    for nm_r, dm_r in race_targets:
                        t_pred = predict_time(active_vd, dm_r)
                        p_pred = t_pred / (dm_r / 1000.0)
                        spd_pred = (dm_r / 1000.0) / (t_pred / 3600.0)
                        actual_e = (bd_period.get(str(dm_r)) or bd_all.get(str(dm_r)))
                        actual_str = f"{fmt_dur(actual_e['sec'])} ({actual_e['date']})" if actual_e else "--"
                        pred_rows.append({
                            "Tekmovalna razdalja": nm_r,
                            "Napovedan čas": fmt_dur(t_pred),
                            "Ciljni tempo (min/km)": f"{fmt_pace(p_pred)} /km",
                            "Hitrost (km/h)": f"{spd_pred:.1f} km/h",
                            "Najboljši zabeležen čas": actual_str,
                        })
                    st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

                st.markdown("**🎯 Priporočene Daniels trenažne cone tempa (glede na vaš VDOT):**")
                st.dataframe(pd.DataFrame(daniels_training_paces(active_vd)), use_container_width=True, hide_index=True)
            else:
                st.info("Za izračun VDOT je potreben vsaj en tek ≥ 1 km ali ročni vnos časa.")
        else:
            # ---------------- KOLESARSKI REKORDI ----------------
            st.subheader("🚴 Kolesarski rekordi: vzponi (1h VAM & največ višincev), hitrost na razdaljah in W/kg")
            scope_c = st.radio("Obseg:", ["Izbrano obdobje", "Vsi časi"], horizontal=True, key="pace_scope_radio_cyc")
            src_c = bd_period if scope_c == "Izbrano obdobje" else bd_all
            _rs, _re = (pc_start, pc_end) if scope_c == "Izbrano obdobje" else (None, None)

            # Zberi vse kolesarske vožnje za lestvico vzponov (obdobje in vsi časi)
            def _collect_climb_rides(r_start=None, r_end=None):
                rides = []
                for aid_, a in _all_acts.items():
                    if sport_group((a.get("activityType") or {}).get("typeKey")) != "Kolesarjenje":
                        continue
                    d_s = (a.get("startTimeLocal") or "")[:10]
                    try:
                        d_o = datetime.strptime(d_s, "%Y-%m-%d").date()
                    except Exception:
                        continue
                    if (r_start and d_o < r_start) or (r_end and d_o > r_end):
                        continue
                    gain_v = float(a.get("elevationGain") or 0)
                    if gain_v <= 0:
                        continue
                    dur_v = float(a.get("movingDuration") or a.get("duration") or 0)
                    dist_v_km = float(a.get("distance") or 0) / 1000.0
                    sm_c = (raw_data.get("streams", {}).get(str(aid_)) or {}).get("summary") or {}
                    avg_w_c = sm_c.get("trueAvgPower") or a.get("avgPower") or a.get("averagePower")
                    np_w_c = sm_c.get("np") or a.get("normPower") or a.get("normalizedPower") or avg_w_c
                    pwr_txt = f"{int(round(avg_w_c))} W (NP {int(round(np_w_c))} W)" if avg_w_c else "--"
                    rides.append({
                        "Datum": d_s,
                        "Vožnja": a.get("activityName", "Kolesarjenje"),
                        "Skupni vzpon (m)": int(round(gain_v)),
                        "Razdalja (km)": round(dist_v_km, 1),
                        "Čas gibanja": fmt_dur(dur_v),
                        "Avg W / NP": pwr_txt,
                        "Povp. VAM celotne vožnje": f"{int(round(gain_v / (dur_v / 3600.0)))} m/h" if dur_v >= 1200 else "--",
                    })
                rides.sort(key=lambda r: r["Skupni vzpon (m)"], reverse=True)
                return rides

            climb_period_list = _collect_climb_rides(pc_start, pc_end)
            climb_all_list = _collect_climb_rides(None, None)
            active_climb_list = climb_period_list if scope_c == "Izbrano obdobje" else climb_all_list

            # --- 1. REKORDI VZPONA (1H VAM & NAJVEČ VIŠINCEV NA VOŽNJI) ---
            st.markdown("#### ⛰️ 1. Rekordi vzpona (Največ višinskih metrov v 1 h & Največ višincev na posamezni vožnji)")
            a_src = asc_period if scope_c == "Izbrano obdobje" else asc_all
            c_a1, c_a2, c_a3 = st.columns(3)

            with c_a1:
                if a_src:
                    vam_extra = ""
                    if a_src.get("vam20") or a_src.get("vam30"):
                        v20 = f"20m VAM: {int(round(a_src['vam20']))} m/h" if a_src.get("vam20") else ""
                        v30 = f"30m VAM: {int(round(a_src['vam30']))} m/h" if a_src.get("vam30") else ""
                        vam_extra = " | ".join([x for x in [v20, v30] if x])
                    st.metric(
                        "Največ višinskih metrov v 1 h (60 min VAM)",
                        f"+{a_src['m']:.0f} m/h",
                        f"{a_src['date']} · {a_src['name']}",
                        delta_color="off",
                    )
                    if vam_extra:
                        st.caption(f"🔥 {vam_extra} ({a_src.get('src', 'tok 1 Hz')})")
                    else:
                        st.caption(f"Vir: {a_src.get('src', 'tok 1 Hz')}")
                else:
                    st.metric("Največ višinskih metrov v 1 h (60 min VAM)", "--", "Ni voženj ≥ 45 min v obdobju", delta_color="off")

            with c_a2:
                if climb_period_list:
                    top_p = climb_period_list[0]
                    st.metric(
                        "Največ višincev na vožnji (Izbrano obdobje)",
                        f"+{top_p['Skupni vzpon (m)']} m",
                        f"{top_p['Datum']} · {top_p['Razdalja (km)']} km ({top_p['Čas gibanja']})",
                        delta_color="off",
                    )
                    st.caption(f"🚴 **{top_p['Vožnja']}** · Moč: {top_p['Avg W / NP']}")
                else:
                    st.metric("Največ višincev na vožnji (Izbrano obdobje)", "--", "Ni voženj v obdobju", delta_color="off")

            with c_a3:
                if climb_all_list:
                    top_a = climb_all_list[0]
                    st.metric(
                        "🏆 Največ višincev na vožnji (Vsi časi)",
                        f"+{top_a['Skupni vzpon (m)']} m",
                        f"{top_a['Datum']} · {top_a['Razdalja (km)']} km ({top_a['Čas gibanja']})",
                        delta_color="off",
                    )
                    st.caption(f"🚴 **{top_a['Vožnja']}** · Moč: {top_a['Avg W / NP']}")
                else:
                    st.metric("🏆 Največ višincev na vožnji (Vsi časi)", "--", delta_color="off")

            if active_climb_list:
                st.markdown(f"**🏔️ Lestvica Top 5 voženj z največ višinskimi metri ({scope_c}):**")
                st.dataframe(pd.DataFrame(active_climb_list[:5]), use_container_width=True, hide_index=True)

            # --- 2. HITROST NA RAZDALJAH ---
            st.markdown("---")
            st.markdown("#### 🚀 2. Najhitrejši časi in povprečne hitrosti na razdaljah (1 km – 100 km)")
            rows, bx, by = [], [], []
            for D in [str(x) for x in CYC_DISTS]:
                e = src_c.get(D)
                if not e:
                    continue
                spd = int(D) / 1000.0 / (e["sec"] / 3600.0)
                label_ = "40 km kronometer" if D == "40000" else DIST_LABELS.get(D, f"{int(D)//1000} km")
                hr_v = e.get("hr")
                rows.append({
                    "Razdalja": label_,
                    "Čas": fmt_dur(e["sec"]),
                    "Hitrost": f"{spd:.1f} km/h",
                    "Povp. HR": f"{int(round(hr_v))} bpm" if hr_v else "--",
                    "Datum": e["date"],
                    "Aktivnost": e["name"],
                    "Vir": e.get("src", ""),
                })
                bx.append(label_)
                by.append(round(spd, 1))
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                fcb2 = go.Figure(go.Bar(x=bx, y=by, text=[f"{v} km/h" for v in by], textposition="outside", marker_color="#2563eb"))
                fcb2.update_layout(height=340, title="Najhitrejše povprečne hitrosti po razdaljah", yaxis=dict(title="km/h"), margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fcb2, use_container_width=True, key="cyc_speed_bar")
            else:
                st.info("Ni kolesarskih voženj z razdaljo v izbranem obsegu.")

            # --- 3. NAJBOLJŠI NAPORI V W/KG ---
            st.markdown("---")
            st.markdown("#### ⚡ 3. Najboljši napori v W in W/kg")
            _wref = float(user_cfg.get("weight_kg", 65.5))
            wk_rec = cycling_wkg_records(raw_data, _rs, _re, _wref)
            if wk_rec:
                wk_rows = [{
                    "Trajanje": DURATION_LABELS.get(d, f"{d}s"),
                    "Moč": f"{v['w']:.0f} W",
                    "W/kg": f"{v['wkg']:.2f} W/kg",
                    "Datum": v["date"],
                    "Aktivnost": v["name"],
                } for d, v in sorted(wk_rec.items())]
                st.dataframe(pd.DataFrame(wk_rows), use_container_width=True, hide_index=True)
                st.caption(f"W/kg je izračunan z referenčno težo {_wref:.1f} kg iz profila.")
            else:
                st.info("Ni podatkov o moči za izbrano obdobje.")
    except Exception as ex_pace:
        st.error(f"Napaka pri izračunu rekordov: {ex_pace}")
        st.exception(ex_pace)
