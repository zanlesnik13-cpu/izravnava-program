import os
import sys
from pathlib import Path

# =============================================================================
# ZAGOTOVITEV PRAVILNIH POTI (ROBUSTNOST ZA STREAMLIT CLOUD IN LOKALNI ZAGON)
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Začetna nastavitev okolja za stabilno delovanje Matplotliba v peskovniku in oblačnem okolju
os.environ["MPLCONFIGDIR"] = os.path.join(str(ROOT_DIR), ".matplotlib_cache")

import streamlit as st

# Varno uvažanje modulov z diagnostiko za Streamlit Cloud in lokalni zagon
try:
    import auth
    import github_sync
    from modules import mnk_izravnava
    from modules import mreze_meritve
    from modules import transformacije
    from modules import deformacije
    from modules import satelitska_vaja3
except ModuleNotFoundError as e_mod:
    # Rezervni poskus uvoza v kolikor bi bili moduli v korenskem imeniku
    try:
        import auth
        import github_sync
        import mnk_izravnava
        import mreze_meritve
        import transformacije
        import deformacije
        import satelitska_vaja3
    except Exception:
        st.error(f"⚠️ Napaka pri uvozu modulov: {e_mod}")
        st.warning("Na GitHub niste naložili mape **`modules/`** ali datoteke **`modules/__init__.py`**!")
        st.info("V Git repozitorij dodajte mapo `modules/` z ukazom: `git add modules/` in `git push`.")
        st.code(f"Korenski imenik: {ROOT_DIR}\nRazpoložljive datoteke: {os.listdir(str(ROOT_DIR))}")
        st.stop()
except Exception as err:
    st.error(f"⚠️ Nepričakovana napaka pri nalaganju modulov: {err}")
    st.stop()


# =============================================================================
# VARNO PREVERJANJE MAPE Z GEODETSKIMI VIRI IN LITERATURO (viri/)
# =============================================================================
VIRI_DIR = ROOT_DIR / "viri"

def preveri_in_pridobi_vire():
    """Varno preveri obstoj mape 'viri' in vrne seznam razpoložljivih referenčnih dokumentov."""
    if not VIRI_DIR.exists():
        try:
            VIRI_DIR.mkdir(parents=True, exist_ok=True)
            gitkeep = VIRI_DIR / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
        except Exception:
            pass
        return []
    try:
        return [f.name for f in VIRI_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]
    except Exception:
        return []


# =============================================================================
# NASTAVITVE STRANI IN SLOGA ZA SAMSUNG GALAXY TAB S11 ULTRA IN DESKTOP
# =============================================================================
st.set_page_config(
    page_title="Geodetski Studio (MNK & Analiza Mrež)",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Prilagoditev velikosti gumbov in tabel za zaslon na dotik, S Pen in miško
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        height: 3.1em;
        font-weight: bold;
        border-radius: 8px;
    }
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #0369a1 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 10px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
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
    .user-badge {
        background-color: #f1f5f9;
        border-left: 4px solid #2563eb;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Inicializacija baze podatkov za uporabnike in shranjevanje
auth.init_storage()

# =============================================================================
# PREVERJANJE PRIJAVE UPORABNIKA
# =============================================================================
if st.session_state.get("current_user") is None:
    auth.render_auth_screen()
    st.stop()
else:
    # Uporabnik je prijavljen
    current_user = st.session_state["current_user"]
    username = current_user["username"]
    display_name = current_user.get("display_name", username)
    role = current_user.get("role", "Študent / Geodet")

    MODULI_SEZNAM = [
        "📐 1. Izravnalni račun in analiza mrež (MNK & gPPP)",
        "🛰️ 2. Satelitska geodezija (Vaja 3)",
        "📊 3. Primerjava koordinat med dvema .txt datotekama",
        "🗺️ 4. Izravnava 1D/2D mrež iz meritev (Beta)",
        "🔄 5. Koordinatne transformacije D48/GK ⇄ D96/TM (Beta)",
        "📈 6. Deformacijska analiza in premiki (Beta)"
    ]

    if "app_selected_module" not in st.session_state or st.session_state["app_selected_module"] not in MODULI_SEZNAM:
        st.session_state["app_selected_module"] = MODULI_SEZNAM[0]
    if "sb_geodetic_module_select" not in st.session_state or st.session_state["sb_geodetic_module_select"] not in MODULI_SEZNAM:
        st.session_state["sb_geodetic_module_select"] = st.session_state["app_selected_module"]
    if "main_geodetic_module_radio" not in st.session_state or st.session_state["main_geodetic_module_radio"] not in MODULI_SEZNAM:
        st.session_state["main_geodetic_module_radio"] = st.session_state["app_selected_module"]

    def _on_sidebar_module_change():
        izb = st.session_state["sb_geodetic_module_select"]
        st.session_state["app_selected_module"] = izb
        st.session_state["main_geodetic_module_radio"] = izb

    def _on_main_module_change():
        izb = st.session_state["main_geodetic_module_radio"]
        st.session_state["app_selected_module"] = izb
        st.session_state["sb_geodetic_module_select"] = izb

    # =============================================================================
    # STRANSKA VRSTICA (SIDEBAR): UPORABNIK, MODULI IN HITRA DEJANJA
    # =============================================================================
    with st.sidebar:
        # Profilna kartica uporabnika
        st.markdown(f"""
        <div class="user-badge">
            <div style="font-weight: bold; color: #1e3a8a; font-size: 1.05rem;">👤 {display_name}</div>
            <div style="color: #64748b; font-size: 0.82rem;">@{username} • {role}</div>
        </div>
        """, unsafe_allow_html=True)

        col_prof1, col_prof2 = st.columns(2)
        with col_prof1:
            if st.button("🚪 Odjava", use_container_width=True):
                st.session_state["current_user"] = None
                st.session_state["mnk_initialized"] = False
                st.rerun()

        with col_prof2:
            backup_json = auth.export_user_backup_json(username)
            st.download_button(
                label="📥 Varnostna kopija",
                data=backup_json,
                file_name=f"geodetski_profil_{username}.json",
                mime="application/json",
                use_container_width=True,
                help="Prenesite vse svoje shranjene izravnave in profil v datoteko JSON"
            )

        # Uvoz varnostne kopije
        with st.expander("📤 Uvozi profil / projekte (.json)", expanded=False):
            uploaded_backup = st.file_uploader("Naložite .json datoteko:", type=["json"], key="sidebar_backup_upload")
            if uploaded_backup is not None:
                raw_str = uploaded_backup.getvalue().decode("utf-8")
                ok_imp, msg_imp = auth.import_user_backup_json(username, raw_str)
                if ok_imp:
                    st.success(msg_imp)
                    st.rerun()
                else:
                    st.error(msg_imp)

        # GitHub Cloud Sinhronizacija (Push / Pull)
        gh_cfg = github_sync.get_github_config()
        last_sync_str = st.session_state.get("gh_last_sync", "—")
        if st.button("🔄 Sinhroniziraj podatke z GitHubom (Push / Pull)", use_container_width=True):
            ok_pull, msg_pull, _ = github_sync.github_restore_all_users_on_startup(force=True)
            ok_push, msg_push, _ = github_sync.github_push_all_local_data()
            if ok_push:
                st.toast("☁️ Sinhronizacija z GitHubom uspešna!")
                st.success(msg_push)
            else:
                st.warning(msg_push)

        with st.expander("☁️ GitHub Cloud Sinhronizacija (Status)", expanded=False):
            if gh_cfg["enabled"]:
                st.markdown(f"**Status:** 🟢 Povezano z GitHubom\n\n- **Repozitorij:** `{gh_cfg['repo']}`\n- **Veja:** `{gh_cfg['branch']}`\n- **Zadnja sinhronizacija:** `{last_sync_str}`")
            else:
                st.markdown("**Status:** 🟡 Lokalni način\n\nZa samodejno shranjevanje na **Streamlit Cloud** dodajte v **App settings ➔ Secrets**:")
                st.code('GITHUB_TOKEN = "ghp_..."\nGITHUB_REPO = "ime/izravnava-program"\nGITHUB_BRANCH = "main"', language="toml")
            s_tok = st.text_input("GITHUB_TOKEN:", value=gh_cfg["token"], type="password", key="sb_gh_tok")
            s_rep = st.text_input("GITHUB_REPO (owner/repo):", value=gh_cfg["repo"], key="sb_gh_rep")
            s_br = st.text_input("GITHUB_BRANCH:", value=gh_cfg["branch"], key="sb_gh_br")
            if st.button("💾 Shrani nastavitve povezave za to sejo", use_container_width=True):
                st.session_state["gh_manual_token"] = s_tok.strip()
                st.session_state["gh_manual_repo"] = s_rep.strip()
                st.session_state["gh_manual_branch"] = s_br.strip() or "main"
                st.rerun()

        st.divider()

        # IZBIRA GEODETSKEGA PODPROGRAMA (MODULA)
        st.markdown("### 🧭 Geodetski Moduli")
        st.selectbox(
            "Izbira geodetskega podprograma:",
            MODULI_SEZNAM,
            key="sb_geodetic_module_select",
            on_change=_on_sidebar_module_change
        )

        izbrani_modul = st.session_state["app_selected_module"]

        if "Satelitska geodezija" in izbrani_modul:
            st.selectbox(
                "Podprogram satelitske geodezije:",
                ["📡 Vaja 3: Koordinatni sistemi pri obravnavi GNSS na območju Slovenije"],
                index=0
            )

        # Če je izbran MNK modul, prikaži hitre šolske primere v stranski vrstici
        if "1. Izravnalni račun" in izbrani_modul:
            st.divider()
            st.markdown("#### 🚀 Hitri šolski primeri iz skripte")
            st.caption("Naložite in takoj izračunajte primere z enim dotikom:")
            if st.button("📦 1. Kocka (Posredna)"):
                mnk_izravnava.nalozi_primer_1()
                mnk_izravnava.izvedi_izravnavo_jedro()
            if st.button("📐 2. Kocka (Pogojna)"):
                mnk_izravnava.nalozi_primer_2()
                mnk_izravnava.izvedi_izravnavo_jedro()
            if st.button("📏 3. Premica (Splošni model)"):
                mnk_izravnava.nalozi_primer_3()
                mnk_izravnava.izvedi_izravnavo_jedro()
            if st.button("🗺️ 4. 2D Mreža (Elipse)"):
                mnk_izravnava.nalozi_primer_4()
                mnk_izravnava.izvedi_izravnavo_jedro()
            if st.button("🛰️ 5. GNSS gPPP (FGG2 — 3 epohe)"):
                mnk_izravnava.nalozi_primer_5_gppp()
                mnk_izravnava.izvedi_izravnavo_jedro()

            st.divider()
            st.markdown("#### ⚡ Hiter izračun")
            if st.button("⚡ IZVEDI IZRAVNAVO IN TESTE", type="primary", use_container_width=True):
                if mnk_izravnava.izvedi_izravnavo_jedro():
                    st.success("Izravnava uspešno izvedena! Rezultati so v zavihkih 3 in 4.")

        st.sidebar.caption("🟢 **Geodetski Studio v2.6** (Vaja 3 & Primerjava .txt aktivna)")

        # Prikaz geodetske referenčne literature (viri)
        seznam_virov = preveri_in_pridobi_vire()
        st.divider()
        with st.expander("📚 Strokovna geodetska literatura (viri)", expanded=False):
            if seznam_virov:
                st.markdown("**Referenčno gradivo v mapi `viri/`:**")
                for v in seznam_virov:
                    st.markdown(f"- 📄 `{v}`")
                st.caption("Vsi matematični modeli, formule in statistični testi v platformi temeljijo na teh uradnih geodetskih virih.")
            else:
                st.caption("Mapa `viri/` je pripravljena. Vanjo lahko naložite svojo geodetsko literaturo in skripte.")

    # =============================================================================
    # GLAVNA GLAVA APLIKACIJE Z OZNAKO RAZLIČICE
    # =============================================================================
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <h2 style="margin: 0; color: white;">📐 GEODETSKI STUDIO — MULTI-MODULARNA PLATFORMA</h2>
            <span style="background-color: #10b981; color: white; padding: 4px 12px; border-radius: 6px; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.5px; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
                v2.6 (Vaja 3 & Primerjava .txt)
            </span>
        </div>
        <p style="margin-top: 6px; color: #bfdbfe;">
            Metoda najmanjših kvadratov (MNK & gPPP) • Satelitska geodezija (Vaja 3: ITRF2020 ⇄ ETRS89 ⇄ D96-17/TM) • Primerjava dveh .txt datotek (Odstopanja)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =============================================================================
    # HITRA HORIZONTALNA NAVIGACIJA NA GLAVNEM ZASLONU (Tab S11 Ultra & Desktop)
    # =============================================================================
    st.markdown("""
    <div style="margin-bottom: 6px;">
        <span style="font-size: 0.85rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;">
            🧭 Izbira geodetskega modula (tudi če je stranska vrstica zaprta):
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.radio(
        "Glavna navigacija po modulih:",
        MODULI_SEZNAM,
        horizontal=True,
        key="main_geodetic_module_radio",
        on_change=_on_main_module_change,
        label_visibility="collapsed"
    )

    active_module = st.session_state["app_selected_module"]
    st.markdown("---")

    # =============================================================================
    # USMERJANJE NA IZBRANI GEODETSKI PODPROGRAM
    # =============================================================================
    if "1. Izravnalni račun" in active_module:
        mnk_izravnava.render_mnk_module(current_user)

    elif "2. Satelitska geodezija" in active_module:
        satelitska_vaja3.render_satelitska_vaja3_module(current_user)

    elif "3. Primerjava koordinat" in active_module:
        satelitska_vaja3.render_primerjava_koordinat_tab(ROOT_DIR)

    elif "4. Izravnava 1D/2D" in active_module:
        mreze_meritve.render_mreze_module(current_user)

    elif "5. Koordinatne transformacije" in active_module:
        transformacije.render_transformacije_module(current_user)

    elif "6. Deformacijska analiza" in active_module:
        deformacije.render_deformacije_module(current_user)
