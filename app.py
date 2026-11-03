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

# Varno uvažanje modulov z diagnostiko za Streamlit Cloud
try:
    import auth
    from modules import mnk_izravnava
    from modules import mreze_meritve
    from modules import transformacije
    from modules import deformacije
except Exception as err:
    st.error(f"⚠️ Napaka pri uvozu geodetskih modulov: {err}")
    st.info("Preverite, ali so vsi moduli v mapi `modules/` naloženi na GitHub skupaj z datoteko `__init__.py`.")
    st.code(f"Sistemska pot sys.path: {sys.path}\nKorenski imenik: {ROOT_DIR}")
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

        st.divider()

        # IZBIRA GEODETSKEGA PODPROGRAMA (MODULA)
        st.markdown("### 🧭 Geodetski Studio Moduli")
        izbrani_modul = st.selectbox(
            "Izbira geodetskega podprograma:",
            [
                "📐 1. Izravnalni račun in analiza mrež (MNK)",
                "🗺️ 2. Izravnava 1D/2D mrež iz meritev (Beta)",
                "🔄 3. Koordinatne transformacije D48/GK ⇄ D96/TM (Beta)",
                "📊 4. Deformacijska analiza in premiki (Beta)"
            ],
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

            st.divider()
            st.markdown("#### ⚡ Hiter izračun")
            if st.button("⚡ IZVEDI IZRAVNAVO IN TESTE", type="primary", use_container_width=True):
                if mnk_izravnava.izvedi_izravnavo_jedro():
                    st.success("Izravnava uspešno izvedena! Rezultati so v zavihkih 3 in 4.")

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
    # GLAVNA GLAVA APLIKACIJE
    # =============================================================================
    st.markdown("""
    <div class="main-header">
        <h2>📐 GEODETSKI STUDIO — MULTI-MODULARNA PLATFORMA ZA IZRAVNAVE IN ANALIZO MREŽ</h2>
        <p>Metoda najmanjših kvadratov (MNK) • Baarda Data Snooping & MDB • Plotly interaktivne elipse • Tab S11 Touch UI</p>
    </div>
    """, unsafe_allow_html=True)

    # =============================================================================
    # USMERJANJE NA IZBRANI GEODETSKI PODPROGRAM
    # =============================================================================
    if "1. Izravnalni račun" in izbrani_modul:
        mnk_izravnava.render_mnk_module(current_user)

    elif "2. Izravnava 1D/2D" in izbrani_modul:
        mreze_meritve.render_mreze_module(current_user)

    elif "3. Koordinatne transformacije" in izbrani_modul:
        transformacije.render_transformacije_module(current_user)

    elif "4. Deformacijska analiza" in izbrani_modul:
        deformacije.render_deformacije_module(current_user)
