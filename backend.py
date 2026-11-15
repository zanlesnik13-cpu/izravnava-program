"""
backend.py - Upravljanje podatkovnega sloja, sinhronizacije in varnostnih kopij
za večuporabniški sistem Garmin Kolesarski Trener.
Integrirano z neposredno GitHub Contents REST API sinhronizacijo (`github_sync.py`)
in lokalno JSON shrambo (`data/users/<username>/...`).
"""

import os
import sys
import json
import subprocess
import auth
import github_sync

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
USERS_DIR = os.path.join(DATA_DIR, "users")
AUTH_PATH = os.path.join(DATA_DIR, "auth_users.json")


def init_storage():
    """Zagotovi obstoj vseh potrebnih map za Streamlit Cloud in lokalni zagon."""
    auth.init_storage()


def sync_user_data(username: str, api_key: str = None, days: int = 365, full: bool = False):
    """Zažene pull.py za določenega uporabnika v popolnoma izoliranem okolju."""
    u_clean = username.lower().strip()
    pull_script = os.path.join(PROJECT_DIR, "pull.py")
    cmd = [sys.executable, pull_script, "--user", u_clean]
    if api_key and api_key.strip():
        cmd.extend(["--api-key", api_key.strip()])
    if full or days == 0:
        cmd.extend(["--all", "--full"])
    else:
        cmd.extend(["--days", str(days)])

    timeout_s = 3600 if (full or days == 0) else 600
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    # Po sinhronizaciji počisti Streamlit predpomnilnik in sproži prenos na GitHub
    if res.returncode == 0:
        try:
            import streamlit as st
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:
            pass
        github_sync.github_push_user_files(u_clean, async_push=True)
    return res


def create_user_backup_zip(username: str, include_streams: bool = True) -> bytes:
    """Ustvari celotno ZIP varnostno kopijo (vključno s podmapami splits, streams, mmp)."""
    return github_sync.create_user_backup_zip_bytes(username, include_streams=include_streams)


def create_user_backup_json(username: str) -> bytes:
    """Ustvari strnjeno JSON varnostno kopijo vseh ključnih datotek profila."""
    return github_sync.create_user_backup_json_bytes(username)


def restore_user_backup(username: str, file_bytes: bytes, filename: str) -> tuple[bool, str]:
    """Obnovi podatke uporabnika iz naložene ZIP ali JSON datoteke ter v ozadju posodobi GitHub."""
    return github_sync.restore_from_uploaded_backup(username, file_bytes, filename)


# ==========================================
# GITHUB CLOUD FUNKCIJE (PER-FILE REST API)
# ==========================================
def github_push_file(local_rel_path: str, commit_message: str = "Update data", async_push: bool = True):
    return github_sync.github_push_file(local_rel_path, commit_message=commit_message, async_push=async_push)


def github_push_user_files(username: str, only_files: list = None, include_streams: bool = False, async_push: bool = True):
    return github_sync.github_push_user_files(
        username=username,
        only_files=only_files,
        include_streams=include_streams,
        async_push=async_push,
    )


def github_restore_all_users_on_startup(force: bool = False):
    return github_sync.github_restore_all_users_on_startup(force=force)


def get_github_status(username: str = ""):
    return github_sync.get_sync_status_info(username)


def save_github_config(token: str, repo: str, branch: str = "main"):
    return github_sync.save_local_github_config(token, repo, branch)


def ensure_full_activity_details(activity_id, username: str, config: dict = None, force: bool = False):
    """
    Zagotovi, da ima izbrana aktivnost (tudi starejša od 14 dni) na disku:
    - veljavne GPS koordinate (`coordinates` / `polyline`),
    - uradne Garmin kroge (`splits/{activity_id}.json`),
    - 1-sekundni tok metrik (`streams/{activity_id}.json`) in 10-minutne bloke.
    Če podatki že obstajajo lokalno na disku, jih naloži takoj brez mrežnega klica!
    """
    import pull
    return pull.ensure_full_activity_details(
        activity_id=activity_id,
        username=username,
        config=config,
        force=force,
    )


# Samodejni zagon inicializacije shrambe
init_storage()
