import os
import re
import json
import base64
from datetime import datetime
from pathlib import Path
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent


# =============================================================================
# 1. AVTOMATSKA DETEKCIJA IN BRANJE KONFIGURACIJE ZA GITHUB REST API
# =============================================================================
def _detect_repo_from_git_config() -> str:
    """Poskusi prebrati 'owner/repo' iz lokalne datoteke .git/config, če obstaja."""
    git_cfg = ROOT_DIR / ".git" / "config"
    if git_cfg.exists():
        try:
            txt = git_cfg.read_text(encoding="utf-8", errors="ignore")
            # Iščemo github.com/owner/repo ali github.com:owner/repo
            m = re.search(r"github\.com[:/]([^/\s]+/[^/\s\.]+?)(?:\.git)?\s*$", txt, re.MULTILINE)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
    return ""


def _auto_resolve_repo_from_token(token: str) -> str:
    """
    Če je nastavljen samo GITHUB_TOKEN, ugotovi uporabniško ime preko /user API-ja
    in sestavi 'owner/<ime_mape_repozitorija>' (na Streamlit Cloud je mapa npr. 'izravnava-program').
    """
    if not token:
        return ""
    cached = st.session_state.get("_gh_auto_resolved_repo")
    if cached:
        return cached
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        resp = requests.get("https://api.github.com/user", headers=headers, timeout=5)
        if resp.status_code == 200:
            owner = resp.json().get("login", "")
            if owner:
                folder_name = ROOT_DIR.name
                for candidate in [folder_name, "izravnava-program", "geodezija_program"]:
                    r_check = requests.get(
                        f"https://api.github.com/repos/{owner}/{candidate}",
                        headers=headers,
                        timeout=5
                    )
                    if r_check.status_code == 200:
                        full_repo = f"{owner}/{candidate}"
                        st.session_state["_gh_auto_resolved_repo"] = full_repo
                        return full_repo
    except Exception:
        pass
    return ""


def get_github_config() -> dict:
    """
    Vrne slovar s konfiguracijo za GitHub Cloud sinhronizacijo:
    {"token": str, "repo": str, "branch": str, "enabled": bool}
    Prioriteta:
    1. Ročna nastavitev v trenutni seji (st.session_state)
    2. Streamlit Secrets (st.secrets["GITHUB_TOKEN"], st.secrets["GITHUB_REPO"], st.secrets["GITHUB_BRANCH"])
    3. Okoljske spremenljivke (os.environ)
    4. Avtomatska detekcija iz .git/config ali GitHub API
    """
    token = ""
    repo = ""
    branch = "main"

    # 1. Preveri st.secrets
    try:
        if "GITHUB_TOKEN" in st.secrets:
            token = str(st.secrets["GITHUB_TOKEN"]).strip()
        if "GITHUB_REPO" in st.secrets:
            repo = str(st.secrets["GITHUB_REPO"]).strip()
        if "GITHUB_BRANCH" in st.secrets:
            branch = str(st.secrets["GITHUB_BRANCH"]).strip() or "main"
    except Exception:
        pass

    # 2. Preveri okoljske spremenljivke
    if not token:
        token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not repo:
        repo = os.environ.get("GITHUB_REPO", "").strip()
    if os.environ.get("GITHUB_BRANCH"):
        branch = os.environ.get("GITHUB_BRANCH", "main").strip()

    # 3. Preveri začasne nastavitve v seji (vnesene v stranski vrstici)
    if st.session_state.get("gh_manual_token"):
        token = str(st.session_state["gh_manual_token"]).strip()
    if st.session_state.get("gh_manual_repo"):
        repo = str(st.session_state["gh_manual_repo"]).strip()
    if st.session_state.get("gh_manual_branch"):
        branch = str(st.session_state["gh_manual_branch"]).strip() or "main"

    # Čiščenje morebitnega polnega URL-ja v GITHUB_REPO (npr. https://github.com/uporabnik/izravnava-program.git)
    if repo.startswith("http://") or repo.startswith("https://"):
        m = re.search(r"github\.com/([^/\s]+/[^/\s\.]+?)(?:\.git)?/?$", repo)
        if m:
            repo = m.group(1)

    # 4. Avtomatska detekcija imena repozitorija, če ni izrecno podan
    if not repo:
        repo = _detect_repo_from_git_config()
    if not repo and token:
        repo = _auto_resolve_repo_from_token(token)

    enabled = bool(token and repo and "/" in repo and not token.startswith("ghp_VasOsebni"))
    return {
        "token": token,
        "repo": repo,
        "branch": branch,
        "enabled": enabled,
    }


def _gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


# =============================================================================
# 2. POTISKANJE (PUSH) IN PRidOBIVANJE (PULL) DATOTEK PREKO GITHUB REST API
# =============================================================================
def github_push_file(local_rel_path: str | Path, commit_message: str = None, silent: bool = False) -> tuple[bool, str]:
    """
    Prebere lokalno datoteko in jo preko GitHub Contents REST API ustvari ali posodobi
    v oddaljenem GitHub repozitoriju.
    """
    cfg = get_github_config()
    if not cfg["enabled"]:
        return False, "GitHub sinhronizacija ni aktivna (manjka GITHUB_TOKEN ali GITHUB_REPO)."

    rel_str = str(local_rel_path).replace("\\", "/").lstrip("/")
    abs_path = ROOT_DIR / rel_str
    if not abs_path.exists() or not abs_path.is_file():
        return False, f"Lokalna datoteka '{rel_str}' ne obstaja."

    try:
        raw_bytes = abs_path.read_bytes()
        b64_content = base64.b64encode(raw_bytes).decode("ascii")
    except Exception as e:
        return False, f"Napaka pri branju lokalne datoteke: {e}"

    token = cfg["token"]
    repo = cfg["repo"]
    branch = cfg["branch"]
    url = f"https://api.github.com/repos/{repo}/contents/{rel_str}"
    headers = _gh_headers(token)

    # 1. Preveri, ali datoteka že obstaja na GitHubu (pridobi SHA)
    sha = None
    try:
        get_resp = requests.get(url, headers=headers, params={"ref": branch}, timeout=8)
        if get_resp.status_code == 200:
            remote_json = get_resp.json()
            sha = remote_json.get("sha")
            remote_b64 = remote_json.get("content", "").replace("\n", "").replace("\r", "")
            # Če je vsebina identična, ne pošiljaj nepotrebnega commita
            if remote_b64 == b64_content:
                st.session_state["gh_last_sync"] = datetime.now().strftime("%H:%M:%S")
                return True, f"Datoteka `{rel_str}` je že posodobljena na GitHubu."
    except Exception as e:
        return False, f"Napaka pri povezavi z GitHub API (GET): {e}"

    # 2. Pošlji PUT zahtevo za ustvarjanje ali posodobitev datoteke
    if not commit_message:
        commit_message = f"Avtomatska sinhronizacija: {rel_str} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"

    payload = {
        "message": commit_message,
        "content": b64_content,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    try:
        put_resp = requests.put(url, headers=headers, json=payload, timeout=10)
        if put_resp.status_code in (200, 201):
            st.session_state["gh_last_sync"] = datetime.now().strftime("%H:%M:%S")
            if not silent:
                try:
                    st.toast(f"☁️ Shranjeno v GitHub oblak (`{rel_str}`)")
                except Exception:
                    pass
            return True, f"Uspešno shranjeno na GitHub (`{rel_str}`)."
        else:
            err_msg = put_resp.json().get("message", put_resp.text) if put_resp.text else f"HTTP {put_resp.status_code}"
            return False, f"GitHub API napaka ({put_resp.status_code}): {err_msg}"
    except Exception as e:
        return False, f"Napaka pri pošiljanju na GitHub (PUT): {e}"


def github_pull_file(remote_path: str, local_rel_path: str = None) -> tuple[bool, bytes | None]:
    """
    Prenese datoteko iz GitHub repozitorija in jo shrani lokalno.
    """
    cfg = get_github_config()
    if not cfg["enabled"]:
        return False, None

    rem_clean = str(remote_path).replace("\\", "/").lstrip("/")
    loc_clean = str(local_rel_path or rem_clean).replace("\\", "/").lstrip("/")
    url = f"https://api.github.com/repos/{cfg['repo']}/contents/{rem_clean}"
    headers = _gh_headers(cfg["token"])

    try:
        resp = requests.get(url, headers=headers, params={"ref": cfg["branch"]}, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and data.get("type") == "file":
                b64_str = data.get("content", "")
                raw_bytes = base64.b64decode(b64_str)
                target_path = ROOT_DIR / loc_clean
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(raw_bytes)
                return True, raw_bytes
    except Exception:
        pass
    return False, None


# =============================================================================
# 3. AVTOMATSKA OBNOVA VSEH UPORABNIKOV IN PROJEKTOV OB ZAGONU STREAMLIT CLOUD
# =============================================================================
def github_restore_all_users_on_startup(force: bool = False) -> tuple[bool, str, int]:
    """
    Ob zagonu aplikacije (ali ob kliku na gumb za sinhronizacijo) preveri GitHub repozitorij
    ter obnovi `data/auth_users.json` in vse uporabniške mape `data/users/<username>/*`.
    Združi oddaljene in lokalne podatke, tako da se noben uporabnik ali projekt nikoli ne izgubi.
    """
    if st.session_state.get("github_startup_restored") and not force:
        return True, "Že obnovljeno v tej seji.", 0

    cfg = get_github_config()
    if not cfg["enabled"]:
        return False, "GitHub sinhronizacija ni nastavljena.", 0

    headers = _gh_headers(cfg["token"])
    repo = cfg["repo"]
    branch = cfg["branch"]
    restored_files = 0

    try:
        # 1. Obnovi in združi data/auth_users.json
        auth_url = f"https://api.github.com/repos/{repo}/contents/data/auth_users.json"
        r_auth = requests.get(auth_url, headers=headers, params={"ref": branch}, timeout=8)
        if r_auth.status_code == 200:
            remote_auth_bytes = base64.b64decode(r_auth.json().get("content", ""))
            remote_users = json.loads(remote_auth_bytes.decode("utf-8"))
            local_auth_path = ROOT_DIR / "data" / "auth_users.json"
            local_auth_path.parent.mkdir(parents=True, exist_ok=True)
            local_users = {}
            if local_auth_path.exists():
                try:
                    local_users = json.loads(local_auth_path.read_text(encoding="utf-8"))
                except Exception:
                    local_users = {}
            # Združimo: oddaljeni uporabniki + morebitni lokalni uporabniki
            merged_users = {**remote_users, **local_users}
            local_auth_path.write_text(json.dumps(merged_users, indent=2, ensure_ascii=False), encoding="utf-8")
            restored_files += 1

        # 2. Preišči mapo data/users na GitHubu
        users_url = f"https://api.github.com/repos/{repo}/contents/data/users"
        r_users = requests.get(users_url, headers=headers, params={"ref": branch}, timeout=8)
        if r_users.status_code == 200 and isinstance(r_users.json(), list):
            for item in r_users.json():
                if item.get("type") == "dir":
                    uname = item.get("name")
                    u_dir_url = f"https://api.github.com/repos/{repo}/contents/data/users/{uname}"
                    r_udir = requests.get(u_dir_url, headers=headers, params={"ref": branch}, timeout=8)
                    if r_udir.status_code == 200 and isinstance(r_udir.json(), list):
                        for f_item in r_udir.json():
                            if f_item.get("type") == "file" and f_item.get("name", "").endswith(".json"):
                                fname = f_item["name"]
                                rel_f = f"data/users/{uname}/{fname}"
                                loc_f = ROOT_DIR / rel_f
                                loc_f.parent.mkdir(parents=True, exist_ok=True)

                                # Prenesimo vsebino datoteke
                                ok_p, raw_b = github_pull_file(rel_f, rel_f)
                                if ok_p:
                                    restored_files += 1

        st.session_state["github_startup_restored"] = True
        st.session_state["gh_last_sync"] = datetime.now().strftime("%H:%M:%S")
        return True, f"Uspešno obnovljenih {restored_files} datotek iz GitHub repozitorija `{repo}`.", restored_files
    except Exception as e:
        return False, f"Napaka pri obnovi podatkov iz GitHuba: {e}", restored_files


def github_push_all_local_data() -> tuple[bool, str, int]:
    """
    Potisne `data/auth_users.json` in vse uporabniške `.json` datoteke iz `data/users/` na GitHub.
    """
    cfg = get_github_config()
    if not cfg["enabled"]:
        return False, "GitHub sinhronizacija ni nastavljena (manjka GITHUB_TOKEN ali GITHUB_REPO).", 0

    pushed = 0
    errors = []

    auth_file = ROOT_DIR / "data" / "auth_users.json"
    if auth_file.exists():
        ok, msg = github_push_file("data/auth_users.json", "Sinhronizacija uporabniške baze (auth_users.json)", silent=True)
        if ok:
            pushed += 1
        else:
            errors.append(msg)

    users_dir = ROOT_DIR / "data" / "users"
    if users_dir.exists():
        for u_folder in users_dir.iterdir():
            if u_folder.is_dir():
                for jf in u_folder.glob("*.json"):
                    rel_p = f"data/users/{u_folder.name}/{jf.name}"
                    ok, msg = github_push_file(rel_p, f"Sinhronizacija uporabnika @{u_folder.name}: {jf.name}", silent=True)
                    if ok:
                        pushed += 1
                    else:
                        errors.append(msg)

    if errors:
        return False, f"Sinhroniziranih {pushed} datotek, napaka: {errors[0]}", pushed
    return True, f"Vsi podatki ({pushed} datotek) so uspešno shranjeni v GitHub repozitorij `{cfg['repo']}`!", pushed
