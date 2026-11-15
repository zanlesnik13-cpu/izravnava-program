import os
import sys
import json
import time
import argparse
from datetime import datetime, timedelta
import urllib.request
import urllib.error

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(PROJECT_DIR, ".env.local")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TOKEN_FILE = os.path.join(DATA_DIR, "token.json")
BASE_URL = "https://personal.connectapi.garmin.com"
OVERRIDE_API_KEY = None
CURRENT_USER = None

def set_target_user(username=None, custom_data_dir=None, custom_api_key=None):
    """Prilagodi podatkovno mapo in API ključ za izbranega uporabnika."""
    global DATA_DIR, TOKEN_FILE, OVERRIDE_API_KEY, CURRENT_USER
    if custom_data_dir:
        DATA_DIR = os.path.abspath(custom_data_dir)
        CURRENT_USER = os.path.basename(os.path.normpath(custom_data_dir))
    elif username and username.strip():
        CURRENT_USER = username.lower().strip()
        DATA_DIR = os.path.join(PROJECT_DIR, "data", "users", CURRENT_USER)
    else:
        DATA_DIR = os.path.join(PROJECT_DIR, "data")
        CURRENT_USER = None

    TOKEN_FILE = os.path.join(DATA_DIR, "token.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    if custom_api_key and custom_api_key.strip():
        OVERRIDE_API_KEY = custom_api_key.strip()

def get_api_key():
    global OVERRIDE_API_KEY
    if OVERRIDE_API_KEY:
        return OVERRIDE_API_KEY

    # 1. Preveri config.json v DATA_DIR
    cfg_path = os.path.join(DATA_DIR, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                c = json.load(f)
                k = c.get("garmin_api_key")
                if k and k.strip():
                    return k.strip()
        except Exception:
            pass

    # 2. Preveri Streamlit secrets ali okoljsko spremenljivko (za Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GARMIN_PERSONAL_API_KEY" in st.secrets:
            sec_val = str(st.secrets["GARMIN_PERSONAL_API_KEY"]).strip()
            if sec_val:
                return sec_val
    except Exception:
        pass

    env_val = os.environ.get("GARMIN_PERSONAL_API_KEY")
    if env_val and env_val.strip():
        return env_val.strip()

    # 3. Preveri .env.local v DATA_DIR
    user_env = os.path.join(DATA_DIR, ".env.local")
    if os.path.exists(user_env):
        try:
            with open(user_env, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("GARMIN_PERSONAL_API_KEY="):
                        val = line.split("=", 1)[1].strip()
                        if val:
                            return val
        except Exception:
            pass

    # 3. Preveri splošni ENV_PATH (.env.local v korenu)
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GARMIN_PERSONAL_API_KEY="):
                    val = line.split("=", 1)[1].strip()
                    if val:
                        return val

    raise ValueError("GARMIN_PERSONAL_API_KEY ni nastavljen v nastavitvah profila ali v .env.local")

CURRENT_TOKEN = {"value": None}

def get_access_token(force=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(TOKEN_FILE) and not force:
        try:
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                token = d.get("access_token")
                created_at = d.get("created_at", 0)
                if token and (time.time() - created_at) < 20 * 3600:
                    return token
        except Exception:
            pass

    api_key = get_api_key()
    url = f"{BASE_URL}/personalapitoken-service/personal-api/personal-api-key/token"
    req = urllib.request.Request(url, data=b"", headers={
        "Authorization": f"ApiKey {api_key}",
        "User-Agent": "GarminDashboard/1.0"
    }, method="POST")

    try:
        with urllib.request.urlopen(req) as resp:
            token = resp.read().decode("utf-8").strip()
            with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                json.dump({"access_token": token, "created_at": time.time()}, f)
            CURRENT_TOKEN["value"] = token
            return token
    except Exception as e:
        hint = ""
        if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
            hint = " – Garmin je zavrnil API ključ (neveljaven ali potekel). Ustvari nov Personal API ključ in ga shrani v nastavitvah profila."
        raise RuntimeError(f"Failed to exchange API key for token: {e}{hint}")

def _is_empty_or_zero(val):
    if val is None or val == "" or val == [] or val == {}:
        return True
    if isinstance(val, bool):
        return False
    if isinstance(val, (int, float)) and val == 0:
        return True
    return False


def _safe_merge_dict(existing, incoming):
    """Varno združi slovar brez prepisovanja veljavnih neničelnih metrik z 0, praznimi zbirkami ali None."""
    out = dict(existing) if isinstance(existing, dict) else {}
    if not isinstance(incoming, dict):
        return out
    for k, v in incoming.items():
        if v is None:
            continue
        old_v = out.get(k)
        if isinstance(v, dict) and isinstance(old_v, dict):
            merged_sub = _safe_merge_dict(old_v, v)
            if merged_sub or not old_v:
                out[k] = merged_sub
        elif isinstance(v, dict):
            if v or not old_v:
                out[k] = dict(v)
        elif isinstance(v, list):
            if k == "hourlySteps":
                new_sum = sum(float(x or 0) for x in v) if v else 0.0
                old_sum = sum(float(x or 0) for x in old_v) if isinstance(old_v, list) else 0.0
                if new_sum > 0 or not isinstance(old_v, list) or len(old_v) == 0:
                    if new_sum > 0 or old_sum == 0:
                        out[k] = v
            elif len(v) > 0 or old_v is None:
                out[k] = v
        else:
            if _is_empty_or_zero(v):
                if k not in out or _is_empty_or_zero(old_v):
                    out[k] = v
            else:
                out[k] = v
    return out


def _get_user_weight_kg():
    """Pridobi zadnjo znano težo uporabnika (kg) za izračun W/kg po krogih."""
    try:
        wt = load_json_file("weight.json", {})
        if isinstance(wt, dict) and wt:
            for d_k in sorted(wt.keys(), reverse=True):
                entry = wt.get(d_k) or {}
                lw = entry.get("latestWeight") if isinstance(entry.get("latestWeight"), dict) else entry
                wg = lw.get("weight") if isinstance(lw, dict) else None
                if wg and float(wg) > 2000:
                    return round(float(wg) / 1000.0, 2)
    except Exception:
        pass
    try:
        cfg = load_json_file("config.json", {})
        if isinstance(cfg, dict) and cfg.get("weight_kg"):
            return float(cfg["weight_kg"])
    except Exception:
        pass
    return 65.5


def api_get(endpoint, token, max_retries=3, _refreshed=False):
    if CURRENT_TOKEN["value"]:
        token = CURRENT_TOKEN["value"]
    url = f"{BASE_URL}{endpoint}" if endpoint.startswith("/") else f"{BASE_URL}/{endpoint}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "User-Agent": "GarminDashboard/1.0"
    }, method="GET")

    for attempt in range(max_retries):
        try:
            time.sleep(0.3)
            with urllib.request.urlopen(req, timeout=20) as resp:
                content = resp.read().decode("utf-8")
                if not content or not content.strip():
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
                    return None
                return json.loads(content)
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503):
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"  [Notice] HTTP {e.code} received. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                return None
            if e.code in (401, 403) and not _refreshed:
                print(f"  [Notice] HTTP {e.code} - osvežujem dostopni žeton in poskušam znova...")
                try:
                    new_tok = get_access_token(force=True)
                except Exception as ex:
                    raise RuntimeError(f"HTTP {e.code}: žetona ni mogoče osvežiti ({ex})")
                return api_get(endpoint, new_tok, max_retries, _refreshed=True)
            if e.code == 404:
                return None
            if e.code == 500:
                raise
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 if isinstance(e, TimeoutError) else 2)
                continue
            raise RuntimeError(f"Network error: {e}")
    return None


def _api_get_safe(endpoint, token, empty_retries=1, retry_delay=0.5):
    """Zaporedni GET klic z varnim ponovnim poskusom (1 retry z time.sleep(0.5)) ob praznem odgovoru ali časovni omejitvi."""
    res = None
    for attempt in range(empty_retries + 1):
        try:
            res = api_get(endpoint, token, max_retries=2)
        except Exception:
            res = None
        if res is not None and res != {} and res != []:
            return res
        if attempt < empty_retries:
            time.sleep(retry_delay)
    return res

def load_json_file(filename, default):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default

def save_json_file(filename, data):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def fetch_activities(token, cutoff_date=None, fetch_all=False):
    import gc
    import github_sync

    existing_activities = load_json_file("activities.json", {})
    # existing_activities is dict of activityId -> activity
    print("\n--- Pulling Activities ---")
    start = 0
    limit = 100 if fetch_all else 20
    new_count = 0
    consecutive_existing = 0
    batch_counter = 0

    while True:
        endpoint = f"/activitylist-service/activities/search/activities?start={start}&limit={limit}"
        try:
            items = api_get(endpoint, token)
        except Exception as e:
            print(f"  Error fetching activities at start={start}: {e}")
            break

        if not items:
            break

        reached_cutoff = False
        for act in items:
            act_id = str(act.get("activityId", ""))
            start_time = act.get("startTimeLocal", "")
            act_date = start_time[:10] if start_time else ""

            if not fetch_all and cutoff_date and act_date and act_date < cutoff_date:
                reached_cutoff = True
                break

            if act_id in existing_activities:
                consecutive_existing += 1
                prev_act = existing_activities.get(act_id)
                if isinstance(prev_act, dict):
                    merged_act = _safe_merge_dict(prev_act, act)
                    if prev_act.get("laps") and not act.get("laps"):
                        merged_act["laps"] = prev_act["laps"]
                    existing_activities[act_id] = merged_act
                else:
                    existing_activities[act_id] = act
            else:
                consecutive_existing = 0
                new_count += 1
                existing_activities[act_id] = act
            batch_counter += 1

            # Auto-save vsakih 50 aktivnosti + GitHub Sync za zaščito pred prekinitvami in OOM
            if fetch_all and batch_counter >= 50:
                save_json_file("activities.json", existing_activities)
                print(f"  [Auto-Save] Shranjenih {len(existing_activities)} aktivnosti na disk...")
                if CURRENT_USER:
                    try:
                        github_sync.github_push_file(f"data/users/{CURRENT_USER}/activities.json", async_push=True)
                    except Exception:
                        pass
                batch_counter = 0
                gc.collect()

        print(f"  Fetched page {start // limit + 1} ({len(items)} activities, {new_count} new so far)...")

        # Shrani po vsaki strani 100 aktivnosti
        save_json_file("activities.json", existing_activities)
        if CURRENT_USER:
            try:
                github_sync.github_push_file(f"data/users/{CURRENT_USER}/activities.json", async_push=True)
            except Exception:
                pass
        gc.collect()

        # Če prenašamo celotno zgodovino (fetch_all), straničimo vse do praznega seznama
        if fetch_all:
            if len(items) < limit:
                break
        else:
            # Če vidimo 20 zaporednih že shranjenih aktivnosti ali dosežemo cutoff_date, ustavimo
            if reached_cutoff or consecutive_existing >= 20 or len(items) < limit:
                break

        start += limit

    save_json_file("activities.json", existing_activities)
    if CURRENT_USER:
        try:
            github_sync.github_push_file(f"data/users/{CURRENT_USER}/activities.json", async_push=True)
        except Exception:
            pass
    gc.collect()
    print(f"Total activities stored: {len(existing_activities)} (+{new_count} new).")
    return existing_activities

def fetch_chunked_range(fetch_func, start_date_str, end_date_str, chunk_days):
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

    curr = start_dt
    results = []
    while curr <= end_dt:
        chunk_end = min(curr + timedelta(days=chunk_days - 1), end_dt)
        s_str = curr.strftime("%Y-%m-%d")
        e_str = chunk_end.strftime("%Y-%m-%d")

        current_span = chunk_days
        success = False
        while current_span >= 1 and not success:
            try:
                sub_end_dt = min(curr + timedelta(days=current_span - 1), end_dt)
                sub_s = curr.strftime("%Y-%m-%d")
                sub_e = sub_end_dt.strftime("%Y-%m-%d")
                res = fetch_func(sub_s, sub_e)
                if res:
                    if isinstance(res, list):
                        results.extend(res)
                    elif isinstance(res, dict):
                        results.append(res)
                curr = sub_end_dt + timedelta(days=1)
                success = True
            except urllib.error.HTTPError as e:
                if e.code == 500 and current_span > 2:
                    current_span = current_span // 2
                    print(f"    Payload limit hit (HTTP 500). Halving range to {current_span} days...")
                else:
                    print(f"    Failed range {curr.strftime('%Y-%m-%d')}: {e}")
                    curr = curr + timedelta(days=1)
                    break
            except Exception as e:
                print(f"    Failed range {curr.strftime('%Y-%m-%d')}: {e}")
                curr = curr + timedelta(days=1)
                break
    return results

def fetch_daily_summaries(token, start_date, end_date):
    existing = load_json_file("dailies.json", {})
    print(f"\n--- Pulling Daily Summaries ({start_date} to {end_date}) ---")

    def call_api(s, e):
        print(f"  Fetching dailies: {s} to {e}...")
        res = _api_get_safe(f"/usersummary-service/v2/dailysummaries?startDate={s}&endDate={e}", token)
        if isinstance(res, dict) and "data" in res:
            return res["data"]
        return res

    chunks = fetch_chunked_range(call_api, start_date, end_date, chunk_days=60)
    new_count = 0
    for day in chunks:
        if isinstance(day, dict):
            c_date = day.get("calendarDate") or day.get("summaryDate")
            if c_date:
                if c_date not in existing:
                    new_count += 1
                existing[c_date] = _safe_merge_dict(existing.get(c_date), day)

    save_json_file("dailies.json", existing)
    print(f"Total daily summaries stored: {len(existing)} (+{new_count} new).")
    return existing

def fetch_sleep(token, start_date, end_date):
    existing = load_json_file("sleep.json", {})
    print(f"\n--- Pulling Sleep Summaries ({start_date} to {end_date}) ---")

    def call_api(s, e):
        print(f"  Fetching sleep: {s} to {e}...")
        res = _api_get_safe(f"/sleep-service/sleep/dailySleepSummariesByDate?startDate={s}&endDate={e}", token)
        return res

    # 30 days max chunk for sleep
    chunks = fetch_chunked_range(call_api, start_date, end_date, chunk_days=30)
    new_count = 0
    for night in chunks:
        if isinstance(night, dict):
            c_date = night.get("calendarDate")
            if c_date:
                if c_date not in existing:
                    new_count += 1
                existing[c_date] = _safe_merge_dict(existing.get(c_date), night)

    save_json_file("sleep.json", existing)
    print(f"Total sleep records stored: {len(existing)} (+{new_count} new).")
    return existing

def fetch_hrv(token, start_date, end_date):
    existing = load_json_file("hrv.json", {})
    print(f"\n--- Pulling HRV Summaries ({start_date} to {end_date}) ---")

    def call_api(s, e):
        print(f"  Fetching HRV: {s} to {e}...")
        res = _api_get_safe(f"/hrv-service/hrv/daily/{s}/{e}", token)
        if isinstance(res, dict) and "hrvSummaries" in res:
            return res["hrvSummaries"]
        return res

    chunks = fetch_chunked_range(call_api, start_date, end_date, chunk_days=180)
    new_count = 0
    for entry in chunks:
        if isinstance(entry, dict):
            c_date = entry.get("calendarDate")
            if c_date:
                if c_date not in existing:
                    new_count += 1
                existing[c_date] = _safe_merge_dict(existing.get(c_date), entry)

    save_json_file("hrv.json", existing)
    print(f"Total HRV records stored: {len(existing)} (+{new_count} new).")
    return existing

def fetch_weight(token, start_date, end_date):
    existing = load_json_file("weight.json", {})
    print(f"\n--- Pulling Weight Summaries ({start_date} to {end_date}) ---")

    def call_api(s, e):
        print(f"  Fetching weight: {s} to {e}...")
        res = _api_get_safe(f"/weight-service/weight/range/{s}/{e}", token)
        if isinstance(res, dict) and "dailyWeightSummaries" in res:
            return res["dailyWeightSummaries"]
        return res

    chunks = fetch_chunked_range(call_api, start_date, end_date, chunk_days=90)
    new_count = 0
    for w in chunks:
        if isinstance(w, dict):
            c_date = w.get("calendarDate") or w.get("date")
            if c_date:
                if c_date not in existing:
                    new_count += 1
                existing[c_date] = _safe_merge_dict(existing.get(c_date), w)

    save_json_file("weight.json", existing)
    print(f"Total weight records stored: {len(existing)} (+{new_count} new).")
    return existing

def fetch_cycling_splits(token, days_back=90):
    import streamlib

    splits_dir = os.path.join(DATA_DIR, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    activities = load_json_file("activities.json", {})
    weight_kg = _get_user_weight_kg()
    recent_14d_cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
    if days_back and days_back > 0:
        cutoff_dt = datetime.now() - timedelta(days=days_back)
        cutoff_str = cutoff_dt.strftime("%Y-%m-%d")
        print(f"\n--- Pulling Splits & Laps (Past {days_back} days: >= {cutoff_str}) ---")
    else:
        cutoff_str = "1970-01-01"
        print("\n--- Pulling Splits & Laps (Celotna zgodovina, brez časovne omejitve) ---")

    target_acts = []
    for act in activities.values():
        st = act.get("startTimeLocal", "")
        if st and st[:10] >= cutoff_str:
            type_str = str(act.get("activityType", {})).lower()
            if any(k in type_str for k in ["cycling", "road_biking", "virtual_ride", "indoor_cycling", "biking", "running", "trail_running", "walking", "hiking", "mountaineer", "trail", "swim"]):
                target_acts.append(act)

    target_acts.sort(key=lambda x: x.get("startTimeLocal", ""), reverse=True)
    print(f"Found {len(target_acts)} activities in the last {days_back} days.")

    pulled_count = 0
    activities_updated = False
    for act in target_acts:
        aid = str(act.get("activityId"))
        split_file = os.path.join(splits_dir, f"{aid}.json")
        act_date = (act.get("startTimeLocal") or "")[:10]
        expected_lc = int(act.get("lapCount") or 0)
        if os.path.exists(split_file):
            try:
                with open(split_file, "r", encoding="utf-8") as f:
                    cached_sp = json.load(f) or {}
            except Exception:
                cached_sp = {}
            cached_laps = cached_sp.get("laps") or []
            has_lap_dtos = bool(cached_sp.get("lapDTOs")) or (cached_sp.get("splitsSource") == "garmin_splits")
            is_complete = (
                (len(cached_laps) > 0 and (expected_lc <= 1 or len(cached_laps) >= expected_lc))
                or (cached_sp.get("splitsChecked") is True and len(cached_laps) == 0 and expected_lc == 0)
            )
            # Za aktivnosti v zadnjih 14 dneh brez uradnih lapDTOs znova preveri /splits (da se ročni krogi nikoli ne izgubijo)
            if is_complete and (has_lap_dtos or act_date < recent_14d_cutoff):
                if cached_laps and not act.get("laps"):
                    act["laps"] = cached_laps
                    activities[aid] = act
                    activities_updated = True
                continue

        print(f"  Fetching splits for activity {aid} ({act.get('startTimeLocal', '')[:10]} - {act.get('activityName', 'Activity')})...")
        time.sleep(0.3)

        act_detail = None
        try:
            act_detail = _api_get_safe(f"/activity-service/activity/{aid}", token)
        except Exception as e:
            print(f"    Warning getting activity details for {aid}: {e}")

        split_res = None
        try:
            split_res = _api_get_safe(f"/activity-service/activity/{aid}/splits", token)
        except Exception as e:
            print(f"    Warning getting /splits for {aid}: {e}")

        laps = _laps_from_split_dtos(split_res, act_detail=act_detail, weight_kg=weight_kg)
        if not laps:
            try:
                req = urllib.request.Request(
                    f"{BASE_URL}/download-service/files/activity/{aid}",
                    headers={"Authorization": f"Bearer {CURRENT_TOKEN['value'] or token}", "User-Agent": "GarminDashboard/1.0"}
                )
                with urllib.request.urlopen(req, timeout=25) as resp:
                    fit_laps = streamlib.parse_fit_laps(resp.read())
                    if fit_laps:
                        laps = _enrich_laps_metrics(fit_laps, weight_kg=weight_kg)
            except Exception as e:
                print(f"    Warning FIT laps fallback for {aid}: {e}")

        raw_lap_dtos = (split_res or {}).get("lapDTOs") if isinstance(split_res, dict) else None
        payload = {
            "activityId": aid,
            "activityName": act.get("activityName") or (act_detail or {}).get("activityName"),
            "startTimeLocal": act.get("startTimeLocal"),
            "summaryDTO": (act_detail or {}).get("summaryDTO") or {},
            "metadataDTO": (act_detail or {}).get("metadataDTO") or {},
            "laps": laps,
            "lapDTOs": raw_lap_dtos if raw_lap_dtos is not None else (laps or []),
            "splitsSource": "garmin_splits",
            "splitsChecked": True,
        }

        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        if laps:
            act["laps"] = laps
            activities[aid] = act
            activities_updated = True

        pulled_count += 1

    if activities_updated:
        save_json_file("activities.json", activities)

    existing_splits_count = len(os.listdir(splits_dir)) if os.path.exists(splits_dir) else 0
    print(f"Cycling splits updated: +{pulled_count} new (Total cached in data/splits: {existing_splits_count}).")

def fetch_cycling_streams(token, days_back=90, repair=False, max_stream_count=None):
    """Prenese in validira 1 Hz tokove (moč, HR, kadenca, hitrost, višina, GPS) za kolesarske, tekaške in pohodniške aktivnosti.
    Indeksi metrik se dinamično preslikajo prek metricDescriptors; FIT se uporabi kot nadomestek."""
    import gc
    import numpy as np
    import streamlib
    import github_sync

    streams_dir = os.path.join(DATA_DIR, "streams")
    mmp_dir = os.path.join(DATA_DIR, "mmp")
    os.makedirs(streams_dir, exist_ok=True)
    os.makedirs(mmp_dir, exist_ok=True)

    activities = load_json_file("activities.json", {})
    if days_back and days_back > 0:
        cutoff_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        print(f"\n--- Pulling Activity Streams (>= {cutoff_str}, repair={repair}) ---")
    else:
        cutoff_str = "1970-01-01"
        print(f"\n--- Pulling Activity Streams (Celotna zgodovina, repair={repair}) ---")

    keys = ["cycl", "bik", "virtual_ride", "run", "walk", "hik"]
    target_acts = []
    for act in activities.values():
        st = act.get("startTimeLocal", "")
        if st and st[:10] >= cutoff_str:
            type_str = str(act.get("activityType", {})).lower()
            if any(k in type_str for k in keys):
                target_acts.append(act)
    target_acts.sort(key=lambda x: x.get("startTimeLocal", ""), reverse=True)
    print(f"Found {len(target_acts)} candidate activities.")

    if max_stream_count and len(target_acts) > max_stream_count:
        target_acts = target_acts[:max_stream_count]
        print(f"  RAM Zaščita: prenašam 1 Hz tokove le za zadnjih {max_stream_count} aktivnosti (starejši se naložijo na zahtevo).")

    pulled_count = 0
    for act in target_acts:
        aid = str(act.get("activityId"))
        stream_file = os.path.join(streams_dir, f"{aid}.json")
        if os.path.exists(stream_file):
            try:
                with open(stream_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
            except Exception:
                cached = None
            if cached is not None and streamlib.stream_is_valid(cached, act):
                continue

        print(f"  Fetching stream for {aid} ({act.get('startTimeLocal', '')[:10]} - {act.get('activityName', 'Activity')})...")
        time.sleep(0.3)

        details_dto, arr = {}, None
        try:
            d_res = api_get(f"/activity-service/activity/{aid}/details?maxChartSize=30000&maxPolylineSize=3000", token)
            if isinstance(d_res, dict):
                details_dto = {"measurementCount": d_res.get("measurementCount"), "metricsCount": d_res.get("metricsCount")}
                arr = streamlib.parse_details(d_res)
        except Exception as e:
            print(f"    Warning details for {aid}: {e}")

        has_hr = arr is not None and bool(np.any(np.asarray(arr.get("heartRate", [0])) > 0))
        has_gps = arr is not None and bool(arr.get("coords"))
        has_power = arr is not None and bool(np.any(np.asarray(arr.get("power", [0])) > 0))
        wants_power = (act.get("avgPower") or act.get("averagePower") or 0) > 0
        if arr is None or not has_hr or not has_gps or (wants_power and not has_power):
            try:
                req = urllib.request.Request(
                    f"{BASE_URL}/download-service/files/activity/{aid}",
                    headers={"Authorization": f"Bearer {token}", "User-Agent": "GarminDashboard/1.0"}
                )
                with urllib.request.urlopen(req) as resp:
                    is_cyc = any(k in str(act.get("activityType", {})).lower() for k in ["cycl", "bik", "virtual_ride"])
                    fit_arr = streamlib.parse_fit(resp.read(), double_cadence=not is_cyc)
                arr = streamlib.merge_arrays(arr, fit_arr)
            except Exception as e:
                print(f"    Warning FIT fallback for {aid}: {e}")

        if arr is None or len(arr.get("speed", [])) == 0:
            print(f"    No stream data obtained for {aid}; keeping existing cache (if any).")
            continue
        if "speed" not in arr:
            ref = next((v for k, v in arr.items() if k != "coords"), None)
            if ref is not None:
                arr["speed"] = np.zeros(len(ref))
        payload, mmp = streamlib.build_payload(act, arr, details_dto)
        payload["repairTried"] = True

        with open(stream_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

        if mmp:
            t_key = act.get("activityType", {}).get("typeKey", "")
            act_name = act.get("activityName", "Kolesarjenje")
            is_indoor = (t_key in ["virtual_ride", "indoor_cycling"]) or any(
                k in act_name.lower() for k in ["zwift", "trainer", "trenažer", "rouvy", "mywhoosh", "indoor"])
            with open(os.path.join(mmp_dir, f"{aid}.json"), "w", encoding="utf-8") as mf:
                json.dump({"activityId": aid, "activityName": act_name, "startTimeLocal": act.get("startTimeLocal"),
                           "is_indoor": is_indoor, "durationSeconds": payload.get("length", 0), "mmp": mmp},
                          mf, indent=2, ensure_ascii=False)
        pulled_count += 1
        gc.collect()

    existing = len(os.listdir(streams_dir))
    print(f"Streams updated: +{pulled_count} (Total cached in data/streams: {existing}).")
    if CURRENT_USER and pulled_count > 0:
        try:
            github_sync.github_push_user_files(CURRENT_USER, async_push=True)
        except Exception:
            pass
    gc.collect()


def fetch_single_activity_stream(activity_id, username=None, token=None):
    """Prenese in shrani 1 Hz tok na zahtevo za posamezno aktivnost ter sprosti RAM."""
    payload, _, _ = fetch_activity_bundle(activity_id=activity_id, username=username, token=token, force=False)
    return payload

def _parse_movement_epochs(raw_movement, d_str):
    """
    Razčleni Garmin 15-minutne epohe gibanja (`/wellness-service/wellness/dailyMovement/{date}?calendarDate={date}`)
    in pretvori `startGMT` / `endGMT` v lokalni čas (`Europe/Ljubljana`).
    Vrne `(epochs_list, hourly_steps_24)`. Ure z 0 koraki imajo strogo vrednost 0.
    """
    from datetime import timezone
    from zoneinfo import ZoneInfo

    lj_tz = ZoneInfo("Europe/Ljubljana")
    hourly_steps = [0] * 24
    epochs = []

    items = []
    if isinstance(raw_movement, list):
        items = raw_movement
    elif isinstance(raw_movement, dict):
        for k in ("movementValues", "dailyMovement", "values", "data", "steps"):
            if isinstance(raw_movement.get(k), list):
                items = raw_movement[k]
                break
        if not items and isinstance(raw_movement.get("startGMT"), str):
            items = [raw_movement]

    for entry in items:
        if not isinstance(entry, dict):
            continue
        s_gmt = entry.get("startGMT") or entry.get("startTimeGMT") or entry.get("startTimestampGMT")
        e_gmt = entry.get("endGMT") or entry.get("endTimeGMT") or entry.get("endTimestampGMT")
        steps_val = int(round(float(entry.get("steps") or 0)))
        act_lvl = entry.get("activityLevelConstant")
        prim_lvl = entry.get("primaryActivityLevel")

        dt_loc = None
        if s_gmt is not None:
            try:
                if isinstance(s_gmt, str) and s_gmt.strip().replace(".", "", 1).isdigit():
                    s_gmt_num = float(s_gmt.strip())
                elif isinstance(s_gmt, (int, float)):
                    s_gmt_num = float(s_gmt)
                else:
                    s_gmt_num = None
                if s_gmt_num is not None:
                    if s_gmt_num > 1e11:  # milisekunde
                        s_gmt_num /= 1000.0
                    dt_loc = datetime.fromtimestamp(s_gmt_num, tz=timezone.utc).astimezone(lj_tz)
                else:
                    s_clean = str(s_gmt).strip().replace("Z", "").replace(" ", "T")
                    if "." in s_clean:
                        s_clean = s_clean.split(".")[0]
                    if "+" in s_clean[10:]:
                        s_clean = s_clean[:10] + s_clean[10:].split("+")[0]
                    dt_utc = datetime.fromisoformat(s_clean).replace(tzinfo=timezone.utc)
                    dt_loc = dt_utc.astimezone(lj_tz)
            except Exception:
                dt_loc = None

        if dt_loc is not None:
            loc_date_str = dt_loc.strftime("%Y-%m-%d")
            # Upoštevaj le epohe, ki v lokalnem času Europe/Ljubljana padejo v izbrani dan d_str
            if loc_date_str == d_str:
                h_idx = dt_loc.hour
                if 0 <= h_idx < 24:
                    hourly_steps[h_idx] += max(0, steps_val)
                epochs.append({
                    "startGMT": s_gmt,
                    "endGMT": e_gmt,
                    "startLocal": dt_loc.strftime("%Y-%m-%d %H:%M:%S"),
                    "hour": h_idx,
                    "steps": max(0, steps_val),
                    "activityLevelConstant": act_lvl,
                    "primaryActivityLevel": prim_lvl,
                })
    return epochs, hourly_steps


_SOCIAL_PROFILE_CACHE = {}


def _get_display_name(token):
    tok_key = token or "default"
    if tok_key in _SOCIAL_PROFILE_CACHE:
        return _SOCIAL_PROFILE_CACHE[tok_key]
    disp_name = None
    try:
        prof = _api_get_safe("/userprofile-service/socialProfile", token)
        if isinstance(prof, dict):
            disp_name = prof.get("displayName") or prof.get("userName")
    except Exception:
        disp_name = None
    if disp_name:
        _SOCIAL_PROFILE_CACHE[tok_key] = disp_name
    return disp_name


def fetch_intraday_day(d_str, token=None, username=None):
    """Prenese in shrani celodnevne (intraday) meritve za izbrani datum `d_str` (YYYY-MM-DD):
    - 2-minutni srčni utrip (vključno z večerom prejšnjega dne za celotno okno spanja)
    - 5-minutni nočni HRV odčitki (`hrvReadings`)
    - Body Battery krivulja in dogodki (`bodyBatteryValuesArray`, `bodyBatteryActivityEvent`)
    - 15-minutni podatki o gibanju in korakih (`dailyMovement` -> `intraday_steps` in `hourlySteps`)
    - morebitni `sleepLevels` / `sleepMovement`
    """
    if username:
        set_target_user(username=username)
    if token is None:
        token = get_access_token()
    intraday_dir = os.path.join(DATA_DIR, "intraday")
    os.makedirs(intraday_dir, exist_ok=True)
    p_out = os.path.join(intraday_dir, f"{d_str}.json")
    existing_out = {}
    if os.path.exists(p_out):
        try:
            with open(p_out, "r", encoding="utf-8") as f:
                existing_out = json.load(f) or {}
        except Exception:
            existing_out = {}

    out = dict(existing_out) if isinstance(existing_out, dict) else {}
    out.setdefault("date", d_str)
    for list_k in ("heartRateValues", "hrvReadings", "bodyBatteryValuesArray", "bodyBatteryActivityEvent", "sleepLevels"):
        if not isinstance(out.get(list_k), list):
            out[list_k] = []

    try:
        dt_cur = datetime.strptime(d_str, "%Y-%m-%d")
        prev_str = (dt_cur - timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        prev_str = None

    hr_pts = []
    if prev_str:
        try:
            hr_prev = _api_get_safe(f"/wellness-service/wellness/dailyHeartRate?date={prev_str}", token)
            if isinstance(hr_prev, dict):
                # Vzemi zadnjih 8 ur prejšnjega dne (za začetek spanja pred polnočjo)
                vals = hr_prev.get("heartRateValues") or []
                if vals:
                    cutoff_ms = vals[-1][0] - 8 * 3600 * 1000
                    hr_pts.extend([p for p in vals if isinstance(p, list) and len(p) >= 2 and p[1] and p[0] >= cutoff_ms])
        except Exception:
            pass
    try:
        hr_cur = _api_get_safe(f"/wellness-service/wellness/dailyHeartRate?date={d_str}", token)
        if isinstance(hr_cur, dict):
            vals = hr_cur.get("heartRateValues") or []
            hr_pts.extend([p for p in vals if isinstance(p, list) and len(p) >= 2 and p[1]])
    except Exception:
        pass
    if hr_pts:
        # Odstrani dvojnike po timestampu in obdrži obstoječe veljavne točke
        dedup = {
            int(p[0]): int(p[1])
            for p in (out.get("heartRateValues") or [])
            if isinstance(p, (list, tuple)) and len(p) >= 2 and p[0] is not None and p[1] is not None
        }
        for p in hr_pts:
            if p[0] is not None and p[1] is not None:
                dedup[int(p[0])] = int(p[1])
        out["heartRateValues"] = [[k, dedup[k]] for k in sorted(dedup)]

    try:
        h_res = _api_get_safe(f"/hrv-service/hrv/{d_str}", token)
        if isinstance(h_res, dict):
            if isinstance(h_res.get("hrvReadings"), list) and len(h_res["hrvReadings"]) > 0:
                out["hrvReadings"] = h_res["hrvReadings"]
            if isinstance(h_res.get("hrvSummary"), dict) and h_res["hrvSummary"]:
                out["hrvSummary"] = _safe_merge_dict(out.get("hrvSummary"), h_res["hrvSummary"])
    except Exception:
        pass

    try:
        bb_res = _api_get_safe(f"/wellness-service/wellness/bodyBattery/reports/daily?startDate={d_str}&endDate={d_str}", token)
        if isinstance(bb_res, list) and len(bb_res) > 0 and isinstance(bb_res[0], dict):
            bb0 = bb_res[0]
            if bb0.get("bodyBatteryValuesArray"):
                out["bodyBatteryValuesArray"] = bb0["bodyBatteryValuesArray"]
            if bb0.get("bodyBatteryActivityEvent"):
                out["bodyBatteryActivityEvent"] = bb0["bodyBatteryActivityEvent"]
            if bb0.get("charged") is not None and (bb0.get("charged") != 0 or not out.get("charged")):
                out["charged"] = bb0.get("charged")
            if bb0.get("drained") is not None and (bb0.get("drained") != 0 or not out.get("drained")):
                out["drained"] = bb0.get("drained")
    except Exception:
        pass

    try:
        steps_res = _api_get_safe(f"/usersummary-service/stats/steps/daily/{d_str}/{d_str}", token)
        if isinstance(steps_res, list) and len(steps_res) > 0 and isinstance(steps_res[0], dict):
            out["stepsDaily"] = _safe_merge_dict(out.get("stepsDaily"), steps_res[0])
    except Exception:
        pass

    # Pravi 15-minutni odčitki korakov in gibanja (dailyMovement)
    mov_endpoints = [
        f"/wellness-service/wellness/dailyMovement/{d_str}?calendarDate={d_str}",
        f"/wellness-service/wellness/dailyMovement?date={d_str}",
        f"/wellness-service/wellness/dailyMovement?calendarDate={d_str}",
    ]
    # garminconnect get_steps_data(): /wellness-service/wellness/dailySummaryChart/{displayName}?date=...
    disp_name = _get_display_name(token)
    if disp_name:
        mov_endpoints.insert(0, f"/wellness-service/wellness/dailySummaryChart/{disp_name}?date={d_str}")
    mov_endpoints.append(f"/wellness-service/wellness/dailySummaryChart?date={d_str}")

    mov_raw = None
    best_epochs, best_hourly = [], [0] * 24
    for mov_ep in mov_endpoints:
        try:
            r_mov = _api_get_safe(mov_ep, token)
        except Exception:
            continue
        if not r_mov:
            continue
        ep_try, hr_try = _parse_movement_epochs(r_mov, d_str)
        if mov_raw is None:
            mov_raw = r_mov
        if ep_try and sum(hr_try) > sum(best_hourly):
            mov_raw, best_epochs, best_hourly = r_mov, ep_try, hr_try
        elif ep_try and not best_epochs:
            best_epochs, best_hourly = ep_try, hr_try
        if sum(best_hourly) > 0:
            break

    if mov_raw:
        out["movementData"] = mov_raw
        if best_epochs:
            old_h_sum = sum(float(x or 0) for x in (out.get("hourlySteps") or []))
            if sum(best_hourly) > 0 or old_h_sum == 0:
                out["intraday_steps"] = best_epochs
                out["hourlySteps"] = best_hourly

    # Če imamo intraday_steps ali hourlySteps, varno osveži tudi zapis v sleep.json za ta datum
    if out.get("intraday_steps") or out.get("hourlySteps") or out.get("stepsDaily"):
        try:
            sleep_all = load_json_file("sleep.json", {})
            if isinstance(sleep_all, dict):
                s_rec = sleep_all.get(d_str)
                if isinstance(s_rec, dict):
                    patch = {}
                    if out.get("intraday_steps"):
                        patch["intraday_steps"] = out["intraday_steps"]
                    if out.get("hourlySteps"):
                        patch["hourlySteps"] = out["hourlySteps"]
                    if out.get("stepsDaily"):
                        patch["stepsDaily"] = out["stepsDaily"]
                    sleep_all[d_str] = _safe_merge_dict(s_rec, patch)
                    save_json_file("sleep.json", sleep_all)
        except Exception:
            pass

    if (
        out["heartRateValues"]
        or out["hrvReadings"]
        or out["bodyBatteryValuesArray"]
        or out.get("stepsDaily")
        or out.get("intraday_steps")
        or out.get("movementData")
    ):
        with open(p_out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    return out


def force_sync_recent(token, days=7, end_date=None):
    """Vedno znova prenese in varno združi zadnjih `days` dni (spanec, HRV, dnevni povzetki, intraday) do jutri."""
    today_dt = datetime.now()
    end_str = end_date or (today_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    start_str = (today_dt - timedelta(days=days)).strftime("%Y-%m-%d")
    print(f"\n--- FORCE sync zadnjih {days} dni ({start_str} -> {end_str}) ---")

    # a) skupni povzetki (zaporedno, z varnim združevanjem brez brisanja obstoječih metrik)
    fetch_daily_summaries(token, start_str, end_str)
    fetch_sleep(token, start_str, end_str)
    fetch_hrv(token, start_str, end_str)

    # b) podrobni spanec po dnevih (/sleep-service/sleep/dailySleep?date=d in dailySleepData) + HRV + Intraday
    sleep = load_json_file("sleep.json", {})
    hrv = load_json_file("hrv.json", {})
    dailies = load_json_file("dailies.json", {})
    disp_name = _get_display_name(token)
    n_days = days + 2
    for i in range(n_days):
        d = (today_dt - timedelta(days=days) + timedelta(days=i)).strftime("%Y-%m-%d")
        res = None
        for ep_sleep in (f"/sleep-service/sleep/dailySleep?date={d}", f"/sleep-service/sleep/dailySleepData/{d}"):
            try:
                r_try = _api_get_safe(ep_sleep, token)
                if isinstance(r_try, dict) and (
                    r_try.get("sleepTimeSeconds")
                    or isinstance(r_try.get("dailySleepDTO"), dict)
                    or r_try.get("sleepScores")
                    or r_try.get("sleepLevels")
                ):
                    res = r_try
                    break
            except Exception:
                pass
        if isinstance(res, dict):
            dto = res.get("dailySleepDTO") if isinstance(res.get("dailySleepDTO"), dict) else res
            merged_dto = dict(dto) if isinstance(dto, dict) else {}
            # Prenesi tudi vrhnje atribute odgovora (sleepLevels, sleepScores, avgOvernightHrv, restingHeartRate, SpO2)
            for extra_k in (
                "sleepLevels",
                "sleepMovement",
                "sleepScores",
                "avgOvernightHrv",
                "restingHeartRate",
                "wellnessSpO2SleepSummaryDTO",
                "avgSkinTempDeviationC",
            ):
                if res.get(extra_k) is not None and extra_k not in merged_dto:
                    merged_dto[extra_k] = res.get(extra_k)
            if merged_dto and (
                merged_dto.get("sleepTimeSeconds")
                or merged_dto.get("deepSleepSeconds")
                or merged_dto.get("sleepScores")
                or merged_dto.get("avgOvernightHrv")
            ):
                merged_dto.setdefault("calendarDate", d)
                sleep[d] = _safe_merge_dict(sleep.get(d, {}), merged_dto)
                print(f"  Spanec {d}: osvežen (podrobno).")

        # HRV po dnevu (zaporedno z varnim ponovnim poskusom)
        h = _api_get_safe(f"/hrv-service/hrv/{d}", token)
        if isinstance(h, dict) and isinstance(h.get("hrvSummary"), dict):
            hs = dict(h["hrvSummary"])
            hs.setdefault("calendarDate", d)
            hrv[d] = _safe_merge_dict(hrv.get(d, {}), hs)
            if hs.get("lastNightAvg") is not None and isinstance(sleep.get(d), dict):
                sleep[d] = _safe_merge_dict(sleep[d], {"avgOvernightHrv": hs["lastNightAvg"]})
        elif isinstance(sleep.get(d), dict) and sleep[d].get("avgOvernightHrv") is not None:
            hrv[d] = _safe_merge_dict(hrv.get(d, {}), {"calendarDate": d, "lastNightAvg": sleep[d]["avgOvernightHrv"]})

        # Če dnevni povzetek nima RHR ali osnovnih statistik, dopolni še iz /usersummary/daily ali sleepDTO
        if disp_name and (d not in dailies or not (dailies.get(d, {}).get("heartRate") or {}).get("restingValue")):
            ds_try = _api_get_safe(f"/usersummary-service/usersummary/daily/{disp_name}?calendarDate={d}", token)
            if isinstance(ds_try, dict) and (ds_try.get("restingHeartRate") or ds_try.get("totalSteps") or ds_try.get("minHeartRate")):
                ds_patch = dict(ds_try)
                if ds_try.get("restingHeartRate") is not None and not isinstance(ds_patch.get("heartRate"), dict):
                    ds_patch["heartRate"] = {"restingValue": ds_try["restingHeartRate"]}
                if ds_try.get("totalSteps") is not None and not isinstance(ds_patch.get("movement"), dict):
                    ds_patch["movement"] = {"steps": {"value": ds_try["totalSteps"], "goal": ds_try.get("dailyStepGoal") or 10000}}
                dailies[d] = _safe_merge_dict(dailies.get(d, {}), ds_patch)

        if isinstance(sleep.get(d), dict) and sleep[d].get("restingHeartRate") is not None:
            rhr_val = sleep[d]["restingHeartRate"]
            if rhr_val:
                dailies[d] = _safe_merge_dict(dailies.get(d, {}), {"calendarDate": d, "heartRate": {"restingValue": rhr_val}})

        # Najprej shrani sleep.json, nato prenesi intraday in združi intraday korake/faze nazaj v sleep[d]!
        if d <= today_dt.strftime("%Y-%m-%d"):
            save_json_file("sleep.json", sleep)
            try:
                intra_out = fetch_intraday_day(d, token=token) or {}
                if isinstance(intra_out, dict):
                    sleep_patch = {}
                    if intra_out.get("intraday_steps"):
                        sleep_patch["intraday_steps"] = intra_out["intraday_steps"]
                    if intra_out.get("hourlySteps"):
                        sleep_patch["hourlySteps"] = intra_out["hourlySteps"]
                    if intra_out.get("stepsDaily"):
                        sleep_patch["stepsDaily"] = intra_out["stepsDaily"]
                    if intra_out.get("sleepLevels"):
                        sleep_patch["sleepLevels"] = intra_out["sleepLevels"]
                    if sleep_patch and isinstance(sleep.get(d), dict):
                        sleep[d] = _safe_merge_dict(sleep[d], sleep_patch)
                    if isinstance(intra_out.get("hrvSummary"), dict) and intra_out["hrvSummary"]:
                        hrv_s = dict(intra_out["hrvSummary"])
                        hrv_s.setdefault("calendarDate", d)
                        hrv[d] = _safe_merge_dict(hrv.get(d, {}), hrv_s)
                    bb_vals = [
                        int(p[1])
                        for p in (intra_out.get("bodyBatteryValuesArray") or [])
                        if isinstance(p, (list, tuple)) and len(p) >= 2 and p[1] is not None
                    ]
                    if bb_vals:
                        dailies[d] = _safe_merge_dict(
                            dailies.get(d, {}),
                            {"calendarDate": d, "bodyBattery": {"minValue": min(bb_vals), "maxValue": max(bb_vals)}},
                        )
            except Exception as ex_i:
                print(f"  Intraday {d}: {ex_i}")

    save_json_file("sleep.json", sleep)
    save_json_file("hrv.json", hrv)
    save_json_file("dailies.json", dailies)
    if sleep:
        last = max(sleep.keys())
        s = sleep[last]
        ts = s.get("sleepEndTimestampGMT")
        try:
            from zoneinfo import ZoneInfo
            tx = datetime.fromtimestamp(ts / 1000.0, tz=ZoneInfo("Europe/Ljubljana")).strftime("%Y-%m-%d %H:%M") if ts else "?"
        except Exception:
            tx = "?"
        print(f"LATEST_SLEEP {last} konec spanca {tx}")
    try:
        with open(os.path.join(DATA_DIR, "last_sync.json"), "w", encoding="utf-8") as f:
            json.dump({"synced_at": datetime.now().isoformat(timespec="seconds"),
                       "latest_sleep": max(sleep.keys()) if sleep else None}, f)
    except Exception:
        pass
    return sleep, hrv, dailies

def fetch_biometrics(token):
    print("\n--- Pulling Heart Rate Zones & Bio Info ---")
    try:
        hr_zones = api_get("/biometric-service/heartRateZones/v2", token)
        if hr_zones:
            save_json_file("heart_rate_zones.json", hr_zones)
            print("Heart rate zones saved.")
    except Exception as e:
        print(f"Heart rate zones not available: {e}")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Pull Garmin Personal API data.")
    parser.add_argument("--days", type=int, default=365, help="Number of past days to pull (default: 365).")
    parser.add_argument("--repair-streams", action="store_true", help="Ponovno prenesi/validiraj tokove (HR, GPS, moč).")
    parser.add_argument("--stream-days", type=int, default=90, help="Okno dni za tokove.")
    parser.add_argument("--full", action="store_true", help="Pull full history.")
    parser.add_argument("--all", action="store_true", help="Prenesi celotno zgodovino od začetka Garmin računa (brez 2-letne ali 5-letne omejitve).")
    parser.add_argument("--user", type=str, default="", help="Uporabniško ime za večuporabniško ločevanje podatkov.")
    parser.add_argument("--data-dir", type=str, default="", help="Neposredna pot do podatkovne mape uporabnika.")
    parser.add_argument("--api-key", type=str, default="", help="Neposredni Garmin Personal API ključ.")
    args = parser.parse_args()

    set_target_user(username=args.user, custom_data_dir=args.data_dir, custom_api_key=args.api_key)
    print(f"Uporabljena podatkovna mapa: {DATA_DIR}")

    token = get_access_token()
    CURRENT_TOKEN["value"] = token
    today_dt = datetime.now()
    # +1 dan, da meja UTC / Europe/Ljubljana ne odreže današnjih podatkov
    today_str = (today_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    is_full_history = bool(args.full or args.all)
    print("--- Starting Garmin Data Pull ---")
    if is_full_history:
        print("Pulling full history of activities first to determine account history...")
        activities = fetch_activities(token, cutoff_date=None, fetch_all=True)
        # Poišči najzgodnejši datum aktivnosti v celotni zgodovini
        earliest_date = today_str
        for act in activities.values():
            st = act.get("startTimeLocal", "")
            if st and st[:10] < earliest_date:
                earliest_date = st[:10]
        start_str = earliest_date
        print(f"Najzgodnejša aktivnost v računu: {earliest_date}. Sinhronizacija celotne baze od {start_str} do {today_str}...")
    else:
        existing_dailies = load_json_file("dailies.json", {})
        if existing_dailies:
            latest_stored = max(existing_dailies.keys())
            # Re-sync last 7 days from latest stored to ensure up-to-date health metrics
            start_dt = max(today_dt - timedelta(days=args.days), min(datetime.strptime(latest_stored, "%Y-%m-%d") - timedelta(days=7), today_dt - timedelta(days=7)))
        else:
            start_dt = today_dt - timedelta(days=args.days)
        start_str = start_dt.strftime("%Y-%m-%d")
        print(f"Incremental sync from {start_str} to {today_str}...")
        activities = fetch_activities(token, cutoff_date=start_str)

    # 2. Daily summaries
    dailies = fetch_daily_summaries(token, start_str, today_str)

    # 3. Sleep
    sleep_data = fetch_sleep(token, start_str, today_str)

    # 4. HRV
    hrv_data = fetch_hrv(token, start_str, today_str)

    # 5. Weight
    weight_data = fetch_weight(token, start_str, today_str)

    # 5b. Vedno prepiši zadnjih 7 dni (spanec, HRV, povzetki)
    sleep_rec, hrv_rec, dailies_rec = force_sync_recent(token, days=7)
    if isinstance(sleep_rec, dict) and sleep_rec:
        sleep_data = sleep_rec
    if isinstance(hrv_rec, dict) and hrv_rec:
        hrv_data = hrv_rec
    if isinstance(dailies_rec, dict) and dailies_rec:
        dailies = dailies_rec

    # 6. Biometrics
    fetch_biometrics(token)

    # 7. Cycling Splits & Intervals (pri celotni zgodovini prenesi za vse aktivnosti)
    fetch_cycling_splits(token, days_back=(0 if is_full_history else 90))

    # 8. Cycling Streams & Progression Blocks
    # RAM Zaščita: pri celotni zgodovini prenesi le zadnjih 30 tokov! Starejši se naložijo na zahtevo (lazy on-demand).
    stream_limit = 30 if is_full_history else None
    fetch_cycling_streams(
        token,
        days_back=(0 if is_full_history else (max(90, args.stream_days) if args.repair_streams else 90)),
        repair=args.repair_streams,
        max_stream_count=stream_limit
    )

    # Summary
    all_dates = set(dailies.keys()).union(sleep_data.keys()).union(hrv_data.keys())
    for act in activities.values():
        st = act.get("startTimeLocal", "")
        if st:
            all_dates.add(st[:10])

    sorted_dates = sorted(list(all_dates))
    date_range_str = f"{sorted_dates[0]} to {sorted_dates[-1]}" if sorted_dates else "None"

    print("\n===============================")
    print("DATA PULL COMPLETE")
    print(f"- Activities: {len(activities)}")
    print(f"- Daily Summaries: {len(dailies)} days")
    print(f"- Sleep records: {len(sleep_data)} nights")
    print(f"- HRV records: {len(hrv_data)} days")
    print(f"- Weight records: {len(weight_data)} entries")
    print(f"- Covered date range: {date_range_str}")
    print("===============================\n")

def _enrich_laps_metrics(laps, weight_kg=None):
    """Dopolni vsak krog z W/kg, NP W/kg, EF in standardiziranimi polji brez prepisa z None/0."""
    out = []
    w_kg = float(weight_kg) if (weight_kg is not None and float(weight_kg) > 0) else None
    for idx, raw in enumerate(laps or []):
        if not isinstance(raw, dict):
            continue
        lp = dict(raw)
        lp["lapIndex"] = lp.get("lapIndex") or (idx + 1)
        dur = lp.get("duration")
        if dur is None:
            dur = lp.get("movingDuration") if lp.get("movingDuration") is not None else lp.get("elapsedDuration")
        lp["duration"] = dur

        avg_p = lp.get("avgPower")
        np_p = lp.get("normalizedPower") if lp.get("normalizedPower") is not None else lp.get("normPower")
        if np_p is not None:
            lp["normalizedPower"] = np_p
        avg_hr = lp.get("avgHR")

        if w_kg:
            if avg_p is not None:
                try:
                    lp["wkg"] = round(float(avg_p) / w_kg, 2)
                except Exception:
                    pass
            if np_p is not None:
                try:
                    lp["npWkg"] = round(float(np_p) / w_kg, 2)
                except Exception:
                    pass

        p_for_ef = np_p if (np_p is not None and float(np_p or 0) > 0) else avg_p
        if p_for_ef is not None and avg_hr is not None:
            try:
                if float(p_for_ef) > 0 and float(avg_hr) > 0:
                    lp["ef"] = round(float(p_for_ef) / float(avg_hr), 2)
            except Exception:
                pass
        out.append(lp)
    return out


def _laps_from_split_dtos(split_res, act_detail=None, weight_kg=None):
    """Pretvori odgovor /activity-service/activity/{id}/splits (lapDTOs / splits / splitSummaries) v normalizirane kroge."""
    laps = []
    dtos = None
    if isinstance(split_res, dict):
        dtos = (
            split_res.get("lapDTOs")
            or split_res.get("splits")
            or split_res.get("splitSummaries")
            or split_res.get("eventDTOs")
        )
    elif isinstance(split_res, list):
        dtos = split_res

    if not dtos and isinstance(act_detail, dict):
        dtos = act_detail.get("lapDTOs") or act_detail.get("splitSummaries")

    for idx, s in enumerate(dtos or []):
        if not isinstance(s, dict):
            continue
        dur = s.get("duration")
        if dur is None:
            dur = s.get("movingDuration") if s.get("movingDuration") is not None else s.get("elapsedDuration")
        avg_p = s.get("averagePower") if s.get("averagePower") is not None else s.get("avgPower")
        np_p = s.get("normalizedPower") if s.get("normalizedPower") is not None else s.get("normPower")
        avg_hr = s.get("averageHR") if s.get("averageHR") is not None else s.get("avgHR")
        laps.append({
            "lapIndex": s.get("lapIndex") or (idx + 1),
            "duration": dur,
            "elapsedDuration": s.get("elapsedDuration"),
            "movingDuration": s.get("movingDuration"),
            "distance": s.get("distance"),
            "avgPower": avg_p,
            "maxPower": s.get("maxPower"),
            "normalizedPower": np_p,
            "wkg": s.get("wkg"),
            "npWkg": s.get("npWkg"),
            "avgHR": avg_hr,
            "maxHR": s.get("maxHR"),
            "avgCadence": (
                s.get("averageBikeCadence")
                if s.get("averageBikeCadence") is not None
                else (
                    s.get("averageRunCadence")
                    if s.get("averageRunCadence") is not None
                    else (s.get("avgCadence") if s.get("avgCadence") is not None else s.get("averageCadence"))
                )
            ),
            "totalAscent": s.get("elevationGain") if s.get("elevationGain") is not None else s.get("totalAscent"),
            "totalDescent": s.get("elevationLoss") if s.get("elevationLoss") is not None else s.get("totalDescent"),
            "ef": s.get("ef"),
            "avgTemperature": s.get("averageTemperature") if s.get("averageTemperature") is not None else s.get("avgTemperature"),
            "maxTemperature": s.get("maxTemperature"),
            "minTemperature": s.get("minTemperature"),
            "intensityType": s.get("intensityType"),
            "lapTrigger": s.get("lapTrigger") or s.get("triggerType"),
            "startLatitude": s.get("startLatitude"),
            "startLongitude": s.get("startLongitude"),
            "endLatitude": s.get("endLatitude"),
            "endLongitude": s.get("endLongitude"),
        })
    return _enrich_laps_metrics(laps, weight_kg=weight_kg)


def _synthesize_arr_from_laps_or_summary(act: dict, laps: list, coords: list = None):
    """
    Rezervna (fallback) pot za zelo stare aktivnosti, kjer Garminov `/details` in FIT datoteka
    ne vrneta več 1-sekundnih meritev, imamo pa uradne kroge (`laps`) ali povzetek aktivnosti (`act`).
    Zgradi sintetičen 1 Hz niz iz krogov ali povzetka, da se 10-minutni bloki (`blocks`) in zemljevid (`coords`)
    vedno prikažejo brez napake!
    """
    import numpy as np

    p_segs, hr_segs, cad_segs, spd_segs, ele_segs = [], [], [], [], []
    cur_ele = float(act.get("minElevation") or 200.0)

    valid_laps = [lp for lp in (laps or []) if isinstance(lp, dict) and float(lp.get("duration") or 0) >= 1.0]
    if valid_laps:
        for lp in valid_laps:
            dur = int(max(1, min(72000, round(float(lp.get("duration") or 60)))))
            dist = float(lp.get("distance") or 0.0)
            spd_v = (dist / dur) if (dist > 0 and dur > 0) else float(act.get("averageSpeed") or 0.0)
            pw_v = float(lp.get("avgPower") or lp.get("normalizedPower") or act.get("avgPower") or act.get("averagePower") or 0.0)
            hr_v = float(lp.get("avgHR") or act.get("averageHR") or 0.0)
            cad_v = float(lp.get("avgCadence") or act.get("averageBikingCadenceInRevPerMinute") or act.get("averageRunningCadenceInStepsPerMinute") or 0.0)
            asc_v = float(lp.get("totalAscent") or 0.0)
            dsc_v = float(lp.get("totalDescent") or 0.0)

            p_segs.append(np.full(dur, pw_v, dtype=float))
            hr_segs.append(np.full(dur, hr_v, dtype=float))
            cad_segs.append(np.full(dur, cad_v, dtype=float))
            spd_segs.append(np.full(dur, spd_v, dtype=float))
            end_ele = cur_ele + asc_v - dsc_v
            ele_segs.append(np.linspace(cur_ele, end_ele, dur, dtype=float))
            cur_ele = end_ele

        out = {
            "power": np.concatenate(p_segs),
            "heartRate": np.concatenate(hr_segs),
            "cadence": np.concatenate(cad_segs),
            "speed": np.concatenate(spd_segs),
            "elevation": np.concatenate(ele_segs),
            "coords": coords or [],
        }
        if act.get("averageTemperature") is not None:
            out["temperature"] = np.full(len(out["speed"]), float(act["averageTemperature"]), dtype=float)
        return out

    # Če ni niti krogov, uporabi celoten povzetek aktivnosti
    total_dur = int(max(60, min(108000, round(float(act.get("movingDuration") or act.get("duration") or 600)))))
    total_dist = float(act.get("distance") or 0.0)
    avg_spd = float(act.get("averageSpeed") or (total_dist / total_dur if total_dur > 0 else 0.0))
    avg_pw = float(act.get("avgPower") or act.get("averagePower") or act.get("normPower") or 0.0)
    avg_hr = float(act.get("averageHR") or 0.0)
    avg_cad = float(act.get("averageBikingCadenceInRevPerMinute") or act.get("averageRunningCadenceInStepsPerMinute") or 0.0)
    gain = float(act.get("elevationGain") or 0.0)

    out = {
        "power": np.full(total_dur, avg_pw, dtype=float),
        "heartRate": np.full(total_dur, avg_hr, dtype=float),
        "cadence": np.full(total_dur, avg_cad, dtype=float),
        "speed": np.full(total_dur, avg_spd, dtype=float),
        "elevation": np.linspace(cur_ele, cur_ele + gain, total_dur, dtype=float),
        "coords": coords or [],
    }
    if act.get("averageTemperature") is not None:
        out["temperature"] = np.full(total_dur, float(act["averageTemperature"]), dtype=float)
    return out


def fetch_activity_bundle(activity_id, username=None, token=None, force=False, config=None, sync_cloud=True):
    """
    Na zahtevo (za katerikoli, tudi večletni trening >14 dni) prenese in predpomni:
    - 1 Hz tok iz /activity-service/activity/{id}/details (+ GPS iz geoPolylineDTO / directLatitude/directLongitude),
    - kroge iz /activity-service/activity/{id}/splits (+ FIT nadomestek prek fitdecode/fitparse),
    - izračuna 10-minutne odseke in povzetke ter shrani v data/users/<user>/streams/{id}.json, mmp/{id}.json in splits/{id}.json,
    - posodobi tudi vnos v activities.json in pošlje posodobljene datoteke na GitHub v ozadju.
    Če podatki že obstajajo lokalno na disku in niso nepopolni, jih takoj naloži z diska BREZ mrežnega klica!
    Vrne (stream_payload|None, splits_payload|None, sporočilo).
    """
    import gc
    import shutil
    import numpy as np
    import streamlib
    import github_sync

    if username:
        custom_key = (config or {}).get("garmin_api_key") if isinstance(config, dict) else None
        set_target_user(username=username, custom_api_key=custom_key)

    streams_dir = os.path.join(DATA_DIR, "streams")
    mmp_dir = os.path.join(DATA_DIR, "mmp")
    splits_dir = os.path.join(DATA_DIR, "splits")
    for d in (streams_dir, mmp_dir, splits_dir):
        os.makedirs(d, exist_ok=True)

    aid = str(activity_id)
    activities_raw = load_json_file("activities.json", {})
    if isinstance(activities_raw, list):
        activities_dict = {str(a.get("activityId")): a for a in activities_raw if isinstance(a, dict) and a.get("activityId")}
    elif isinstance(activities_raw, dict):
        activities_dict = activities_raw
    else:
        activities_dict = {}

    act = activities_dict.get(aid, {}) or {}
    stream_file = os.path.join(streams_dir, f"{aid}.json")
    split_file = os.path.join(splits_dir, f"{aid}.json")

    # Preveri tudi staro (legacy) mapo `data/streams/{aid}.json` oz. `data/splits/{aid}.json` in jo po potrebi prekopiraj v profil
    legacy_stream_file = os.path.join(PROJECT_DIR, "data", "streams", f"{aid}.json")
    legacy_split_file = os.path.join(PROJECT_DIR, "data", "splits", f"{aid}.json")
    if not os.path.exists(stream_file) and os.path.exists(legacy_stream_file):
        try:
            shutil.copy2(legacy_stream_file, stream_file)
        except Exception:
            pass
    if not os.path.exists(split_file) and os.path.exists(legacy_split_file):
        try:
            shutil.copy2(legacy_split_file, split_file)
        except Exception:
            pass

    # Če datoteke še ni lokalno, poskusi hitri prenos iz GitHuba (če bila že predpomnjena v repozitoriju)
    if not os.path.exists(stream_file) and CURRENT_USER and github_sync.is_github_configured():
        github_sync.github_pull_file(f"data/users/{CURRENT_USER}/streams/{aid}.json", overwrite=False)
    if not os.path.exists(split_file) and CURRENT_USER and github_sync.is_github_configured():
        github_sync.github_pull_file(f"data/users/{CURRENT_USER}/splits/{aid}.json", overwrite=False)

    # 1) Že veljaven predpomnilnik na lokalnem disku? (Brez mrežnega klica v Garmin!)
    cached_stream, cached_split = None, None
    if os.path.exists(stream_file):
        try:
            with open(stream_file, "r", encoding="utf-8") as f:
                cached_stream = json.load(f)
        except Exception:
            cached_stream = None
    if os.path.exists(split_file):
        try:
            with open(split_file, "r", encoding="utf-8") as f:
                cached_split = json.load(f)
        except Exception:
            cached_split = None

    stream_ok = cached_stream is not None and streamlib.stream_is_valid(cached_stream, act)
    expected_laps = int(act.get("lapCount") or 0)
    cached_laps = (cached_split or {}).get("laps") if isinstance(cached_split, dict) else None
    cached_dtos = (cached_split or {}).get("lapDTOs") if isinstance(cached_split, dict) else None
    has_nonempty_laps = bool((isinstance(cached_laps, list) and len(cached_laps) > 0) or (isinstance(cached_dtos, list) and len(cached_dtos) > 0))
    checked_zero = bool(isinstance(cached_split, dict) and cached_split.get("splitsChecked") and expected_laps == 0)
    count_incomplete = bool(expected_laps > 1 and isinstance(cached_laps, list) and len(cached_laps) < expected_laps)
    split_ok = bool(cached_split and (has_nonempty_laps or checked_zero) and not count_incomplete)
    if stream_ok and split_ok and not force:
        return cached_stream, cached_split, "Podrobnosti so že shranjene na disku."

    # 2) Pridobi dostopni žeton šele takrat, ko je prenos iz Garmina res potreben
    if token is None:
        # Najprej preveri, ali je žeton že v st.session_state["garmin_token"]
        try:
            import streamlit as st
            tok_ss = st.session_state.get("garmin_token")
            tok_ts = st.session_state.get("garmin_token_ts", 0)
            if tok_ss and (time.time() - tok_ts) < 18 * 3600 and not force:
                token = tok_ss
                CURRENT_TOKEN["value"] = token
        except Exception:
            pass
    if token is None:
        try:
            token = get_access_token()
            CURRENT_TOKEN["value"] = token
            try:
                import streamlit as st
                st.session_state["garmin_token"] = token
                st.session_state["garmin_token_ts"] = time.time()
            except Exception:
                pass
        except Exception as e:
            return cached_stream, cached_split, f"⚠️ Garmin seja ni aktivna ali je API ključ potekel: {e}"

    msgs = []
    fit_bytes_cache = None

    def _get_fit_bytes_once():
        nonlocal fit_bytes_cache
        if fit_bytes_cache is not None:
            return fit_bytes_cache
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/download-service/files/activity/{aid}",
                headers={"Authorization": f"Bearer {CURRENT_TOKEN['value'] or token}", "User-Agent": "GarminDashboard/1.0"},
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                fit_bytes_cache = resp.read()
                return fit_bytes_cache
        except Exception as e:
            msgs.append(f"FIT prenos: {e}")
            fit_bytes_cache = b""
            return b""

    # 3) Krogi / splits in podrobnosti aktivnosti (prenesemo najprej, da imamo tudi lape za morebitni fallback!)
    split_payload = cached_split if (split_ok and not force) else None
    act_detail, split_res = None, None
    if split_payload is None:
        weight_kg = _get_user_weight_kg()
        try:
            act_detail = _api_get_safe(f"/activity-service/activity/{aid}", token)
        except Exception as e:
            msgs.append(f"activity: {e}")
        try:
            split_res = _api_get_safe(f"/activity-service/activity/{aid}/splits", token)
        except Exception as e:
            msgs.append(f"splits: {e}")

        laps = _laps_from_split_dtos(split_res, act_detail=act_detail, weight_kg=weight_kg)
        raw_dtos = split_res.get("lapDTOs") if isinstance(split_res, dict) else None
        fit_used = False
        if not laps or (expected_laps > 1 and len(laps) < expected_laps):
            fb = _get_fit_bytes_once()
            if fb:
                fit_laps = _enrich_laps_metrics(streamlib.parse_fit_laps(fb), weight_kg=weight_kg)
                if len(fit_laps) > len(laps):
                    laps = fit_laps
                    fit_used = True

        # Ne prepiši obstoječih shranjenih krogov s praznim seznamom, če je API začasno zatajil
        prev_laps = (cached_split or {}).get("laps") if isinstance(cached_split, dict) else None
        if not laps and isinstance(prev_laps, list) and prev_laps:
            laps = _enrich_laps_metrics(prev_laps, weight_kg=weight_kg)

        split_payload = {
            "activityId": aid,
            "activityName": act.get("activityName") or (act_detail or {}).get("activityName"),
            "startTimeLocal": act.get("startTimeLocal"),
            "lapCount": expected_laps or len(laps),
            "summaryDTO": (act_detail or {}).get("summaryDTO") or ((cached_split or {}).get("summaryDTO") or {}),
            "metadataDTO": (act_detail or {}).get("metadataDTO") or ((cached_split or {}).get("metadataDTO") or {}),
            "laps": laps,
            "lapDTOs": raw_dtos if isinstance(raw_dtos, list) and raw_dtos else (laps or []),
            "splitsSource": "fit" if fit_used else "garmin_splits",
            "splitsChecked": True,
        }
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(split_payload, f, indent=2, ensure_ascii=False)
        if sync_cloud and CURRENT_USER and github_sync.is_github_configured():
            github_sync.github_push_file(f"data/users/{CURRENT_USER}/splits/{aid}.json", async_push=True)

    # 4) Tok 1 Hz + GPS zemljevid + 10-minutni bloki
    payload = cached_stream if (stream_ok and not force) else None
    if payload is None:
        details_dto, arr, d_res = {}, None, None
        try:
            d_res = api_get(f"/activity-service/activity/{aid}/details?maxChartSize=30000&maxPolylineSize=3000", token)
            if isinstance(d_res, dict):
                details_dto = {"measurementCount": d_res.get("measurementCount"), "metricsCount": d_res.get("metricsCount")}
                arr = streamlib.parse_details(d_res)
        except Exception as e:
            msgs.append(f"details: {e}")

        has_hr = arr is not None and bool(np.any(np.asarray(arr.get("heartRate", [0])) > 0))
        has_gps = arr is not None and bool(arr.get("coords"))
        has_power = arr is not None and bool(np.any(np.asarray(arr.get("power", [0])) > 0))
        wants_power = (act.get("avgPower") or act.get("averagePower") or 0) > 0

        if arr is None or not has_hr or not has_gps or (wants_power and not has_power):
            fb = _get_fit_bytes_once()
            if fb:
                try:
                    is_cyc = any(k in str(act.get("activityType", {})).lower() for k in ["cycl", "bik", "virtual_ride"])
                    fit_arr = streamlib.parse_fit(fb, double_cadence=not is_cyc)
                    arr = streamlib.merge_arrays(arr, fit_arr)
                except Exception as e:
                    msgs.append(f"FIT razčlenjevanje: {e}")

        # Izlušči GPS koordinate iz vseh razpoložljivih virov (geoPolylineDTO, directLatitude/Longitude, krogov ali start/end točk)
        extracted_coords = (arr or {}).get("coords") or []
        if len(extracted_coords) < 2 and isinstance(d_res, dict):
            poly = (d_res.get("geoPolylineDTO") or {}).get("polyline") or d_res.get("polyline") or []
            extracted_coords = streamlib.clean_coords(poly)
        if len(extracted_coords) < 2 and split_payload and split_payload.get("laps"):
            lap_pts = []
            for lp in split_payload["laps"]:
                if lp.get("startLatitude") is not None and lp.get("startLongitude") is not None:
                    lap_pts.append((lp["startLatitude"], lp["startLongitude"]))
                if lp.get("endLatitude") is not None and lp.get("endLongitude") is not None:
                    lap_pts.append((lp["endLatitude"], lp["endLongitude"]))
            extracted_coords = streamlib.clean_coords(lap_pts)
        if len(extracted_coords) < 2:
            se_pts = []
            if act.get("startLatitude") is not None and act.get("startLongitude") is not None:
                se_pts.append((act["startLatitude"], act["startLongitude"]))
            if act.get("endLatitude") is not None and act.get("endLongitude") is not None:
                se_pts.append((act["endLatitude"], act["endLongitude"]))
            if len(se_pts) == 1:
                se_pts.append(se_pts[0])
            extracted_coords = streamlib.clean_coords(se_pts)

        # Če za zelo star trening 1 Hz tok iz /details in FIT ni na voljo (ali vsebuje le koordinate),
        # sintetiziraj 1 Hz tok iz uradnih krogov (laps) ali povzetka aktivnosti, da zemljevida in 10-min blokov nikoli ne manjka!
        if arr is None or len(arr.get("speed", [])) == 0:
            laps_for_synth = (split_payload or {}).get("laps") or []
            arr = _synthesize_arr_from_laps_or_summary(act, laps_for_synth, coords=extracted_coords)
            details_dto["synthesizedFromLaps"] = True
        else:
            if not arr.get("coords") and extracted_coords:
                arr["coords"] = extracted_coords

        if arr is not None and len(arr.get("speed", [])) > 0:
            payload, mmp = streamlib.build_payload(act, arr, details_dto)
            payload["repairTried"] = True
            with open(stream_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            if sync_cloud and CURRENT_USER and github_sync.is_github_configured():
                github_sync.github_push_file(f"data/users/{CURRENT_USER}/streams/{aid}.json", async_push=True)
            if mmp:
                t_key = (act.get("activityType") or {}).get("typeKey", "")
                act_name = act.get("activityName", "Kolesarjenje")
                is_indoor = (t_key in ["virtual_ride", "indoor_cycling"]) or any(
                    k in act_name.lower() for k in ["zwift", "trainer", "trenažer", "rouvy", "mywhoosh", "indoor"])
                mmp_path = os.path.join(mmp_dir, f"{aid}.json")
                with open(mmp_path, "w", encoding="utf-8") as mf:
                    json.dump(
                        {
                            "activityId": aid,
                            "activityName": act_name,
                            "startTimeLocal": act.get("startTimeLocal"),
                            "is_indoor": is_indoor,
                            "durationSeconds": payload.get("length", 0),
                            "mmp": mmp,
                        },
                        mf,
                        indent=2,
                        ensure_ascii=False,
                    )
                if sync_cloud and CURRENT_USER and github_sync.is_github_configured():
                    github_sync.github_push_file(f"data/users/{CURRENT_USER}/mmp/{aid}.json", async_push=True)
        else:
            msgs.append("1 Hz tok ni na voljo")
        del arr, d_res
        gc.collect()

    # 5) Posodobi zapis v activities.json s povzetkom krogov/toka, če obstaja
    if aid in activities_dict and (payload or split_payload):
        try:
            act_entry = activities_dict[aid]
            changed_act = False
            if payload and isinstance(payload.get("summary"), dict):
                summ = payload["summary"]
                if summ.get("avgTemp") is not None and act_entry.get("averageTemperature") is None:
                    act_entry["averageTemperature"] = summ["avgTemp"]
                    changed_act = True
                if summ.get("minTemp") is not None and act_entry.get("minTemperature") is None:
                    act_entry["minTemperature"] = summ["minTemp"]
                    changed_act = True
                if summ.get("maxTemp") is not None and act_entry.get("maxTemperature") is None:
                    act_entry["maxTemperature"] = summ["maxTemp"]
                    changed_act = True
            if split_payload and isinstance(split_payload.get("summaryDTO"), dict):
                sdto = split_payload["summaryDTO"]
                for k_src, k_dst in (
                    ("averageTemperature", "averageTemperature"),
                    ("minTemperature", "minTemperature"),
                    ("maxTemperature", "maxTemperature"),
                    ("trainingEffect", "aerobicTrainingEffect"),
                    ("anaerobicTrainingEffect", "anaerobicTrainingEffect"),
                    ("activityTrainingLoad", "activityTrainingLoad"),
                ):
                    if sdto.get(k_src) is not None and act_entry.get(k_dst) is None:
                        act_entry[k_dst] = sdto[k_src]
                        changed_act = True
            if split_payload and isinstance(split_payload.get("laps"), list) and split_payload["laps"]:
                existing_act_laps = act_entry.get("laps")
                if not isinstance(existing_act_laps, list) or len(split_payload["laps"]) >= len(existing_act_laps):
                    act_entry["laps"] = split_payload["laps"]
                    changed_act = True
            if changed_act:
                activities_dict[aid] = act_entry
                save_json_file("activities.json", activities_dict)
                if sync_cloud and CURRENT_USER and github_sync.is_github_configured():
                    github_sync.github_push_file(f"data/users/{CURRENT_USER}/activities.json", async_push=True)
        except Exception:
            pass

    return payload, split_payload, ("; ".join(msgs) if msgs else "OK")


def ensure_full_activity_details(activity_id, username=None, config=None, force=False, token=None, sync_cloud=True):
    """
    Preveri, ali ima aktivnost `activity_id` na disku (`streams/{id}.json` in `splits/{id}.json`) že:
    - veljavne GPS točke (`coordinates` / `coords` / `polyline`),
    - uradne Garmin kroge (`laps` / `lapDTOs`),
    - 1-sekundni tok metrik (`details`) in izračunane 10-minutne bloke (`summary.blocks`).
    Če podatki že obstajajo lokalno na disku, jih naloži v hipu BREZ mrežnega klica v Garmin!
    Če manjkajo (npr. pri treningih starejših od 14 dni), jih samodejno prenese iz Garmina in trajno shrani.
    """
    return fetch_activity_bundle(
        activity_id=activity_id,
        username=username,
        token=token,
        force=force,
        config=config,
        sync_cloud=sync_cloud,
    )


if __name__ == "__main__":
    main()
