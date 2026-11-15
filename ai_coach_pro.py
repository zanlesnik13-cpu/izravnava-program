"""
ai_coach_pro.py - Napredni modul "🤖 AI Osebni Trener & Pametni Koledar (Rest or Train Pro)"
PART 1 nadgradnja:
1. Hibridni sistem Gemini API ključev in modelov (Privzeti Flash vs. Osebni ključ za Pro modele)
2. Izračun dnevne pripravljenosti "Rest or Train" (0–100 %) s semaforjem (🟢 TRAIN HARD / 🟡 MODULATE / 🔴 REST)
3. Večnitna arhitektura pogovorov (Multi-Thread Chat kot Gemini / Claude) v ai_chat_history.json:
   - Upravljalnik pogovorov v levem stolpcu [1, 3] z gumbom "➕ Začni nov pogovor", seznamom pogovorov (Datum, Naslov, Kratek povzetek), preimenovanjem in brisanjem
   - Hitri ukazi (1-klik gumbi) samodejno odprejo novo, ločeno tematsko nit pogovora
   - Pametno zlaganje sporočil po izmenjavah (User + Assistant): zadnje 3 izmenjave razširjene v celoti, starejše zložene v st.expander z naslovom:
     "🕒 [Datum in ura] | 💬 Vprašanje: <prvih 50 znakov> ➔ 🤖 Povzetek: <prvi stavek / ključna odločitev>"
   - Vedno dostopno pripeto vnosno polje st.chat_input("Vprašaj AI trenerja v tem pogovoru...") pod drsnim oknom pogovora (st.container(height=650))
4. Podzavihek 3: "🧠 Moja trenerska pravila, cilji in navodila (@kolesarski-trening)" shranjen v data/users/<username>/custom_coach_rules.json (z auto-save in najvišjo prioriteto v System Promptu)
5. Podzavihek 2: "📅 Pametni Koledar Treningov" (training_calendar.json) s primerjavo Planirano vs. Opravljeno, AI vpisom (<calendar_update>) in izvozom (.ics / Markdown)
"""

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
import re
import time
import json
import uuid
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import streamlit as st
import streamlib

detect_stream_intervals = streamlib.detect_stream_intervals


# ==============================================================================
# 0. ČISTA PREDLOGA PROFILA IN NAVODIL TRENERJA (@kolesarski-trening) & MODELI
# ==============================================================================
MODEL_FLASH_DEFAULT_LABEL = "Gemini Flash - Privzeti (gemini-3.8-flash)"
GEMINI_CUSTOM_MODEL_LABEL = "Poljuben model (vnesi ime modela)"

GEMINI_MODEL_OPTIONS = [
    "gemini-4-hydrogen",
    "gemini-3.8-flash",
    "gemini-3.8-pro",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    GEMINI_CUSTOM_MODEL_LABEL,
]

CLAUDE_MODEL_OPTIONS = [
    "🟣 Claude Haiku 5.5 (Anthropic – Najnovejši, izjemno hiter in zelo ugoden, ~5€ za več mesecev)",
    "🟣 Claude Sonnet 4.6 (Anthropic – Najnovejši za vrhunsko globoko analizo)",
    "🟣 Claude Opus 4.6 (Anthropic – Najzmogljivejši model za kompleksno planiranje)",
]

MODEL_OPTIONS_PRO = GEMINI_MODEL_OPTIONS + CLAUDE_MODEL_OPTIONS

DEFAULT_COACH_TEMPLATE = """Ti si moj osebni trener za vzdržljivostne športe (cestno kolesarstvo / trenažer / tek). Tvoja naloga je dolgoročno izboljševati mojo zmogljivost in mi pomagati sprejemati dobre odločitve glede treninga, regeneracije, dirk in periodizacije. Deluj kot izkušen trener za tekmovalnega rekreativnega športnika. Pri odločanju vedno upoštevaj mojo celotno zgodovino treninga, trenutno pripravljenost, zadnje treninge, spanje, utrujenost, počutje, življenjske obveznosti in dolgoročne cilje. Ne obravnavaj vsakega treninga ločeno. Vedno poskušaj razumeti širši kontekst zadnjih dni in tednov. Če podatki manjkajo ali so nejasni, me vprašaj za dodatne informacije, namesto da si jih izmisliš.

==================================================
NAČELA TRENINGA IN METODOLOGIJA
==================================================
- Daj prednost dolgoročnemu napredku pred kratkoročnimi dobitki.
- Ne povečuj trening obremenitve brez jasnega razloga.
- Ne dodajaj intenzivnosti samo zato, ker se na posamezen dan počutim dobro.
- Ne zmanjšuj treninga samo zaradi enega slabega občutka, če širši kontekst kaže, da sem dobro regeneriran.
- Vedno upoštevaj kumulativno utrujenost in celostno oceno pripravljenosti (kombinacija nočnega HRV, povprečnega nočnega srčnega utripa, ure bujenja, globokega spanca in preteklega TSS).
- Kadar obstaja večje tveganje za preutrujenost, predlagaj bolj konservativen pristop.
- Če je kakovost ključnega treninga pomembnejša od količine, raje zmanjšaj količino kot kakovost.
- Ne poskušaj vsak trening narediti maksimalno težak; razporejaj intenzivne treninge tako, da omogočajo dobro regeneracijo.
- Ne sklepaj prehitro na podlagi enega samega treninga; pri ocenjevanju napredka uporabljaj več tednov podatkov.
- Če se moja predpostavka ne ujema s podatki, mi to neposredno povej in razloži zakaj. Ne potrjuj mojih idej samo zato, ker jih predlagam jaz.

==================================================
JUTRANJI PREGLED STANJA (7 TOČK)
==================================================
Na podlagi podatkov o spancu (trajanje, kakovost, sleep score, faze spanca, HRV, povprečni nočni HR, jutranji RHR, ura bujenja) in subjektivnega počutja skupaj z zgodovino zadnjih treningov vsako jutro oceni:
1. Trenutno pripravljenost.
2. Stopnjo utrujenosti (loči med dejansko utrujenostjo in zgodnjim bujenjem z budilko ob mirnem nočnem utripu).
3. Ali je današnji načrtovani trening primeren.
4. Ali naj trening ostane nespremenjen.
5. Ali naj ga zmanjšam ali prilagodim.
6. Ali naj naredim lahek Z2 ali regeneracijsko vožnjo/tek.
7. Ali je smiselno izvesti popoln počitek.

==================================================
PRIPRAVA TRENINGOV (10 TOČK)
==================================================
Ko pripraviš posamezen trening, vedno navedi:
1. Namen treninga.
2. Fiziološki cilj.
3. Trajanje.
4. Ciljno moč v W (oz. ciljni tempo /km za tek).
5. Odstotek FTP (oz. pragovnega tempa), kadar je smiseln.
6. Predviden RPE (1–10).
7. Strukturo intervalov in počitkov (dovolj konkretno za izvedbo zunaj ali na trenažerju brez ugibanja).
8. Kaj naj spremljam med treningom (HR drift, kadenca, občutek v nogah).
9. Kdaj naj zmanjšam intenzivnost.
10. Kdaj naj trening predčasno zaključim.

==================================================
ANALIZA TRENINGA & ANALIZA FORME
==================================================
Ko analiziraš končan trening, preveri 10-minutne bloke, samodejno zaznane intervale/klance, razmerje moč/HR (EF), srčni drift (decoupling), vpliv terena/semaforjev ter kakovost zadnjih intervalov. Odgovori v 3 sklopih: (1) Pregled ključnih metrik, (2) Trenerska ocena in natančna razlaga dinamike/drifta po odsekih, (3) Prehrana, regeneracija in načrt za naslednje dni.

==================================================
# MOJ PROFIL ŠPORTNIKA
==================================================
- Višina: [Vpiši svojo višino, npr. 175 cm]
- Teža med sezono / trenutna teža: [Vpiši svojo težo, npr. 68 kg med sezono / 70 kg pozimi]
- Šport in oprema: [Vpiši svoj primarni šport in opremo, npr. Cestno kolesarjenje + pametni trenažer / Tek]
- Tip športnika: [Vpiši svoj tip, npr. Klančar / Kronometrist / Vsestranski kolesar / Maratonec]
- Glavni fokus: [Vpiši fokus, npr. 20–45 min klanci, granfondo dirke, vzdržljivost]

==================================================
# TRENUTNA PRIPRAVLJENOST IN CONE
==================================================
- Trenutni FTP zunaj / na trenažerju (oz. Pragovni tempo za tek): [Vpiši svoj FTP zunaj in na trenažerju, npr. Zunaj 310 W / Trenažer 295 W ali T-Pace 4:15/km]
- Osebni rekordi (5m / 20m / 30m / 40m moč ali tekaški rekordi): [Vpiši svoje najboljše dosežke, npr. 5m: 380 W, 20m: 325 W, 40m: 310 W]
- Posebnosti pri conah: [Vpiši želene vate za Z2 bazo, Sweet Spot, Threshold, VO2max]

==================================================
# KAJ MI NAJBOLJ USTREZA IN MOJE SLABOSTI
==================================================
- Kateri intervali mi prinašajo največ napredka: [Npr. 3x15 min Sweet Spot, 2x20 min Threshold, 30/15s VO2max, dolge Z2 vožnje]
- Ciljni tedenski volumen v urah: [Vpiši okvir ur na teden, npr. 8–12 ur/teden, 2 ključna intervalna treninga + Z2 baza]
- Občutljivost na vročino / mraz in druge posebnosti: [Npr. občutljivost na vročino nad 30 °C, življenjski urnik, služba/faks]
- Poškodbe ali omejitve: [Vpiši morebitne pretekle poškodbe ali vaje za preventivo, npr. občutljiva kolenska tetiva, vaje za jedro]

==================================================
# MOJI KRATKOROČNI IN DOLGOROČNI CILJI (TEKME)
==================================================
- Glavne dirke / tekme in datumi: [Vpiši glavne ciljne tekme in datume v sezoni]
- Ciljni vati / W/kg / časi na glavnih ciljih: [Vpiši konkretne cilje, npr. ciljni čas vzpona, ciljni FTP ali W/kg]"""

DEFAULT_COACH_RESPONSE_STYLE = """Bodi neposreden, strokoven in praktičen v slovenščini. Ne potrebujem nepotrebnega motivacijskega govorjenja ("no-fluff").
- Ko je nekaj dobra ideja, povej zakaj. Ko je nekaj slaba ideja, povej neposredno in razloži razlog.
- Pri treningu daj prednost kakovosti pred nepotrebno količino.
- Če imam zaradi slabega spanca, vročine (>30 °C) ali utrujenosti slab dan, ne obravnavaj tega kot izgubo forme, dokler podatki ne kažejo dejanskega padca zmogljivosti.
- Če imam zaradi dobrega počutja zelo dober dan, ne sklepaj avtomatsko, da je moj FTP višji.
- Vedno jasno loči med: (1) dejstvi iz podatkov, (2) verjetnimi fiziološkimi sklepi ter (3) ocenami in negotovostjo.
- Pri pomembnih odločitvah povej tudi, zakaj si izbral določeno možnost pred alternativami."""

# Privzete konstante kažejo na čisto predlogo (DEFAULT_COACH_TEMPLATE) za vse nove uporabnike
DEFAULT_KOLESARSKI_TRENING_INSTRUCTIONS = DEFAULT_COACH_TEMPLATE
DEFAULT_KOLESARSKI_TRENING_RULES = DEFAULT_COACH_TEMPLATE


def load_custom_coach_rules(user_dir: str, user_cfg: dict = None) -> dict:
    """
    Naloži uporabnikova osebna trenerska pravila, slog odgovarjanja in pripete dogovore iz
    data/users/<username>/custom_coach_rules.json.
    Če datoteka še ne obstaja, jo samodejno inicializira s čisto predlogo DEFAULT_COACH_TEMPLATE
    (oz. z obstoječimi navodili uporabnika v config.json).
    """
    user_dir = str(user_dir or DATA_DIR)
    path = os.path.join(user_dir, "custom_coach_rules.json")
    default_rules = (user_cfg or {}).get("custom_coach_instructions") or DEFAULT_COACH_TEMPLATE
    default_pins = []
    defaults = {
        "rules_and_profile": default_rules,
        "response_style": DEFAULT_COACH_RESPONSE_STYLE,
        "pinned_memories": default_pins,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if data.get("rules_and_profile"):
                    defaults["rules_and_profile"] = str(data["rules_and_profile"])
                if data.get("response_style"):
                    defaults["response_style"] = str(data["response_style"])
                if isinstance(data.get("pinned_memories"), list):
                    defaults["pinned_memories"] = [str(x).strip() for x in data["pinned_memories"] if str(x).strip()]
                if data.get("updated_at"):
                    defaults["updated_at"] = str(data["updated_at"])
                return defaults
        except Exception:
            pass
    save_custom_coach_rules(user_dir, defaults)
    return defaults


def _trigger_cloud_sync_for_dir(user_dir: str, changed_filename: str = None):
    """Nenasilno sproži sinhronizacijo posamezne spremenjene datoteke na GitHub v ozadju."""
    try:
        if user_dir:
            u_clean = os.path.basename(os.path.normpath(user_dir))
            if u_clean and u_clean.lower() not in ("data", "", "."):
                import github_sync
                if changed_filename:
                    rel_p = f"data/users/{u_clean}/{changed_filename}"
                    github_sync.github_push_file(rel_p, commit_message=f"Update {changed_filename} for {u_clean}", async_push=True)
                else:
                    github_sync.github_push_user_files(
                        u_clean,
                        only_files=["custom_coach_rules.json", "coach_memory.json", "ai_chat_history.json", "training_calendar.json"],
                        async_push=True,
                    )
    except Exception:
        pass


def save_custom_coach_rules(user_dir: str, rules_data: dict) -> bool:
    """Shrani trenerska pravila, slog odgovarjanja in pripete dogovore v data/users/<username>/custom_coach_rules.json."""
    user_dir = str(user_dir or DATA_DIR)
    path = os.path.join(user_dir, "custom_coach_rules.json")
    try:
        os.makedirs(user_dir, exist_ok=True)
        existing_pins = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as rf:
                    old_d = json.load(rf)
                    if isinstance(old_d, dict) and isinstance(old_d.get("pinned_memories"), list):
                        existing_pins = old_d["pinned_memories"]
            except Exception:
                pass
        pins = rules_data.get("pinned_memories") if isinstance(rules_data.get("pinned_memories"), list) else existing_pins
        payload = {
            "rules_and_profile": str(rules_data.get("rules_and_profile") or DEFAULT_COACH_TEMPLATE).strip(),
            "response_style": str(rules_data.get("response_style") or DEFAULT_COACH_RESPONSE_STYLE).strip(),
            "pinned_memories": [str(x).strip() for x in pins if str(x).strip()],
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        _trigger_cloud_sync_for_dir(user_dir, "custom_coach_rules.json")
        return True
    except Exception:
        return False


COACH_MEMORY_FILE = "coach_memory.json"


def get_coach_memory_path(user_dir: str) -> str:
    """Vrne absolutno pot do trajnega spomina trenerja za uporabnika."""
    return os.path.join(user_dir, COACH_MEMORY_FILE)


def load_coach_memory(user_dir: str) -> dict:
    """
    Naloži trajni dolgoročni spomin trenerja (coach_memory.json).
    Če datoteka še ne obstaja, samodejno migrira obstoječe pripete dogovore
    iz custom_coach_rules.json, da se nič ne izgubi.
    """
    if not user_dir:
        return {"facts": [], "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    p = get_coach_memory_path(user_dir)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
                if isinstance(d, dict) and "facts" in d:
                    return d
                elif isinstance(d, list):
                    return {"facts": [str(x).strip() for x in d if str(x).strip()], "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
        except Exception:
            pass

    # Samodejna migracija iz custom_coach_rules.json ob prvem zagonu
    migrated_facts = []
    rules_p = os.path.join(user_dir, "custom_coach_rules.json")
    if os.path.exists(rules_p):
        try:
            with open(rules_p, "r", encoding="utf-8") as rf:
                r_data = json.load(rf)
                pins = r_data.get("pinned_memories") or []
                if isinstance(pins, list):
                    migrated_facts = [str(x).strip() for x in pins if str(x).strip()]
        except Exception:
            pass

    mem_obj = {"facts": migrated_facts[:35], "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    if migrated_facts:
        save_coach_memory(user_dir, mem_obj)
    return mem_obj


def save_coach_memory(user_dir: str, memory_data: dict) -> bool:
    """Varno shrani strukturiran dolgoročni spomin trenerja (omejen na max 35 dejstev / ~500 žetonov)."""
    if not user_dir:
        return False
    try:
        p = get_coach_memory_path(user_dir)
        facts = list(memory_data.get("facts") or [])
        clean_facts = []
        for f in facts:
            fs = str(f).strip()
            if fs and fs not in clean_facts:
                clean_facts.append(fs)
        payload = {
            "facts": clean_facts[:35],
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        _trigger_cloud_sync_for_dir(user_dir, "coach_memory.json")
        return True
    except Exception:
        return False


def add_coach_memory_fact(user_dir: str, fact: str) -> bool:
    """Doda novo ključno dejstvo v dolgoročni spomin trenerja (z deduplikacijo in mejo 35 dejstev)."""
    clean = str(fact or "").strip()
    if not clean or not user_dir:
        return False
    clean = re.sub(r"^[-*•\s]+", "", clean).strip()
    mem = load_coach_memory(user_dir)
    facts = mem.get("facts", [])
    clean_lower = clean.lower()
    for existing in facts:
        if clean_lower == existing.lower() or (len(clean_lower) > 20 and clean_lower in existing.lower()):
            return False
    facts.insert(0, clean)
    mem["facts"] = facts[:35]
    res = save_coach_memory(user_dir, mem)
    try:
        rules_obj = load_custom_coach_rules(user_dir)
        rules_obj["pinned_memories"] = mem["facts"]
        save_custom_coach_rules(user_dir, rules_obj)
    except Exception:
        pass
    return res


def remove_coach_memory_fact(user_dir: str, index_to_remove: int) -> bool:
    """Odstrani dejstvo iz dolgoročnega spomina po indeksu."""
    if not user_dir:
        return False
    mem = load_coach_memory(user_dir)
    facts = mem.get("facts", [])
    if 0 <= index_to_remove < len(facts):
        facts.pop(index_to_remove)
        mem["facts"] = facts
        res = save_coach_memory(user_dir, mem)
        try:
            rules_obj = load_custom_coach_rules(user_dir)
            rules_obj["pinned_memories"] = facts
            save_custom_coach_rules(user_dir, rules_obj)
        except Exception:
            pass
        return res
    return False


def add_pinned_coach_memory(user_dir: str, note: str, user_cfg: dict = None) -> bool:
    return add_coach_memory_fact(user_dir, note)


def remove_pinned_coach_memory(user_dir: str, index_to_remove: int, user_cfg: dict = None) -> bool:
    return remove_coach_memory_fact(user_dir, index_to_remove)


def extract_memory_updates_from_text(text: str) -> tuple[str, list[str]]:
    """
    Poišče in izvleče vse <memory_update>...</memory_update> oznake, ki jih AI trener izpiše,
    jih odstrani iz vidnega odgovora uporabniku ter vrne (očiščeno_besedilo, seznam_novih_dejstev).
    """
    if not text or "<memory_update>" not in text:
        return text or "", []
    pattern = re.compile(r"<memory_update>(.*?)</memory_update>", re.DOTALL | re.IGNORECASE)
    facts = []
    for match in pattern.finditer(text):
        f = match.group(1).strip()
        if f and f not in facts:
            facts.append(f)
    cleaned = pattern.sub("", text).strip()
    return cleaned, facts


_LAST_TOKEN_USAGE: dict = {}


def _set_last_token_usage(usage: dict):
    global _LAST_TOKEN_USAGE
    clean = {
        "prompt_tokens": int((usage or {}).get("prompt_tokens") or 0),
        "cached_tokens": int((usage or {}).get("cached_tokens") or 0),
        "candidates_tokens": int((usage or {}).get("candidates_tokens") or 0),
        "total_tokens": int((usage or {}).get("total_tokens") or 0),
    }
    _LAST_TOKEN_USAGE = dict(clean)
    try:
        st.session_state["_last_ai_token_usage"] = dict(clean)
    except Exception:
        pass


def get_last_token_usage() -> dict:
    """Vrne zadnjo zabeleženo porabo žetonov iz response.usage_metadata / st.session_state."""
    try:
        ss_u = st.session_state.get("_last_ai_token_usage")
        if isinstance(ss_u, dict) and ss_u:
            return dict(ss_u)
    except Exception:
        pass
    return dict(_LAST_TOKEN_USAGE) if isinstance(_LAST_TOKEN_USAGE, dict) else {}


def format_token_usage_caption(usage: dict) -> str:
    """Oblikuje prikaz porabe žetonov pod odgovorom AI trenerja."""
    if not isinstance(usage, dict):
        return ""
    p_tok = int(usage.get("prompt_tokens") or 0)
    c_tok = int(usage.get("cached_tokens") or 0)
    o_tok = int(usage.get("candidates_tokens") or 0)
    t_tok = int(usage.get("total_tokens") or (p_tok + o_tok))
    if t_tok <= 0 and p_tok <= 0:
        return ""
    return (
        f"📊 Žetoni: {p_tok:,} vhodnih (od tega {c_tok:,} iz predpomnilnika) "
        f"+ {o_tok:,} izhodnih = skupaj {t_tok:,}"
    )


def clean_assistant_message_for_history(content: str, max_chars: int = 1500) -> str:
    """
    Očisti sporočilo asistenta pred ponovnim pošiljanjem v zgodovini API-ja:
    - odstrani <calendar_update> in <memory_update> XML/JSON bloke,
    - odstrani odvečne ločnice markdown tabel (|---|---|) in redundantno oblikovanje (###, **, ---),
    - zgošča prazne vrstice in po potrebi ohrani uvodno diagnozo + zaključna priporočila.
    """
    txt = str(content or "").strip()
    if not txt:
        return ""
    txt = re.sub(r"<calendar_update>.*?</calendar_update>", "", txt, flags=re.DOTALL | re.IGNORECASE)
    txt = re.sub(r"<memory_update>.*?</memory_update>", "", txt, flags=re.DOTALL | re.IGNORECASE)
    out_lines = []
    for raw_ln in txt.splitlines():
        s = raw_ln.strip()
        if not s:
            continue
        if re.match(r"^\|?[\s:\-]+\|[\s\|:\-]*$", s):
            continue
        if re.match(r"^[\-\*_=]{3,}$", s):
            continue
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            s = " | ".join(c for c in cells if c)
        s = re.sub(r"^#{1,6}\s+", "", s)
        s = re.sub(r"\*\*|__|~~|`", "", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        if s:
            out_lines.append(s)
    cleaned = "\n".join(out_lines)
    if max_chars and len(cleaned) > max_chars:
        head_n = int(max_chars * 0.65)
        tail_n = max(120, max_chars - head_n - 8)
        cleaned = cleaned[:head_n].rstrip() + "\n[...]\n" + cleaned[-tail_n:].lstrip()
    return cleaned


def _extract_dense_coach_bullets(content: str, max_items: int = 2) -> str:
    """Izlušči ključne sklepe s številkami (W, NP, bpm, ms, TSS, EF, Z1-Z5, min, km) za jedrnat povzetek niti."""
    cleaned = clean_assistant_message_for_history(content, max_chars=2400)
    if not cleaned:
        return ""
    metric_pat = re.compile(
        r"\b(?:\d+(?:[.,]\d+)?\s*(?:W|w|bpm|ms|TSS|EF|km|min|h|%|kcal|kJ|W/kg|rpm|spm|/km)|Z[1-5]|FTP|HRV|RHR|ACWR)\b"
    )
    chosen = []
    for ln in cleaned.splitlines():
        s = re.sub(r"^[\-\*•\d\.\)\s]+", "", ln).strip()
        if len(s) < 10:
            continue
        if metric_pat.search(s) or any(k in s.lower() for k in ("priporoč", "odločit", "ohrani", "zmanjš", "povečaj", "počitek", "interval", "naslednj")):
            if len(s) > 130:
                s = s[:127].rstrip() + "…"
            if s not in chosen:
                chosen.append(s)
            if len(chosen) >= max_items:
                break
    if not chosen:
        return summarize_message_short(cleaned, 120)
    return "; ".join(chosen)


def build_compact_thread_summary(older_messages: list, existing_summary: str = "", max_words: int = 180) -> str:
    """
    Samodejno stisne starejša sporočila (0 .. -5) v strnjen alinejni povzetek (max 180 besed)
    ključnih številk, fizioloških ugotovitev in dogovorjenih prilagoditev treninga.
    """
    if not older_messages:
        return ""
    exchanges = group_messages_into_exchanges(older_messages)
    bullets = []
    for idx, ex in enumerate(exchanges, 1):
        u_txt = str((ex.get("user") or {}).get("content") or "").strip()
        a_msg = ex.get("assistant") or {}
        a_txt = str(a_msg.get("content") or "").strip()
        q_short = summarize_message_short(u_txt, 50) if u_txt else f"Korak {idx}"
        a_dense = _extract_dense_coach_bullets(a_txt, max_items=2) if a_txt else ""
        if a_dense:
            bullets.append(f"• {q_short} -> {a_dense}")
        elif u_txt:
            bullets.append(f"• {q_short}")

    if not bullets and existing_summary:
        bullets = [ln.strip() for ln in str(existing_summary).splitlines() if ln.strip()]

    # Vedno ohrani najnovejše sklepe znotraj omejitve max_words (privzeto 180 besed)
    selected_rev = []
    used_words = 0
    for b in reversed(bullets):
        bw = len(b.split())
        if selected_rev and used_words + bw > max_words:
            break
        selected_rev.append(b)
        used_words += bw
    selected = list(reversed(selected_rev)) if selected_rev else bullets[-1:]
    summary = "\n".join(selected)
    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]).rstrip(".,;:") + "…"
    return summary


def estimate_context_tokens(
    context_text: str,
    chat_history: list = None,
    system_instruction: str = None,
    thread_summary: str = "",
) -> int:
    """
    Ocenjevanje števila žetonov za slovenska besedila (~3.5 znaka na žeton)
    glede na dejansko obrezano zgodovino, ki se pošlje API-ju.
    """
    total_chars = len(context_text or "")
    if system_instruction:
        total_chars += len(str(system_instruction))
    if chat_history:
        trimmed = trim_chat_history_for_api(chat_history, thread_summary=thread_summary)
        for m in trimmed:
            total_chars += len(str(m.get("content") or ""))
    return max(150, int(total_chars / 3.5))


def trim_chat_history_for_api(chat_history: list, thread_summary: str = "") -> list:
    """
    Optimizira zgodovino aktivne niti za API klice (Layer 1 Rolling Window + Thread Summary):
    - Zadnja 4 sporočila (zadnji 2 uporabniški vprašanji + zadnja 2 odgovora trenerja) obdrži na voljo kot neposreden kontekst,
      pri čemer odgovorom asistenta očisti odvečne markdown tabele in XML oznake.
    - Če ima nit več kot 4 sporočila, starejša sporočila (0 .. -5) strne v kompakten povzetek (max 180 besed)
      pod oznako '[Povzetek dosedanjega pogovora]: ...' pred zadnjimi 4 sporočili.
    """
    if not chat_history:
        return []

    recent_raw = chat_history[-4:]
    recent_clean = []
    for m in recent_raw:
        role = "user" if m.get("role") == "user" else "assistant"
        raw_c = str(m.get("content") or "").strip()
        if not raw_c:
            continue
        txt_c = clean_assistant_message_for_history(raw_c, max_chars=1500) if role == "assistant" else raw_c
        if txt_c:
            recent_clean.append({"role": role, "content": txt_c})

    if len(chat_history) <= 4:
        return recent_clean

    older_msgs = chat_history[:-4]
    summary_txt = str(thread_summary or "").strip() or build_compact_thread_summary(older_msgs, max_words=180)
    if not summary_txt:
        return recent_clean

    return [{"role": "user", "content": f"[Povzetek dosedanjega pogovora]:\n{summary_txt}"}] + recent_clean


def build_cross_thread_memory_summary(
    user_dir: str,
    user_cfg: dict = None,
    exclude_thread_id: str = None,
    include_other_threads: bool = True,
) -> str:
    """
    Pregleda coach_memory.json (ter po želji kratke sklepe zadnjih 3 niti)
    in sestavi strnjen razdelek dolgoročnega spomina.
    Opomba: Za prefix-cache stabilen system_instruction naj bo include_other_threads=False,
    da dodajanje sporočil v tekočem pogovoru ne spreminja sistemske predpone bajt za bajtom.
    """
    lines = []
    mem_obj = load_coach_memory(user_dir) if user_dir else {}
    facts = mem_obj.get("facts") or []
    if not facts and user_dir:
        rules_obj = load_custom_coach_rules(user_dir, user_cfg)
        facts = rules_obj.get("pinned_memories") or []
    if facts:
        lines.append("📌 **Ključna dejstva, urnik in posebnosti športnika (Dolgoročni spomin):**")
        for f_txt in facts[:35]:
            lines.append(f"  - {f_txt}")

    if user_dir and include_other_threads:
        store = load_ai_chat_store(user_dir)
        threads = store.get("threads", {})
        sorted_tids = sorted(
            threads.keys(),
            key=lambda k: str(threads[k].get("updated_at", "")),
            reverse=True,
        )
        recent_thread_notes = []
        for tid in sorted_tids:
            if exclude_thread_id and tid == exclude_thread_id:
                continue
            th = threads[tid]
            msgs = th.get("messages", [])
            if not msgs:
                continue
            ex_list = group_messages_into_exchanges(msgs)
            if not ex_list:
                continue
            t_title = th.get("title", tid)
            t_tag = th.get("topic_tag", "Splošno")
            last_ex = ex_list[-1]
            u_q = summarize_message_short((last_ex.get("user") or {}).get("content", ""), 60)
            a_s = (last_ex.get("assistant") or {}).get("short_summary") or summarize_message_short((last_ex.get("assistant") or {}).get("content", ""), 95)
            recent_thread_notes.append(f"  - [{t_tag}] **{t_title}**: Vprašanje: *{u_q}* ➔ Sklep: **{a_s}**")
            if len(recent_thread_notes) >= 3:
                break
        if recent_thread_notes:
            lines.append("🗂️ **Zadnji sklepi iz ostalih pogovornih niti:**")
            lines.extend(recent_thread_notes)

    if not lines:
        return "Ni shranjenih posebnih dejstev ali preteklih pogovorov."
    return "\n".join(lines)


# ==============================================================================
# 1. HIBRIDNI GEMINI & CLAUDE (ANTHROPIC) API KLJUČI, MODELI & 503 RETRY LOGIKA
# ==============================================================================
def read_env_file_key(env_path: str, key: str) -> str:
    """Prebere ključ iz .env ali .env.local datoteke, če obstaja."""
    if not env_path or not os.path.exists(env_path):
        return ""
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def get_master_gemini_key(project_dir: str) -> str:
    """Pridobi privzeti (master) Gemini API ključ iz st.secrets, os.environ ali .env.local."""
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            val = str(st.secrets["GEMINI_API_KEY"]).strip()
            if val:
                return val
    except Exception:
        pass
    env_val = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_val:
        return env_val
    for fname in (".env.local", ".env"):
        val = read_env_file_key(os.path.join(project_dir, fname), "GEMINI_API_KEY")
        if val:
            return val
    return ""


def get_available_model_options(user_cfg: dict) -> list:
    """
    Vrne seznam odklenjenih modelov v izbirniku glede na vnesene osebne ključe:
    - Brez osebnih ključev: samo privzeti Gemini Flash (`Gemini Flash - Privzeti`)
    - Z osebnim Gemini ključem: Gemini Flash, Gemini Pro in Poljuben model
    - Z osebnim Claude ključem: Claude 3.5 Haiku, Claude Sonnet (3.7/3.5), Claude 3 Haiku
    """
    has_gem = bool(str((user_cfg or {}).get("gemini_api_key") or "").strip())
    has_cld = bool(str((user_cfg or {}).get("claude_api_key") or "").strip())
    opts = []
    if has_gem:
        opts.extend(GEMINI_MODEL_OPTIONS)
    else:
        opts.append(MODEL_FLASH_DEFAULT_LABEL)
    if has_cld:
        opts.extend(CLAUDE_MODEL_OPTIONS)
        if GEMINI_CUSTOM_MODEL_LABEL not in opts:
            opts.append(GEMINI_CUSTOM_MODEL_LABEL)
    return opts


def _dedup_models(seq: list) -> list:
    out = []
    for m in seq:
        ms = str(m or "").strip()
        if ms and ms not in out:
            out.append(ms)
    return out


def resolve_gemini_auth(user_cfg: dict, project_dir: str) -> dict:
    """
    Vrne slovar z informacijami o aktivnem ponudniku (Gemini ali Claude), API ključu in kandidatih modelov:
    - provider: "gemini" ali "claude"
    - has_personal_key: True, če je uporabnik vnesel lasten Gemini ali Claude ključ
    - has_personal_gemini_key / has_personal_claude_key
    - api_key: dejanski ključ za klic
    - selected_model_ui: oznaka modela v UI
    - display_model_id: prikazno ime aktivnega modela
    - candidate_models: seznam modelov po vrstnem redu preizkušanja
    """
    personal_gem_key = str((user_cfg or {}).get("gemini_api_key") or "").strip()
    personal_cld_key = str((user_cfg or {}).get("claude_api_key") or "").strip()
    master_gem_key = get_master_gemini_key(project_dir)

    avail_opts = get_available_model_options(user_cfg)
    raw_choice = str((user_cfg or {}).get("ai_model_choice") or avail_opts[0])
    legacy_map = {
        "Gemini 3.8 Flash (Hiter, visoka kvota)": "gemini-3.8-flash",
        "Gemini 3.1 Pro / 2.5 Pro (Globoka analiza in kompleksno planiranje)": "gemini-2.5-pro",
    }
    mapped_choice = legacy_map.get(raw_choice, raw_choice)
    choice = mapped_choice if mapped_choice in avail_opts else avail_opts[0]
    custom_name = str((user_cfg or {}).get("ai_custom_model") or "").strip()
    custom_wants_claude = bool(custom_name and custom_name.lower().startswith("claude"))
    custom_wants_gemini = bool(custom_name and not custom_wants_claude)

    # Preveri, ali je izbran Claude model (in ima uporabnik vnesen Claude ključ)
    is_claude_choice = (
        personal_cld_key
        and not custom_wants_gemini and (
            choice in CLAUDE_MODEL_OPTIONS
            or "claude" in raw_choice.lower()
            or "claude" in choice.lower()
            or custom_wants_claude
        )
    )

    if is_claude_choice:
        target_label = raw_choice if "claude" in raw_choice.lower() else choice
        if custom_wants_claude:
            candidates = _dedup_models([custom_name, "claude-haiku-5-5-latest", "claude-haiku-5-5", "claude-haiku-4-5-latest", "claude-3-5-haiku-latest"])
            display_id = str(custom_name)
        elif "haiku" in target_label.lower():
            candidates = [
                "claude-haiku-5-5-latest",
                "claude-haiku-5-5",
                "claude-5-5-haiku-latest",
                "claude-haiku-4-5-latest",
                "claude-haiku-4-5",
                "claude-3-5-haiku-latest",
                "claude-3-5-haiku-20241022",
            ]
            display_id = "claude-haiku-5-5"
        elif "sonnet" in target_label.lower():
            candidates = [
                "claude-sonnet-4-6-latest",
                "claude-sonnet-4-6",
                "claude-sonnet-4-5-latest",
                "claude-sonnet-4-5",
                "claude-3-7-sonnet-latest",
                "claude-3-5-sonnet-latest",
            ]
            display_id = "claude-sonnet-4-6"
        elif "opus" in target_label.lower():
            candidates = [
                "claude-opus-4-6-latest",
                "claude-opus-4-6",
                "claude-opus-4-5-latest",
                "claude-opus-4-5",
                "claude-3-opus-latest",
            ]
            display_id = "claude-opus-4-6"
        elif custom_name:
            candidates = [
                custom_name,
                "claude-haiku-5-5-latest",
                "claude-haiku-5-5",
                "claude-haiku-4-5-latest",
                "claude-3-5-haiku-latest",
            ]
            display_id = custom_name if custom_name else "gemini-4-hydrogen"
        else:
            candidates = [
                "claude-haiku-5-5-latest",
                "claude-haiku-5-5",
                "claude-5-5-haiku-latest",
                "claude-haiku-4-5-latest",
                "claude-3-5-haiku-latest",
            ]
            display_id = "claude-haiku-5-5"
        return {
            "provider": "claude",
            "has_personal_key": True,
            "has_personal_gemini_key": bool(personal_gem_key),
            "has_personal_claude_key": True,
            "api_key": personal_cld_key,
            "selected_model_ui": choice,
            "display_model_id": display_id,
            "candidate_models": candidates,
            "available_options": avail_opts,
        }

    if personal_gem_key:
        if custom_wants_gemini:
            candidates = _dedup_models([custom_name, "gemini-2.5-flash", "gemini-3.8-flash"])
            display_id = str(custom_name)
        elif choice == "gemini-4-hydrogen":
            candidates = ["gemini-4-hydrogen", "gemini-2.5-flash", "gemini-3.8-flash"]
            display_id = "gemini-4-hydrogen"
        elif choice == "gemini-3.8-pro":
            candidates = ["gemini-3.8-pro", "gemini-3.1-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.8-flash"]
            display_id = "gemini-3.8-pro"
        elif choice == "gemini-2.5-pro" or "pro" in raw_choice.lower():
            candidates = ["gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-2.5-flash", "gemini-3.8-flash"]
            display_id = "gemini-2.5-pro"
        elif choice == "gemini-2.5-flash":
            candidates = ["gemini-2.5-flash", "gemini-3.8-flash"]
            display_id = "gemini-2.5-flash"
        elif choice == "gemini-3.8-flash":
            candidates = _dedup_models(["gemini-3.8-flash", "gemini-2.5-flash"])
            display_id = str("gemini-3.8-flash")
        elif choice == GEMINI_CUSTOM_MODEL_LABEL:
            candidates = _dedup_models([custom_name, "gemini-2.5-flash", "gemini-3.8-flash"]) if custom_name else ["gemini-4-hydrogen", "gemini-2.5-flash", "gemini-3.8-flash"]
            display_id = "gemini-4-hydrogen"




            display_id = custom_name if custom_name else "gemini-4-hydrogen"
        else:
            candidates = ["gemini-4-hydrogen", "gemini-2.5-flash", "gemini-3.8-flash"]
            display_id = "gemini-4-hydrogen"
        return {
            "provider": "gemini",
            "has_personal_key": True,
            "has_personal_gemini_key": True,
            "has_personal_claude_key": bool(personal_cld_key),
            "api_key": personal_gem_key,
            "selected_model_ui": choice,
            "display_model_id": display_id,
            "candidate_models": candidates,
            "available_options": avail_opts,
        }

    return {
        "provider": "gemini",
        "has_personal_key": bool(personal_cld_key),
        "has_personal_gemini_key": False,
        "has_personal_claude_key": bool(personal_cld_key),
        "api_key": master_gem_key,
        "selected_model_ui": MODEL_FLASH_DEFAULT_LABEL,
        "display_model_id": custom_name if custom_wants_gemini else "gemini-3.8-flash (Privzeti)",
        "candidate_models": _dedup_models([custom_name, "gemini-2.5-flash", "gemini-3.8-flash"]) if custom_wants_gemini else ["gemini-3.8-flash", "gemini-2.5-flash"],
        "available_options": avail_opts,
    }


def _call_claude_with_caching(
    user_prompt: str,
    api_key: str,
    candidate_models: list,
    system_instruction: str = None,
    chat_history: list = None,
    status_placeholder=None,
    thread_summary: str = "",
) -> tuple[bool, str, str]:
    """
    Pokliče Anthropic Claude API z vključenim Prompt Cachingom ('cache_control': {'type': 'ephemeral'})
    za velik sistemski prompt in Garminov podatkovni kontekst ter samodejnim 503/529/429 retry mehanizmom (do 4 poskusi).
    """
    try:
        import anthropic
    except ImportError:
        return (
            False,
            "ℹ️ Knjižnica `anthropic` ni nameščena (`pip install anthropic>=0.39.0`).",
            candidate_models[0] if candidate_models else "claude-haiku-5-5",
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Če user_prompt vsebuje velik podatkovni kontekst + '=== VPRAŠANJE / UKAZ ŠPORTNIKA ===',
    # premaknemo podatkovni kontekst v predpomnjen (ephemeral) system blok, da se pri naslednjih vprašanjih
    # v isti seji bere iz predpomnilnika (do 90 % nižji strošek!).
    cached_context_part = ""
    actual_user_question = user_prompt
    split_marker = "=== VPRAŠANJE / UKAZ ŠPORTNIKA ==="
    if split_marker in str(user_prompt):
        parts = str(user_prompt).split(split_marker, 1)
        cached_context_part = parts[0].strip()
        actual_user_question = parts[1].strip() or user_prompt

    system_blocks = []
    if system_instruction:
        system_blocks.append({
            "type": "text",
            "text": str(system_instruction),
            "cache_control": {"type": "ephemeral"},
        })
    if cached_context_part:
        system_blocks.append({
            "type": "text",
            "text": f"=== CELOTNI PODATKOVNI KONTEKST ŠPORTNIKA (GARMIN & DNEVNIK) ===\n{cached_context_part}",
            "cache_control": {"type": "ephemeral"},
        })

    # Sestavi izmenične user/assistant messages za Anthropic API (Rolling 4-message window + povzetek niti)
    raw_msgs = []
    trimmed_chat_history = trim_chat_history_for_api(chat_history, thread_summary=thread_summary)
    if trimmed_chat_history:
        for msg in trimmed_chat_history:
            r = "user" if msg.get("role") == "user" else "assistant"
            c = str(msg.get("content") or "").strip()
            if not c:
                continue
            if raw_msgs and raw_msgs[-1]["role"] == r:
                raw_msgs[-1]["content"] += f"\n\n{c}"
            else:
                raw_msgs.append({"role": r, "content": c})

    if raw_msgs and raw_msgs[-1]["role"] == "user":
        raw_msgs[-1]["content"] += f"\n\n{actual_user_question}"
    else:
        raw_msgs.append({"role": "user", "content": actual_user_question})

    # Zagotovi, da se messages začne z vlogo 'user'
    while raw_msgs and raw_msgs[0]["role"] != "user":
        raw_msgs.pop(0)
    if not raw_msgs:
        raw_msgs = [{"role": "user", "content": actual_user_question}]

    retry_delays = [2, 4, 7]
    last_err = None
    primary_model = candidate_models[0] if candidate_models else "claude-haiku-5-5"

    for attempt_idx in range(4):
        model_to_try = primary_model if attempt_idx < 2 else (candidate_models[min(attempt_idx - 1, len(candidate_models) - 1)])
        try:
            kwargs = {
                "model": model_to_try,
                "max_tokens": 4096,
                "messages": raw_msgs,
            }
            if system_blocks:
                kwargs["system"] = system_blocks
            resp = client.messages.create(**kwargs)
            text_parts = []
            for blk in getattr(resp, "content", []) or []:
                if hasattr(blk, "text") and blk.text:
                    text_parts.append(blk.text)
            out_text = "\n".join(text_parts).strip() or str(resp)
            try:
                cu = getattr(resp, "usage", None)
                in_t = int(getattr(cu, "input_tokens", 0) or 0)
                rd_c = int(getattr(cu, "cache_read_input_tokens", 0) or 0)
                ou_t = int(getattr(cu, "output_tokens", 0) or 0)
                p_all = in_t + rd_c + int(getattr(cu, "cache_creation_input_tokens", 0) or 0)
                if p_all <= 0:
                    p_all = estimate_context_tokens(user_prompt, chat_history, system_instruction, thread_summary=thread_summary)
                _set_last_token_usage({
                    "prompt_tokens": p_all,
                    "cached_tokens": rd_c,
                    "candidates_tokens": ou_t or max(1, int(len(out_text) / 3.5)),
                    "total_tokens": p_all + (ou_t or max(1, int(len(out_text) / 3.5))),
                })
            except Exception:
                pass
            if status_placeholder is not None:
                status_placeholder.empty()
            return True, out_text, model_to_try
        except Exception as e:
            err_str = str(e)
            last_err = err_str
            err_lower = err_str.lower()
            if "404" in err_lower or "not_found" in err_lower or "not found" in err_lower:
                # Če model (npr. -latest alias) ni najden, takoj preizkusi naslednjega kandidata
                for alt_m in candidate_models[1:]:
                    try:
                        kwargs["model"] = alt_m
                        resp = client.messages.create(**kwargs)
                        out_text = "\n".join([b.text for b in getattr(resp, "content", []) if hasattr(b, "text") and b.text]).strip()
                        if status_placeholder is not None:
                            status_placeholder.empty()
                        return True, out_text, alt_m
                    except Exception as e2:
                        last_err = str(e2)
                break
            is_overloaded = any(k in err_lower for k in ("503", "529", "overloaded", "unavailable", "429", "rate_limit", "rate limit"))
            if is_overloaded and attempt_idx < 3:
                wait_s = retry_delays[attempt_idx]
                msg_wait = f"⏳ Strežnik modela je trenutno zaseden (503). Samodejno poskušam znova (poskus {attempt_idx + 2}/4 čez {wait_s}s)..."
                if status_placeholder is not None:
                    status_placeholder.info(msg_wait)
                time.sleep(wait_s)
                continue
            break

    if status_placeholder is not None:
        status_placeholder.empty()
    return (
        False,
        f"❌ **Napaka pri klicu Claude API:** {last_err}",
        primary_model,
    )


def call_gemini_hybrid(
    user_prompt: str,
    user_cfg: dict,
    project_dir: str,
    system_instruction: str = None,
    chat_history: list = None,
    status_placeholder=None,
    thread_summary: str = "",
) -> tuple[bool, str, str]:
    """
    Pokliče izbrani AI model (Gemini ali Claude) z uporabo hibridne logike ključev,
    samodejnega 503/429 ponavljanja (do 4 poskusi z zamikom 2s, 4s, 7s) in fallbackom na Flash model.
    Vrne: (uspeh: bool, odgovor: str, uporabljen_model: str)
    """
    auth_info = resolve_gemini_auth(user_cfg, project_dir)

    # 1) ČE JE IZBRAN CLAUDE (ANTHROPIC) MODEL
    if auth_info.get("provider") == "claude":
        return _call_claude_with_caching(
            user_prompt=user_prompt,
            api_key=auth_info["api_key"],
            candidate_models=auth_info["candidate_models"],
            system_instruction=system_instruction,
            chat_history=chat_history,
            status_placeholder=status_placeholder,
            thread_summary=thread_summary,
        )

    # 2) GEMINI POT (S SAMODEJNIM 503 RETRY LOOPOM DO 4 POSKUSOV + FALLBACK NA FLASH)
    api_key = auth_info["api_key"]
    if not api_key:
        return (
            False,
            "⚠️ **GEMINI_API_KEY ni nastavljen.** Vnesi svoj osebni Gemini ali Claude API ključ v nastavitvah ali nastavi privzeti ključ v Streamlit Secrets / `.env.local`.",
            auth_info["display_model_id"],
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return (
            False,
            "ℹ️ Knjižnica `google-genai` ni nameščena v tem okolju (`pip install google-genai`).",
            auth_info["display_model_id"],
        )

    client = genai.Client(api_key=api_key)

    contents_payload = []
    trimmed_chat_history = trim_chat_history_for_api(chat_history, thread_summary=thread_summary)
    if trimmed_chat_history:
        for msg in trimmed_chat_history:
            role = "user" if msg.get("role") == "user" else "model"
            txt = str(msg.get("content") or "").strip()
            if txt:
                contents_payload.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=txt)])
                )
    contents_payload.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
    )

    cfg_kwargs = {}
    if system_instruction:
        cfg_kwargs["system_instruction"] = system_instruction
    gen_config = types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None

    candidates = list(auth_info.get("candidate_models") or ["gemini-4-hydrogen", "gemini-2.5-flash", "gemini-3.8-flash"])
    ordered_models = _dedup_models(candidates + ["gemini-2.5-flash", "gemini-3.8-flash"])
    requested_model = ordered_models[0]
    primary_model = requested_model
    fallback_flash_models = ["gemini-2.5-flash", "gemini-3.8-flash"]
    fallback_reason = None
    dead_models = set()
    retry_delays = [2, 4, 7]
    last_err = None

    for attempt_idx in range(4):
        # Po 2 neuspešnih 503 poskusih na primarnem modelu samodejno preklopi na hitri Gemini Flash fallback
        if (attempt_idx < 2 or fallback_reason is not None) and primary_model not in dead_models:
            model_name = primary_model
        else:
            avail_fb = [m for m in fallback_flash_models if m not in dead_models and m != primary_model]
            if not avail_fb:
                avail_fb = [m for m in candidates if m not in dead_models]
            model_name = avail_fb[0] if avail_fb else "gemini-3.8-flash"

        try:
            if gen_config:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=contents_payload,
                    config=gen_config,
                )
            else:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=contents_payload,
                )
            out_text = resp.text if hasattr(resp, "text") and resp.text else str(resp)
            try:
                um = getattr(resp, "usage_metadata", None)
                prompt_tokens = int(getattr(um, "prompt_token_count", 0) or 0)
                cached_tokens = int(getattr(um, "cached_content_token_count", 0) or 0)
                candidates_tokens = int(getattr(um, "candidates_token_count", 0) or 0)
                total_tokens = int(getattr(um, "total_token_count", 0) or 0)
                if prompt_tokens <= 0:
                    prompt_tokens = estimate_context_tokens(
                        user_prompt,
                        chat_history,
                        system_instruction,
                        thread_summary=thread_summary,
                    )
                if candidates_tokens <= 0:
                    candidates_tokens = max(1, int(len(out_text) / 3.5))
                if total_tokens <= 0:
                    total_tokens = prompt_tokens + candidates_tokens
                _set_last_token_usage({
                    "prompt_tokens": prompt_tokens,
                    "cached_tokens": cached_tokens,
                    "candidates_tokens": candidates_tokens,
                    "total_tokens": total_tokens,
                })
            except Exception:
                pass
            if fallback_reason and model_name != requested_model:
                _fb_msg = f"Model `{requested_model}` ni uspel ({fallback_reason}) — odgovor ustvarjen s stabilnim modelom `{model_name}`."
                try:
                    st.session_state["_pending_ai_fallback_toast"] = _fb_msg
                    st.toast(_fb_msg, icon="⚡")
                except Exception:
                    pass
            try:
                st.session_state["_last_gemini_model_used"] = model_name
            except Exception:
                pass
            shown_model = (
                auth_info["display_model_id"]
                if model_name in ("gemini-3.8-flash", "gemini-2.5-flash") and not auth_info["has_personal_gemini_key"] and not fallback_reason
                else model_name
            )
            if status_placeholder is not None:
                status_placeholder.empty()
            return True, out_text, shown_model
        except Exception as e:
            err_str = str(e)
            last_err = err_str
            err_lower = err_str.lower()

            # Če model vrne 404 Not Found ali no longer available, ga označi kot mrtvega in takoj preizkusi preostale kandidate
            is_404_err = ("404" in err_lower or "not_found" in err_lower or "not found" in err_lower or "no longer available" in err_lower or "is not supported" in err_lower)
            is_429_err = ("429" in err_lower or "resource_exhausted" in err_lower or "quota" in err_lower or "rate limit" in err_lower)
            if is_404_err or is_429_err:
                fallback_reason = "429 RESOURCE_EXHAUSTED" if is_429_err else "404 NOT_FOUND"
                dead_models.add(model_name)
                # Če Google v sporočilu 404 izrecno predlaga nov model (npr. models/gemini-3.8-flash), ga dodaj na vrh
                m_sug = re.search(r"models/([a-zA-Z0-9\.\-\_]+)", err_str)
                if m_sug:
                    sug_id = m_sug.group(1)
                    if sug_id not in dead_models and sug_id not in candidates:
                        candidates.insert(0, sug_id)
                for alt_cand in ordered_models:
                    if alt_cand not in dead_models:
                        primary_model = alt_cand
                        break
                continue

            is_retryable_503_429 = any(
                tok in err_lower
                for tok in ("503", "unavailable", "overloaded", "429", "resource_exhausted", "quota", "rate limit", "500", "internal")
            )
            if is_retryable_503_429 and attempt_idx < 3:
                wait_s = retry_delays[attempt_idx]
                next_model_hint = " (preklop na hitri Flash model)" if attempt_idx >= 1 else ""
                spinner_txt = f"⏳ Strežnik modela je trenutno zaseden (503). Samodejno poskušam znova (poskus {attempt_idx + 2}/4{next_model_hint})..."
                if status_placeholder is not None:
                    status_placeholder.warning(spinner_txt)
                time.sleep(wait_s)
                continue
            break

    if status_placeholder is not None:
        status_placeholder.empty()
    return (
        False,
        f"⏳ **Strežnik izbranega AI modela je trenutno preobremenjen (503 / 429) po 4 samodejnih poskusih.**\nPodrobnosti: `{last_err}`\nKlikni spodnji gumb **🔄 Poskusi znova** ali v nastavitvah vklopi **Claude Haiku 5.5 / Gemini 3.8 Flash**.",
        auth_info["display_model_id"],
    )


# ==============================================================================
# 2. SHRANJEVANJE KOLEDARJA IN VEČNITNE ZGODOVINE POGOVOROV (MULTI-THREAD JSON)
# ==============================================================================
TOPIC_TAGS = [
    "Jutranji pregled",
    "Trening",
    "Planiranje",
    "Ocena forme",
    "Prehrana & Regeneracija",
    "Splošno"
]


def summarize_message_short(content: str, max_len: int = 90) -> str:
    """Izlušči prvi smiselni stavek ali ključno odločitev sporočila za zložen prikaz (collapsible history)."""
    if not content:
        return "Brez vsebine"
    lines = [ln.strip() for ln in str(content).splitlines() if ln.strip()]
    for ln in lines:
        clean = re.sub(r"^[#*\->\d\.\)\s]+", "", ln)
        clean = re.sub(r"\*\*|__|`|~~", "", clean).strip()
        if len(clean) >= 6:
            first_sent = re.split(r"(?<=[\.\!\?])\s+", clean)[0]
            cand = first_sent if 12 <= len(first_sent) <= max_len else clean
            return (cand[:max_len].rstrip() + "…") if len(cand) > max_len else cand
    flat = re.sub(r"\s+", " ", str(content)).strip()
    return (flat[:max_len].rstrip() + "…") if len(flat) > max_len else flat


def _make_thread_id() -> str:
    """Ustvari unikatni identifikator pogovorne niti v obliki thread_<timestamp>."""
    ts = int(time.time() * 1000)
    return f"thread_{ts}_{uuid.uuid4().hex[:4]}"


def create_new_thread_dict(
    title: str = None,
    topic_tag: str = "Splošno",
    thread_id: str = None,
) -> dict:
    """Ustvari nov slovar pogovorne niti po specifikaciji."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    now_disp = datetime.now().strftime("%d. %m. %Y")
    tid = thread_id or _make_thread_id()
    t_title = title or f"💬 Nov pogovor ({now_disp})"
    return {
        "id": tid,
        "title": t_title,
        "topic_tag": topic_tag if topic_tag in TOPIC_TAGS else "Splošno",
        "created_at": now_str,
        "updated_at": now_str,
        "summary_preview": "Nov pogovor z AI osebnim trenerjem...",
        "thread_summary": "",
        "messages": [],
    }


def load_ai_chat_store(user_dir: str) -> dict:
    """
    Naloži večnitno strukturo pogovorov iz data/users/<user>/ai_chat_history.json.
    Samodejno pretvori staro enonivojsko listo sporočil v novo večnitno strukturo!
    """
    user_dir = str(user_dir or DATA_DIR)
    path = os.path.join(user_dir, "ai_chat_history.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "threads" in data and isinstance(data["threads"], dict):
                threads = data["threads"]
                active_id = data.get("active_thread_id")
                for tid, th in threads.items():
                    th.setdefault("id", tid)
                    th.setdefault("title", f"💬 Pogovor ({tid[-6:]})")
                    th.setdefault("topic_tag", "Splošno")
                    th.setdefault("created_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
                    th.setdefault("updated_at", th["created_at"])
                    msgs = th.setdefault("messages", [])
                    for m in msgs:
                        if not m.get("id"):
                            m["id"] = uuid.uuid4().hex[:8]
                        if not m.get("short_summary"):
                            m["short_summary"] = summarize_message_short(m.get("content", ""))
                    th.setdefault("thread_summary", "")
                    if len(msgs) > 4 and not th.get("thread_summary"):
                        th["thread_summary"] = build_compact_thread_summary(msgs[:-4], max_words=180)
                    elif len(msgs) <= 4 and th.get("thread_summary"):
                        th["thread_summary"] = ""
                    if not th.get("summary_preview"):
                        th["summary_preview"] = (
                            msgs[-1]["short_summary"] if msgs else "Nov pogovor z AI osebnim trenerjem..."
                        )
                if not threads:
                    new_th = create_new_thread_dict()
                    threads[new_th["id"]] = new_th
                    active_id = new_th["id"]
                elif not active_id or active_id not in threads:
                    sorted_ids = sorted(threads.keys(), key=lambda k: threads[k].get("updated_at", ""), reverse=True)
                    active_id = sorted_ids[0]
                return {"active_thread_id": active_id, "threads": threads}

            elif isinstance(data, list):
                migrated_th = create_new_thread_dict(
                    title=f"💬 Prvi pogovor ({datetime.now().strftime('%d. %m. %Y')})",
                    topic_tag="Splošno",
                    thread_id="thread_migrated_01",
                )
                for m in data:
                    if isinstance(m, dict):
                        m_copy = dict(m)
                        m_copy.setdefault("id", uuid.uuid4().hex[:8])
                        m_copy["short_summary"] = summarize_message_short(m_copy.get("content", ""))
                        migrated_th["messages"].append(m_copy)
                if migrated_th["messages"]:
                    migrated_th["summary_preview"] = migrated_th["messages"][-1]["short_summary"]
                store = {
                    "active_thread_id": migrated_th["id"],
                    "threads": {migrated_th["id"]: migrated_th},
                }
                save_ai_chat_store(user_dir, store)
                return store
        except Exception:
            pass

    init_th = create_new_thread_dict()
    return {
        "active_thread_id": init_th["id"],
        "threads": {init_th["id"]: init_th},
    }


def save_ai_chat_store(user_dir: str, store: dict) -> bool:
    """Shrani večnitni slovar pogovorov v data/users/<user>/ai_chat_history.json."""
    user_dir = str(user_dir or DATA_DIR)
    path = os.path.join(user_dir, "ai_chat_history.json")
    try:
        os.makedirs(user_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
        _trigger_cloud_sync_for_dir(user_dir, "ai_chat_history.json")
        return True
    except Exception:
        return False


def append_exchange_to_thread(
    user_dir: str,
    store: dict,
    thread_id: str,
    user_content: str,
    assistant_content: str,
    used_model: str = "gemini-3.8-flash",
    calendar_updates: list = None,
    auto_applied: int = 0,
    auto_title_from_user: bool = False,
    token_usage: dict = None,
) -> dict:
    """Doda uporabniško vprašanje in odgovor trenerja v izbrano pogovorno nit ter posodobi summary_preview in thread_summary."""
    threads = store.setdefault("threads", {})
    if thread_id not in threads:
        threads[thread_id] = create_new_thread_dict(thread_id=thread_id)
    th = threads[thread_id]
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M")

    user_short = summarize_message_short(user_content, 50)
    asst_short = summarize_message_short(assistant_content, 85)

    if auto_title_from_user and len(th.get("messages", [])) == 0:
        clean_q = summarize_message_short(user_content, 42)
        th["title"] = f"💬 {clean_q} ({datetime.now().strftime('%d. %m.')})"

    usage_rec = token_usage if isinstance(token_usage, dict) and token_usage else get_last_token_usage()

    th.setdefault("messages", []).append({
        "id": uuid.uuid4().hex[:8],
        "role": "user",
        "content": user_content,
        "timestamp": now_iso,
        "short_summary": user_short,
    })
    th["messages"].append({
        "id": uuid.uuid4().hex[:8],
        "role": "assistant",
        "content": assistant_content,
        "timestamp": now_iso,
        "short_summary": asst_short,
        "model": used_model,
        "calendar_updates": calendar_updates or [],
        "auto_applied": auto_applied,
        "token_usage": usage_rec or {},
    })
    msgs_now = th["messages"]
    if len(msgs_now) > 4:
        th["thread_summary"] = build_compact_thread_summary(
            msgs_now[:-4],
            existing_summary=th.get("thread_summary", ""),
            max_words=180,
        )
    else:
        th["thread_summary"] = ""
    th["updated_at"] = now_iso
    th["summary_preview"] = asst_short
    store["active_thread_id"] = thread_id
    save_ai_chat_store(user_dir, store)
    return th


def delete_exchange_from_thread(user_dir: str, store: dict, thread_id: str, exchange_idx: int) -> bool:
    """
    Izbriše izbrani par vprašanje/odgovor (ali posamično sporočilo / napako) iz thread["messages"]
    in shrani posodobljeno zgodovino v ai_chat_history.json.
    Ker kontekst in spomin bereta iz thread["messages"], izbrisano sporočilo ne vpliva več na prihodnje odgovore!
    """
    th = store.get("threads", {}).get(thread_id)
    if not th:
        return False
    msgs = th.get("messages", [])
    exchanges = group_messages_into_exchanges(msgs)
    if not (0 <= exchange_idx < len(exchanges)):
        return False
    target_ex = exchanges[exchange_idx]
    ids_to_remove = set()
    for role_k in ("user", "assistant"):
        m_obj = target_ex.get(role_k)
        if isinstance(m_obj, dict) and m_obj.get("id"):
            ids_to_remove.add(m_obj["id"])

    if ids_to_remove:
        new_msgs = [m for m in msgs if m.get("id") not in ids_to_remove]
    else:
        # Rezervno brisanje po referenci objekta, če sporočilo nima id-ja
        objs_to_remove = [target_ex.get("user"), target_ex.get("assistant")]
        new_msgs = [m for m in msgs if m not in objs_to_remove]

    th["messages"] = new_msgs
    if len(new_msgs) > 4:
        th["thread_summary"] = build_compact_thread_summary(new_msgs[:-4], max_words=180)
    else:
        th["thread_summary"] = ""
    th["summary_preview"] = (
        summarize_message_short(new_msgs[-1].get("content", ""), 85)
        if new_msgs
        else "Nov pogovor z AI osebnim trenerjem..."
    )
    save_ai_chat_store(user_dir, store)
    return True


def replace_exchange_assistant_response(
    user_dir: str,
    store: dict,
    thread_id: str,
    exchange_idx: int,
    new_assistant_content: str,
    used_model: str,
    calendar_updates: list = None,
    auto_applied: int = 0,
    token_usage: dict = None,
) -> bool:
    """
    Nadomesti stari/neuspešni odgovor asistenta v izbrani izmenjavi (Retry / Ponovi vprašanje)
    z novim generiranim odgovorom in shrani v ai_chat_history.json.
    """
    th = store.get("threads", {}).get(thread_id)
    if not th:
        return False
    msgs = th.get("messages", [])
    exchanges = group_messages_into_exchanges(msgs)
    if not (0 <= exchange_idx < len(exchanges)):
        return False
    target_ex = exchanges[exchange_idx]
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M")
    asst_short = summarize_message_short(new_assistant_content, 85)
    usage_rec = token_usage if isinstance(token_usage, dict) and token_usage else get_last_token_usage()

    old_a = target_ex.get("assistant")
    old_u = target_ex.get("user")
    if isinstance(old_a, dict):
        old_a["content"] = new_assistant_content
        old_a["timestamp"] = now_iso
        old_a["short_summary"] = asst_short
        old_a["model"] = used_model
        old_a["calendar_updates"] = calendar_updates or []
        old_a["auto_applied"] = auto_applied
        old_a["token_usage"] = usage_rec or {}
    elif isinstance(old_u, dict):
        # Če odgovora še ni bilo, ga vstavimo takoj za uporabniškim vprašanjem
        try:
            u_pos = msgs.index(old_u)
        except ValueError:
            u_pos = len(msgs) - 1
        msgs.insert(u_pos + 1, {
            "id": uuid.uuid4().hex[:8],
            "role": "assistant",
            "content": new_assistant_content,
            "timestamp": now_iso,
            "short_summary": asst_short,
            "model": used_model,
            "calendar_updates": calendar_updates or [],
            "auto_applied": auto_applied,
            "token_usage": usage_rec or {},
        })
    if len(msgs) > 4:
        th["thread_summary"] = build_compact_thread_summary(msgs[:-4], max_words=180)
    else:
        th["thread_summary"] = ""
    th["updated_at"] = now_iso
    th["summary_preview"] = asst_short
    save_ai_chat_store(user_dir, store)
    return True


def group_messages_into_exchanges(messages: list) -> list:
    """
    Združi sporočila v pare (user_msg, assistant_msg) za pametno zlaganje zgodovine:
    zadnje 3 izmenjave (odgovori) so razširjene, starejše pa zložene z 1-vrstičnim povzetkom:
    🕒 [Datum in ura] | 💬 Vprašanje: <prvih 50 znakov> ➔ 🤖 Povzetek: <prvi stavek / ključna odločitev>
    """
    exchanges = []
    i = 0
    n = len(messages or [])
    while i < n:
        m = messages[i]
        if m.get("role") == "user":
            u_msg = m
            a_msg = None
            if i + 1 < n and messages[i + 1].get("role") != "user":
                a_msg = messages[i + 1]
                i += 2
            else:
                i += 1
            exchanges.append({"user": u_msg, "assistant": a_msg})
        else:
            exchanges.append({"user": None, "assistant": m})
            i += 1
    return exchanges


def format_collapsed_exchange_title(ex: dict) -> str:
    """
    Oblikuje naslov zloženega st.expander bloka natanko po specifikaciji:
    🕒 [Datum in ura] | 💬 Vprašanje: <prvih 50 znakov vprašanja> ➔ 🤖 Povzetek: <prvi stavek / ključna odločitev odgovora>
    """
    u_msg = ex.get("user") or {}
    a_msg = ex.get("assistant") or {}
    ts = a_msg.get("timestamp") or u_msg.get("timestamp") or ""
    q_raw = str(u_msg.get("content") or "").strip()
    q_clean = re.sub(r"\s+", " ", re.sub(r"[#*\->`]", "", q_raw)).strip()
    q_50 = (q_clean[:50].rstrip() + "…") if len(q_clean) > 50 else (q_clean or "Hitri ukaz")
    a_sum = a_msg.get("short_summary") or summarize_message_short(a_msg.get("content", ""), 75)
    return f"🕒 [{ts}] | 💬 Vprašanje: {q_50} ➔ 🤖 Povzetek: {a_sum}"


# Ohranimo pomožni funkciji za kompatibilnost
def load_ai_chat_history(user_dir: str) -> list:
    store = load_ai_chat_store(user_dir)
    aid = store.get("active_thread_id")
    return store.get("threads", {}).get(aid, {}).get("messages", [])


def save_ai_chat_history(user_dir: str, messages: list) -> bool:
    store = load_ai_chat_store(user_dir)
    aid = store.get("active_thread_id")
    if aid and aid in store.get("threads", {}):
        store["threads"][aid]["messages"] = messages
        return save_ai_chat_store(user_dir, store)
    return False


def load_training_calendar(user_dir: str) -> list:
    """Naloži seznam planiranih treningov iz data/users/<user>/training_calendar.json."""
    user_dir = str(user_dir or DATA_DIR)
    path = os.path.join(user_dir, "training_calendar.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and isinstance(data.get("workouts"), list):
                    return data["workouts"]
        except Exception:
            pass
    return []


def save_training_calendar(user_dir: str, workouts: list) -> bool:
    """Shrani seznam planiranih treningov v data/users/<user>/training_calendar.json."""
    user_dir = str(user_dir or DATA_DIR)
    path = os.path.join(user_dir, "training_calendar.json")
    try:
        os.makedirs(user_dir, exist_ok=True)
        workouts_sorted = sorted(workouts, key=lambda x: (str(x.get("date", "")), str(x.get("title", ""))))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(workouts_sorted, f, indent=2, ensure_ascii=False)
        _trigger_cloud_sync_for_dir(user_dir, "training_calendar.json")
        return True
    except Exception:
        return False


def upsert_calendar_workouts(user_dir: str, new_items: list, replace_same_date: bool = True) -> int:
    """Vpiše ali posodobi treninge v training_calendar.json."""
    if not new_items:
        return 0
    existing = load_training_calendar(user_dir)
    count = 0
    for item in new_items:
        if not isinstance(item, dict):
            continue
        d_str = str(item.get("date") or "").strip()[:10]
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", d_str):
            continue
        entry = {
            "id": str(item.get("id") or uuid.uuid4().hex[:10]),
            "date": d_str,
            "sport": str(item.get("sport") or "Kolesarjenje").strip(),
            "title": str(item.get("title") or "Trening").strip(),
            "type": str(item.get("type") or "Z2 Aerobno").strip(),
            "duration_min": int(float(item.get("duration_min") or 0)),
            "target_power_w": str(item.get("target_power_w") or item.get("target_intensity") or "").strip(),
            "target_tss": int(float(item.get("target_tss") or 0)),
            "description": str(item.get("description") or "").strip(),
            "source": str(item.get("source") or "ai"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        if replace_same_date:
            existing = [w for w in existing if str(w.get("date", ""))[:10] != d_str]
        existing.append(entry)
        count += 1
    save_training_calendar(user_dir, existing)
    return count


def delete_calendar_workout(user_dir: str, workout_id: str = None, date_str: str = None) -> bool:
    """Izbriše planiran trening po ID-ju ali datumu."""
    existing = load_training_calendar(user_dir)
    if workout_id:
        filtered = [w for w in existing if str(w.get("id")) != str(workout_id)]
    elif date_str:
        filtered = [w for w in existing if str(w.get("date", ""))[:10] != str(date_str)[:10]]
    else:
        return False
    return save_training_calendar(user_dir, filtered)


def extract_calendar_updates_from_text(text: str) -> tuple[str, list]:
    """Iz odgovora AI trenerja izlušči JSON blok znotraj <calendar_update>...</calendar_update>."""
    if not text:
        return "", []
    pattern = re.compile(r"<calendar_update>\s*(.*?)\s*</calendar_update>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    workouts = []
    for raw_block in matches:
        clean_json = raw_block.strip()
        clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json, flags=re.IGNORECASE)
        clean_json = re.sub(r"\s*```$", "", clean_json)
        try:
            parsed = json.loads(clean_json)
            if isinstance(parsed, list):
                workouts.extend([x for x in parsed if isinstance(x, dict)])
            elif isinstance(parsed, dict):
                workouts.append(parsed)
        except Exception:
            pass
    cleaned_text = pattern.sub("", text).strip()
    return cleaned_text, workouts


# ==============================================================================
# 3. "REST OR TRAIN" READINESS ENGINE (0–100 % & SEMAFOR)
# ==============================================================================
def compute_activity_tss(a: dict, user_cfg: dict, compute_run_load_fn=None, get_sport_cat_fn=None) -> float:
    """Izračuna ali vrne TSS/obremenitev za posamezno aktivnost."""
    tss_val = a.get("trainingStressScore") or a.get("activityTrainingLoad")
    if tss_val and float(tss_val) > 0:
        return float(tss_val)
    m_sec = float(a.get("movingDuration") or a.get("duration") or 0)
    np_val = float(a.get("normPower") or a.get("avgPower") or 0)
    ftp = float(user_cfg.get("ftp_outdoor") or 300)
    if np_val > 0 and ftp > 0:
        if_val = np_val / ftp
        return round((m_sec * np_val * if_val) / (ftp * 36.0), 1)
    if get_sport_cat_fn and compute_run_load_fn:
        sp = get_sport_cat_fn(a.get("activityType", {}).get("typeKey"), a.get("activityName", ""))
        if sp == "Tek":
            rl = compute_run_load_fn(a, None, user_cfg)
            if rl and rl.get("rtss"):
                return float(rl["rtss"])
    avg_hr = float(a.get("averageHR") or 0)
    max_hr = float(user_cfg.get("max_hr") or 195)
    if m_sec > 0 and avg_hr > 0 and max_hr > 0:
        hr_ratio = min(1.05, max(0.4, avg_hr / (max_hr * 0.88)))
        return round((m_sec / 3600.0) * (hr_ratio ** 2) * 75.0, 1)
    return round((m_sec / 3600.0) * 45.0, 1)


def _parse_wake_minutes_from_bedtime_str(bedtime_str: str):
    """Iz niza 'HH:MM – HH:MM' ali 'HH:MM ➔ HH:MM' izlušči jutranjo uro bujenja v minutah od polnoči (npr. '06:07' -> 367)."""
    if not bedtime_str:
        return None, None
    try:
        matches = re.findall(r"(\d{1,2}):(\d{2})", str(bedtime_str))
        if len(matches) < 2:
            return None, None
        hh, mm = int(matches[1][0]), int(matches[1][1])
        if 2 <= hh <= 14:
            return hh * 60 + mm, f"{hh:02d}:{mm:02d}"
    except Exception:
        pass
    return None, None


def _fmt_minutes_to_hhmm(mins_val: float) -> str:
    if mins_val is None or np.isnan(mins_val):
        return "--:--"
    m_int = int(round(float(mins_val))) % 1440
    return f"{m_int // 60:02d}:{m_int % 60:02d}"


def compute_rest_or_train_readiness(
    df_daily_all: pd.DataFrame,
    activities_all: list,
    user_cfg: dict,
    user_journal: dict,
    ref_date_str: str = None,
    compute_run_load_fn=None,
    get_sport_cat_fn=None,
) -> dict:
    """
    Izračuna celovito oceno pripravljenosti 'Rest or Train' (0–100 %) na podlagi 5 stebrov
    z napredno fiziološko interpretacijo (zgodnje bujenje / budilka vs. dejanska utrujenost,
    povprečni nočni HR, globoki spanec, včerajšnji počitek in EWMA ACWR s kroničnim dnom 40 TSS/dan).
    """
    if df_daily_all is None or df_daily_all.empty:
        return {
            "score": 70,
            "status_code": "MODULATE",
            "badge": "🟡 MODULATE / Z2 BAZA (70/100)",
            "headline": "Priporočena aerobna Z2 vožnja ali prilagojen trening (manjkajo zadnji podatki spanca)",
            "color": "#f59e0b",
            "bg_color": "rgba(245, 158, 11, 0.12)",
            "reasons": ["Ni zabeleženih dnevnih zdravstvenih metrik za izračun odstopanj."],
            "metrics": {},
        }

    df = df_daily_all.sort_values("date").reset_index(drop=True)
    if not ref_date_str or ref_date_str not in df["date"].values:
        valid_rows = df[df["sleep_hours"].notna() | df["hrv_night"].notna() | df["rhr"].notna()]
        target_date = valid_rows.iloc[-1]["date"] if not valid_rows.empty else df.iloc[-1]["date"]
    else:
        target_date = ref_date_str

    idx_list = df.index[df["date"] == target_date].tolist()
    t_idx = idx_list[-1] if idx_list else len(df) - 1
    row_today = df.iloc[t_idx].to_dict()

    df_7d = df.iloc[max(0, t_idx - 6):t_idx + 1]
    df_14d_prev = df.iloc[max(0, t_idx - 14):t_idx]
    df_30d = df.iloc[max(0, t_idx - 29):t_idx + 1]

    reasons = []

    # 0. ANALIZA URE BUJENJA, NOČNEGA SRČNEGA UTRIPA IN VČERAJŠNJE OBREMENITVE
    wake_min_today, wake_str_today = _parse_wake_minutes_from_bedtime_str(row_today.get("bedtime_str"))
    prev_wake_mins = []
    for _, r_prev in (df_14d_prev if not df_14d_prev.empty else df_7d).iterrows():
        wm, _ = _parse_wake_minutes_from_bedtime_str(r_prev.get("bedtime_str"))
        if wm is not None:
            prev_wake_mins.append(wm)
    avg_wake_14d_min = float(np.mean(prev_wake_mins)) if prev_wake_mins else 495.0  # privzeto ~08:15
    avg_wake_14d_str = _fmt_minutes_to_hhmm(avg_wake_14d_min)
    wake_time_diff_min = int(round(wake_min_today - avg_wake_14d_min)) if wake_min_today is not None else 0

    sleep_hr_today = row_today.get("avg_sleep_hr")
    shr_7d_series = df_7d["avg_sleep_hr"].dropna()
    shr_30d_series = df_30d["avg_sleep_hr"].dropna()
    avg_overnight_hr_7d = float(shr_7d_series.mean()) if not shr_7d_series.empty else (float(shr_30d_series.mean()) if not shr_30d_series.empty else None)
    avg_overnight_hr_diff = round(float(sleep_hr_today) - avg_overnight_hr_7d, 1) if (pd.notna(sleep_hr_today) and avg_overnight_hr_7d is not None) else 0.0

    # Včerajšnji TSS & EWMA ACWR (7d EWMA / 28d EWMA s kroničnim dnom 40 TSS/dan)
    ref_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
    daily_tss = {}
    yesterday_kj = 0.0
    for a in (activities_all or []):
        d_s = str(a.get("startTimeLocal") or "")[:10]
        if not d_s:
            continue
        t_v = compute_activity_tss(a, user_cfg, compute_run_load_fn, get_sport_cat_fn)
        daily_tss[d_s] = daily_tss.get(d_s, 0.0) + t_v
        if d_s == (ref_dt - timedelta(days=1)).strftime("%Y-%m-%d"):
            avg_p = float(a.get("avgPower") or a.get("averagePower") or 0)
            m_s = float(a.get("movingDuration") or a.get("duration") or 0)
            yesterday_kj += (avg_p * m_s / 1000.0) if avg_p > 0 else 0.0

    tss_yesterday = daily_tss.get((ref_dt - timedelta(days=1)).strftime("%Y-%m-%d"), 0.0)
    tss_7d_sum = sum(daily_tss.get((ref_dt - timedelta(days=i)).strftime("%Y-%m-%d"), 0.0) for i in range(1, 8))
    tss_28d_sum = sum(daily_tss.get((ref_dt - timedelta(days=i)).strftime("%Y-%m-%d"), 0.0) for i in range(1, 29))

    # Standardni 7d EWMA / 28d EWMA z minimalnim kroničnim dnom 40 TSS/dan
    ewma_window_days = 56
    tss_series_56 = [daily_tss.get((ref_dt - timedelta(days=i)).strftime("%Y-%m-%d"), 0.0) for i in range(ewma_window_days, 0, -1)]
    s_tss = pd.Series(tss_series_56, dtype=float)
    acute_ewma = float(s_tss.ewm(span=7, min_periods=1).mean().iloc[-1])
    chronic_ewma = max(40.0, float(s_tss.ewm(span=28, min_periods=1).mean().iloc[-1]))
    acwr = round(acute_ewma / chronic_ewma, 2)

    deep_m = float(row_today.get("deep_min") or 0)
    rem_m = float(row_today.get("rem_min") or 0)
    sleep_h = row_today.get("sleep_hours")
    sleep_sc = row_today.get("sleep_score")

    # Zaznava fiziološkega artefakta zgodnje budilke (Rule 1):
    # Zgodnje bujenje (wake_time_diff_min <= -45 min ALI rem_m < 50 ob nizkem nočnem HR <= 7d povprečja + 0.5 bpm,
    # solidnem globokem spancu >= 70 min in včerajšnjem počitku / lahkem dnevu <= 45 TSS).
    overnight_hr_low_and_stable = (pd.notna(sleep_hr_today) and avg_overnight_hr_diff <= 0.6)
    yesterday_was_rest_or_easy = (tss_yesterday <= 45.0)
    early_wake_artifact = bool(
        overnight_hr_low_and_stable
        and deep_m >= 70.0
        and yesterday_was_rest_or_easy
        and (wake_time_diff_min <= -40 or (rem_m > 0 and rem_m <= 50 and (pd.isna(sleep_h) or float(sleep_h) <= 7.0)))
    )

    # 1. HRV STEBER (0–25 točk)
    hrv_today = row_today.get("hrv_night")
    hrv_7d_series = df_7d["hrv_night"].dropna()
    hrv_30d_series = df_30d["hrv_night"].dropna()
    hrv_7d_mean = float(hrv_7d_series.mean()) if not hrv_7d_series.empty else None
    hrv_30d_mean = float(hrv_30d_series.mean()) if not hrv_30d_series.empty else None
    hrv_30d_std = float(hrv_30d_series.std()) if len(hrv_30d_series) >= 4 else 6.0
    if not hrv_30d_std or np.isnan(hrv_30d_std) or hrv_30d_std < 3.0:
        hrv_30d_std = 5.0

    hrv_z = 0.0
    if pd.notna(hrv_today) and hrv_30d_mean:
        hrv_z = (float(hrv_today) - hrv_30d_mean) / hrv_30d_std
        if early_wake_artifact and hrv_z < 0:
            # Ne kaznuj HRV padca ostro, če je posledica zgodnje budilke ob nizkem nočnem utripu!
            hrv_pts = float(np.clip(20.5 + hrv_z * 2.5, 15.5, 25.0))
            wake_info_txt = f"bujenje ob `{wake_str_today}` vs. 14d povp. `{avg_wake_14d_str}` ({wake_time_diff_min:+d} min)" if wake_str_today else "zgodnje prebujanje z budilko"
            reasons.append(
                f"⏰ **Fiziološki artefakt zgodnje budilke (ni sistemska utrujenost!):** Nočni HRV (`{int(hrv_today)} ms` vs. 7d `{hrv_7d_mean:.0f} ms`) in nižji REM (`{int(rem_m)} min`) "
                f"sta posledica zgodnjega bujenja ({wake_info_txt}), ki je odrezalo zadnji jutranji cikel spanca (kjer sta REM in parasimpatični HRV najvišja) ter jutranjega kortizolskega odziva. "
                f"**Povprečni nočni HR je izjemno miren (`{int(sleep_hr_today)} bpm`, {avg_overnight_hr_diff:+.1f} bpm proti 7d povp. `{avg_overnight_hr_7d:.1f} bpm`), globoki spanec odličen (`{int(deep_m)} min`), včeraj pa je bil dan počitka (`{int(tss_yesterday)} TSS`)** — polna aerobna Z2 vožnja ostane nespremenjena!"
            )
        else:
            hrv_pts = float(np.clip(19.0 + hrv_z * 6.5, 4.0, 25.0))
            if hrv_z >= 0.4:
                reasons.append(f"✅ **HRV nad izhodiščem:** Nočni HRV ({int(hrv_today)} ms) je nad 30-dnevnim povprečjem ({hrv_30d_mean:.0f} ms, Z={hrv_z:+.2f}).")
            elif hrv_z <= -0.8:
                reasons.append(f"⚠️ **Znižan HRV:** Nočni HRV ({int(hrv_today)} ms) je pod 30-dnevnim povprečjem ({hrv_30d_mean:.0f} ms, Z={hrv_z:+.2f}) – znak nepopolne parasimpatične regeneracije.")
            else:
                reasons.append(f"ℹ️ **Stabilen HRV:** Nočni HRV ({int(hrv_today)} ms) je v območju 30d normale ({hrv_30d_mean:.0f} ms).")
    else:
        hrv_pts = 17.5

    # 2. RHR & NOČNI HR STEBER (0–15 točk)
    rhr_today = row_today.get("rhr")
    rhr_30d = df_30d["rhr"].dropna()
    shr_30d = df_30d["avg_sleep_hr"].dropna()
    rhr_base = float(rhr_30d.mean()) if not rhr_30d.empty else float(user_cfg.get("rhr_base") or 48)
    shr_base = float(shr_30d.mean()) if not shr_30d.empty else rhr_base

    rhr_delta = (float(rhr_today) - rhr_base) if pd.notna(rhr_today) else 0.0
    shr_delta = (float(sleep_hr_today) - shr_base) if pd.notna(sleep_hr_today) else 0.0
    eff_hr_delta = shr_delta if (early_wake_artifact and pd.notna(sleep_hr_today)) else max(rhr_delta, shr_delta)

    hr_pts = float(np.clip(13.0 - eff_hr_delta * 1.6, 2.0, 15.0))
    if pd.notna(sleep_hr_today) and avg_overnight_hr_diff <= 0.0:
        reasons.append(f"✅ **Miren nočni srčni utrip:** Povprečni nočni HR `{int(sleep_hr_today)} bpm` ({avg_overnight_hr_diff:+.1f} bpm glede na 7d povprečje `{avg_overnight_hr_7d:.1f} bpm`) potrjuje popolno kardiovaskularno spočitost.")
    elif pd.notna(rhr_today):
        if eff_hr_delta >= 3.5:
            reasons.append(f"⚠️ **Povišan srčni utrip v mirovanju:** RHR/nočni HR odstopa za `+{eff_hr_delta:.1f} bpm` nad 30d bazo ({rhr_base:.0f} bpm).")
        elif eff_hr_delta <= -1.5:
            reasons.append(f"✅ **Nizek jutranji RHR:** `{int(rhr_today)} bpm` ({rhr_delta:+.1f} bpm glede na 30d povprečje).")

    # 3. SPANEC & ARHITEKTURA (0–25 točk)
    if pd.notna(sleep_h) or pd.notna(sleep_sc):
        sh_val = float(sleep_h) if pd.notna(sleep_h) else 7.2
        sc_val = float(sleep_sc) if pd.notna(sleep_sc) else 75.0
        dur_pts = float(np.clip((sh_val / 8.0) * 11.0, 2.0, 11.0))
        score_pts = float(np.clip((sc_val / 90.0) * 10.0, 2.0, 10.0))
        arch_pts = float(np.clip(((deep_m * 1.35 + rem_m) / 180.0) * 4.0, 1.0, 4.0))
        sleep_pts = float(np.clip(dur_pts + score_pts + arch_pts + (2.5 if early_wake_artifact else 0.0), 4.0, 25.0))
        if early_wake_artifact:
            reasons.append(f"✅ **Močna fizična obnova (Globoki spanec {int(deep_m)} min):** Kljub krajšemu trajanju ({sh_val:.1f} h zaradi zgodnje budilke) je bila faza globokega spanca vrhunska.")
        elif sh_val < 6.3 or sc_val < 65:
            reasons.append(f"⚠️ **Okrnjen spanec:** {sh_val:.1f} h (Garmin ocena {int(sc_val)}/100) – priporočena previdnost pri visokointenzivnih intervalih.")
        elif sh_val >= 7.5 and sc_val >= 80:
            reasons.append(f"✅ **Odlična nočna regeneracija:** {sh_val:.1f} h spanca (ocena {int(sc_val)}/100, Globoki: {int(deep_m)} min, REM: {int(rem_m)} min).")
    else:
        sleep_pts = 17.5

    # 4. KUMULATIVNA UTRUJENOST: EWMA ACWR (7d vs 28d s kroničnim dnom 40 TSS/dan) & VČERAJŠNJI TRENING (0–20 točk)
    load_pts = 16.5
    if acwr > 1.45:
        load_pts -= min(9.0, (acwr - 1.30) * 14.0)
        reasons.append(f"⚠️ **Povišana akutna obremenitev (EWMA ACWR {acwr:.2f}):** 7-dnevni EWMA ({acute_ewma:.0f} TSS/dan) presega kronično bazo ({chronic_ewma:.0f} TSS/dan).")
    elif 0.75 <= acwr <= 1.30:
        load_pts += 2.5
    if tss_yesterday == 0:
        load_pts += 1.0
        reasons.append("✅ **Včerajšnji dan počitka (0 TSS):** Mišični glikogen je zapolnjen, ni akutne periferne utrujenosti od včeraj.")
    elif tss_yesterday >= 180 or yesterday_kj >= 2200:
        load_pts -= 5.5
        reasons.append(f"🔥 **Težak včerajšnji trening:** Včeraj opravljeno `{int(tss_yesterday)} TSS` ({int(yesterday_kj)} kJ) – mišični glikogen in živčni sistem sta pod vplivom akutne utrujenosti.")
    elif tss_yesterday >= 110:
        load_pts -= 2.0
    load_pts = float(np.clip(load_pts, 3.0, 20.0))

    # 5. SUBJEKTIVNI JUTRANJI DNEVNIK (0–15 točk)
    m_saved = (user_journal or {}).get("morning_entries", {}).get(target_date, {})
    if m_saved:
        s_feel = float(m_saved.get("sleep_feel", 8))
        s_fat = float(m_saved.get("fatigue", 3))
        s_mot = float(m_saved.get("motivation", 8))
        leg_str = str(m_saved.get("leg_feel", "2 - Dobre"))
        leg_num = int(leg_str[0]) if leg_str and leg_str[0].isdigit() else 2
        pain_str = str(m_saved.get("pain", "Brez")).strip().lower()

        subj_pts = (
            ((s_feel - 1.0) / 9.0) * 3.5
            + ((10.0 - s_fat) / 9.0) * 4.5
            + ((6.0 - leg_num) / 5.0) * 4.5
            + ((s_mot - 1.0) / 9.0) * 2.5
        )
        if pain_str and pain_str not in ("brez", "ni", "ne", "ok", "-"):
            subj_pts -= 5.0
            reasons.append(f"🚨 **Zabeležena bolečina / znaki bolezni:** *{m_saved.get('pain')}*.")
        if leg_num >= 5 or s_fat >= 7:
            reasons.append(f"⚠️ **Subjektivna utrujenost nog ({leg_str}):** Utrujenost {int(s_fat)}/10.")
        elif leg_num <= 2 and s_fat <= 3:
            reasons.append(f"✅ **Odličen subjektivni občutek:** Spočite noge in visoka motivacija ({int(s_mot)}/10).")
        subj_pts = float(np.clip(subj_pts, 1.0, 15.0))
    else:
        subj_pts = 11.5

    total_score = int(round(np.clip(hrv_pts + hr_pts + sleep_pts + load_pts + subj_pts, 15.0, 99.0)))

    if total_score >= 75:
        status_code = "TRAIN_HARD"
        badge = f"🟢 TRAIN ({total_score}/100)"
        if early_wake_artifact:
            headline = "Zelena luč za načrtovano polno Z2 aerobno bazo (nižji HRV/REM je le posledica zgodnje budilke ob mirnem nočnem HR)"
        else:
            headline = "Pripravljen na načrtovan polni trening (Z2 aerobna baza / Threshold / VO2max)"
        color = "#16a34a"
        bg_color = "rgba(22, 163, 74, 0.12)"
    elif total_score >= 50:
        status_code = "MODULATE"
        badge = f"🟡 MODULATE / Z2 BAZA ({total_score}/100)"
        if early_wake_artifact:
            headline = "Ohrani načrtovano polno Z2 aerobno vožnjo (nadzorovan spodnji Z2 zaradi zgodnjega bujenja)"
        else:
            headline = "Priporočena aerobna Z2 vožnja ali prilagojen/nadzorovan trening"
        color = "#d97706"
        bg_color = "rgba(245, 158, 11, 0.13)"
    else:
        status_code = "REST"
        badge = f"🔴 REST / AKTIVNA REGENERACIJA ({total_score}/100)"
        headline = "Visoka kumulativna utrujenost – priporočen počitek ali lahkotna Z1 razpeljava"
        color = "#dc2626"
        bg_color = "rgba(220, 38, 38, 0.13)"

    return {
        "date": target_date,
        "score": total_score,
        "status_code": status_code,
        "badge": badge,
        "headline": headline,
        "color": color,
        "bg_color": bg_color,
        "reasons": reasons,
        "metrics": {
            "hrv_pts": round(hrv_pts, 1),
            "hr_pts": round(hr_pts, 1),
            "sleep_pts": round(sleep_pts, 1),
            "load_pts": round(load_pts, 1),
            "subj_pts": round(subj_pts, 1),
            "hrv_today": hrv_today,
            "hrv_7d": round(hrv_7d_mean, 1) if hrv_7d_mean else None,
            "hrv_30d": round(hrv_30d_mean, 1) if hrv_30d_mean else None,
            "hrv_z": round(hrv_z, 2),
            "rhr_today": rhr_today,
            "rhr_base": round(rhr_base, 1),
            "rhr_delta": round(rhr_delta, 1),
            "avg_sleep_hr_today": sleep_hr_today,
            "avg_sleep_hr_7d": round(avg_overnight_hr_7d, 1) if avg_overnight_hr_7d is not None else None,
            "avg_overnight_hr_diff": avg_overnight_hr_diff,
            "wake_time_today": wake_str_today or "--:--",
            "wake_time_14d_avg": avg_wake_14d_str,
            "wake_time_diff_min": wake_time_diff_min,
            "early_wake_artifact": early_wake_artifact,
            "deep_min": deep_m,
            "rem_min": rem_m,
            "sleep_hours": sleep_h,
            "sleep_score": sleep_sc,
            "tss_yesterday": round(tss_yesterday, 0),
            "yesterday_kj": round(yesterday_kj, 0),
            "tss_7d": round(tss_7d_sum, 0),
            "tss_28d_weekly_avg": round(tss_28d_sum / 4.0, 0),
            "acute_ewma": round(acute_ewma, 1),
            "chronic_ewma": round(chronic_ewma, 1),
            "acwr": acwr,
        },
    }


# ==============================================================================
# 4. VGRAJENI TRENERSKI MOŽGANI (SYSTEM PROMPT Z NAJVIŠJO PRIORITETO PRAVIL)
# ==============================================================================
def build_coach_system_prompt(user_cfg: dict, display_name: str, user_dir: str = None, exclude_thread_id: str = None) -> str:
    """
    Sestavi stabilen, Prefix-Caching prijazen sistemski poziv elitnega trenerja (brez spremenljivih časovnih žigov),
    ki na NAJVIŠJO PRIORITETO postavi uporabnikova pravila in slog iz custom_coach_rules.json ter coach_memory.json,
    napredna fiziološka pravila in direktivo za jedrnate odgovore brez ponavljanja surovih tabel.
    """
    sport_prof = user_cfg.get("sport_profile", "🚴♂️ Kolesar (Cesta / Trenažer / MTB)")
    ftp_out = user_cfg.get("ftp_outdoor", 320)
    ftp_in = user_cfg.get("ftp_indoor", 305)
    weight = user_cfg.get("weight_kg", 65.5)
    t_pace = user_cfg.get("threshold_pace") or "4:15"
    lthr = user_cfg.get("lthr") or 174

    if user_dir:
        rules_obj = load_custom_coach_rules(user_dir, user_cfg)
        rules_text = rules_obj.get("rules_and_profile") or DEFAULT_KOLESARSKI_TRENING_RULES
        style_text = rules_obj.get("response_style") or DEFAULT_COACH_RESPONSE_STYLE
        # include_other_threads=False ohrani system_instruction bajt-na-bajt enak skozi celo sejo (Prefix Caching)
        cross_memory_text = build_cross_thread_memory_summary(
            user_dir,
            user_cfg,
            exclude_thread_id=exclude_thread_id,
            include_other_threads=False,
        )
    else:
        rules_text = str(user_cfg.get("custom_coach_instructions") or DEFAULT_KOLESARSKI_TRENING_RULES).strip()
        style_text = DEFAULT_COACH_RESPONSE_STYLE
        cross_memory_text = "Ni shranjenih posebnih dejstev."

    return f"""=== NAJVIŠJA PRIORITETA: OSEBNA TRENERSKA PRAVILA, PROFIL ŠPORTNIKA IN SLOG ODGOVARJANJA ===
Športnik: **{display_name}** (Primarni profil: **{sport_prof}**)

[A. OSEBNA NAVODILA, PROFIL ŠPORTNIKA IN PRAVILA TRENIRANJA (@kolesarski-trening)]
{rules_text}

[B. ZAHTEVAN SLOG IN NAČIN ODGOVARJANJA AI TRENERJA]
{style_text}
- Bodi maksimalno strokoven, konkreten in jedrnat. Ne ponavljaj surovih tabel podatkov, ki jih uporabnik že vidi na zaslonu – takoj preidi na fiziološko interpretacijo, ključne številke in jasna priporočila brez odvečnega uvoda ali zaključka.

[C. AKTUALNI FIZIOLOŠKI PARAMETRI IZ NASTAVITEV PROFILA]
- Zunanji FTP: **{ftp_out} W** ({round(ftp_out/max(40.0, float(weight)), 2)} W/kg) | Trenažer FTP: **{ftp_in} W** ({round(ftp_in/max(40.0, float(weight)), 2)} W/kg)
- Teža v profilu: **{weight} kg** | Tekaški pragovni tempo (T-Pace): **{t_pace} /km** | LTHR: **{lthr} bpm**

[D. 🧠 SPOMIN PRETEKLIH POGOVOROV IN DOGOVOROV (CROSS-THREAD MEMORY)]
Spodaj so zbrani vsi trajno pripeti dogovori (npr. nadomeščanje ob petkih, občutljivost tetive ob kolenu pri bolgarskih počepih, trenažer med tednom / zunaj za vikend) ter povzetki preteklih pogovornih niti. Te informacije UPOŠTEVAJ v vsakem odgovoru, ne glede na to, v kateri pogovorni niti se nahajamo:
{cross_memory_text}

[E. NAPREDNA FIZIOLOŠKA PRAVILA RAZMIŠLJANJA (OBVEZNO UPOŠTEVAJ BREZ ROBOTSKEGA REZANJA TRENINGOV!)]
1. **PRAVILO 1 — Nianse jutranjega spanca, zgodnje budilke in HRV (Wake-Up Time & Overnight HR Check):**
   - Če je nočni `HRV` nižji (npr. padec s 84 ms na 65 ms) in je `REM` spanec kratek (npr. 37 min), **NIKOLI avtomatsko in robotsko ne skrajšaj načrtovane 2h Z2 vožnje na 45 min Z1!**
   - Najprej obvezno preveri 4 ključne kazalnike iz konteksta: `Avg Overnight HR` (`avg_sleep_hr`), `Deep Sleep` (`deep_min`), `Yesterday's Load` (`tss_yesterday`) in `Wake-up Time` (`wake_time_diff_min` proti 14-dnevnemu povprečju):
     * Če je `Avg Overnight HR` zelo nizek/miren (npr. 51 bpm <= 7d povprečja ~52.3 bpm), je bil `Globoki spanec` odličen (npr. 80+ min), včeraj pa je bil dan počitka ali lahek dan, športniku jasno razloži, da sta nižji HRV in nižji REM **matematični in fiziološki artefakt zgodnje budilke** (npr. bujenje ob `06:07` namesto običajnih `08:15`), ki je odrezala zadnji jutranji cikel spanca (kjer sta REM in parasimpatični HRV najvišja), skupaj z jutranjim kortizolskim odzivom ob alarmu!
     * V tem primeru **OHRANI načrtovan polni Z2 aerobni volumen nespremenjen** (npr. polni `2h 00m` kontroliran Z2 na trenažerju pri ~57–62 % Indoor FTP, tj. `175–190 W` pri Indoor FTP 305 W, HR `120–132 bpm`), hkrati pa svetuj kontrolirano izvedbo v spodnji polovici Z2, da noge ostanejo sveže za jutrišnji ključni intervalni trening.
   - Pri interpretaciji `ACWR` vedno uporabi podani standardni 7d EWMA / 28d EWMA (z dnom 40 TSS/dan), da posamezna sobotna vožnja po premoru ne ustvari lažne panike.

2. **PRAVILO 2 — Inteligentna analiza končanega treninga in srčnega drifta (Pa:Hr Decoupling):**
   - Pri analizi opravljenega treninga (npr. cestna vožnja z izračunanim celotnim driftom +10.2 %) **VEDNO najprej preglej 10-minutne odseke, profil višine, % prostega teka (coasting) in zaključni mestni/prometni odsek, preden sodiš o srčnem driftu!**
   - Preveri, ali je povprečna moč v 2. polčasu padla zgolj zaradi **spusta** ali **prihoda v mestni promet / semaforje v zadnjih 15–20 minutah** (kjer naraste prosti tek z 0 W, povprečna moč pade na ~125–130 W, stop-and-go pospeševanja in toplota pa držijo utrip pri ~136 bpm, kar umetno napihne celotni Pa:Hr decoupling!).
   - Vedno poišči zadnji čisti mirni odsek pred mestom (npr. odsek 100–120 min pri ~194 W / NP 200 W @ 138 bpm ➔ `EF = 1.45`) in ta pozni `EF` neposredno primerjaj z `EF` prve ure (`1.39–1.47`) ter s prejšnjimi vožnjami (`EF 1.38`).
   - Analizo treninga strukturiraj v 3 jasne sklope:
     1. **Pregled ključnih metrik (Dejstva):** Čas gibanja vs. skupni čas, trasa, prava moč z 0 W vs. NP, IF, kJ, HR, EF (primerjava s prejšnjimi vožnjami), kadenca, % prostega teka in samodejno zaznani mikro-intervali / klanci iz 1s toka.
     2. **Trenerska ocena in natančna razlaga dinamike / srčnega drifta (Po odsekih in terenu):** Kaj se je dejansko dogajalo v 1. polčasu, na klancih/spustih, pri mikro-intervalih (npr. 30/15, 40/20) in v zaključku.
     3. **Prehrana, regeneracija in konkreten načrt za naslednje dni (Koledarski kontekst):** Kako ta trening vpliva na jutrišnji trening (kolo + moč/trup/koleno) in prihajajoči blok.

3. **PRAVILO 3 — Večšportna inteligenca (Kolesarjenje, Tek, Plavanje, Hoja, Moč/Jedro):**
   - Kadar je obravnavani trening ali cilj **Tek**, samodejno preklopi terminologijo in analitiko na tekaško fiziologijo: pragovni tempo (`Threshold Pace min/km`), GAP (`Grade Adjusted Pace`), `rTSS`, `Pace:HR EF`, tekaška kadenca (`spm`), mehanska obremenitev tetiv/kolen in Jack Daniels cone.
   - Kadar je zabeleženo ali načrtovano **Plavanje**, **Hoja/Pohodništvo** ali **Moč / Fitnes / Jedro (Core)**, to celostno vključi v regeneracijo in tedensko periodizacijo (npr. kako večerno plavanje razbremeni hrbtenico in ramenski obroč ter pospeši parasimpatično regeneracijo, ali kako bolgarski počepi brez bolečine v kolenski tetivi in glute bridge sodijo pred dan počitka, da ne okrnijo ključnih kolesarskih/tekaških intervalov).

[F. PROTOKOL ZA SAMODEJNI VPIS IN POSODABLJANJE KOLEDARJA TRENINGOV (<calendar_update>)]
Kadar te uporabnik prosi, da:
- sestaviš tedenski ali večdnevni plan treningov,
- dodaš trening v koledar, ALI
- prilagodiš današnji/jutrišnji trening v koledarju glede na utrujenost/počutje,
na KONCU svojega odgovora OBVEZNO dodaj strojno berljiv JSON blok zavit v oznaki `<calendar_update>` in `</calendar_update>`, ki vsebuje seznam treningov za vpis v koledar:

<calendar_update>
[
  {{
    "date": "YYYY-MM-DD",
    "sport": "Kolesarjenje",
    "title": "Kratek jasen naslov (npr. Z2 Aerobna baza / Threshold 2x20m / Počitek)",
    "type": "Z2 Aerobno | Threshold / Prag | VO2max | Sweet Spot | Regeneracija / Z1 | Počitek | Tek Z2 | Tek Intervali",
    "duration_min": 120,
    "target_power_w": "175-190 W (57-62% Indoor FTP) ali 200-230 W zunaj",
    "target_tss": 75,
    "description": "Ogrevanje 15 min Z1/Z2, glavni del ..., ohlajanje 10 min."
  }}
]
</calendar_update>
Poskrbi, da so datumi (`YYYY-MM-DD`) točni glede na današnji datum v kontekstu!

[G. PROTOKOL ZA DOLGOROČNI SPOMIN TRENERJA (<memory_update>)]
Če uporabnik v pogovoru omeni novo pomembno dolgoročno informacijo (npr. sprememba urnika, faks/služba, bolečina/občutljivost tetive ali kolena, oprema/trenažer, prehranske preference, nov tekmovalni cilj ali ugotovitev, kaj njegovemu telesu najbolj ustreza), na samem koncu odgovora dodaj skrito oznako:
<memory_update>Kratko, jedrnato dejstvo v eni povedi</memory_update>
Primer: <memory_update>Petek je uradni dan počitka zaradi faksa.</memory_update>
Aplikacija bo to dejstvo samodejno shranila v tvoj trajni dolgoročni spomin (coach_memory.json), ki se bo nato upošteval v vseh prihodnjih pogovorih!"""


def is_followup_turn(chat_history: list, user_question: str = "") -> bool:
    """
    Določi, ali gre za nadaljevalno vprašanje (follow-up) v aktivnem pogovoru.
    True kadar ima zgodovina >= 2 sporočili IN uporabnikovo vprašanje NE omenja novega datuma
    (npr. '2026-...') ali ključnih besed za podrobno analizo treninga/odsekov.
    """
    if not chat_history or len(chat_history) < 2:
        return False
    q = str(user_question or "").lower()
    if "2026-" in q or re.search(r"\b202\d-\d{2}-\d{2}\b", q) or re.search(r"\b\d{1,2}\.\s?\d{1,2}\.\s?(?:202\d)?\b", q):
        return False
    trigger_keywords = ("analiziraj trening", "včerajšnji trening", "zadnji trening", "interval", "odsek")
    if any(kw in q for kw in trigger_keywords):
        return False
    return True


def _fine_segments_compact(segs: list) -> str:
    """Kompaktna, cevno ločena oblika 2-minutnih visokoločljivostnih odsekov (~35–45 % manj znakov/žetonov brez izgube podatkov)."""
    if not segs:
        return "Visokoločljivostni odseki niso na voljo."
    f = lambda v, u="": f"{v}{u}" if v is not None else "--"
    out = ["Odsek(s) | W(Avg/NP/Mx/30s/1m/Wkg) | σ/VI | 0W% | HR(mn->mx/avg) | rpm | Naklon/+m | EF"]
    for g in segs:
        ele = f"{f(g.get('slope'), '%')}/+{g['elevGain']}m" if g.get("elevGain") is not None else "--"
        out.append(
            f"{g['label']}({g['dur']}s) | {g['avgW']}/{g['np']}/{g['maxW']}/{g['max30']}/{g['max60']}/{f(g.get('wkg'))} | "
            f"{g['stdW']}/{g['vi']} | {g['coastPct']}% | {f(g.get('minHR'))}->{f(g.get('maxHR'))}/{f(g.get('avgHR'))} | "
            f"{f(g.get('avgCad'))} | {ele} | {f(g.get('ef'))}"
        )
    return "\n".join(out)


def build_full_athlete_context(
    user_cfg: dict,
    user_journal: dict,
    df_daily_all: pd.DataFrame,
    activities_all: list,
    raw_data: dict,
    calendar_workouts: list,
    readiness_info: dict,
    today_date: date,
    get_sport_cat_fn,
    compute_fixed_blocks_fn,
    compute_run_blocks_fn,
    compute_run_load_fn,
    parse_stream_fn,
    user_dir: str = None,
    user_question: str = "",
    is_followup: bool = False,
    chat_history: list = None,
) -> str:
    """
    Turn-aware podatkovni kontekst športnika (Layer 2 Context Optimization):
    - Kadar is_followup == False (1. poteza ali izrecna zahteva za analizo treninga/datuma):
      vključi 100 % polno granularnost (2–3 min odseki, variabilnost, vsi uradni krogi,
      14-dnevni spanec/HRV/koraki/kalorije, 90-dnevna tabela treningov, sezonske MMP krivulje)
      v kompaktnih vrsticah z znakom '|'.
    - Kadar is_followup == True (nadaljevalni pogovor brez novega datuma/treninga):
      ohrani 100 % osebna trenerska pravila (@kolesarski-trening), coach_memory.json,
      današnji in včerajšnji spanec/HRV/pripravljenost + 7d povprečja ter strnjeno 1-vrstično
      tabelo zadnjih 10 treningov (brez ponovnega pošiljanja tisočih vrstic 2-min odsekov).
    """
    effective_followup = bool(is_followup) or (chat_history is not None and is_followup_turn(chat_history, user_question))
    lines = []
    today_str = today_date.strftime("%Y-%m-%d")
    weekday_slo = ["Ponedeljek", "Torek", "Sreda", "Četrtek", "Petek", "Sobota", "Nedelja"][today_date.weekday()]
    mode_tag = "NADALJEVALNI KONTEKST" if effective_followup else "POLNI GRANULARNI KONTEKST"
    lines.append(f"=== STRNJEN PODATKOVNI KONTEKST ŠPORTNIKA ({mode_tag} | Danes: {today_str}, {weekday_slo}) ===")

    user_cfg = user_cfg or {}
    ftp_out = int(user_cfg.get("ftp_outdoor", 320))
    ftp_in = int(user_cfg.get("ftp_indoor", 305))
    w_kg = float(user_cfg.get("weight_kg", 65.5))
    wpkg_out = round(ftp_out / max(40.0, w_kg), 2)
    wpkg_in = round(ftp_in / max(40.0, w_kg), 2)
    max_hr = int(user_cfg.get("max_hr", 202))
    rhr_b = int(user_cfg.get("rhr_base", 48))
    prof = user_cfg.get("sport_profile", "Kolesar")
    t_pace = user_cfg.get("threshold_pace") or "4:15"
    lthr = user_cfg.get("lthr") or 174

    lines.append("\n[1. PROFIL ŠPORTNIKA & CONE]")
    lines.append(f"- Profil: {prof} | Teža: {w_kg:.1f} kg | Max HR: {max_hr} bpm | RHR baza: {rhr_b} bpm")
    lines.append(f"- Kolesarski FTP: Zunanji {ftp_out} W ({wpkg_out} W/kg) | Trenažer (Indoor) {ftp_in} W ({wpkg_in} W/kg)")
    lines.append(
        f"- Kolesarske cone ZUNAJ (FTP {ftp_out}W): Z1 <{int(ftp_out*0.55)}W | Z2 {int(ftp_out*0.56)}-{int(ftp_out*0.75)}W | "
        f"Z3 {int(ftp_out*0.76)}-{int(ftp_out*0.90)}W | SweetSpot {int(ftp_out*0.88)}-{int(ftp_out*0.94)}W | Z4 Prag {int(ftp_out*0.91)}-{int(ftp_out*1.05)}W | Z5 VO2 {int(ftp_out*1.06)}-{int(ftp_out*1.20)}W"
    )
    lines.append(
        f"- Kolesarske cone TRENAŽER (FTP {ftp_in}W): Z1 <{int(ftp_in*0.55)}W | Z2 {int(ftp_in*0.56)}-{int(ftp_in*0.75)}W (kontroliran Z2 {int(ftp_in*0.57)}-{int(ftp_in*0.62)}W) | "
        f"Z3 {int(ftp_in*0.76)}-{int(ftp_in*0.90)}W | SweetSpot {int(ftp_in*0.88)}-{int(ftp_in*0.94)}W | Z4 Prag {int(ftp_in*0.91)}-{int(ftp_in*1.05)}W | Z5 VO2 {int(ftp_in*1.06)}-{int(ftp_in*1.20)}W"
    )
    lines.append(f"- Tekaški parametri: Pragovni tempo (T-Pace) = {t_pace} /km | LTHR = {lthr} bpm | Kadenca = {user_cfg.get('run_cadence', 180)} spm")

    goals = user_cfg.get("goals", [])
    if goals:
        lines.append("- Aktivni cilji in tekme:")
        for g in goals:
            g_date = str(g.get("date", ""))[:10]
            try:
                d_left = (datetime.strptime(g_date, "%Y-%m-%d").date() - today_date).days
                d_txt = f"še {d_left} dni" if d_left >= 0 else f"pred {abs(d_left)} dnevi"
            except Exception:
                d_txt = ""
            lines.append(f"  * {g.get('name', 'Cilj')} ({g_date}, {d_txt}) | Šport: {g.get('sport', 'Kolo')} | Cilj: {g.get('target_ftp') or g.get('target_time') or ''} | {g.get('metric', '')}")

    # 2. Današnja ocena pripravljenosti
    lines.append("\n[2. DANAŠNJA OCENA PRIPRAVLJENOSTI (REST OR TRAIN)]")
    if readiness_info:
        lines.append(f"- Status: **{readiness_info.get('status', 'Zelena / Treniraj')}** (Semafor: {readiness_info.get('color', 'green')})")
        m = readiness_info.get("metrics") or {}
        w_diff = m.get("wake_time_diff_min")
        w_diff_str = f"{int(w_diff):+d} min" if w_diff is not None and pd.notna(w_diff) else "--"
        ohr_diff = m.get("avg_overnight_hr_diff")
        ohr_diff_str = f"{float(ohr_diff):+.1f} bpm" if ohr_diff is not None and pd.notna(ohr_diff) else "--"
        lines.append(
            f"- Analiza bujenja in nočnega utripa (Rule 1): Ura bujenja={m.get('wake_time_today') or '--'} vs. 14d povp.={m.get('wake_time_14d_avg') or '--'} "
            f"(`wake_time_diff_min` = {w_diff_str}) | Nočni HR=`{m.get('avg_sleep_hr_today') or '--'} bpm` vs. 7d povp.=`{m.get('avg_sleep_hr_7d') or '--'} bpm` (`avg_overnight_hr_diff` = {ohr_diff_str}) | "
            f"Globoki spanec=`{m.get('deep_min') or '--'} min` | REM=`{m.get('rem_min') or '--'} min` | Zaznan artefakt zgodnje budilke: **{'DA' if m.get('early_wake_artifact') else 'NE'}**"
        )
        lines.append(
            f"- Ključni kazalniki: HRV danes={m.get('hrv_today')} ms (7d={m.get('hrv_7d')} ms, 30d={m.get('hrv_30d')} ms, Z={m.get('hrv_z')}) | "
            f"RHR={m.get('rhr_today')} bpm (baza {m.get('rhr_base')} bpm) | "
            f"Včerajšnji trening=`{m.get('tss_yesterday')} TSS` ({m.get('yesterday_kj', 0)} kJ) | "
            f"7d TSS={m.get('tss_7d')} | 28d tedensko povp.={m.get('tss_28d_weekly_avg')} | "
            f"EWMA ACWR={m.get('acwr')}"
        )
        for r_line in readiness_info.get("reasons", []):
            lines.append(f"  * {r_line}")

    # 2b. Stres in gibanje prejšnjega dne (celoten dan) + današnji tekoči koraki/kalorije
    try:
        _dls = (raw_data or {}).get("dailies") or {}
        def _mv(_dk, _lbl):
            _d = _dls.get(_dk) or {}
            if not _d:
                return f"- **{_lbl}:** ni podatkov."
            _mov = _d.get("movement") if isinstance(_d.get("movement"), dict) else {}
            _stp_obj = _mov.get("steps") if isinstance(_mov.get("steps"), dict) else {}
            _dst_obj = _mov.get("distance") if isinstance(_mov.get("distance"), dict) else {}
            _cal_obj = _d.get("calories") if isinstance(_d.get("calories"), dict) else {}
            _str_obj = _d.get("stress") if isinstance(_d.get("stress"), dict) else {}
            _stp_v = int(_d.get("totalSteps") or _stp_obj.get("value") or 0)
            _dst_v = float(_d.get("totalDistanceMeters") or _dst_obj.get("value") or (_stp_v * 0.78))
            _km_v = _dst_v / 1000.0 if _dst_v > 50 else _dst_v
            _b = float(_d.get("bmrKilocalories") or _cal_obj.get("burnedResting") or 0)
            _a = float(_d.get("activeKilocalories") or _cal_obj.get("burnedActive") or 0)
            _st_v = _d.get("averageStressLevel") if _d.get("averageStressLevel") is not None else _str_obj.get("avgLevel", "--")
            _t = float(_d.get("totalKilocalories") or (_a + _b))
            _wk = []
            for _x in acts_all_for_mv:
                if str(_x.get("startTimeLocal", ""))[:10] == _dk:
                    _sec = int(_x.get("movingDuration") or _x.get("duration") or 0)
                    _wk.append(f"{_x.get('activityName', 'Trening')} ({_sec // 60} min)")
            return (f"- **{_lbl}:** Prehojeni koraki: {_stp_v:,} korakov "
                    f"({_km_v:.1f} km) | Skupna poraba: {int(round(_t))} kcal "
                    f"(Aktivne kalorije: {int(round(_a))} kcal, BMR: {int(round(_b))} kcal) | "
                    f"Povprečni dnevni stres: {_st_v} | Treningi: [{'; '.join(_wk) or 'brez treninga'}]")
        acts_all_for_mv = list(((raw_data or {}).get("activities") or {}).values())
        _yd = (today_date - timedelta(days=1)).strftime("%Y-%m-%d")
        lines.append(_mv(_yd, f"Stres in gibanje prejšnjega dne (Včeraj - {_yd}) — celoten dan"))
        lines.append(_mv(today_date.strftime("%Y-%m-%d"), "Danes (trenutno)"))
    except Exception:
        pass

    # 3. Dolgoročni spomin trenerja (coach_memory.json) - 100 % brez rezanja
    lines.append("\n[3. 🧠 DOLGOROČNI SPOMIN TRENERJA (KLJUČNA DEJSTVA IZ COACH_MEMORY.JSON)]")
    mem_obj = load_coach_memory(user_dir) if user_dir else {}
    facts = mem_obj.get("facts") or []
    if facts:
        for f_txt in facts:
            lines.append(f"  • {f_txt}")
    else:
        lines.append("  (V spominu še ni posebnih trajnih dogovorov.)")

    # A) Zadnjih 14 dni spanca in regeneracije (turn-aware: na nadaljevalnih potezah samo danes/včeraj + 7d povp.)
    if df_daily_all is not None and not df_daily_all.empty and "date" in df_daily_all.columns:
        if "date_obj" not in df_daily_all.columns:
            df_daily_all["date_obj"] = pd.to_datetime(df_daily_all["date"]).dt.date
        cutoff_14d = today_date - timedelta(days=14)
        df_14d = df_daily_all[df_daily_all["date_obj"] >= cutoff_14d].sort_values("date")
        m_entries = (user_journal or {}).get("morning_entries", {})

        def _wake_min(bed_s):
            try:
                tail = re.split(r"[–-]", str(bed_s))[-1].strip()
                hh, mm = tail.split(":")[:2]
                return int(hh) * 60 + int(mm)
            except Exception:
                return None

        wake_all = [(_k, _wake_min(_b)) for _k, _b in zip(df_14d["date"], df_14d.get("bedtime_str", [None] * len(df_14d)))]
        wake_vals = [w for _, w in wake_all if w is not None]
        wake_base = (sum(wake_vals) / len(wake_vals)) if wake_vals else None

        def _f(v, fmt="{:.0f}", suf=""):
            return (fmt.format(v) + suf) if v is not None and pd.notna(v) else "--"

        def _sleep_compact_row(r):
            d_k = str(r.get("date"))
            wk = _wake_min(r.get("bedtime_str"))
            wk_txt = "--"
            if wk is not None:
                wk_txt = f"{wk // 60:02d}:{wk % 60:02d}"
                if wake_base is not None:
                    wk_txt += f"({int(round(wk - wake_base)):+d}m)"
            me = m_entries.get(d_k, {})
            subj = ""
            if me:
                subj = f" | Dnevnik: sp={me.get('sleep_feel')}/10, utr={me.get('fatigue')}/10, noge='{me.get('leg_feel', '')}'"
                if me.get("notes"):
                    subj += f", opomba='{me.get('notes')}'"
            hrv_7 = r.get("hrv_7d_mean") if "hrv_7d_mean" in r else r.get("hrv_weekly")
            return (
                f"{d_k} | {_f(r.get('sleep_hours'), '{:.1f}', 'h')}[{r.get('bedtime_str') or '--'}] buj.{wk_txt} | "
                f"Sc{_f(r.get('sleep_score'))} | D{_f(r.get('deep_min'))}/R{_f(r.get('rem_min'))}/L{_f(r.get('light_min'))}/B{_f(r.get('awake_min'))}m | "
                f"HRV {_f(r.get('hrv_night'))}(7d:{_f(hrv_7)})ms | RHR {_f(r.get('rhr'))}/nHR {_f(r.get('avg_sleep_hr'))}bpm | "
                f"Stp {_f(r.get('steps'))} | Kcal {_f(r.get('cal_act'))}/{_f(r.get('cal_tot'))}{subj}"
            )

        if effective_followup:
            lines.append("\n[TIER 1 — DANES, VČERAJ IN 7-DNEVNO POVPREČJE SPANCA TER REGENERACIJE]")
            lines.append("Datum | Spanec[ura] bujenje | Sc | Faze(D/R/L/B) | HRV(7d) | RHR/nHR | Koraki | Kcal(akt/sk) | Dnevnik")
            yest_str = (today_date - timedelta(days=1)).strftime("%Y-%m-%d")
            df_recent = df_14d[df_14d["date"].astype(str).isin([yest_str, today_str])]
            if df_recent.empty:
                df_recent = df_14d.tail(2)
            for _, r in df_recent.iterrows():
                lines.append(_sleep_compact_row(r))
            cutoff_7d = today_date - timedelta(days=7)
            df_7d = df_14d[df_14d["date_obj"] >= cutoff_7d]
            if not df_7d.empty:
                def _col_avg(col, fmt="{:.0f}", suf=""):
                    if col not in df_7d.columns:
                        return "--"
                    val = pd.to_numeric(df_7d[col], errors="coerce").mean()
                    return (fmt.format(val) + suf) if pd.notna(val) else "--"
                lines.append(
                    f"7d-povp. | Spanec {_col_avg('sleep_hours', '{:.1f}', 'h')} | Sc {_col_avg('sleep_score')} | "
                    f"D{_col_avg('deep_min')}/R{_col_avg('rem_min')}m | HRV {_col_avg('hrv_night', suf='ms')} | "
                    f"RHR {_col_avg('rhr')}/nHR {_col_avg('avg_sleep_hr')}bpm | Koraki {_col_avg('steps')}"
                )
        else:
            lines.append("\n[TIER 1 — ZADNJIH 14 DNI SPANCA, REGENERACIJE IN POČUTJA]")
            lines.append("Datum | Spanec[ura] bujenje | Sc | Faze(D/R/L/B) | HRV(7d) | RHR/nHR | Koraki | Kcal(akt/sk) | Dnevnik")
            for _, r in df_14d.iterrows():
                lines.append(_sleep_compact_row(r))

    def _append_calendar_block():
        lines.append("\n[4. PLANIRANI KOLEDAR TRENINGOV (-2 dni do +10 dni)]")
        w_start = (today_date - timedelta(days=2)).strftime("%Y-%m-%d")
        w_end = (today_date + timedelta(days=10)).strftime("%Y-%m-%d")
        win_cal = [w for w in (calendar_workouts or []) if w_start <= str(w.get("date", ""))[:10] <= w_end]
        if win_cal:
            for cw in sorted(win_cal, key=lambda x: str(x.get("date", ""))):
                lines.append(f"{cw.get('date')} | [{cw.get('sport', 'Kolo')}] {cw.get('title')} ({cw.get('type')}) | {cw.get('duration_min')}m | {cw.get('target_power_w')} | TSS{cw.get('target_tss')} | {cw.get('description')}")
        else:
            lines.append("- V koledarju še ni vpisanih planiranih treningov za prihodnje dni.")

    # B) Treningi (turn-aware: nadaljevalne poteze dobijo 1-vrstični povzetek zadnjih 10 treningov; 1. poteza/datum polno anatomijo)
    streams_map = (raw_data or {}).get("streams", {})
    splits_map = (raw_data or {}).get("splits", {})
    w_entries = (user_journal or {}).get("workout_entries", {})
    acts_list = list(activities_all or [])

    def _workout_line(a):
        aid = str(a.get("activityId"))
        d_s = str(a.get("startTimeLocal", ""))[:16]
        name = a.get("activityName", "Trening")
        sp = get_sport_cat_fn(a.get("activityType", {}).get("typeKey"), name) if get_sport_cat_fn else "Šport"
        m_sec = int(a.get("movingDuration") or a.get("duration") or 0)
        dist_km = round((a.get("distance") or 0) / 1000.0, 1)
        elev_m = int(round(a.get("elevationGain") or 0))
        st_sum = (streams_map.get(aid) or {}).get("summary", {}) or {}
        avg_w = st_sum.get("trueAvgPower") or a.get("avgPower") or a.get("averagePower") or 0
        np_w = st_sum.get("np") or a.get("normPower") or a.get("normalizedPower") or avg_w
        avg_hr = a.get("averageHR")
        max_hr_a = a.get("maxHR")
        tss_v = compute_activity_tss(a, user_cfg, compute_run_load_fn, get_sport_cat_fn)
        work_kj = st_sum.get("work_kJ") or (round(avg_w * m_sec / 1000.0) if avg_w else 0)
        drift_obj = st_sum.get("cardiacDrift") or {}
        decoup = drift_obj.get("decouplingPct") if isinstance(drift_obj, dict) else None
        coast_pct = st_sum.get("coastingPct")
        ef_v = round(np_w / avg_hr, 2) if (np_w and avg_hr and avg_hr > 0) else None
        wn = w_entries.get(aid, {})
        extra = ""
        if wn.get("rpe"):
            extra += f" | RPE {wn.get('rpe')}/10"
        if wn.get("comment"):
            extra += f" | Komentar: '{wn.get('comment')}'"
        if coast_pct is not None:
            extra += f" | 0W {coast_pct:.1f}%"
        dec_s = f"{decoup:+.1f}%" if decoup is not None else "--"
        return (f"{d_s} | {name} ({sp}) | {timedelta(seconds=m_sec)} | {dist_km}km | +{elev_m}m | "
                f"{int(round(avg_w))}W(NP{int(round(np_w))}) | HR {int(round(avg_hr)) if avg_hr else '--'}/{int(round(max_hr_a)) if max_hr_a else '--'} | "
                f"TSS{tss_v:.0f} | {work_kj}kJ | EF{ef_v or '--'} | Drift{dec_s}{extra}"), sp, st_sum

    if effective_followup:
        lines.append("\n[TIER 1B — POVZETEK ZADNJIH 10 TRENINGOV (1 VRSTICA NA TRENING)]")
        lines.append("Datum | Trening (Šport) | Čas | km | +m | AvgW(NP) | HR(avg/max) | TSS | kJ | EF | Drift | Opombe")
        for a in acts_list[:10]:
            lines.append("  - " + _workout_line(a)[0])
        _append_calendar_block()
        return "\n".join(lines)

    lines.append("\n[TIER 1B — ZADNJI 3 TRENINGI IN OMENJENI TRENINGI: POLNA ANATOMIJA (KROGI, ODSEKI, MIKRO-INTERVALI)]")
    detail_ids = [str(a.get("activityId")) for a in acts_list[:3]]
    q = str(user_question or "").lower()
    if q:
        q_dates = set(re.findall(r"\d{4}-\d{2}-\d{2}", q))
        for dd, mm_, yy in re.findall(r"(\d{1,2})\.\s?(\d{1,2})\.\s?(\d{4})?", q):
            yy = yy or str(today_date.year)
            q_dates.add(f"{int(yy):04d}-{int(mm_):02d}-{int(dd):02d}")
        for a in acts_list:
            aid_ = str(a.get("activityId"))
            nm = str(a.get("activityName") or "").lower()
            if aid_ in detail_ids:
                continue
            if str(a.get("startTimeLocal", ""))[:10] in q_dates or (len(nm) >= 5 and nm in q):
                detail_ids.append(aid_)

    by_id = {str(a.get("activityId")): a for a in acts_list}
    for n_i, aid in enumerate(detail_ids, 1):
        a = by_id.get(aid)
        if not a:
            continue
        head, sp, st_sum = _workout_line(a)
        tag = "NAJNOVEJŠI" if n_i == 1 else ("ZADNJI" if n_i <= 3 else "OMENJEN V VPRAŠANJU")
        lines.append(f"  • Trening #{n_i} ({tag}): {head}")
        drift_d = st_sum.get("cardiacDrift") or {}
        if isinstance(drift_d, dict) and "decouplingPct" in drift_d:
            lines.append(
                f"    - Srčni drift: {drift_d['decouplingPct']:+.1f}% (1. polčas {drift_d.get('avgPower1')}W @ {drift_d.get('avgHR1')}bpm [EF {drift_d.get('ef1')}] | "
                f"2. polčas {drift_d.get('avgPower2')}W @ {drift_d.get('avgHR2')}bpm [EF {drift_d.get('ef2')}])")
        sf = streams_map.get(aid) or {}
        if not sf or ("power" not in sf and "speed" not in sf):
            _cand_st_list = ([os.path.join(str(user_dir), "streams", f"{aid}.json")] if user_dir else []) + [os.path.join(PROJECT_DIR, "data", "streams", f"{aid}.json")]
            for _cand_st in _cand_st_list:
                if os.path.exists(_cand_st):
                    try:
                        with open(_cand_st, "r", encoding="utf-8") as _stf:
                            sf = json.load(_stf) or {}
                        break
                    except Exception:
                        pass
        try:
            has_pw = bool(sf.get("power"))
        except Exception:
            has_pw = False
        if sf:
            try:
                if sp == "Tek" and compute_run_blocks_fn:
                    r_blocks = compute_run_blocks_fn(sf, block_sec=600) or []
                    if r_blocks:
                        lines.append("    - 10-minutni odseki teka:")
                        for b in r_blocks:
                            lines.append(f"      * {b.get('label')}: {b.get('dist_km', 0):.2f} km | Tempo {b.get('pace_str')} /km | HR {b.get('avgHR')}/{b.get('maxHR')} bpm")
                elif compute_fixed_blocks_fn and has_pw:
                    c_blocks = compute_fixed_blocks_fn(
                        sf.get("power"), sf.get("heartRate") or sf.get("heart_rate"), sf.get("cadence"),
                        sf.get("speed"), sf.get("temperature"), block_sec=600, ele_arr=sf.get("elevation")) or []
                    if c_blocks:
                        lines.append("    - 10-min odseki [Odsek | AvgW(NP/Mx) | HR(min->mx/avg) | VI | 0W% | +m | EF]:")
                        for b in c_blocks:
                            lines.append(
                                f"      * {b.get('label')} | {b.get('avgW')}W(NP{b.get('np')}/Mx{b.get('maxW', b.get('np', '--'))}) | "
                                f"HR:{b.get('minHR', '--')}->{b.get('maxHR')}/{b.get('avgHR')} | VI{b.get('vi', 1.0)} | "
                                f"{b.get('coastingPct', 0)}% | +{b.get('elevGain', b.get('elev', 0))}m | EF:{b.get('ef')}"
                            )
                if has_pw:
                    try:
                        _fs = streamlib.compute_fine_segments(
                            sf.get("power"), sf.get("heartRate") or sf.get("heart_rate"), sf.get("cadence"),
                            sf.get("speed"), sf.get("elevation"), seg_sec=120, weight_kg=w_kg)
                        if _fs:
                            lines.append("    - 🔬 2-min visokoločljivostni odseki (notranja variabilnost/HR/30s/1m):")
                            for _l in _fine_segments_compact(_fs).splitlines():
                                lines.append(f"      {_l}")
                        _dyn = streamlib.compute_dynamic_effort_segments(sf.get("power"), sf.get("heartRate") or sf.get("heart_rate"), sf.get("cadence"), sf.get("speed"), sf.get("elevation"), weight_kg=w_kg)
                        if _dyn:
                            lines.append("    - 🔄 Dinamični odseki [# Odsek(s) | AvgW(NP/Mx) | HR(min->mx/avg) | VI | 0W% | +m | EF]:")
                            for _ds in _dyn:
                                lines.append(
                                    f"      * #{_ds['index']} {_ds['label']}({_ds['dur_sec']}s) | {_ds['avgW']}W(NP{_ds['np']}/Mx{_ds['maxW']}) | "
                                    f"HR:{_ds.get('minHR', '--')}->{_ds.get('maxHR', '--')}/{_ds.get('avgHR', '--')} | VI{_ds['vi']} | "
                                    f"{_ds['coastPct']}% | +{_ds['elevGain']}m | EF:{_ds.get('ef', '--')}"
                                )
                    except Exception as _fe:
                        lines.append(f"    - (2-min odseki niso na voljo: {_fe})")
                det_int = streamlib.detect_stream_intervals(
                    sf=sf, act_obj=a, sport="Tek" if sp == "Tek" else "Kolesarjenje",
                    ftp=float(ftp_in if ("trenažer" in sp.lower() or "indoor" in sp.lower()) else ftp_out), weight_kg=w_kg)
                if det_int and det_int.get("markdown_summary"):
                    lines.append("    - ⚡ Samodejno zaznani mikro-intervali in klanci (iz 1 Hz toka):")
                    for md_l in det_int["markdown_summary"].splitlines():
                        lines.append(f"      {md_l}")
            except Exception as _e:
                lines.append(f"    - (razčlenitev toka ni bila mogoča: {_e})")
        else:
            lines.append("    - (1 Hz tok za ta trening ni naložen; odprite ga v zavihku Analiza treningov, da se samodejno prenese)")
        sp_obj = splits_map.get(aid) or {}
        if not sp_obj:
            _cand_sp_list = ([os.path.join(str(user_dir), "splits", f"{aid}.json")] if user_dir else []) + [os.path.join(PROJECT_DIR, "data", "splits", f"{aid}.json")]
            for _cand_sp in _cand_sp_list:
                if os.path.exists(_cand_sp):
                    try:
                        with open(_cand_sp, "r", encoding="utf-8") as _spf:
                            sp_obj = json.load(_spf) or {}
                        break
                    except Exception:
                        pass
        laps = sp_obj.get("lapDTOs") or sp_obj.get("laps") or []
        if laps:
            lines.append(f"    - Garmin krogi ({len(laps)}) [Krog | Čas(km) | AvgW(NP/Mx) | HR(min->mx/avg) | VI | 0W% | +m | EF]:")
            _lp_t0 = 0
            _p_lp_arr = np.nan_to_num(np.asarray((sf.get("power") if sf else None) or [], dtype=float))
            _h_lp_arr = np.nan_to_num(np.asarray(((sf.get("heartRate") or sf.get("heart_rate")) if sf else None) or [], dtype=float))
            _e_lp_arr = np.nan_to_num(np.asarray((sf.get("elevation") if sf else None) or [], dtype=float))
            for idx_l, lp in enumerate(laps, 1):
                l_dur = int(round(lp.get("duration") or lp.get("movingDuration") or 0))
                l_dist = round((lp.get("distance") or 0) / 1000.0, 2)
                l_w = lp.get("averagePower") or lp.get("avgPower")
                l_hr = lp.get("averageHR") or lp.get("avgHR")
                l_np = lp.get("normalizedPower") or lp.get("normPower") or l_w
                l_max_w = lp.get("maxPower")
                l_max_hr = lp.get("maxHR")
                l_min_hr = None
                l_coast = None
                l_elev = lp.get("totalAscent") if lp.get("totalAscent") is not None else lp.get("elevationGain")
                if l_dur > 0 and len(_p_lp_arr) >= _lp_t0 + l_dur:
                    _pw_s = _p_lp_arr[_lp_t0:_lp_t0 + l_dur]
                    if len(_pw_s):
                        if not l_w:
                            l_w = float(np.mean(_pw_s))
                        if not l_np:
                            l_np = float(np.mean(np.convolve(_pw_s, np.ones(30)/30.0, mode="valid") ** 4) ** 0.25) if len(_pw_s) >= 30 else float(l_w)
                        if not l_max_w:
                            l_max_w = float(np.max(_pw_s))
                        l_coast = round(float(np.mean(_pw_s == 0)) * 100.0, 1)
                if l_dur > 0 and len(_h_lp_arr) >= _lp_t0 + l_dur:
                    _hv_s = _h_lp_arr[_lp_t0:_lp_t0 + l_dur]
                    _hv_s = _hv_s[_hv_s > 0]
                    if len(_hv_s):
                        if not l_hr:
                            l_hr = float(np.mean(_hv_s))
                        if not l_max_hr:
                            l_max_hr = int(np.max(_hv_s))
                        l_min_hr = int(np.min(_hv_s))
                if l_elev is None and l_dur > 1 and len(_e_lp_arr) >= _lp_t0 + l_dur:
                    l_elev = int(round(float(np.sum(np.clip(np.diff(_e_lp_arr[_lp_t0:_lp_t0 + l_dur]), 0, 2.8)))))
                _lp_t0 += max(0, l_dur)
                l_vi = round(float(l_np) / float(l_w), 2) if (l_np and l_w and float(l_w) > 0) else 1.0
                l_ef = round(float(l_np or l_w) / float(l_hr), 2) if ((l_np or l_w) and l_hr and float(l_hr) > 0) else "--"
                lines.append(
                    f"      * K{idx_l} | {l_dur // 60}:{l_dur % 60:02d}({l_dist}km) | "
                    f"{int(round(l_w)) if l_w else '--'}W(NP{int(round(l_np)) if l_np else '--'}/Mx{int(round(l_max_w)) if l_max_w else '--'}) | "
                    f"HR:{l_min_hr if l_min_hr else '--'}->{int(round(l_max_hr)) if l_max_hr else '--'}/{int(round(l_hr)) if l_hr else '--'} | "
                    f"VI{l_vi} | {f'{l_coast}%' if l_coast is not None else '--'} | +{int(round(float(l_elev))) if l_elev is not None else 0}m | EF:{l_ef}"
                )

    # Dnevi 15-90: ključna vrstica vsakega treninga + tedenski povzetki
    lines.append("\n[TIER 2 — DNEVI 15 DO 90: VSI TRENINGI (KLJUČNA VRSTICA) + TEDENSKI POVZETKI]")
    lo_d, hi_d = today_date - timedelta(days=90), today_date - timedelta(days=15)
    for a in acts_list:
        try:
            ad = datetime.strptime(str(a.get("startTimeLocal", ""))[:10], "%Y-%m-%d").date()
        except Exception:
            continue
        if lo_d <= ad <= hi_d and str(a.get("activityId")) not in detail_ids:
            lines.append("  - " + _workout_line(a)[0])
    lines.append("Tedenski povzetki (ure | TSS | kJ | višinci | Z1-Z2 vs Z4-Z5 | spanje | HRV | RHR):")
    for w_idx in range(1, 14):
        w_end_dt = today_date - timedelta(days=(w_idx - 1) * 7 + 8)
        w_start_dt = w_end_dt - timedelta(days=6)
        w_acts = []
        for a in acts_list:
            try:
                ad = datetime.strptime(str(a.get("startTimeLocal", ""))[:10], "%Y-%m-%d").date()
            except Exception:
                continue
            if w_start_dt <= ad <= w_end_dt:
                w_acts.append(a)
        w_sec = sum(float(a.get("movingDuration") or a.get("duration") or 0) for a in w_acts)
        w_tss = int(sum(compute_activity_tss(a, user_cfg, compute_run_load_fn, get_sport_cat_fn) for a in w_acts))
        w_elev = int(sum(float(a.get("elevationGain") or 0) for a in w_acts))
        w_kj = int(sum(float(a.get("avgPower") or 0) * float(a.get("movingDuration") or 0) / 1000.0 for a in w_acts))
        z12_s = z45_s = 0.0
        for wa in w_acts:
            np_val = float(wa.get("normPower") or wa.get("avgPower") or 0)
            dur_v = float(wa.get("movingDuration") or 0)
            if np_val > 0 and ftp_out > 0:
                ifr = np_val / float(ftp_out)
                if ifr < 0.78:
                    z12_s += dur_v
                elif ifr >= 0.90:
                    z45_s += dur_v
        tot_dur = max(1.0, w_sec)
        sl_mean = hrv_mean = rhr_mean = "--"
        if df_daily_all is not None and not df_daily_all.empty and "date_obj" in df_daily_all.columns:
            df_w = df_daily_all[(df_daily_all["date_obj"] >= w_start_dt) & (df_daily_all["date_obj"] <= w_end_dt)]
            if not df_w.empty:
                if pd.notna(df_w["sleep_hours"].mean()):
                    sl_mean = f"{df_w['sleep_hours'].mean():.1f}h"
                if pd.notna(df_w["hrv_night"].mean()):
                    hrv_mean = f"{df_w['hrv_night'].mean():.0f}ms"
                if pd.notna(df_w["rhr"].mean()):
                    rhr_mean = f"{df_w['rhr'].mean():.0f}bpm"
        lines.append(f"Teden {w_start_dt.strftime('%d.%m')}–{w_end_dt.strftime('%d.%m')}: {w_sec / 3600.0:.1f}h ({len(w_acts)}x) | {w_tss} TSS | {w_kj} kJ | +{w_elev}m | "
                     f"Z1-Z2 {int(round(z12_s / tot_dur * 100))}% vs Z4-Z5 {int(round(z45_s / tot_dur * 100))}% | {sl_mean} | {hrv_mean} | {rhr_mean}")

    # TIER 3: Dolgoročni večletni arhiv (Vsa pretekla zgodovina – Sezonska & All-Time tabela)
    lines.append("\n[TIER 3 — VEČLETNI SEZONSKI & ALL-TIME BAZNI ARHIV (CELOTNA ZGODOVINA)]")
    durs = [(5, "5s"), (60, "1m"), (300, "5m"), (600, "10m"), (1200, "20m"), (1800, "30m"), (2400, "40m"), (3600, "60m")]
    lines.append("Sezona | Tedenski volumen | " + " | ".join(f"Max {n}" for _, n in durs) + " | Povp. EF")
    years = sorted(set(str(a.get("startTimeLocal", ""))[:4] for a in (activities_all or []) if str(a.get("startTimeLocal", ""))[:4].isdigit()), reverse=True)
    mmp_map = (raw_data or {}).get("mmp", {})
    for yr in years:
        y_acts = [a for a in (activities_all or []) if str(a.get("startTimeLocal", ""))[:4] == yr]
        y_sec = sum(float(a.get("movingDuration") or a.get("duration") or 0) for a in y_acts)
        best = {d: 0 for d, _ in durs}
        ef_list = []
        for ya in y_acts:
            md = (mmp_map.get(str(ya.get("activityId"))) or {}).get("mmp") or {}
            for d, _ in durs:
                v = md.get(str(d)) or md.get(d) or 0
                try:
                    best[d] = max(best[d], int(v))
                except Exception:
                    pass
            np_w = float(ya.get("normPower") or ya.get("avgPower") or 0)
            ahr = float(ya.get("averageHR") or 0)
            if np_w > 120 and ahr > 100:
                ef_list.append(np_w / ahr)
        avg_ef = f"{np.mean(ef_list):.2f}" if ef_list else "--"
        peaks = " | ".join(f"{n}: {best[d] or '--'}W" for d, n in durs)
        lines.append(f"{yr}: ~{round(y_sec / 3600.0 / 52.0, 1)} h/teden ({len(y_acts)} tren.) | {peaks} | EF {avg_ef}")
    lines.append("Mesečni volumen (ure | km | višinci):")
    months = {}
    for a in (activities_all or []):
        mk = str(a.get("startTimeLocal", ""))[:7]
        if len(mk) == 7:
            m = months.setdefault(mk, [0.0, 0.0, 0.0])
            m[0] += float(a.get("movingDuration") or a.get("duration") or 0) / 3600.0
            m[1] += float(a.get("distance") or 0) / 1000.0
            m[2] += float(a.get("elevationGain") or 0)
    for mk in sorted(months, reverse=True):
        h, k, e = months[mk]
        lines.append(f"  {mk}: {h:.1f}h | {k:.0f} km | +{e:.0f} m")

    _append_calendar_block()

    return "\n".join(lines)


# Alias za zahtevo v specifikaciji (build_ai_coach_context)
build_ai_coach_context = build_full_athlete_context


# ==============================================================================
# 5. IZVOZ KOLEDARJA V MARKDOWN IN .ICS (iCal ZA TELEFON)
# ==============================================================================
def export_week_markdown(week_days: list, cal_by_date: dict, acts_by_date: dict) -> str:
    """Pripravi pregleden Markdown izpis tedenskega koledarja treningov."""
    slo_days = ["Ponedeljek", "Torek", "Sreda", "Četrtek", "Petek", "Sobota", "Nedelja"]
    md = [f"### 📅 Tedenski plan treningov ({week_days[0].strftime('%d.%m.%Y')} – {week_days[-1].strftime('%d.%m.%Y')})\n"]
    md.append("| Dan | Datum | Planiran trening | Šport & Cona | Trajanje & Cilj | Opravljen trening (Dejansko) |")
    md.append("|---|---|---|---|---|---|")
    for d_obj in week_days:
        d_s = d_obj.strftime("%Y-%m-%d")
        d_name = slo_days[d_obj.weekday()]
        cw = cal_by_date.get(d_s)
        act_list = acts_by_date.get(d_s, [])
        if cw:
            p_title = f"**{cw.get('title')}**" + (f" — _{cw.get('description')}_" if cw.get("description") else "")
            p_sport = f"{cw.get('sport', 'Kolo')} ({cw.get('type', '')})"
            p_dur = f"{cw.get('duration_min', 0)} min | {cw.get('target_power_w', '--')} ({cw.get('target_tss', 0)} TSS)"
        else:
            p_title, p_sport, p_dur = "— Ni plana —", "--", "--"
        if act_list:
            a_strs = [f"✅ {a['name']} ({a['dur_min']} min, {a['tss']:.0f} TSS)" for a in act_list]
            act_col = "<br>".join(a_strs)
        else:
            act_col = "—"
        md.append(f"| **{d_name}** | {d_s} | {p_title} | {p_sport} | {p_dur} | {act_col} |")
    return "\n".join(md)


def export_calendar_ics(workouts: list) -> bytes:
    """Ustvari standardno .ics (iCalendar) datoteko za uvoz planiranih treningov v telefon (Google / Apple Calendar)."""
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Garmin AI Osebni Trener//Rest or Train Pro//SL",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]
    now_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    for w in workouts:
        d_str = str(w.get("date") or "")[:10]
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", d_str):
            continue
        dt_clean = d_str.replace("-", "")
        uid = f"{w.get('id') or uuid.uuid4().hex}@garmin-ai-trener"
        title = f"[{w.get('sport', 'Trening')}] {w.get('title', 'Trening')} ({w.get('duration_min', 0)} min)"
        desc_parts = [
            f"Tip / Cona: {w.get('type', '')}",
            f"Planirano trajanje: {w.get('duration_min', 0)} min",
            f"Ciljna moč / tempo: {w.get('target_power_w', '--')}",
            f"Ciljni TSS: {w.get('target_tss', 0)}",
            f"Navodila: {w.get('description', '')}",
        ]
        desc_escaped = "\\n".join(desc_parts).replace(",", "\\,").replace(";", "\\;")
        title_escaped = title.replace(",", "\\,").replace(";", "\\;")
        lines.extend([
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{now_utc}",
            f"DTSTART;VALUE=DATE:{dt_clean}",
            f"SUMMARY:{title_escaped}",
            f"DESCRIPTION:{desc_escaped}",
            "END:VEVENT",
        ])
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines).encode("utf-8")


# ==============================================================================
# 6. GLAVNI UI MODUL ZA ZAVIHEK "🤖 AI TRENER & PLANIRANJE" (PART 1 SPECIFIKACIJA)
# ==============================================================================
SLO_WEEKDAYS = ["Ponedeljek", "Torek", "Sreda", "Četrtek", "Petek", "Sobota", "Nedelja"]
SLO_MONTHS = [
    "Januar", "Februar", "Marec", "April", "Maj", "Junij",
    "Julij", "Avgust", "September", "Oktober", "November", "December"
]

WORKOUT_TYPE_COLORS = {
    "Z2 Aerobno": "#2563eb",
    "Sweet Spot": "#d97706",
    "Threshold / Prag": "#ea580c",
    "VO2max": "#dc2626",
    "Regeneracija / Z1": "#10b981",
    "Počitek": "#64748b",
    "Tek Z2": "#0891b2",
    "Tek Intervali": "#e11d48",
    "Moč / Fitnes": "#7c3aed",
}


def _group_actual_activities_by_date(activities_all: list, user_cfg: dict, compute_run_load_fn=None, get_sport_cat_fn=None) -> dict:
    """Združi dejansko opravljene Garmin aktivnosti po datumu (YYYY-MM-DD)."""
    by_date = {}
    for a in (activities_all or []):
        d_s = str(a.get("startTimeLocal") or "")[:10]
        if not d_s:
            continue
        m_sec = float(a.get("movingDuration") or a.get("duration") or 0)
        dur_min = int(round(m_sec / 60.0))
        avg_w = a.get("avgPower") or a.get("averagePower") or 0
        np_w = a.get("normPower") or a.get("normalizedPower") or avg_w
        avg_hr = a.get("averageHR")
        dist_km = round((a.get("distance") or 0) / 1000.0, 1)
        tss_v = compute_activity_tss(a, user_cfg, compute_run_load_fn, get_sport_cat_fn)
        sp = get_sport_cat_fn(a.get("activityType", {}).get("typeKey"), a.get("activityName", "")) if get_sport_cat_fn else "Trening"
        by_date.setdefault(d_s, []).append({
            "id": str(a.get("activityId")),
            "name": a.get("activityName", "Trening"),
            "sport": sp,
            "dur_min": dur_min,
            "dist_km": dist_km,
            "avg_w": int(round(avg_w)) if avg_w else 0,
            "np_w": int(round(np_w)) if np_w else 0,
            "avg_hr": int(round(avg_hr)) if avg_hr else 0,
            "tss": round(tss_v, 0),
        })
    return by_date


def _evaluate_day_compliance(planned: dict, actual_list: list, day_date: date, today_date: date) -> tuple[str, str]:
    """Primerja planiran in opravljen trening za dan ter vrne (status_badge, barva)."""
    is_rest_planned = False
    if planned:
        t_low = (str(planned.get("type", "")) + " " + str(planned.get("title", ""))).lower()
        is_rest_planned = ("počitek" in t_low or "prost" in t_low or int(planned.get("duration_min") or 0) == 0)

    total_act_min = sum(x["dur_min"] for x in actual_list)
    total_act_tss = sum(x["tss"] for x in actual_list)

    if not planned:
        if actual_list:
            return f"✅ Opravljeno ({total_act_min} min · {int(total_act_tss)} TSS)", "#16a34a"
        return ("—", "#94a3b8")

    if is_rest_planned:
        if not actual_list or total_act_min <= 30:
            return ("✅ Počitek opravljen", "#16a34a") if day_date <= today_date else ("⏳ Planiran počitek", "#64748b")
        return (f"⚠️ Trening na dan počitka ({total_act_min} min)", "#d97706")

    plan_min = max(1, int(planned.get("duration_min") or 60))
    plan_tss = max(1, int(planned.get("target_tss") or 50))

    if not actual_list:
        if day_date < today_date:
            return ("❌ Izpuščen trening", "#dc2626")
        elif day_date == today_date:
            return ("⏳ Danes na sporedu", "#2563eb")
        else:
            return ("📅 Načrtovano", "#64748b")

    dur_ratio = total_act_min / float(plan_min)
    tss_ratio = total_act_tss / float(plan_tss) if plan_tss > 0 else dur_ratio
    if 0.78 <= dur_ratio <= 1.30 or 0.75 <= tss_ratio <= 1.35:
        return (f"✅ Opravljeno ({total_act_min}/{plan_min} min · {int(total_act_tss)}/{plan_tss} TSS)", "#16a34a")
    return (f"⚠️ Prilagojeno ({total_act_min}/{plan_min} min · {int(total_act_tss)}/{plan_tss} TSS)", "#d97706")


def render_ai_coach_and_calendar_tab(
    user_cfg: dict,
    save_user_config_fn,
    user_journal: dict,
    df_daily_all: pd.DataFrame,
    activities_all: list,
    raw_data: dict,
    user_dir: str,
    project_dir: str,
    display_name: str,
    today_date: date,
    get_sport_cat_fn,
    compute_fixed_blocks_fn,
    compute_run_blocks_fn,
    compute_run_load_fn,
    parse_stream_fn,
):
    """
    Izriše celoten zavihek '🤖 AI Trener & Planiranje (Rest or Train Pro)' s 3 podzavihki:
    - Sub-tab 1: 💬 Klepet z AI Trenerjem (Readiness bar + Quick Action gumbi zgoraj, levi stolpec [1] upravljalnik pogovorov, desni stolpec [3] drsno okno pogovora + pripet st.chat_input spodaj)
    - Sub-tab 2: 📅 Pametni Koledar Treningov (vizualni tedenski/mesečni koledar in ročni vnos)
    - Sub-tab 3: 🧠 Moja trenerska pravila, cilji in navodila (@kolesarski-trening)
    """
    # Sticky CSS za pripeto vnosno polje na dnu pogovora
    st.markdown(
        """<style>
        div[data-testid="stChatInput"] {
            position: sticky !important;
            bottom: 0 !important;
            z-index: 999 !important;
            background: var(--background-color, #ffffff) !important;
            padding-top: 6px !important;
            padding-bottom: 6px !important;
            border-top: 1px solid rgba(148, 163, 184, 0.25) !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    st.header("🤖 AI Osebni Trener & Pametni Koledar (Rest or Train Pro)")
    _fb_toast = st.session_state.pop("_pending_ai_fallback_toast", None)
    if _fb_toast:
        st.toast(_fb_toast, icon="⚡")
        st.caption(f"⚡ {_fb_toast}")

    # Hibridni Gemini & Claude (Anthropic) API ključi in izbira modela
    auth_info = resolve_gemini_auth(user_cfg, project_dir)
    avail_models = auth_info.get("available_options", get_available_model_options(user_cfg))
    with st.container(border=True):
        col_k1, col_k2 = st.columns([3, 2])
        with col_k1:
            if not auth_info["has_personal_key"]:
                st.info(
                    "ℹ️ Uporabljaš privzeti hitri model (**Gemini Flash - Privzeti**). "
                    "Za odklep **Gemini Pro** ali **Claude Haiku / Sonnet** modelov vnesi svoj osebni API ključ v stranski vrstici ali v spodnjem razdelku."
                )
            else:
                active_keys_labels = []
                if auth_info.get("has_personal_gemini_key"):
                    active_keys_labels.append("Gemini API")
                if auth_info.get("has_personal_claude_key"):
                    active_keys_labels.append("🟣 Claude API (Prompt Caching aktiven)")
                st.success(
                    f"🔑 **Aktivni osebni ključi ({', '.join(active_keys_labels)})** — trenutno izbran model: `{auth_info['display_model_id']}`."
                )

            with st.expander("🔑 Hitra nastavitev Gemini / Claude (Anthropic) API ključev", expanded=False):
                render_api_key_help("ai")
                qk_c1, qk_c2 = st.columns(2)
                with qk_c1:
                    q_gem_key = st.text_input(
                        "🔑 Moj osebni Gemini API ključ (opcijsko):",
                        value=user_cfg.get("gemini_api_key", ""),
                        type="password",
                        key="ai_tab_quick_gemini_key",
                    )
                with qk_c2:
                    q_cld_key = st.text_input(
                        "🟣 Claude (Anthropic) API ključ (opcijsko – npr. za Claude Haiku / Sonnet):",
                        value=user_cfg.get("claude_api_key", ""),
                        type="password",
                        key="ai_tab_quick_claude_key",
                    )
                if st.button("💾 Shrani API ključe", key="btn_save_quick_ai_keys", use_container_width=True):
                    user_cfg["gemini_api_key"] = q_gem_key.strip()
                    user_cfg["personal_gemini_key_set"] = bool(q_gem_key.strip())
                    user_cfg["claude_api_key"] = q_cld_key.strip()
                    # Če je uporabnik vnesel samo Claude ključ in je bil prej izbran Gemini, samodejno izberi Claude 3.5 Haiku
                    new_avail = get_available_model_options(user_cfg)
                    if q_cld_key.strip() and not q_gem_key.strip() and user_cfg.get("ai_model_choice") not in CLAUDE_MODEL_OPTIONS:
                        user_cfg["ai_model_choice"] = CLAUDE_MODEL_OPTIONS[0]
                    elif user_cfg.get("ai_model_choice") not in new_avail:
                        user_cfg["ai_model_choice"] = new_avail[0]
                    save_user_config_fn(user_cfg)
                    st.toast("✅ API ključi so shranjeni!")
                    st.rerun()

        with col_k2:
            if auth_info["has_personal_key"]:
                cur_choice = user_cfg.get("ai_model_choice", avail_models[0])
                idx_m = avail_models.index(cur_choice) if cur_choice in avail_models else 0
                new_choice = st.selectbox(
                    "🧠 Izberi AI model trenerja:",
                    avail_models,
                    index=idx_m,
                    key="ai_tab_model_selector",
                )
                custom_m = user_cfg.get("ai_custom_model", "")
                custom_m = st.text_input(
                    "Ali vpiši točen API ID modela po meri (opcijsko):",
                    value=str(custom_m),
                    placeholder="npr. gemini-4-hydrogen",
                    key="ai_tab_custom_model_input",
                    )
                if new_choice != cur_choice or custom_m.strip() != str(user_cfg.get("ai_custom_model", "")).strip():
                    user_cfg["ai_model_choice"] = new_choice
                    user_cfg["ai_custom_model"] = custom_m.strip()
                    save_user_config_fn(user_cfg)
                    auth_info = resolve_gemini_auth(user_cfg, project_dir)
            else:
                st.selectbox(
                    "🧠 Aktivni AI model (zaklenjeno na privzetem ključu):",
                    [MODEL_FLASH_DEFAULT_LABEL],
                    index=0,
                    disabled=True,
                    key="ai_tab_model_locked",
                )

    # Naloži koledar, večnitno zgodovino pogovorov in osebna trenerska pravila
    calendar_workouts = load_training_calendar(user_dir)
    cal_by_date = {str(w.get("date", ""))[:10]: w for w in calendar_workouts}
    acts_by_date = _group_actual_activities_by_date(activities_all, user_cfg, compute_run_load_fn, get_sport_cat_fn)
    chat_store = load_ai_chat_store(user_dir)
    coach_rules_data = load_custom_coach_rules(user_dir, user_cfg)

    readiness = compute_rest_or_train_readiness(
        df_daily_all=df_daily_all,
        activities_all=activities_all,
        user_cfg=user_cfg,
        user_journal=user_journal,
        ref_date_str=today_date.strftime("%Y-%m-%d"),
        compute_run_load_fn=compute_run_load_fn,
        get_sport_cat_fn=get_sport_cat_fn,
    )
    rm = readiness.get("metrics", {})
    today_str = today_date.strftime("%Y-%m-%d")
    today_disp = today_date.strftime("%d. %m. %Y")
    today_planned = cal_by_date.get(today_str)

    # 3 PODZAVIHKI PO SPECIFIKACIJI
    sub_chat, sub_cal, sub_rules = st.tabs([
        "💬 Klepet z AI Trenerjem",
        "📅 Pametni Koledar Treningov",
        "🧠 Moja trenerska pravila, cilji in navodila (@kolesarski-trening)"
    ])

    # ==================================================================
    # SUB-TAB 1: 💬 KLEPET Z AI TRENERJEM
    # (Readiness bar + Quick Action gumbi zgoraj, [1, 3] Thread Manager levo + drsni pogovor & pripet chat_input desno)
    # ==================================================================
    with sub_chat:
        # 1. READINESS BAR NA VRHU
        st.markdown(
            f"""
            <div style="background:{readiness['bg_color']}; border-left: 6px solid {readiness['color']}; padding: 14px 18px; border-radius: 10px; margin-bottom: 12px;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div>
                        <span style="font-size:1.3rem; font-weight:800; color:{readiness['color']};">{readiness['badge']}</span>
                        <div style="font-size:1.0rem; font-weight:600; margin-top:3px;">{readiness['headline']}</div>
                    </div>
                    <div style="text-align:right; font-size:0.88rem; opacity:0.9;">
                        📅 Referenčni datum: <b>{readiness.get('date', today_str)}</b><br>
                        🎯 Današnji plan v koledarju: <b>{today_planned.get('title') + ' (' + str(today_planned.get('duration_min')) + ' min)' if today_planned else 'Ni vpisanega plana'}</b>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        with rc1:
            hrv_val_str = f"{int(rm['hrv_today'])} ms" if pd.notna(rm.get("hrv_today")) and rm.get("hrv_today") is not None else "--"
            hrv_z_val = float(rm.get("hrv_z") or 0.0)
            hrv_sub = f"7d: {rm.get('hrv_7d') or '--'} | 30d: {rm.get('hrv_30d') or '--'} (Z={hrv_z_val:+.2f})"
            st.metric("💓 Nočni HRV vs. Baza", hrv_val_str, hrv_sub, delta_color="off")
        with rc2:
            rhr_val_str = f"{int(rm['rhr_today'])} bpm" if pd.notna(rm.get("rhr_today")) and rm.get("rhr_today") is not None else "--"
            rhr_d_val = float(rm.get("rhr_delta") or 0.0)
            rhr_sub = f"{rhr_d_val:+.1f} bpm proti 30d ({rm.get('rhr_base') or '--'} bpm)"
            st.metric("🫀 Jutranji RHR", rhr_val_str, rhr_sub, delta_color="off")
        with rc3:
            sh_v = rm.get("sleep_hours")
            sc_v = rm.get("sleep_score")
            sh_str = f"{float(sh_v):.1f} h" if pd.notna(sh_v) and sh_v is not None else "--"
            sc_str = f"Sleep Score: {int(sc_v)}/100" if pd.notna(sc_v) and sc_v is not None else "Točke: --"
            st.metric("💤 Spanec & Kakovost", sh_str, sc_str, delta_color="off")
        with rc4:
            acwr_v = rm.get("acwr", 1.0)
            st.metric("⚡ Akutna obremenitev (ACWR)", f"{acwr_v:.2f}", f"7d: {int(rm.get('tss_7d', 0))} TSS | Včeraj: {int(rm.get('tss_yesterday', 0))} TSS", delta_color="off")
        with rc5:
            st.metric("🧠 Sestava Readiness", f"{readiness['score']} / 100", f"HRV {rm.get('hrv_pts', 0):.0f} · Spanec {rm.get('sleep_pts', 0):.0f} · Load {rm.get('load_pts', 0):.0f}", delta_color="off")

        if readiness.get("reasons"):
            with st.expander("🔍 Podrobna fiziološka obrazložitev današnjega semaforja (Rest or Train)", expanded=False):
                for r_txt in readiness["reasons"]:
                    st.markdown(f"- {r_txt}")

        # 2. HITRI 1-KLIK UKAZI (VSAK ODPRE NOVO NAMENSKO POGOVORNO NIT!)
        st.markdown("##### ⚡ Hitri ukazi za AI Trenerja (vsak klik samodejno ustvari novo namensko nit pogovora)")
        q_col1, q_col2, q_col3, q_col4 = st.columns(4)

        triggered_prompt = None
        triggered_thread_title = None
        triggered_topic_tag = "Splošno"
        auto_apply_calendar = False

        with q_col1:
            if st.button("🌅 Jutranja ocena & Prilagoditev današnjega plana", use_container_width=True, key="btn_ai_morning_eval"):
                plan_desc = (
                    f"V koledarju imam danes ({today_str}) načrtovano: '{today_planned.get('title')}' ({today_planned.get('type')}, {today_planned.get('duration_min')} min, cilj {today_planned.get('target_power_w')}, {today_planned.get('target_tss')} TSS)."
                    if today_planned
                    else f"V koledarju za danes ({today_str}) še nimam vpisanega treninga."
                )
                triggered_thread_title = f"🌅 Jutranji pregled – {today_disp}"
                triggered_topic_tag = "Jutranji pregled"
                triggered_prompt = (
                    f"🌅 **Jutranja ocena in prilagoditev današnjega plana ({today_str}):**\n"
                    f"Analiziraj moj zadnji spanec, nočni HRV, RHR in kumulativno utrujenost zadnjih 14 dni. "
                    f"{plan_desc}\n"
                    f"Odgovori po vseh 7 točkah jutranjega pregleda (trenutna pripravljenost, stopnja utrujenosti, ustreznost plana, ali naj ostane nespremenjen / prilagojen / Z2 / počitek) "
                    f"ter predpiši današnji trening in ga na koncu zapiši v `<calendar_update>` blok za datum `{today_str}`."
                )
                auto_apply_calendar = True

        with q_col2:
            if st.button("🏅 Analiziraj zadnji opravljen trening", use_container_width=True, key="btn_ai_last_workout"):
                latest_act = activities_all[0] if activities_all else {}
                l_nm = latest_act.get("activityName", "Trening")
                l_dt_raw = str(latest_act.get("startTimeLocal", today_str))[:10]
                try:
                    l_dt_short = datetime.strptime(l_dt_raw, "%Y-%m-%d").strftime("%d. %m.")
                except Exception:
                    l_dt_short = today_date.strftime("%d. %m.")
                triggered_thread_title = f"🏅 Analiza treninga: {l_nm[:28]} ({l_dt_short})"
                triggered_topic_tag = "Trening"
                triggered_prompt = (
                    f"🏅 **Globoka fiziološka analiza zadnjega opravljenega treninga ({l_nm} – {l_dt_raw}):**\n"
                    "Podrobno analiziraj moj zadnji opravljen trening iz konteksta (moč/tempo po 10-minutnih odsekih in intervalih, srčni utrip, "
                    "aerobna učinkovitost EF, srčni drift / decoupling, W/kg, vpliv temperature ter kakovost zadnjih intervalov). "
                    "Odgovori po 5 točkah analize treninga: ali sem dosegel namen, kakovost izvedbe, kaj se naučimo, vpliv na naslednji trening in sprememba pripravljenosti."
                )

        with q_col3:
            if st.button("📅 Sestavi in vpiši tedenski plan v koledar", use_container_width=True, key="btn_ai_weekly_plan"):
                start_plan_dt = today_date
                end_plan_dt = today_date + timedelta(days=6)
                triggered_thread_title = f"📅 Tedenski plan – {today_disp}"
                triggered_topic_tag = "Planiranje"
                triggered_prompt = (
                    f"📅 **Sestavi optimalen 7-dnevni plan treningov ({start_plan_dt.strftime('%Y-%m-%d')} do {end_plan_dt.strftime('%Y-%m-%d')}) in ga vpiši v koledar:**\n"
                    f"Glede na mojo trenutno pripravljenost (Readiness {readiness['score']}/100), preteklo 14-dnevno obremenitev, moj športni profil in ciljne tekme "
                    f"sestavi strukturiran 7-dnevni plan za vsak dan posebej od `{start_plan_dt.strftime('%Y-%m-%d')}` do `{end_plan_dt.strftime('%Y-%m-%d')}` "
                    f"(okvir ~10–12 ur/teden prilagojen regeneraciji, obilica Z2 baze, 2 ključna treninga – dolg Threshold in zmerni VO2max – ter dovolj počitka). "
                    f"Za vsak dan navedi jasen strukturiran opis, na koncu odgovora pa OBVEZNO dodaj `<calendar_update>` JSON blok z vsemi 7 dnevi!"
                )
                auto_apply_calendar = True

        with q_col4:
            if st.button("📈 Ocena trenutne forme (FTP, VO2max, Klanci & Trendi)", use_container_width=True, key="btn_ai_form_eval"):
                triggered_thread_title = f"📈 Ocena trenutne forme – {today_disp}"
                triggered_topic_tag = "Ocena forme"
                triggered_prompt = (
                    "📈 **Celovita ocena trenutne forme, napredka in pripravljenosti na cilje:**\n"
                    "Ocenjuj moj večtedenski trend moči (5m, 10m, 20m, 30m, 40–45m MMP, FTP in W/kg) oz. tekaškega tempa, gibanje aerobne učinkovitosti (EF = NP/HR), "
                    "sposobnost vožnje v dolge klance, ponavljanje naporov in regeneracijo. "
                    "Kje sem trenutno glede na moje nastavljene cilje (npr. Rogla), kaj je moja glavna omejitev in na kaj se morava osredotočiti v naslednjih 3 tednih?"
                )

        if triggered_prompt:
            new_th = create_new_thread_dict(
                title=triggered_thread_title,
                topic_tag=triggered_topic_tag,
            )
            chat_store["threads"][new_th["id"]] = new_th
            chat_store["active_thread_id"] = new_th["id"]
            st.session_state["_force_selected_thread_key"] = new_th["id"]
            save_ai_chat_store(user_dir, chat_store)

            full_ctx = build_full_athlete_context(
                user_cfg=user_cfg,
                user_journal=user_journal,
                df_daily_all=df_daily_all,
                activities_all=activities_all,
                raw_data=raw_data,
                calendar_workouts=calendar_workouts,
                readiness_info=readiness,
                today_date=today_date,
                get_sport_cat_fn=get_sport_cat_fn,
                compute_fixed_blocks_fn=compute_fixed_blocks_fn,
                compute_run_blocks_fn=compute_run_blocks_fn,
                compute_run_load_fn=compute_run_load_fn,
                parse_stream_fn=parse_stream_fn,
                user_dir=user_dir,
                user_question=str(triggered_prompt),
                is_followup=False,
                chat_history=[],
            )
            sys_prompt = build_coach_system_prompt(user_cfg, display_name, user_dir=user_dir)
            combined_prompt = f"{full_ctx}\n\n=== VPRAŠANJE / UKAZ ŠPORTNIKA ===\n{triggered_prompt}"

            retry_status_box = st.empty()
            with st.spinner(f"🤖 AI Trener ({auth_info['display_model_id']}) pripravlja analizo v novi niti »{triggered_thread_title}«..."):
                ok, reply_raw, used_model = call_gemini_hybrid(
                    user_prompt=combined_prompt,
                    user_cfg=user_cfg,
                    project_dir=project_dir,
                    system_instruction=sys_prompt,
                    chat_history=[],
                    thread_summary="",
                    status_placeholder=retry_status_box,
                )

            if ok:
                tok_usage = get_last_token_usage()
                clean_reply, cal_updates = extract_calendar_updates_from_text(reply_raw)
                clean_reply, mem_facts = extract_memory_updates_from_text(clean_reply)
                for f_txt in mem_facts:
                    if add_coach_memory_fact(user_dir, f_txt):
                        st.toast(f"🧠 Trener si je zapomnil: '{f_txt}'", icon="🧠")
                applied_n = 0
                if cal_updates and auto_apply_calendar:
                    applied_n = upsert_calendar_workouts(user_dir, cal_updates, replace_same_date=True)
                    calendar_workouts = load_training_calendar(user_dir)
                    cal_by_date = {str(w.get("date", ""))[:10]: w for w in calendar_workouts}
                    st.toast(f"📅 AI Trener je samodejno vpisal/posodobil {applied_n} treningov v tvoj koledar!")

                append_exchange_to_thread(
                    user_dir=user_dir,
                    store=chat_store,
                    thread_id=new_th["id"],
                    user_content=triggered_prompt,
                    assistant_content=clean_reply or reply_raw,
                    used_model=used_model,
                    calendar_updates=cal_updates,
                    auto_applied=applied_n,
                    auto_title_from_user=False,
                    token_usage=tok_usage,
                )
                st.session_state.pop("ai_last_failed_exchange", None)
            else:
                # NE shrani 503/napake kot trajnega odgovora v zgodovino – shrani v session_state za gumb "🔄 Poskusi znova"
                st.session_state["ai_last_failed_exchange"] = {
                    "thread_id": new_th["id"],
                    "user_prompt": triggered_prompt,
                    "error_msg": reply_raw,
                    "auto_apply_calendar": auto_apply_calendar,
                    "auto_title": False,
                }
            st.rerun()

        st.markdown("---")

        # 3. POSTAVITEV [1, 3]: LEVO UPRAVLJALNIK POGOVOROV, DESNO DRSNO OKNO KLEPETA + PRIPET VNOS SPODAJ
        threads_dict = chat_store.get("threads", {})
        sorted_thread_ids = sorted(
            threads_dict.keys(),
            key=lambda tid: threads_dict[tid].get("updated_at", ""),
            reverse=True,
        )
        # Če je bil v prejšnjem koraku sprožen preklop pogovora, ga uveljavi PRED izrisom widgeta selected_thread_key
        forced_tid = st.session_state.pop("_force_selected_thread_key", None)
        if forced_tid and forced_tid in threads_dict:
            chat_store["active_thread_id"] = forced_tid
            save_ai_chat_store(user_dir, chat_store)

        active_tid = chat_store.get("active_thread_id")
        if not active_tid or active_tid not in threads_dict:
            active_tid = sorted_thread_ids[0]
            chat_store["active_thread_id"] = active_tid

        # Sinhroniziraj session_state ključa izbirnika z active_tid PREDEN se ustvari st.selectbox(key="selected_thread_key")
        if st.session_state.get("selected_thread_key") != active_tid:
            st.session_state["selected_thread_key"] = active_tid

        col_left_mgr, col_right_chat = st.columns([1.15, 3.0], gap="medium")

        # --- LEVI STOLPEC [1]: CONVERSATION MANAGER UI ---
        with col_left_mgr:
            st.markdown("##### 🗂️ Moji pogovori")
            if st.button("➕ Začni nov pogovor", type="primary", use_container_width=True, key="btn_new_ai_thread"):
                fresh_th = create_new_thread_dict(
                    title=f"💬 Nov pogovor ({datetime.now().strftime('%d. %m. %Y %H:%M')})",
                    topic_tag="Splošno",
                )
                chat_store["threads"][fresh_th["id"]] = fresh_th
                chat_store["active_thread_id"] = fresh_th["id"]
                st.session_state["selected_thread_key"] = fresh_th["id"]
                st.session_state["_force_selected_thread_key"] = fresh_th["id"]
                st.session_state["confirm_delete_thread_id"] = None
                save_ai_chat_store(user_dir, chat_store)
                st.rerun()

            def _fmt_thread_option(tid: str) -> str:
                t_obj = threads_dict.get(tid, {})
                ttl = t_obj.get("title", tid)
                upd = str(t_obj.get("updated_at", ""))[:16]
                return f"{ttl} ({upd})"

            def _on_thread_select_change():
                chosen = st.session_state.get("selected_thread_key")
                if chosen and chosen in chat_store.get("threads", {}):
                    chat_store["active_thread_id"] = chosen
                    st.session_state["confirm_delete_thread_id"] = None
                    save_ai_chat_store(user_dir, chat_store)

            sel_tid = st.selectbox(
                "Izberi aktivni pogovor:",
                sorted_thread_ids,
                format_func=_fmt_thread_option,
                key="selected_thread_key",
                on_change=_on_thread_select_change,
            )
            if sel_tid != active_tid and sel_tid in threads_dict:
                chat_store["active_thread_id"] = sel_tid
                active_tid = sel_tid
                save_ai_chat_store(user_dir, chat_store)
                st.rerun()

            active_thread = chat_store["threads"][chat_store["active_thread_id"]]

            # Preimenovanje, urejanje teme in VARNO BRISANJE Z DVOKORAČNO POTRDITVIJO v zloženem meniju
            with st.expander("⚙️ Možnosti pogovora (Preimenuj / Izbriši)", expanded=bool(st.session_state.get("confirm_delete_thread_id") or st.session_state.get("confirm_clear_all_threads"))):
                ren_title = st.text_input(
                    "Naslov pogovora:",
                    value=active_thread.get("title", ""),
                    key=f"ren_title_inp_{active_tid}",
                )
                cur_tag = active_thread.get("topic_tag", "Splošno")
                ren_tag = st.selectbox(
                    "Oznaka teme:",
                    TOPIC_TAGS,
                    index=TOPIC_TAGS.index(cur_tag) if cur_tag in TOPIC_TAGS else len(TOPIC_TAGS) - 1,
                    key=f"ren_tag_sel_{active_tid}",
                )
                if st.button("💾 Shrani naslov", use_container_width=True, key=f"btn_save_ren_{active_tid}"):
                    if ren_title.strip():
                        active_thread["title"] = ren_title.strip()
                    active_thread["topic_tag"] = ren_tag
                    save_ai_chat_store(user_dir, chat_store)
                    st.toast("✅ Naslov pogovora posodobljen!")
                    st.rerun()

                st.markdown("---")
                st.caption("🗑️ **Varno upravljanje izbrisa pogovora:**")
                if st.button("🗑️ Izbriši pogovor", use_container_width=True, key="btn_del_active_thread"):
                    st.session_state["confirm_delete_thread_id"] = active_tid
                    st.rerun()

                if st.session_state.get("confirm_delete_thread_id") == active_tid:
                    st.warning(f'⚠️ Ali res želiš trajno izbrisati pogovor "{active_thread.get("title", active_tid)}"?')
                    c_yes, c_no = st.columns(2)
                    with c_yes:
                        if st.button("✅ Da, trajno izbriši", type="primary", use_container_width=True, key=f"btn_confirm_yes_del_{active_tid}"):
                            if len(chat_store["threads"]) > 1:
                                chat_store["threads"].pop(active_tid, None)
                                rem_ids = sorted(
                                    chat_store["threads"].keys(),
                                    key=lambda k: chat_store["threads"][k].get("updated_at", ""),
                                    reverse=True,
                                )
                                chat_store["active_thread_id"] = rem_ids[0]
                            else:
                                fresh_th = create_new_thread_dict()
                                chat_store["threads"] = {fresh_th["id"]: fresh_th}
                                chat_store["active_thread_id"] = fresh_th["id"]
                            st.session_state["_force_selected_thread_key"] = chat_store["active_thread_id"]
                            st.session_state["confirm_delete_thread_id"] = None
                            save_ai_chat_store(user_dir, chat_store)
                            st.toast("🗑️ Pogovor je bil trajno izbrisan.")
                            st.rerun()
                    with c_no:
                        if st.button("❌ Prekliči", use_container_width=True, key=f"btn_confirm_no_del_{active_tid}"):
                            st.session_state["confirm_delete_thread_id"] = None
                            st.rerun()

                if st.button("🧹 Počisti vse pogovore", use_container_width=True, key="btn_clear_all_threads"):
                    st.session_state["confirm_clear_all_threads"] = True
                    st.rerun()

                if st.session_state.get("confirm_clear_all_threads"):
                    st.warning("⚠️ Ali res želiš trajno izbrisati VSE pogovorne niti?")
                    ca_y, ca_n = st.columns(2)
                    with ca_y:
                        if st.button("✅ Da, počisti vse", type="primary", use_container_width=True, key="btn_confirm_clear_all_yes"):
                            fresh_th = create_new_thread_dict()
                            chat_store = {"active_thread_id": fresh_th["id"], "threads": {fresh_th["id"]: fresh_th}}
                            st.session_state["_force_selected_thread_key"] = fresh_th["id"]
                            st.session_state["confirm_clear_all_threads"] = False
                            save_ai_chat_store(user_dir, chat_store)
                            st.toast("Vsi pogovori so bili počiščeni.")
                            st.rerun()
                    with ca_n:
                        if st.button("❌ Prekliči", use_container_width=True, key="btn_confirm_clear_all_no"):
                            st.session_state["confirm_clear_all_threads"] = False
                            st.rerun()

            st.caption("📜 **Seznam vseh pogovorov (Datum, Naslov & Povzetek):**")
            with st.container(height=390, border=True):
                for tid_item in sorted_thread_ids:
                    th_item = threads_dict[tid_item]
                    is_curr = (tid_item == active_tid)
                    n_ex = len(group_messages_into_exchanges(th_item.get("messages", [])))
                    preview_txt = th_item.get("summary_preview") or "Brez sporočil"
                    border_style = "2px solid #2563eb" if is_curr else "1px solid rgba(148,163,184,0.3)"
                    bg_style = "rgba(37, 99, 235, 0.08)" if is_curr else "transparent"
                    st.markdown(
                        f"""<div style="border:{border_style}; background:{bg_style}; border-radius:8px; padding:8px 10px; margin-bottom:6px;">
                            <div style="font-size:0.75rem; opacity:0.78;">📅 {th_item.get('updated_at', '')} · <b>{th_item.get('topic_tag', 'Splošno')}</b> ({n_ex} odgovorov)</div>
                            <div style="font-size:0.9rem; font-weight:700; margin:2px 0;">{'🔹 ' if is_curr else ''}{th_item.get('title', tid_item)}</div>
                            <div style="font-size:0.78rem; opacity:0.85;">{preview_txt[:85]}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    if not is_curr:
                        def _switch_to_thread_cb(target_tid: str = tid_item):
                            chat_store["active_thread_id"] = target_tid
                            st.session_state["selected_thread_key"] = target_tid
                            st.session_state["_force_selected_thread_key"] = target_tid
                            st.session_state["confirm_delete_thread_id"] = None
                            save_ai_chat_store(user_dir, chat_store)

                        if st.button(
                            "Odpri ta pogovor",
                            key=f"open_th_btn_{tid_item}",
                            use_container_width=True,
                            on_click=_switch_to_thread_cb,
                            kwargs={"target_tid": tid_item},
                        ):
                            st.rerun()

        # --- DESNI STOLPEC [3]: AKTIVNI POGOVOR (ZADNJIH 3 ODGOVOROV RAZŠIRJENIH) + PRIPET CHAT INPUT ---
        with col_right_chat:
            active_thread = chat_store["threads"][chat_store["active_thread_id"]]
            active_messages = active_thread.get("messages", [])
            exchanges = group_messages_into_exchanges(active_messages)
            total_ex = len(exchanges)

            hdr_c1, hdr_c2 = st.columns([3.2, 1.8])
            with hdr_c1:
                st.markdown(
                    f"##### 💬 {active_thread.get('title', 'Pogovor')}  \n"
                    f"<span style='font-size:0.82rem; opacity:0.8;'>📌 Tema: <b>{active_thread.get('topic_tag', 'Splošno')}</b> · Ustvarjeno: {active_thread.get('created_at', '')} · Skupaj odgovorov: {total_ex}</span>",
                    unsafe_allow_html=True,
                )
            with hdr_c2:
                auto_write_chk = st.checkbox(
                    "📅 Samodejno vpiši AI plan v koledar",
                    value=True,
                    key="ai_chat_auto_write_cal",
                    help="Če AI trener v odgovoru predlaga spremembo koledarja (<calendar_update>), jo takoj zapiše v tvoj koledar.",
                )

            # PART 2 UX NADGRADNJI:
            # 1) 📌 Pripni pomemben dogovor v spomin trenerja (Pin to Coach Memory)
            # 2) 🔍 Predogled konteksta (Kaj AI trener trenutno vidi)
            ux_c1, ux_c2 = st.columns(2)
            with ux_c1:
                with st.expander("🧠 Dolgoročni spomin trenerja - pomembna dejstva (coach_memory.json)", expanded=False):
                    st.caption("Trajni spomin trenerja (urnik, faks, poškodbe, oprema, specifični odzivi telesa). AI trener si ob pogovoru samodejno zapomni nova dejstva (oznaka `<memory_update>`), tukaj pa jih lahko poljubno pregleduješ, dodajaš ali brišeš.")
                    mem_obj_ui = load_coach_memory(user_dir) if user_dir else {}
                    current_facts = mem_obj_ui.get("facts") or []

                    pin_inp = st.text_input(
                        "Novo dejstvo za trajni spomin:",
                        placeholder="Npr. Ob petkih imam faks – uradni dan počitka",
                        key="ai_pin_memory_input",
                    )
                    p_b1, p_b2 = st.columns([1.4, 1.6])
                    with p_b1:
                        if st.button("➕ Dodaj v spomin", type="primary", use_container_width=True, key="btn_add_pin_mem"):
                            if pin_inp and pin_inp.strip():
                                if add_coach_memory_fact(user_dir, pin_inp.strip()):
                                    st.toast("🧠 Dejstvo shranjeno v trajni spomin!")
                                st.rerun()
                    with p_b2:
                        if st.button("⚡ Pripni: Petek = Rest Day", use_container_width=True, key="btn_quick_pin_fri"):
                            if add_coach_memory_fact(user_dir, "Ob petkih imam nadomeščanje na faksu – uradni Rest Day."):
                                st.toast("🧠 Pripeto: Ob petkih uradni Rest Day!")
                            st.rerun()

                    if current_facts:
                        st.markdown(f"**Aktivna shranjena dejstva ({len(current_facts)} / 35):**")
                        for p_i, p_val in enumerate(current_facts):
                            pc_txt, pc_del = st.columns([4.2, 0.8])
                            with pc_txt:
                                st.caption(f"• {p_val}")
                            with pc_del:
                                if st.button("✖", key=f"del_mem_fact_{p_i}", help="Odstrani to dejstvo iz spomina"):
                                    remove_coach_memory_fact(user_dir, p_i)
                                    st.rerun()
                    else:
                        st.caption("V spominu še ni shranjenih posebnih dejstev.")

            with ux_c2:
                with st.expander("🔍 Predogled konteksta (Kaj AI trener trenutno vidi)", expanded=False):
                    preview_ctx_str = build_full_athlete_context(
                        user_cfg=user_cfg,
                        user_journal=user_journal,
                        df_daily_all=df_daily_all,
                        activities_all=activities_all,
                        raw_data=raw_data,
                        calendar_workouts=calendar_workouts,
                        readiness_info=readiness,
                        today_date=today_date,
                        get_sport_cat_fn=get_sport_cat_fn,
                        compute_fixed_blocks_fn=compute_fixed_blocks_fn,
                        compute_run_blocks_fn=compute_run_blocks_fn,
                        compute_run_load_fn=compute_run_load_fn,
                        parse_stream_fn=parse_stream_fn,
                        user_dir=user_dir,
                    )
                    cross_mem_preview = build_cross_thread_memory_summary(user_dir, user_cfg)
                    st.caption("Celoten strukturiran povzetek (analiza ure bujenja, mikro-intervali iz 1s toka, 45-dnevni treningi, mesečni trendi in spomin vseh pogovorov), ki se pošlje modelu:")
                    st.code(f"=== SPOMIN PRETEKLIH POGOVOROV ===\n{cross_mem_preview}\n\n{preview_ctx_str}", language="markdown")

            retry_exchange_target = None

            # Drsno okno za sporočila (st.container(height=650))
            chat_scroll_box = st.container(height=650, border=True)
            with chat_scroll_box:
                if total_ex == 0 and not st.session_state.get("ai_last_failed_exchange"):
                    st.info(
                        "👋 **Ta pogovorna nit je pripravljena.**  \n"
                        "Spodaj v vedno dostopno polje vpiši svoje vprašanje (npr. *'Kako naj prilagodim trening ta teden?'*, *'Kakšno prehrano priporočaš za 3h Z2 vožnjo?'*) "
                        "ali pa uporabi enega izmed **4 hitrih gumbov zgoraj**, ki odprejo namensko analizo!"
                    )
                else:
                    cutoff_ex_idx = max(0, total_ex - 3)
                    if cutoff_ex_idx > 0:
                        st.caption(
                            f"📂 **Starejši odgovori ({cutoff_ex_idx}) so samodejno zloženi** za hitro preglednost. "
                            f"Klikni na vrstico za celoten izpis. **Zadnji {min(3, total_ex)} odgovori** so razširjeni v celoti."
                        )

                    for ex_idx, ex in enumerate(exchanges):
                        u_msg = ex.get("user")
                        a_msg = ex.get("assistant")
                        ex_uid = (a_msg or u_msg or {}).get("id", str(ex_idx))

                        if ex_idx < cutoff_ex_idx:
                            # STAREJŠA IZMENJAVA -> ZLOŽENA V EXPANDERJU Z 1-VRSTIČNIM POVZETKOM
                            exp_title = format_collapsed_exchange_title(ex)
                            with st.expander(exp_title, expanded=False):
                                if u_msg:
                                    st.markdown(f"**👤 Vprašanje ({u_msg.get('timestamp', '')}):**")
                                    st.markdown(u_msg.get("content", ""))
                                    st.markdown("---")
                                if a_msg:
                                    m_tag = f" · Model: `{a_msg.get('model')}`" if a_msg.get("model") else ""
                                    st.markdown(f"**🤖 Odgovor AI Trenerja ({a_msg.get('timestamp', '')}{m_tag}):**")
                                    st.markdown(a_msg.get("content", ""))
                                    _tok_cap = format_token_usage_caption(a_msg.get("token_usage"))
                                    if _tok_cap:
                                        st.caption(_tok_cap)
                                    c_upds = a_msg.get("calendar_updates") or []
                                    if c_upds:
                                        if st.button(
                                            f"📅 Vpiši / Posodobi ta plan v moj koledar ({len(c_upds)} treningov)",
                                            key=f"btn_apply_coll_{active_tid}_{ex_uid}_{ex_idx}",
                                            type="primary",
                                        ):
                                            n_up = upsert_calendar_workouts(user_dir, c_upds, replace_same_date=True)
                                            st.success(f"Uspešno vpisanih {n_up} treningov v koledar!")
                                            st.rerun()

                                act_c1, act_c2 = st.columns(2)
                                with act_c1:
                                    if u_msg and st.button(
                                        "🔄 Ponovi vprašanje (Retry)",
                                        key=f"btn_retry_coll_{active_tid}_{ex_uid}_{ex_idx}",
                                        use_container_width=True,
                                    ):
                                        retry_exchange_target = (ex_idx, u_msg.get("content", ""))
                                with act_c2:
                                    if st.button(
                                        "🗑️ Izbriši to sporočilo / par",
                                        key=f"btn_del_coll_{active_tid}_{ex_uid}_{ex_idx}",
                                        use_container_width=True,
                                    ):
                                        delete_exchange_from_thread(user_dir, chat_store, active_tid, ex_idx)
                                        st.toast("🗑️ Sporočilo / par je bil izbrisan iz zgodovine in spomina trenerja.")
                                        st.rerun()
                        else:
                            # ZADNJE 3 IZMENJAVE -> RAZŠIRJENE V CELOTI (EXPANDED)
                            if u_msg:
                                with st.chat_message("user"):
                                    st.caption(f"**👤 Ti** · *{u_msg.get('timestamp', '')}*")
                                    st.markdown(u_msg.get("content", ""))
                            if a_msg:
                                with st.chat_message("assistant"):
                                    meta_line = f"**🤖 AI Trener** · *{a_msg.get('timestamp', '')}*"
                                    if a_msg.get("model"):
                                        meta_line += f" · Model: `{a_msg.get('model')}`"
                                    if a_msg.get("auto_applied"):
                                        meta_line += f" · ✅ *Samodejno vpisanih {a_msg['auto_applied']} treningov v koledar*"
                                    st.caption(meta_line)
                                    st.markdown(a_msg.get("content", ""))
                                    _tok_cap = format_token_usage_caption(a_msg.get("token_usage"))
                                    if _tok_cap:
                                        st.caption(_tok_cap)
                                    c_upds = a_msg.get("calendar_updates") or []
                                    if c_upds:
                                        if st.button(
                                            f"📅 Vpiši / Posodobi ta plan v moj koledar ({len(c_upds)} treningov)",
                                            key=f"btn_apply_exp_{active_tid}_{ex_uid}_{ex_idx}",
                                            type="primary",
                                        ):
                                            n_up = upsert_calendar_workouts(user_dir, c_upds, replace_same_date=True)
                                            st.success(f"Uspešno vpisanih {n_up} treningov v koledar!")
                                            st.rerun()

                            act_e1, act_e2, _ = st.columns([1.5, 1.7, 2.5])
                            with act_e1:
                                if u_msg and st.button(
                                    "🔄 Ponovi vprašanje (Retry)",
                                    key=f"btn_retry_exp_{active_tid}_{ex_uid}_{ex_idx}",
                                    use_container_width=True,
                                ):
                                    retry_exchange_target = (ex_idx, u_msg.get("content", ""))
                            with act_e2:
                                if st.button(
                                    "🗑️ Izbriši to sporočilo / par",
                                    key=f"btn_del_exp_{active_tid}_{ex_uid}_{ex_idx}",
                                    use_container_width=True,
                                ):
                                    delete_exchange_from_thread(user_dir, chat_store, active_tid, ex_idx)
                                    st.toast("🗑️ Sporočilo / par je bil izbrisan iz zgodovine in spomina trenerja.")
                                    st.rerun()

                # Če je zadnji klic spodletel (npr. 503 po 4 poskusih), prikaži opozorilno okno z gumbom "🔄 Poskusi znova" (brez trajnega zapisa napake v zgodovino!)
                failed_ex = st.session_state.get("ai_last_failed_exchange")
                if isinstance(failed_ex, dict) and failed_ex.get("thread_id") == active_tid:
                    st.error(
                        f"**Vprašanje:** *{summarize_message_short(failed_ex.get('user_prompt', ''), 120)}*\n\n"
                        f"{failed_ex.get('error_msg', 'Strežnik je trenutno zaseden.')}"
                    )
                    f_c1, f_c2, _ = st.columns([1.6, 1.6, 2.5])
                    with f_c1:
                        if st.button("🔄 Poskusi znova", type="primary", use_container_width=True, key="btn_retry_failed_temp"):
                            retry_exchange_target = ("__FAILED_TEMP__", failed_ex.get("user_prompt", ""))
                    with f_c2:
                        if st.button("🗑️ Odstrani obvestilo o napaki", use_container_width=True, key="btn_dismiss_failed_temp"):
                            st.session_state.pop("ai_last_failed_exchange", None)
                            st.rerun()

            # Izvedba "🔄 Ponovi vprašanje (Retry)" ali "🔄 Poskusi znova"
            if retry_exchange_target is not None:
                r_idx, r_user_q = retry_exchange_target
                # Zgodovina pred ponovljenim vprašanjem
                if r_idx == "__FAILED_TEMP__":
                    hist_before = active_messages
                    r_thread_summary = active_thread.get("thread_summary", "")
                else:
                    ex_before = exchanges[:int(r_idx)]
                    hist_before = []
                    for eb in ex_before:
                        if eb.get("user"):
                            hist_before.append(eb["user"])
                        if eb.get("assistant"):
                            hist_before.append(eb["assistant"])
                    r_thread_summary = build_compact_thread_summary(hist_before[:-4]) if len(hist_before) > 4 else ""

                full_ctx = build_full_athlete_context(
                    user_cfg=user_cfg,
                    user_journal=user_journal,
                    df_daily_all=df_daily_all,
                    activities_all=activities_all,
                    raw_data=raw_data,
                    calendar_workouts=calendar_workouts,
                    readiness_info=readiness,
                    today_date=today_date,
                    get_sport_cat_fn=get_sport_cat_fn,
                    compute_fixed_blocks_fn=compute_fixed_blocks_fn,
                    compute_run_blocks_fn=compute_run_blocks_fn,
                    compute_run_load_fn=compute_run_load_fn,
                    parse_stream_fn=parse_stream_fn,
                    user_dir=user_dir,
                    user_question=str(r_user_q),
                    is_followup=is_followup_turn(hist_before, str(r_user_q)),
                    chat_history=hist_before,
                )
                sys_prompt = build_coach_system_prompt(user_cfg, display_name, user_dir=user_dir)
                combined_prompt = f"{full_ctx}\n\n=== VPRAŠANJE / UKAZ ŠPORTNIKA ===\n{r_user_q}"

                retry_status_box = st.empty()
                with st.spinner(f"🔄 Ponovno pošiljam vprašanje AI Trenerju ({auth_info['display_model_id']})..."):
                    ok, reply_raw, used_model = call_gemini_hybrid(
                        user_prompt=combined_prompt,
                        user_cfg=user_cfg,
                        project_dir=project_dir,
                        system_instruction=sys_prompt,
                        chat_history=hist_before,
                        thread_summary=r_thread_summary,
                        status_placeholder=retry_status_box,
                    )
                if ok:
                    tok_usage = get_last_token_usage()
                    clean_reply, cal_updates = extract_calendar_updates_from_text(reply_raw)
                    clean_reply, mem_facts = extract_memory_updates_from_text(clean_reply)
                    for f_txt in mem_facts:
                        if add_coach_memory_fact(user_dir, f_txt):
                            st.toast(f"🧠 Trener si je zapomnil: '{f_txt}'", icon="🧠")
                    applied_n = 0
                    if cal_updates and auto_write_chk:
                        applied_n = upsert_calendar_workouts(user_dir, cal_updates, replace_same_date=True)
                        st.toast(f"📅 V koledar je bilo samodejno vpisanih {applied_n} treningov!")

                    if r_idx == "__FAILED_TEMP__":
                        append_exchange_to_thread(
                            user_dir=user_dir,
                            store=chat_store,
                            thread_id=active_tid,
                            user_content=r_user_q,
                            assistant_content=clean_reply or reply_raw,
                            used_model=used_model,
                            calendar_updates=cal_updates,
                            auto_applied=applied_n,
                            auto_title_from_user=True,
                            token_usage=tok_usage,
                        )
                        st.session_state.pop("ai_last_failed_exchange", None)
                    else:
                        replace_exchange_assistant_response(
                            user_dir=user_dir,
                            store=chat_store,
                            thread_id=active_tid,
                            exchange_idx=int(r_idx),
                            new_assistant_content=clean_reply or reply_raw,
                            used_model=used_model,
                            calendar_updates=cal_updates,
                            auto_applied=applied_n,
                            token_usage=tok_usage,
                        )
                    st.toast("✅ Odgovor uspešno posodobljen!")
                else:
                    st.session_state["ai_last_failed_exchange"] = {
                        "thread_id": active_tid,
                        "user_prompt": r_user_q,
                        "error_msg": reply_raw,
                        "auto_apply_calendar": auto_write_chk,
                        "auto_title": True,
                    }
                st.rerun()

            # PRIPETO VNOSNO POLJE SPODAJ (st.chat_input - vedno pri roki takoj pod drsnim oknom)
            _active_sum = active_thread.get("thread_summary", "")
            est_ctx_sample = build_full_athlete_context(
                user_cfg=user_cfg,
                user_journal=user_journal,
                df_daily_all=df_daily_all,
                activities_all=activities_all,
                raw_data=raw_data,
                calendar_workouts=calendar_workouts,
                readiness_info=readiness,
                today_date=today_date,
                get_sport_cat_fn=get_sport_cat_fn,
                compute_fixed_blocks_fn=compute_fixed_blocks_fn,
                compute_run_blocks_fn=compute_run_blocks_fn,
                compute_run_load_fn=compute_run_load_fn,
                parse_stream_fn=parse_stream_fn,
                user_dir=user_dir,
                is_followup=is_followup_turn(active_messages, ""),
                chat_history=active_messages,
            )
            est_sys_sample = build_coach_system_prompt(user_cfg, display_name, user_dir=user_dir)
            est_tokens_cnt = estimate_context_tokens(
                est_ctx_sample,
                active_messages,
                est_sys_sample,
                thread_summary=_active_sum,
            )
            st.caption(f"⚡ **Optimiran kontekst:** ~{est_tokens_cnt:,} žetonov (< 0,002 € na Flash modelu)")

            user_chat_text = st.chat_input(
                "Vprašaj AI trenerja v tem pogovoru...",
                key="ai_coach_sticky_chat_input",
            )
            if user_chat_text and user_chat_text.strip():
                _u_q = user_chat_text.strip()
                full_ctx = build_full_athlete_context(
                    user_cfg=user_cfg,
                    user_journal=user_journal,
                    df_daily_all=df_daily_all,
                    activities_all=activities_all,
                    raw_data=raw_data,
                    calendar_workouts=calendar_workouts,
                    readiness_info=readiness,
                    today_date=today_date,
                    get_sport_cat_fn=get_sport_cat_fn,
                    compute_fixed_blocks_fn=compute_fixed_blocks_fn,
                    compute_run_blocks_fn=compute_run_blocks_fn,
                    compute_run_load_fn=compute_run_load_fn,
                    parse_stream_fn=parse_stream_fn,
                    user_dir=user_dir,
                    user_question=_u_q,
                    is_followup=is_followup_turn(active_messages, _u_q),
                    chat_history=active_messages,
                )
                sys_prompt = est_sys_sample
                combined_prompt = f"{full_ctx}\n\n=== VPRAŠANJE / UKAZ ŠPORTNIKA ===\n{_u_q}"

                retry_status_box = st.empty()
                with st.spinner(f"🤖 AI Trener ({auth_info['display_model_id']}) pripravlja odgovor..."):
                    ok, reply_raw, used_model = call_gemini_hybrid(
                        user_prompt=combined_prompt,
                        user_cfg=user_cfg,
                        project_dir=project_dir,
                        system_instruction=sys_prompt,
                        chat_history=active_messages,
                        thread_summary=_active_sum,
                        status_placeholder=retry_status_box,
                    )

                if ok:
                    tok_usage = get_last_token_usage()
                    clean_reply, cal_updates = extract_calendar_updates_from_text(reply_raw)
                    clean_reply, mem_facts = extract_memory_updates_from_text(clean_reply)
                    for f_txt in mem_facts:
                        if add_coach_memory_fact(user_dir, f_txt):
                            st.toast(f"🧠 Trener si je zapomnil: '{f_txt}'", icon="🧠")
                    applied_n = 0
                    if cal_updates and auto_write_chk:
                        applied_n = upsert_calendar_workouts(user_dir, cal_updates, replace_same_date=True)
                        st.toast(f"📅 V koledar je bilo samodejno vpisanih {applied_n} treningov!")

                    append_exchange_to_thread(
                        user_dir=user_dir,
                        store=chat_store,
                        thread_id=active_thread["id"],
                        user_content=_u_q,
                        assistant_content=clean_reply or reply_raw,
                        used_model=used_model,
                        calendar_updates=cal_updates,
                        auto_applied=applied_n,
                        auto_title_from_user=True,
                        token_usage=tok_usage,
                    )
                    st.session_state.pop("ai_last_failed_exchange", None)
                else:
                    # Če vsi 4 samodejni poskusi spodletijo, NE zapišemo napake kot trajnega sporočila v zgodovino!
                    st.session_state["ai_last_failed_exchange"] = {
                        "thread_id": active_thread["id"],
                        "user_prompt": user_chat_text.strip(),
                        "error_msg": reply_raw,
                        "auto_apply_calendar": auto_write_chk,
                        "auto_title": True,
                    }
                st.rerun()

    # ==================================================================
    # SUB-TAB 2: 📅 PAMETNI KOLEDAR TRENINGOV
    # ==================================================================
    with sub_cal:
        st.subheader("📅 Pametni Koledar Treningov (Planirano vs. Opravljeno)")
        if "cal_week_offset" not in st.session_state:
            st.session_state["cal_week_offset"] = 0

        nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns([1.2, 1.2, 1.2, 1.2, 2.2])
        with nav_c1:
            if st.button("◀ Prejšnji teden", use_container_width=True, key="cal_prev_wk"):
                st.session_state["cal_week_offset"] -= 1
                st.rerun()
        with nav_c2:
            if st.button("📍 Ta teden", use_container_width=True, key="cal_this_wk"):
                st.session_state["cal_week_offset"] = 0
                st.rerun()
        with nav_c3:
            if st.button("⏭️ Naslednji teden", use_container_width=True, key="cal_next1_wk"):
                st.session_state["cal_week_offset"] = 1
                st.rerun()
        with nav_c4:
            if st.button("Naslednji ▶", use_container_width=True, key="cal_next_wk"):
                st.session_state["cal_week_offset"] += 1
                st.rerun()

        base_monday = today_date - timedelta(days=today_date.weekday())
        active_monday = base_monday + timedelta(weeks=int(st.session_state["cal_week_offset"]))
        week_days = [active_monday + timedelta(days=i) for i in range(7)]

        wk_plan_min = sum(int((cal_by_date.get(d.strftime("%Y-%m-%d")) or {}).get("duration_min") or 0) for d in week_days)
        wk_plan_tss = sum(int((cal_by_date.get(d.strftime("%Y-%m-%d")) or {}).get("target_tss") or 0) for d in week_days)
        wk_act_min = sum(sum(x["dur_min"] for x in acts_by_date.get(d.strftime("%Y-%m-%d"), [])) for d in week_days)
        wk_act_tss = sum(sum(x["tss"] for x in acts_by_date.get(d.strftime("%Y-%m-%d"), [])) for d in week_days)

        with nav_c5:
            st.markdown(
                f"**📆 Teden: {week_days[0].strftime('%d.%m.%Y')} – {week_days[-1].strftime('%d.%m.%Y')}**  \n"
                f"Plan: **{wk_plan_min/60.0:.1f} h** ({wk_plan_tss} TSS) · Opravljeno: **{wk_act_min/60.0:.1f} h** ({int(wk_act_tss)} TSS)"
            )

        for d_obj in week_days:
            d_s = d_obj.strftime("%Y-%m-%d")
            day_name = SLO_WEEKDAYS[d_obj.weekday()]
            is_today = (d_obj == today_date)
            planned = cal_by_date.get(d_s)
            actual_list = acts_by_date.get(d_s, [])
            status_txt, status_col = _evaluate_day_compliance(planned, actual_list, d_obj, today_date)
            today_tag = " 🔹 **[DANES]**" if is_today else ""

            with st.container(border=True):
                c_day, c_plan, c_act, c_btn = st.columns([1.4, 3.3, 3.0, 1.1])
                with c_day:
                    st.markdown(f"**{day_name}**{today_tag}  \n`{d_obj.strftime('%d.%m.%Y')}`")
                    st.markdown(
                        f"<span style='font-size:0.82rem; font-weight:700; color:{status_col};'>{status_txt}</span>",
                        unsafe_allow_html=True,
                    )
                with c_plan:
                    if planned:
                        w_type = planned.get("type", "Trening")
                        t_col = WORKOUT_TYPE_COLORS.get(w_type, "#2563eb")
                        st.markdown(
                            f"<span style='background:{t_col}; color:white; padding:2px 8px; border-radius:6px; font-size:0.76rem; font-weight:600;'>"
                            f"{planned.get('sport', 'Kolesarjenje')} · {w_type}</span> "
                            f"**{planned.get('title', 'Trening')}**",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"⏱️ Plan: **{planned.get('duration_min', 0)} min** | "
                            f"🎯 Cilj: **{planned.get('target_power_w') or '--'}** | "
                            f"⚡ Plan TSS: **{planned.get('target_tss', 0)}**"
                        )
                        if planned.get("description"):
                            st.markdown(f"<div style='font-size:0.86rem;'>{planned.get('description')}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("— Ni načrtovanega treninga v koledarju —")

                with c_act:
                    if actual_list:
                        for act in actual_list:
                            pwr_part = f" · ⚡ {act['avg_w']} W (NP {act['np_w']} W)" if act["avg_w"] else ""
                            hr_part = f" · ❤️ {act['avg_hr']} bpm" if act["avg_hr"] else ""
                            st.markdown(
                                f"🏅 **{act['name']}** ({act['sport']})  \n"
                                f"<span style='font-size:0.84rem;'>⏱️ {act['dur_min']} min · 📏 {act['dist_km']} km{pwr_part}{hr_part} · 🔥 <b>{int(act['tss'])} TSS</b></span>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("Opravljeno: —")

                with c_btn:
                    if st.button("✏️ Uredi", key=f"cal_edit_btn_{d_s}", use_container_width=True):
                        st.session_state["cal_edit_date"] = d_s
                    if planned:
                        if st.button("🗑️ Izbriši", key=f"cal_del_btn_{d_s}", use_container_width=True):
                            delete_calendar_workout(user_dir, date_str=d_s)
                            st.toast(f"Plan za {d_s} izbrisan.")
                            st.rerun()

        default_edit_dt = today_date
        if st.session_state.get("cal_edit_date"):
            try:
                default_edit_dt = datetime.strptime(st.session_state["cal_edit_date"], "%Y-%m-%d").date()
            except Exception:
                default_edit_dt = today_date

        with st.expander("➕ Ročni vnos ali urejanje planiranega treninga v koledarju", expanded=bool(st.session_state.get("cal_edit_date"))):
            f_c1, f_c2, f_c3, f_c4 = st.columns(4)
            with f_c1:
                edit_dt = st.date_input("Datum treninga:", value=default_edit_dt, key="cal_form_date")
                edit_d_str = edit_dt.strftime("%Y-%m-%d")
                existing_for_dt = cal_by_date.get(edit_d_str, {})
            with f_c2:
                sport_opts = ["Kolesarjenje", "Trenažer / Indoor", "Tek", "Hoja / Pohodništvo", "Moč / Fitnes", "Počitek"]
                cur_sp = existing_for_dt.get("sport", "Kolesarjenje")
                edit_sport = st.selectbox(
                    "Šport:",
                    sport_opts,
                    index=sport_opts.index(cur_sp) if cur_sp in sport_opts else 0,
                    key="cal_form_sport",
                )
            with f_c3:
                type_opts = list(WORKOUT_TYPE_COLORS.keys())
                cur_tp = existing_for_dt.get("type", "Z2 Aerobno")
                edit_type = st.selectbox(
                    "Tip treninga / Cona:",
                    type_opts,
                    index=type_opts.index(cur_tp) if cur_tp in type_opts else 0,
                    key="cal_form_type",
                )
            with f_c4:
                edit_title = st.text_input(
                    "Naslov treninga:",
                    value=existing_for_dt.get("title", "Z2 Aerobna baza"),
                    placeholder="npr. Threshold 2x20 min ali Z2 Baza",
                    key="cal_form_title",
                )

            f_c5, f_c6, f_c7 = st.columns([1, 1.5, 1])
            with f_c5:
                edit_dur = st.number_input(
                    "Planirano trajanje (min):",
                    min_value=0,
                    max_value=600,
                    value=int(existing_for_dt.get("duration_min", 90)),
                    step=5,
                    key="cal_form_dur",
                )
            with f_c6:
                edit_pwr = st.text_input(
                    "Ciljna moč (W / % FTP) ali tempo (/km):",
                    value=existing_for_dt.get("target_power_w", f"{int(user_cfg.get('ftp_outdoor', 300)*0.60)}-{int(user_cfg.get('ftp_outdoor', 300)*0.72)} W"),
                    placeholder="npr. 200-230 W (65-72% FTP) ali 4:35 /km",
                    key="cal_form_pwr",
                )
            with f_c7:
                edit_tss = st.number_input(
                    "Ciljni TSS / Load:",
                    min_value=0,
                    max_value=500,
                    value=int(existing_for_dt.get("target_tss", 80)),
                    step=5,
                    key="cal_form_tss",
                )

            edit_desc = st.text_area(
                "Podrobna struktura intervalov in opombe:",
                value=existing_for_dt.get("description", ""),
                placeholder="npr. 20 min ogrevanje Z1/Z2, 2x20 min @ 95-100% FTP (5 min vmes počitek Z1), 15 min ohlajanje.",
                height=85,
                key="cal_form_desc",
            )

            b_save, b_clear = st.columns([1.5, 1.5])
            with b_save:
                if st.button("💾 Shrani trening v koledar", type="primary", use_container_width=True, key="cal_form_save_btn"):
                    upsert_calendar_workouts(
                        user_dir,
                        [{
                            "date": edit_d_str,
                            "sport": edit_sport,
                            "type": edit_type,
                            "title": edit_title,
                            "duration_min": edit_dur,
                            "target_power_w": edit_pwr,
                            "target_tss": edit_tss,
                            "description": edit_desc,
                            "source": "manual",
                        }],
                        replace_same_date=True,
                    )
                    st.session_state["cal_edit_date"] = None
                    st.success(f"Trening za {edit_d_str} uspešno shranjen v koledar!")
                    st.rerun()
            with b_clear:
                if existing_for_dt and st.button("🗑️ Odstrani plan za izbrani dan", use_container_width=True, key="cal_form_remove_btn"):
                    delete_calendar_workout(user_dir, date_str=edit_d_str)
                    st.session_state["cal_edit_date"] = None
                    st.rerun()

        col_m_exp, col_ics_exp = st.columns(2)
        with col_m_exp:
            with st.expander("🗓️ Mesečni pregled koledarja (Tabela vseh dni v mesecu)"):
                m_year = active_monday.year
                m_month = active_monday.month
                first_m = date(m_year, m_month, 1)
                next_m = date(m_year + (1 if m_month == 12 else 0), 1 if m_month == 12 else m_month + 1, 1)
                days_in_m = (next_m - first_m).days
                m_rows = []
                for d_i in range(days_in_m):
                    cur_d = first_m + timedelta(days=d_i)
                    cur_s = cur_d.strftime("%Y-%m-%d")
                    cw = cal_by_date.get(cur_s)
                    al = acts_by_date.get(cur_s, [])
                    st_b, _ = _evaluate_day_compliance(cw, al, cur_d, today_date)
                    m_rows.append({
                        "Datum": cur_s,
                        "Dan": SLO_WEEKDAYS[cur_d.weekday()][:3],
                        "Planiran trening": f"{cw.get('title')} ({cw.get('duration_min')} min, {cw.get('target_tss')} TSS)" if cw else "—",
                        "Opravljen trening": ", ".join(f"{x['name']} ({x['dur_min']}m, {int(x['tss'])} TSS)" for x in al) if al else "—",
                        "Status": st_b,
                    })
                st.caption(f"Pregled za **{SLO_MONTHS[m_month-1]} {m_year}**:")
                st.dataframe(pd.DataFrame(m_rows), use_container_width=True, hide_index=True, height=320)

        with col_ics_exp:
            with st.expander("📲 Izvoz koledarja (Markdown kopiranje & .ics za telefon)"):
                wk_md = export_week_markdown(week_days, cal_by_date, acts_by_date)
                st.code(wk_md, language="markdown")
                ics_bytes = export_calendar_ics(calendar_workouts)
                st.download_button(
                    label="📲 Prenesi .ics koledar (Google / Apple / Outlook Calendar)",
                    data=ics_bytes,
                    file_name=f"trening_koledar_{today_date.strftime('%Y%m%d')}.ics",
                    mime="text/calendar",
                    use_container_width=True,
                )

    # ==================================================================
    # SUB-TAB 3: 🧠 MOJA TRENERSKA PRAVILA, CILJI IN NAVODILA (@kolesarski-trening)
    # Shranjeno v data/users/<username>/custom_coach_rules.json z auto-save
    # ==================================================================
    with sub_rules:
        st.subheader("🧠 Moja trenerska pravila, cilji in navodila (`@kolesarski-trening`)")
        st.caption(
            "Vsebina teh dveh polj se trajno shranjuje v `data/users/<uporabnik>/custom_coach_rules.json` in se ob vsakem pogovoru "
            "z najvišjo prioriteto vključi v sistemska navodila (System Prompt) tvojega AI trenerja. "
            "Novi uporabniki prejmejo strukturirano predlogo s polji `[Vpiši svoje podatke]`, ki jo lahko poljubno izpolnijo."
        )

        # Če je bil kliknjen gumb "📥 Naloži predlogo" ali če stanje še ni inicializirano, nastavi vrednosti pred izrisom
        if st.session_state.pop("_force_reset_custom_rules", False):
            st.session_state["custom_coach_rules_main_ta"] = DEFAULT_COACH_TEMPLATE
            st.session_state["custom_coach_style_ta"] = DEFAULT_COACH_RESPONSE_STYLE
        else:
            if "custom_coach_rules_main_ta" not in st.session_state:
                st.session_state["custom_coach_rules_main_ta"] = coach_rules_data.get("rules_and_profile", DEFAULT_COACH_TEMPLATE)
            if "custom_coach_style_ta" not in st.session_state:
                st.session_state["custom_coach_style_ta"] = coach_rules_data.get("response_style", DEFAULT_COACH_RESPONSE_STYLE)

        rules_val = st.text_area(
            "📋 Osebna navodila, profil športnika in pravila treniranja (npr. @kolesarski-trening skill):",
            height=450,
            key="custom_coach_rules_main_ta",
        )

        style_val = st.text_area(
            "🎯 Kako želim, da mi AI trener odgovarja (slog, dolžina, specifične zahteve):",
            height=175,
            key="custom_coach_style_ta",
        )

        new_rules_payload = {
            "rules_and_profile": rules_val.strip(),
            "response_style": style_val.strip(),
        }

        # Auto-save ob spremembi
        if (
            new_rules_payload["rules_and_profile"] != coach_rules_data.get("rules_and_profile", "").strip()
            or new_rules_payload["response_style"] != coach_rules_data.get("response_style", "").strip()
        ):
            save_custom_coach_rules(user_dir, new_rules_payload)
            user_cfg["custom_coach_instructions"] = new_rules_payload["rules_and_profile"]
            save_user_config_fn(user_cfg)
            st.toast("✅ Trenerska pravila samodejno shranjena v custom_coach_rules.json!")

        r_btn1, r_btn2 = st.columns([1.6, 1.6])
        with r_btn1:
            if st.button("💾 Shrani moja pravila", type="primary", use_container_width=True, key="btn_save_custom_rules"):
                save_custom_coach_rules(user_dir, new_rules_payload)
                user_cfg["custom_coach_instructions"] = new_rules_payload["rules_and_profile"]
                save_user_config_fn(user_cfg)
                st.success("✅ Trenerska pravila in slog odgovarjanja so shranjeni v `custom_coach_rules.json`!")
        with r_btn2:
            def _reset_rules_cb():
                reset_payload = {
                    "rules_and_profile": DEFAULT_COACH_TEMPLATE,
                    "response_style": DEFAULT_COACH_RESPONSE_STYLE,
                }
                save_custom_coach_rules(user_dir, reset_payload)
                user_cfg["custom_coach_instructions"] = DEFAULT_COACH_TEMPLATE
                save_user_config_fn(user_cfg)
                st.session_state["custom_coach_rules_main_ta"] = DEFAULT_COACH_TEMPLATE
                st.session_state["custom_coach_style_ta"] = DEFAULT_COACH_RESPONSE_STYLE
                st.session_state["_force_reset_custom_rules"] = True

            if st.button(
                "📥 Naloži predlogo (Osnutek za vnos lastnih ciljev)",
                use_container_width=True,
                key="btn_reset_custom_rules",
                on_click=_reset_rules_cb,
            ):
                st.toast("📥 Naložena je čista strukturirana predloga za vnos lastnih ciljev in podatkov.")
                st.rerun()


def render_api_key_help(key_suffix="sb"):
    """Gumb ❓ z navodili (v slovenščini) za pridobitev Gemini / Claude API ključa."""
    import streamlit as st
    with st.popover("❓ Navodila za pridobitev API ključev", use_container_width=True):
        st.markdown(
            """
### 🔹 Kako brezplačno pridobiti Gemini API ključ (Google AI Studio – priporočeno)
- **1. korak:** Odpri brskalnik in pojdi na uradno stran **[https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)** ter se prijavi s svojim običajnim Google (Gmail) računom.
- **2. korak:** Če stran odpiraš prvič, potrdi pogoje uporabe (*Accept terms of service*).
- **3. korak:** V zgornjem desnem kotu (ali na sredini zaslona) klikni na modri gumb **"Create API key"** (*Ustvari API ključ*).
- **4. korak:** V pojavnem oknu izberi **"Create API key in new project"** (ali izberi obstoječi projekt iz spustnega seznama in klikni *Create API key in existing project*).
- **5. korak:** Počakaj nekaj sekund, da se prikaže tvoj novi ključ (začne se z `AIzaSy...`), klikni gumb **"Copy"** poleg ključa, ga prilepi v polje **Gemini API ključ** v tej aplikaciji in klikni **Shrani**.

💡 *Nasvet glede stroškov:* Brez dodane plačilne kartice v Google Cloud računu (Free Tier) je uporaba modela **Gemini Flash** brezplačna (do ~1.500 klicev na dan). Če imaš vklopljen Billing / brezplačne mesečne kredite, uporabi model Flash za minimalno porabo (< 0,002 € na klic).

---

### 🔸 Kako pridobiti Claude API ključ (Anthropic Console – opcijsko za model Haiku/Sonnet)
- **1. korak:** Pojdi na **[https://console.anthropic.com/](https://console.anthropic.com/)** in ustvari račun ali se prijavi.
- **2. korak:** Za delovanje Anthropic API-ja je treba v meniju **Settings → Billing** najprej naložiti minimalno predplačniško dobroimetje (npr. 5 \\$) ali povezati plačilno metodo.
- **3. korak:** V levem meniju (ali na glavni plošči) klikni na **"API Keys"**.
- **4. korak:** Klikni gumb **"+ Create Key"**, vpiši poljubno ime (npr. *Kolesarski trener*) in klikni **Add**.
- **5. korak:** Takoj kopiraj prikazani ključ (začne se s `sk-ant-api03-...`), saj ga Anthropic pokaže samo enkrat! Prilepi ga v polje **Claude API ključ** v aplikaciji in klikni **Shrani**.
"""
        )
