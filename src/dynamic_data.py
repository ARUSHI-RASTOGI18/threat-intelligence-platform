"""
Dynamic Event Feed Ingestion & Normalization Module (ACLED API Integration)
Location: ./src/dynamic_data.py
Purpose: Ingests real-world conflict surveillance data via ACLED API,
         normalizes schema fields, caches locally, and handles resilient fallbacks.
"""

import os
import json
import time
from datetime import datetime
from typing import Tuple, Dict, Any
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "dynamic_events_cache.parquet")
META_FILE = os.path.join(CACHE_DIR, "dynamic_cache_meta.json")
TOKEN_CACHE_FILE = os.path.join(CACHE_DIR, "acled_token_cache.json")

ACLED_OAUTH_URL = "https://acleddata.com/oauth/token"
ACLED_API_URL = "https://acleddata.com/api/acled/read"

DEFAULT_TTL_MINUTES = 60
DEFAULT_TIMEOUT_SECONDS = 20


def _extract_clean_error(resp: requests.Response) -> str:
    """Safely extracts error text without dumping raw HTML/script blocks."""
    content_type = resp.headers.get("Content-Type", "").lower()
    if "application/json" in content_type:
        try:
            data = resp.json()
            if isinstance(data, dict):
                return str(
                    data.get("error_description")
                    or data.get("message")
                    or data.get("error")
                    or f"HTTP {resp.status_code}"
                )
        except Exception:
            pass

    status_reasons = {
        400: "Bad Request",
        401: "Unauthorized (Check ACLED credentials)",
        403: "Forbidden / Access Denied",
        404: "Endpoint Not Found",
        429: "Rate Limit Exceeded",
        500: "ACLED Server Error",
        503: "ACLED Service Unavailable",
    }
    return f"HTTP {resp.status_code}: {status_reasons.get(resp.status_code, resp.reason or 'Request Failed')}"


def _get_oauth_token() -> Tuple[str, str]:
    """Authenticates using ACLED OAuth2 endpoint with caching."""
    email = os.getenv("ACLED_EMAIL", "").strip()
    password = os.getenv("ACLED_PASSWORD", "").strip()

    if not email or not password:
        return "", "Missing ACLED_EMAIL or ACLED_PASSWORD in .env file."

    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(TOKEN_CACHE_FILE):
        try:
            with open(TOKEN_CACHE_FILE, "r", encoding="utf-8") as f:
                t_cache = json.load(f)
                if t_cache.get("expires_at", 0) > time.time() + 60:
                    return t_cache.get("access_token", ""), ""
        except Exception:
            pass

    payload = {
        "grant_type": "password",
        "username": email,
        "password": password,
        "client_id": "acled",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
    }

    try:
        resp = requests.post(
            ACLED_OAUTH_URL,
            data=payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "").lower()
            if "application/json" in content_type:
                data = resp.json()
                tok = data.get("access_token")
                exp = time.time() + data.get("expires_in", 3600)
                with open(TOKEN_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump({"access_token": tok, "expires_at": exp}, f)
                return tok, ""
        return "", f"OAuth Failed: {_extract_clean_error(resp)}"
    except Exception as e:
        return "", f"Connection Error: {type(e).__name__}"


def _normalize_acled_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes raw ACLED API output into standard schema."""
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()
    col_map = {
        "data_id": "event_id",
        "event_id_cnty": "event_code",
        "event_date": "event_date",
        "year": "year",
        "country": "country",
        "region": "region",
        "latitude": "latitude",
        "longitude": "longitude",
        "event_type": "event_type",
        "sub_event_type": "sub_event_type",
        "actor1": "actor",
        "fatalities": "fatalities",
        "notes": "notes",
    }

    rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(2026).astype(int)
    if "month" not in df.columns and "event_date" in df.columns:
        df["month"] = pd.to_datetime(df["event_date"], errors="coerce").dt.month.fillna(1).astype(int)
    if "fatalities" in df.columns:
        df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0).astype(int)
    if "injured" not in df.columns:
        df["injured"] = 0
    if "latitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    if "longitude" in df.columns:
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df["data_source"] = "ACLED"
    return df


def fetch_dynamic_events(force_refresh: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetches ACLED events via OAuth/API key with local caching and clean fallback handling."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Check local cache
    if not force_refresh and os.path.exists(CACHE_FILE) and os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
            ttl_sec = int(os.getenv("ACLED_CACHE_TTL_MINUTES", DEFAULT_TTL_MINUTES)) * 60
            if time.time() - meta.get("timestamp", 0) < ttl_sec:
                df = pd.read_parquet(CACHE_FILE)
                return df, {
                    "source": "CACHE",
                    "last_updated": meta.get("last_updated"),
                    "records": len(df),
                    "success": True,
                    "message": "Serving cached ACLED feed.",
                }
        except Exception:
            pass

    # 2. Acquire OAuth Token or check Direct Key
    token, err = _get_oauth_token()

    params = {
        "limit": 5000,
        "year": "2017:2026",
        "year_where": "BETWEEN",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        email = os.getenv("ACLED_EMAIL", "").strip()
        key = os.getenv("ACLED_KEY", os.getenv("ACLED_PASSWORD", "")).strip()
        params["email"] = email
        params["key"] = key

    try:
        resp = requests.get(
            ACLED_API_URL,
            params=params,
            headers=headers,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )

        content_type = resp.headers.get("Content-Type", "").lower()
        if "application/json" not in content_type:
            safe_msg = f"HTTP {resp.status_code}: ACLED returned non-JSON payload"
            return pd.DataFrame(), {
                "source": "API_INVALID_FORMAT",
                "message": safe_msg,
                "success": False,
                "records": 0,
            }

        res_json = resp.json()

        if res_json.get("status") == 0 or "error" in res_json:
            err_details = res_json.get("error", {})
            err_msg = err_details.get("message") if isinstance(err_details, dict) else str(err_details)
            return pd.DataFrame(), {
                "source": "API_ERROR",
                "message": f"ACLED API Error: {err_msg or 'Access Denied'}",
                "success": False,
                "records": 0,
            }

        raw_data = res_json.get("data", [])
        if raw_data:
            df_raw = pd.DataFrame(raw_data)
            df_norm = _normalize_acled_dataframe(df_raw)
            df_norm.to_parquet(CACHE_FILE, index=False)

            meta = {
                "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "timestamp": time.time(),
                "total_records": len(df_norm),
            }
            with open(META_FILE, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            return df_norm, {
                "source": "LIVE_API",
                "message": f"Successfully ingested {len(df_norm):,} events spanning 2017–2026 via ACLED API.",
                "success": True,
                "records": len(df_norm),
                "last_updated": meta["last_updated"],
            }

        return pd.DataFrame(), {"source": "API_EMPTY", "message": "API returned 0 records.", "success": False, "records": 0}

    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), {"source": "EXCEPTION", "message": f"Connection Failure: {type(e).__name__}", "success": False, "records": 0}


def get_dynamic_feed_status() -> Dict[str, Any]:
    """Returns current operational status."""
    if os.path.exists(META_FILE) and os.path.exists(CACHE_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {
                "status": "CONNECTED",
                "status_color": "#39D353",
                "last_updated": meta.get("last_updated", "N/A"),
                "total_records": meta.get("total_records", 0),
            }
        except Exception:
            pass
    return {
        "status": "DISCONNECTED",
        "status_color": "#F85149",
        "last_updated": "Never",
        "total_records": 0,
    }


def compute_dynamic_country_signal(df_gtd: pd.DataFrame, df_dyn: pd.DataFrame, country_name: str) -> Dict[str, Any]:
    """Compares GTD baseline against live/cached ACLED activity."""
    if df_gtd.empty:
        return {
            "baseline_annual": 0.0,
            "dynamic_events": 0,
            "percentage_deviation": 0.0,
            "signal_tier": "NO DATA",
            "signal_color": "#8B949E",
        }

    df_c_gtd = df_gtd[df_gtd["country"] == country_name]
    num_years = max(1, df_gtd["year"].nunique()) if "year" in df_gtd.columns else 47
    baseline_annual = round(len(df_c_gtd) / num_years, 1)

    if df_dyn.empty:
        return {
            "baseline_annual": baseline_annual,
            "dynamic_events": 0,
            "percentage_deviation": -100.0 if baseline_annual > 0 else 0.0,
            "signal_tier": "SUB-BASELINE CONCENTRATION",
            "signal_color": "#39D353",
        }

    df_c_dyn = df_dyn[df_dyn["country"] == country_name]
    dyn_cnt = len(df_c_dyn)

    if baseline_annual == 0:
        dev = 100.0 if dyn_cnt > 0 else 0.0
    else:
        dev = round(((dyn_cnt - baseline_annual) / baseline_annual) * 100.0, 1)

    if dev >= 50.0:
        tier, color = "ELEVATED CONFLICT ACTIVITY", "#F85149"
    elif dev <= -30.0:
        tier, color = "SUB-BASELINE CONCENTRATION", "#39D353"
    else:
        tier, color = "ACTIVE / BASELINE CONCORDANT", "#58A6FF"

    return {
        "baseline_annual": baseline_annual,
        "dynamic_events": dyn_cnt,
        "percentage_deviation": dev,
        "signal_tier": tier,
        "signal_color": color,
    }