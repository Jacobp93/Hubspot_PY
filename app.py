import time
from datetime import datetime, timedelta, timezone

import altair as alt
import requests
import pandas as pd
import streamlit as st


# ----------------------------
# Config
# ----------------------------

HUBSPOT_TOKEN = st.secrets["hubspot"]["token"]
BASE = st.secrets["hubspot"]["base_url"]

HEADERS = {
    "Authorization": f"Bearer {HUBSPOT_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

DEFAULT_TTL_SECONDS = 9000
DEFAULT_LIMIT = 100  # per-request page size; we paginate until done

# Ticket properties to request (these live under t["properties"])
PROPERTIES = [
    "subject",
    "content",
    "hs_createdate",
    "hs_lastmodifieddate",
    "hs_pipeline",
    "hs_pipeline_stage",
    "hs_ticket_priority",
    "hs_ticket_category",
    "hs_ticket_source",
]

# Pipeline dropdown (ID -> Label)
PIPELINES = {
    "ALL": "All Pipelines",
    "0": "Support Pipeline",
    "2568198332": "Onboarding Pipeline",
    "2939463891": "Success Pipeline",
    "2854155450": "Pre-Sales",
    "2813942000": "Red Flag",
}

# Stage labels (used for Open/Closed KPI logic)
OPEN_STAGE_LABELS = {"New", "Waiting on contact", "Waiting on us"}
CLOSED_STAGE_LABEL = "Closed"

st.set_page_config(page_title="HubSpot Tickets Reporting Suite", layout="wide")


# ----------------------------
# Helpers
# ----------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def week_start_utc(now: datetime) -> datetime:
    # Monday 00:00 UTC
    return (now - timedelta(days=now.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def parse_hs_dt(val):
    """
    HubSpot can return:
      - ISO strings: '2025-11-06T14:31:16.451Z'
      - ms since epoch (as string or int)
      - None/empty
    Returns pandas Timestamp (UTC) or NaT.
    """
    if val is None or val == "":
        return pd.NaT

    if isinstance(val, (int, float)):
        return pd.to_datetime(val, unit="ms", errors="coerce", utc=True)

    if isinstance(val, str):
        s = val.strip()
        if s.isdigit():
            return pd.to_datetime(int(s), unit="ms", errors="coerce", utc=True)
        return pd.to_datetime(s, errors="coerce", utc=True)

    return pd.NaT


def _post_with_retry(url: str, payload: dict, retries: int = 6) -> dict:
    last = None
    for i in range(retries):
        r = requests.post(url, headers=HEADERS, json=payload, timeout=60)
        last = r
        if r.status_code == 429:
            time.sleep(min(2**i, 30))
            continue
        r.raise_for_status()
        return r.json()
    if last is not None:
        last.raise_for_status()
    raise RuntimeError("HubSpot request failed with no response")


def _get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, headers=HEADERS, params=params or {}, timeout=60)
    r.raise_for_status()
    return r.json()


def _first_assoc_id(obj: dict, assoc_name: str):
    """
    Safely read first association id from:
      obj["associations"][assoc_name]["results"] = [{"id": "..."}]
    """
    results = (
        obj.get("associations", {})
        .get(assoc_name, {})
        .get("results", [])
    )
    return results[0].get("id") if results else None


# ----------------------------
# Cached loaders
# ----------------------------
@st.cache_data(ttl=DEFAULT_TTL_SECONDS)
def load_pipeline_stage_map() -> dict:
    """
    stage_id -> stage_label across all ticket pipelines.
    Requires token scope: crm.pipelines.read
    """
    data = _get(f"{BASE}/crm/v3/pipelines/tickets")
    stage_map = {}
    for pipe in data.get("results", []):
        for stage in pipe.get("stages", []):
            stage_map[stage.get("id")] = stage.get("label")
    return stage_map


@st.cache_data(ttl=DEFAULT_TTL_SECONDS)
def load_company_names(company_ids: list[str]) -> dict:
    """
    Returns {company_id: company_name} using HubSpot batch read.
    Requires token scope: crm.objects.companies.read (or equivalent)
    """
    company_ids = [str(x) for x in company_ids if x]
    if not company_ids:
        return {}

    unique_ids = sorted(set(company_ids))
    out = {}

    for i in range(0, len(unique_ids), 100):
        batch = unique_ids[i : i + 100]
        payload = {
            "inputs": [{"id": cid} for cid in batch],
            "properties": ["name"],
        }
        data = _post_with_retry(f"{BASE}/crm/v3/objects/companies/batch/read", payload)

        for r in data.get("results", []):
            out[r["id"]] = (r.get("properties", {}) or {}).get("name")

    return out
@st.cache_data(ttl=DEFAULT_TTL_SECONDS)
def load_ticket_company_ids(ticket_ids: list[str]) -> dict:
    """
    v3 Associations Batch Read: tickets -> companies
    Returns {ticket_id: company_id} (first company if multiple)
    """
    ticket_ids = [str(x) for x in ticket_ids if x]
    if not ticket_ids:
        return {}

    out = {}

    for i in range(0, len(ticket_ids), 100):
        batch = ticket_ids[i : i + 100]
        payload = {"inputs": [{"id": tid} for tid in batch]}

        data = _post_with_retry(
            f"{BASE}/crm/v3/associations/tickets/companies/batch/read",
            payload,
        )

        # Expected shape:
        # results: [{ "from": {"id": "ticketId"}, "to": [{"id":"companyId"}, ...] }, ...]
        for r in data.get("results", []):
            from_id = (r.get("from") or {}).get("id")
            to_list = r.get("to") or []
            company_id = to_list[0].get("id") if (to_list and isinstance(to_list[0], dict)) else None
            if from_id:
                out[str(from_id)] = company_id

    return out


@st.cache_data(ttl=DEFAULT_TTL_SECONDS)
def load_contact_company_ids(contact_ids: list[str]) -> dict:
    """
    Returns {contact_id: associatedcompanyid} by batch-reading contacts.
    Requires token scope: crm.objects.contacts.read (or equivalent)
    """
    contact_ids = [str(x) for x in contact_ids if x]
    if not contact_ids:
        return {}

    unique_ids = sorted(set(contact_ids))
    out = {}

    for i in range(0, len(unique_ids), 100):
        batch = unique_ids[i : i + 100]
        payload = {
            "inputs": [{"id": cid} for cid in batch],
            "properties": ["associatedcompanyid"],
        }
        data = _post_with_retry(f"{BASE}/crm/v3/objects/contacts/batch/read", payload)

        for r in data.get("results", []):
            props = r.get("properties", {}) or {}
            out[r["id"]] = props.get("associatedcompanyid")

    return out


@st.cache_data(ttl=DEFAULT_TTL_SECONDS)
def fetch_all_tickets(limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    """
    Fetch tickets across ALL pipelines using list endpoint with pagination.
    Capture top-level createdAt/updatedAt + requested properties + associations (companies + contacts).
    """
    rows = []
    after = None

    while True:
        params = {
            "limit": limit,
            "archived": "false",
            "properties": ",".join(PROPERTIES),
            "associations": "companies,contacts",  # ✅ both
        }
        if after:
            params["after"] = after

        data = _get(f"{BASE}/crm/v3/objects/tickets", params=params)

        for t in data.get("results", []):
            props = t.get("properties", {}) or {}

            rows.append(
                {
                    "ticket_id": t.get("id"),
                    "company_id": _first_assoc_id(t, "companies"),
                    "contact_id": _first_assoc_id(t, "contacts"),
                    "createdAt_raw": t.get("createdAt"),
                    "updatedAt_raw": t.get("updatedAt"),
                    **props,
                }
            )

        after = data.get("paging", {}).get("next", {}).get("after")
        if not after:
            break

    return pd.DataFrame(rows)


@st.cache_data(ttl=DEFAULT_TTL_SECONDS)
def fetch_tickets_in_pipeline(pipeline_id: str, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    """
    Fetch tickets in a single pipeline using Search API with pagination.
    Capture top-level createdAt/updatedAt + requested properties + associations (companies + contacts).
    """
    rows = []
    after = None

    while True:
        body = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "hs_pipeline",
                            "operator": "EQ",
                            "value": pipeline_id,
                        }
                    ]
                }
            ],
            "properties": PROPERTIES,
            "associations": ["companies", "contacts"],  # ✅ both
            "limit": limit,
        }
        if after:
            body["after"] = after

        data = _post_with_retry(f"{BASE}/crm/v3/objects/tickets/search", body)

        for t in data.get("results", []):
            props = t.get("properties", {}) or {}

            rows.append(
                {
                    "ticket_id": t.get("id"),
                    "company_id": _first_assoc_id(t, "companies"),
                    "contact_id": _first_assoc_id(t, "contacts"),
                    "createdAt_raw": t.get("createdAt"),
                    "updatedAt_raw": t.get("updatedAt"),
                    **props,
                }
            )

        after = data.get("paging", {}).get("next", {}).get("after")
        if not after:
            break

    return pd.DataFrame(rows)


# ----------------------------
# Sidebar UI (pipeline + refresh)
# ----------------------------
st.sidebar.title("Controls")

pipeline_ids = list(PIPELINES.keys())
default_index = pipeline_ids.index("0") if "0" in pipeline_ids else 0

selected_pipeline_id = st.sidebar.selectbox(
    "Pipeline",
    options=pipeline_ids,
    index=default_index,
    format_func=lambda k: f"{PIPELINES[k]} ({k})",
)

if st.sidebar.button("Refresh"):
    st.cache_data.clear()


# ----------------------------
# Load data
# ----------------------------
try:
    stage_map = load_pipeline_stage_map()

    if selected_pipeline_id == "ALL":
        df_raw = fetch_all_tickets(limit=DEFAULT_LIMIT).copy()
    else:
        df_raw = fetch_tickets_in_pipeline(
            pipeline_id=selected_pipeline_id, limit=DEFAULT_LIMIT
        ).copy()

except requests.HTTPError as e:
    st.error(f"HubSpot API error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error: {e}")
    st.stop()

if df_raw.empty:
    st.title("HubSpot Tickets Reporting Suite")
    st.info("No tickets returned.")
    st.stop()


# ----------------------------
# CANONICAL DATES (fixes “missing createdate”)
# ----------------------------
df_raw["createdAt"] = df_raw["createdAt_raw"].apply(parse_hs_dt)
df_raw["updatedAt"] = df_raw["updatedAt_raw"].apply(parse_hs_dt)

df_raw["hs_createdate_parsed"] = df_raw.get(
    "hs_createdate", pd.Series([None] * len(df_raw))
).apply(parse_hs_dt)

df_raw["hs_lastmodifieddate_parsed"] = df_raw.get(
    "hs_lastmodifieddate", pd.Series([None] * len(df_raw))
).apply(parse_hs_dt)

df_raw["createdate"] = df_raw["createdAt"].combine_first(df_raw["hs_createdate_parsed"])
df_raw["hs_lastmodifieddate"] = df_raw["updatedAt"].combine_first(
    df_raw["hs_lastmodifieddate_parsed"]
)


# ----------------------------
# COMPANY NAMES (direct ticket->company, fallback ticket->contact->associatedcompanyid)
# ----------------------------
# ----------------------------
# COMPANY NAMES (v3 associations batch: ticket -> company -> name)
# ----------------------------
df_raw["company_name"] = None

ticket_ids = df_raw["ticket_id"].dropna().astype(str).tolist()
ticket_to_company = load_ticket_company_ids(ticket_ids)

# This becomes the canonical company id for the ticket (first associated company)
df_raw["company_id"] = df_raw["ticket_id"].astype(str).map(ticket_to_company)

company_ids = df_raw["company_id"].dropna().astype(str).tolist()
company_map = load_company_names(company_ids)

# IMPORTANT: do NOT astype(str) on company_id before mapping names
df_raw["company_name"] = df_raw["company_id"].map(company_map)




# Debug counters (handy)
tickets_with_company = int(df_raw.get("company_id", pd.Series(dtype=object)).notna().sum())
tickets_with_contact = int(df_raw.get("contact_id", pd.Series(dtype=object)).notna().sum())


# Resolve company_id
resolved_company_id = pd.Series([None] * len(df_raw), index=df_raw.index, dtype=object)

if "company_id" in df_raw.columns:
    resolved_company_id = df_raw["company_id"].where(df_raw["company_id"].notna(), None)

# Fallback via contact -> associatedcompanyid
if tickets_with_contact > 0:
    contact_ids = df_raw["contact_id"].dropna().astype(str).tolist()
    contact_to_company = load_contact_company_ids(contact_ids)

    company_from_contact = df_raw["contact_id"].astype(str).map(contact_to_company)
    # only fill where no direct company
    resolved_company_id = resolved_company_id.combine_first(company_from_contact)

df_raw["resolved_company_id"] = resolved_company_id

# Batch load names for all resolved company ids
company_ids = df_raw["resolved_company_id"].dropna().astype(str).tolist()
company_map = load_company_names(company_ids)

# IMPORTANT: map without turning NaNs into strings
df_raw["company_name"] = df_raw["resolved_company_id"].map(company_map)


# ----------------------------
# Stage labels (human readable)
# ----------------------------
if "hs_pipeline_stage" in df_raw.columns:
    df_raw["stage_label"] = (
        df_raw["hs_pipeline_stage"]
        .astype(str)
        .map(stage_map)
        .fillna(df_raw["hs_pipeline_stage"].astype(str))
    )
else:
    df_raw["stage_label"] = None


# ----------------------------
# Sidebar UI (filters)
# ----------------------------


all_companies = sorted(df_raw["company_name"].dropna().astype(str).unique().tolist())
selected_companies = st.sidebar.multiselect("Companies", options=all_companies)

keyword = st.sidebar.text_input("Keyword in subject/content", value="").strip()

all_stages = sorted(df_raw["stage_label"].dropna().astype(str).unique().tolist())
selected_stages = st.sidebar.multiselect(
    "Stages", options=all_stages, default=all_stages
)



date_field = st.sidebar.selectbox(
    "Date filter based on",
    options=["Created date", "Last modified date"],
    index=1,
)
date_col = "createdate" if date_field == "Created date" else "hs_lastmodifieddate"


date_range = None
if date_col in df_raw.columns:
    valid_dates = df_raw[date_col].dropna()
    if not valid_dates.empty:
        min_d = valid_dates.min().date()
        max_d = valid_dates.max().date()
        date_range = st.sidebar.date_input(
            "Date range (UTC)",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
        )
        st.sidebar.caption(
            f"{date_field} present for {valid_dates.shape[0]:,} / {df_raw.shape[0]:,} tickets"
        )
    else:
        st.sidebar.warning(
            f"No valid {date_field.lower()} values found; date filtering disabled."
        )
else:
    st.sidebar.warning(f"{date_col} not in data; date filtering disabled.")




# ----------------------------
# KPI calculations (UNFILTERED)
# ----------------------------
now = utc_now()
wk_start = week_start_utc(now)

total_tickets = len(df_raw)

label_series = df_raw.get("stage_label", pd.Series([None] * len(df_raw))).astype(str)
is_open = label_series.isin(OPEN_STAGE_LABELS)
is_closed = label_series.eq(CLOSED_STAGE_LABEL)

total_open = int(is_open.sum())

created_this_week = df_raw["createdate"].notna() & (df_raw["createdate"] >= wk_start)
total_open_this_week = int((is_open & created_this_week).sum())

modified_this_week = df_raw["hs_lastmodifieddate"].notna() & (
    df_raw["hs_lastmodifieddate"] >= wk_start
)
total_closed_this_week = int((is_closed & modified_this_week).sum())


# ----------------------------
# Apply filters (FILTERED VIEW)
# ----------------------------
df = df_raw.copy()

if date_range is not None:
    start_date, end_date = date_range
    start_ts = pd.Timestamp(start_date).tz_localize("UTC")
    end_ts = pd.Timestamp(end_date + timedelta(days=1)).tz_localize("UTC")

    df = df[df[date_col].notna()]
    df = df[(df[date_col] >= start_ts) & (df[date_col] < end_ts)]

if keyword:
    subj = df.get("subject", pd.Series([""] * len(df))).fillna("")
    cont = df.get("content", pd.Series([""] * len(df))).fillna("")
    mask = subj.str.contains(keyword, case=False, na=False) | cont.str.contains(
        keyword, case=False, na=False
    )
    df = df[mask]

if selected_stages:
    df = df[df["stage_label"].astype(str).isin(selected_stages)]

if selected_companies:
    df = df[df["company_name"].astype(str).isin(selected_companies)]


# ----------------------------
# UI
# ----------------------------
st.title("HubSpot Tickets Reporting")
subtitle = PIPELINES.get(selected_pipeline_id, selected_pipeline_id)
st.caption(f"Pipeline: **{subtitle}**")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total tickets", f"{total_tickets:,}")
k2.metric("Total tickets open", f"{total_open:,}")
k3.metric("Open tickets created this week", f"{total_open_this_week:,}")
k4.metric("Closed tickets updated this week", f"{total_closed_this_week:,}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Tickets by stage")
    if len(df) == 0:
        st.info("No tickets in the current filters.")
    else:
        stage_counts = df["stage_label"].value_counts().reset_index()
        stage_counts.columns = ["Stage", "Count"]
        st.bar_chart(stage_counts.set_index("Stage"))

with right:
    st.subheader("Tickets created over time (monthly)")

    if "createdate" in df.columns and df["createdate"].notna().any():
        tmp = df.dropna(subset=["createdate"]).copy()

        # last 365 days anchored to today (UTC)
        end_ts = pd.Timestamp.now(tz="UTC")
        start_ts = end_ts - pd.Timedelta(days=365)
        tmp = tmp[(tmp["createdate"] >= start_ts) & (tmp["createdate"] <= end_ts)]

        tmp["month"] = (
            tmp["createdate"].dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
        )

        monthly = tmp.groupby("month").size().rename("Tickets").sort_index()

        first_month = start_ts.to_period("M").to_timestamp().tz_localize("UTC")
        last_month = end_ts.to_period("M").to_timestamp().tz_localize("UTC")
        full_months = pd.date_range(first_month, last_month, freq="MS", tz="UTC")
        monthly = monthly.reindex(full_months, fill_value=0)

        monthly_df = monthly.reset_index()
        monthly_df.columns = ["month", "Tickets"]
        monthly_df["month_label"] = monthly_df["month"].dt.strftime("%b %Y")
        month_order = monthly_df["month_label"].tolist()

        chart = (
            alt.Chart(monthly_df)
            .mark_bar(size=28)
            .encode(
                x=alt.X(
                    "month_label:O",
                    sort=month_order,
                    axis=alt.Axis(title=None, labelAngle=0),
                ),
                y=alt.Y("Tickets:Q", axis=alt.Axis(title=None)),
                tooltip=[
                    alt.Tooltip("month_label:O", title="Month"),
                    alt.Tooltip("Tickets:Q"),
                ],
            )
            .properties(height=280)
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No createdate values to chart.")

st.divider()

st.subheader("Tickets (details)")
cols = [
    c
    for c in [
        "ticket_id",
        "company_name",
        "subject",
        "stage_label",
        "hs_ticket_priority",
        "createdate",
        "updatedAt",
    ]
    if c in df.columns
]

if cols and len(df) > 0:
    st.dataframe(
        df[cols].sort_values(by="createdate", ascending=False),
        use_container_width=True,
    )
else:
    st.dataframe(df, use_container_width=True)

st.download_button(
    label="Download filtered CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="hubspot_tickets_filtered.csv",
    mime="text/csv",
)
