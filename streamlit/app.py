import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Streamlit config (must be first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Airline Passenger Satisfaction", page_icon="✈️", layout="centered")

# -----------------------------------------------------------------------------
# Styling (simple + centered, no sidebar)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {max-width: 980px; padding-top: 1.8rem; padding-bottom: 3rem;}
      .stApp {
        background:
          radial-gradient(1200px 700px at 30% 10%, rgba(63,94,251,0.12), transparent 55%),
          radial-gradient(900px 600px at 80% 20%, rgba(252,70,107,0.10), transparent 55%),
          linear-gradient(180deg, rgba(0,0,0,0.0), rgba(0,0,0,0.0));
      }

      /* Hide sidebar completely */
      section[data-testid="stSidebar"] {display:none !important;}
      div[data-testid="collapsedControl"] {display:none !important;}

      /* Cards */
      .card {
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        padding: 16px;
        border-radius: 14px;
      }

      /* Reduce giant gaps */
      .stMarkdown {margin-bottom: 0.25rem;}

      /* Make radio stars look cleaner */
      div[role="radiogroup"] {gap: 0.35rem;}
      label[data-baseweb="radio"] {
        background: rgba(255,255,255,0.04);
        padding: 6px 10px;
        border-radius: 10px;
      }
      label[data-baseweb="radio"]:hover {background: rgba(255,255,255,0.08);}

      /* ⭐ Star color (WORKING): default grey, checked gold */
      label[data-baseweb="radio"] { color: rgba(255,255,255,0.40) !important; }
      label[data-baseweb="radio"] * { color: rgba(255,255,255,0.40) !important; }

      /* BaseWeb radio: label > input + div ... so target input:checked */
      label[data-baseweb="radio"] input:checked + div,
      label[data-baseweb="radio"] input:checked + div * {
        color: #f5c542 !important;  /* gold */
      }

      /* Make expanders slightly tighter */
      details {border-radius: 14px !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Helpers: schema + API autodetect
# -----------------------------------------------------------------------------
STREAMLIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STREAMLIT_DIR.parent


def pick_schema_path() -> Path:
    env_path = os.getenv("SCHEMA_PATH")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        return p

    local = PROJECT_ROOT / "data" / "data_schema.json"
    docker = Path("/app/data/data_schema.json")
    if local.exists():
        return local.resolve()
    return docker


def pick_api_base_url() -> str:
    env_url = os.getenv("API_URL")
    if env_url:
        return env_url

    for url in ("http://localhost:8000", "http://api:8000"):
        try:
            r = requests.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                return url
        except Exception:
            pass

    return "http://localhost:8000"


@st.cache_resource
def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Schema file not found: {path}\n"
            f"Local: ensure data/data_schema.json exists\n"
            f"Docker: ensure ./data is mounted to /app/data"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stars_line(v: int) -> str:
    v = int(v)
    v = max(1, min(5, v))
    return "★" * v + "☆" * (5 - v)


def rating_word(v: int) -> str:
    v = int(v)
    return {1: "Poor", 2: "Meh", 3: "Okay", 4: "Good", 5: "Excellent"}.get(v, "Okay")


def api_health(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def star_picker(label: str, key: str) -> int:
    """Click-to-rate stars (review-style click behavior)."""
    v = st.radio(
        label,
        options=[1, 2, 3, 4, 5],
        key=key,
        horizontal=True,
        format_func=lambda x: stars_line(int(x)),
    )
    st.caption(f"{stars_line(int(v))} — **{rating_word(int(v))}**")
    return int(v)


def pretty_name(col: str) -> str:
    return col.replace("_", " ").title()


# -----------------------------------------------------------------------------
# Load schema + derive features
# -----------------------------------------------------------------------------
SCHEMA_PATH = pick_schema_path()
API_BASE_URL = pick_api_base_url()
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

schema = load_schema(SCHEMA_PATH)
numerical_features: Dict[str, Any] = schema.get("numerical", {})
categorical_features: Dict[str, Any] = schema.get("categorical", {})

NUM_MAIN = ["age", "flight_distance", "departure_delay_minutes", "arrival_delay_minutes"]

RATING_ORDER = [
    "inflight_wifi_service",
    "departure_arrival_time_convenient",
    "ease_of_online_booking",
    "gate_location",
    "food_and_drink",
    "online_boarding",
    "seat_comfort",
    "inflight_entertainment",
    "on_board_service",
    "leg_room_service",
    "baggage_handling",
    "checkin_service",
    "inflight_service",
    "cleanliness",
]

NUM_RATINGS = [k for k in RATING_ORDER if k in numerical_features]
CAT_COLS = list(categorical_features.keys())

NO_PRESET = "— No preset —"


# -----------------------------------------------------------------------------
# Defaults + Presets
# -----------------------------------------------------------------------------
def defaults_dict() -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "age": 40,
        "flight_distance": 900,
        "departure_delay_minutes": 0,
        "arrival_delay_minutes": 0,
    }
    for k in NUM_RATINGS:
        d[k] = 3

    for k, info in categorical_features.items():
        uniq = info.get("unique_values", []) or []
        counts = info.get("value_counts", {}) or {}
        if counts:
            d[k] = max(counts, key=counts.get)
        else:
            d[k] = uniq[0] if uniq else ""
    return d


PRESETS: Dict[str, Dict[str, Any]] = {
    "😄 Great experience": {
        "departure_delay_minutes": 0,
        "arrival_delay_minutes": 0,
        "flight_distance": 1200,
        **{k: 5 for k in NUM_RATINGS},
        "type_of_travel": "Business travel",
        "travel_class": "Business",
        "customer_type": "Loyal Customer",
    },
    "🙂 Average trip": {
        "departure_delay_minutes": 10,
        "arrival_delay_minutes": 15,
        "flight_distance": 900,
        **{k: 3 for k in NUM_RATINGS},
        "type_of_travel": "Personal Travel",
        "travel_class": "Eco",
    },
    "😣 Rough trip": {
        "departure_delay_minutes": 90,
        "arrival_delay_minutes": 120,
        "flight_distance": 1400,
        **{k: 1 for k in NUM_RATINGS},
        "type_of_travel": "Personal Travel",
        "travel_class": "Eco",
    },
    "🏖️ Vacation vibe": {
        "departure_delay_minutes": 20,
        "arrival_delay_minutes": 25,
        "flight_distance": 1600,
        **{k: 4 for k in NUM_RATINGS},
        "type_of_travel": "Personal Travel",
        "travel_class": "Eco Plus",
    },
}


# -----------------------------------------------------------------------------
# Session state (init BEFORE widgets to avoid warnings)
# -----------------------------------------------------------------------------
def init_state() -> None:
    base = defaults_dict()

    if "preset_name" not in st.session_state:
        st.session_state.preset_name = NO_PRESET
    if "show_details" not in st.session_state:
        st.session_state.show_details = False

    for k, v in base.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.session_state._base_defaults = base


def apply_values(values: Dict[str, Any]) -> None:
    for k, v in values.items():
        st.session_state[k] = v


def on_preset_change() -> None:
    name = st.session_state.preset_name
    if name == NO_PRESET:
        return
    preset_vals = PRESETS.get(name, {})
    base = st.session_state._base_defaults.copy()
    base.update(preset_vals)
    apply_values(base)


def on_random_click() -> None:
    st.session_state.preset_name = NO_PRESET

    st.session_state["age"] = random.randint(18, 80)
    st.session_state["flight_distance"] = random.randint(100, 3500)

    def delay_sample() -> int:
        r = random.random()
        if r < 0.75:
            return random.randint(0, 20)
        if r < 0.95:
            return random.randint(21, 90)
        return random.randint(91, 180)

    st.session_state["departure_delay_minutes"] = delay_sample()
    st.session_state["arrival_delay_minutes"] = delay_sample()

    for k in NUM_RATINGS:
        st.session_state[k] = int(random.choices([1, 2, 3, 4, 5], weights=[1, 2, 4, 3, 2])[0])

    for k, info in categorical_features.items():
        uniq = info.get("unique_values", []) or []
        if uniq:
            st.session_state[k] = random.choice(uniq)


def on_clear_click() -> None:
    st.session_state.preset_name = NO_PRESET
    apply_values(st.session_state._base_defaults)


init_state()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("✈️ Airline Passenger Satisfaction")
st.write("Fill passenger details, then click **Predict**.")

online = api_health(API_BASE_URL)
if online:
    st.success("✅ API Online")
else:
    st.error("❌ API Offline (start FastAPI / docker-compose first)")

# -----------------------------------------------------------------------------
# Preset + buttons
# -----------------------------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([2.2, 1, 1])
with c1:
    st.selectbox(
        "Quick preset",
        options=[NO_PRESET] + list(PRESETS.keys()),
        key="preset_name",
        on_change=on_preset_change,
    )
with c2:
    st.button("🎲 Random", on_click=on_random_click, use_container_width=True)
with c3:
    st.button("🧹 Clear", on_click=on_clear_click, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("")

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
with st.expander("🔢 Numbers", expanded=True):
    left, right = st.columns(2)
    with left:
        st.slider("Age", min_value=7, max_value=85, key="age")
        st.slider("Flight distance", min_value=50, max_value=4983, key="flight_distance")
    with right:
        st.slider("Departure delay (minutes)", min_value=0, max_value=180, key="departure_delay_minutes")
        st.slider("Arrival delay (minutes)", min_value=0, max_value=180, key="arrival_delay_minutes")

with st.expander("⭐ Service ratings (click stars)", expanded=True):
    st.caption("Click a rating — 1 = poor, 5 = excellent.")
    colA, colB = st.columns(2)
    half = (len(NUM_RATINGS) + 1) // 2
    left_r = NUM_RATINGS[:half]
    right_r = NUM_RATINGS[half:]

    with colA:
        for k in left_r:
            star_picker(pretty_name(k), key=k)

    with colB:
        for k in right_r:
            star_picker(pretty_name(k), key=k)

with st.expander("🏷️ Categories", expanded=True):
    colA, colB = st.columns(2)
    for i, k in enumerate(CAT_COLS):
        info = categorical_features.get(k, {}) or {}
        uniq = info.get("unique_values", []) or []
        if not uniq:
            continue

        current = st.session_state.get(k, uniq[0])
        idx = uniq.index(current) if current in uniq else 0

        target_col = colA if i % 2 == 0 else colB
        with target_col:
            st.selectbox(pretty_name(k), options=uniq, index=idx, key=k)

# -----------------------------------------------------------------------------
# Predict area
# -----------------------------------------------------------------------------
st.markdown("---")
top_left, top_right = st.columns([1.5, 2.5], vertical_alignment="center")
with top_left:
    st.toggle("Show details (table + raw JSON)", key="show_details")
with top_right:
    do_predict = st.button("Predict", type="primary", use_container_width=True)


def collect_instance() -> Dict[str, Any]:
    inst: Dict[str, Any] = {}

    inst["age"] = int(st.session_state["age"])
    inst["flight_distance"] = int(st.session_state["flight_distance"])
    inst["departure_delay_minutes"] = int(st.session_state["departure_delay_minutes"])
    inst["arrival_delay_minutes"] = int(st.session_state["arrival_delay_minutes"])

    for k in NUM_RATINGS:
        inst[k] = int(st.session_state[k])

    for k in CAT_COLS:
        inst[k] = st.session_state.get(k, "")

    return inst


if do_predict:
    instance = collect_instance()
    payload = {"instances": [instance]}

    if not online:
        st.error("API is offline. Start the FastAPI service first.")
    else:
        with st.spinner("Predicting..."):
            try:
                resp = requests.post(PREDICT_ENDPOINT, json=payload, timeout=30)
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
            else:
                if resp.status_code != 200:
                    st.error(f"API error (HTTP {resp.status_code}): {resp.text}")
                else:
                    data = resp.json()
                    labels = data.get("labels", [])
                    probs = data.get("probabilities", None)

                    if not labels:
                        st.warning("No prediction returned.")
                    else:
                        label = str(labels[0]).strip().lower()
                        p_sat = float(probs[0]) if (probs and len(probs) > 0) else None

                        is_sat = ("satisfied" in label) and ("neutral" not in label) and ("dissatisfied" not in label)

                        if is_sat:
                            msg = "✅ Satisfied? **YES**"
                            if p_sat is not None:
                                msg += f"  (P≈{p_sat:.3f})"
                            st.success(msg)
                        else:
                            msg = "❌ Satisfied? **NO**"
                            if p_sat is not None:
                                msg += f"  (P≈{p_sat:.3f})"
                            st.error(msg)

                        if st.session_state.show_details:
                            st.markdown("### Details")
                            out = {
                                "prediction_label": labels[0],
                                "prediction_binary": int(data.get("predictions", [0])[0]),
                                "p_satisfied": p_sat,
                            }
                            st.dataframe(pd.DataFrame([out]), use_container_width=True)

                            with st.expander("Raw API response", expanded=False):
                                st.json(data)

                            with st.expander("Input payload", expanded=False):
                                st.json(payload)
