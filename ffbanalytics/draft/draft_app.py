"""
Streamlit UI for the Fantasy Football Draft Agent.

This file does NOT modify draft_agent.py — it imports the reusable
pieces (DraftTracker, chat-session helpers, prompt builders) and wraps them in
a browser UI. The original file still works unchanged from the terminal via
`python draft_agent.py`; this is purely an alternate front end.

Run with:
    streamlit run streamlit_app.py
"""

import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from google import genai

from draft_agent import (
    DraftTracker,
    load_context_folder,
    build_system_instructions,
    create_chat_session,
    reset_chat_with_recap,
    RESET_EVERY_N_TURNS,
)

load_dotenv()

CSV_PATH = "FantasyPros_2026_Overall_ADP_Rankings.csv"
CONTEXT_FOLDER = "context_docs"
NUM_TEAMS = 12  # your league size — used to convert overall pick number into round number

st.set_page_config(page_title="Fantasy Draft Assistant", layout="wide")

if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY not found in environment variables. Check your .env file.")
    st.stop()


def build_round_position_chart_data(tracker: DraftTracker, num_teams: int) -> pd.DataFrame:
    """Derive a Round x Position pick-count table from the tracker's existing drafted
    data. Doesn't touch DraftTracker itself — just reads its public state."""
    if not tracker.drafted:
        return pd.DataFrame()

    rows = []
    for info in tracker.drafted.values():
        player, pick_no = info["player"], info["pick_no"]
        round_no = (pick_no - 1) // num_teams + 1
        match = tracker.df[tracker.df["Player"] == player]
        position = match["Position"].iloc[0] if not match.empty and "Position" in tracker.df.columns else "Unknown"
        rows.append({"Round": round_no, "Position": position})

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="Round", columns="Position", values="Position", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(range(1, pivot.index.max() + 1), fill_value=0)  # fill any skipped rounds
    return pivot


def find_keeper_files(folder_path: str) -> list:
    """Auto-detect any file in the context folder whose name contains 'keeper'."""
    if not os.path.isdir(folder_path):
        return []
    supported_ext = {".pdf", ".txt", ".md", ".csv"}
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if "keeper" in f.lower() and os.path.splitext(f)[1].lower() in supported_ext
    ]


def _extract_raw_text(fpath: str) -> str:
    ext = os.path.splitext(fpath)[1].lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(fpath)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext == ".csv":
            return pd.read_csv(fpath).to_string()
        else:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        return ""


def extract_keeper_names(folder_path: str, tracker: DraftTracker) -> set:
    """Detect keeper files, then check every known player name from the rankings
    pool against the raw extracted text. This sidesteps parsing the PDF's actual
    layout (table, list, whatever) — it just checks whether each known name shows
    up anywhere in the document, using the same normalization DraftTracker uses."""
    files = find_keeper_files(folder_path)
    if not files:
        return set()
    combined_text = " ".join(_extract_raw_text(f) for f in files)
    normalized_text = DraftTracker._normalize(combined_text)
    if not normalized_text:
        return set()
    keeper_names = set()
    for player in tracker.df["Player"].tolist():
        norm_player = DraftTracker._normalize(player)
        if norm_player and norm_player in normalized_text:
            keeper_names.add(norm_player)
    return keeper_names


def filter_out_keepers(df: pd.DataFrame, keeper_names: set) -> pd.DataFrame:
    if df.empty or not keeper_names:
        return df
    mask = ~df["Player"].apply(lambda p: DraftTracker._normalize(p) in keeper_names)
    return df[mask]


def build_filtered_turn_message(tracker: DraftTracker, user_input: str, keeper_names: set) -> str:
    """Same shape/purpose as draft_agent.build_turn_message, but built from
    a pool that also excludes keepers — so the model never sees them as draftable.
    Kept local (not imported) so draft_agent.py stays untouched."""
    filtered = filter_out_keepers(tracker.available_df(), keeper_names).head(tracker.top_n)
    table_md = filtered.to_markdown(index=False) if not filtered.empty else "No players loaded."
    return f"""### CURRENT AVAILABLE PLAYERS (top {tracker.top_n}, already excludes drafted and keeper players)
{table_md}

### USER MESSAGE
{user_input}
"""


def load_adp_lookup(csv_path: str) -> dict:
    """Read the ADP consensus column (AVG) directly from the CSV, independent of
    DraftTracker, so draft_agent.py stays untouched. Returns
    {normalized_player_name: avg_adp}."""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if "AVG" not in df.columns or "Player" not in df.columns:
        return {}
    return {
        DraftTracker._normalize(row["Player"]): row["AVG"]
        for _, row in df.iterrows()
        if pd.notna(row["AVG"])
    }


def get_best_value_picks(tracker: DraftTracker, adp_lookup: dict, keeper_names: set, top_n: int = 5) -> pd.DataFrame:
    """Players still available whose consensus ADP suggests they should already be
    gone, given how many picks have happened league-wide so far. Bigger gap = bigger
    value. Recomputed fresh every call, so it naturally updates as picks are made."""
    avail = filter_out_keepers(tracker.available_df(), keeper_names).copy()
    if avail.empty or not adp_lookup:
        return pd.DataFrame()
    avail["ADP"] = avail["Player"].apply(lambda p: adp_lookup.get(DraftTracker._normalize(p)))
    avail = avail.dropna(subset=["ADP"])
    if avail.empty:
        return pd.DataFrame()
    avail["Value Gap"] = (tracker.pick_counter - avail["ADP"]).round(1)
    avail = avail[avail["Value Gap"] > 0].sort_values("Value Gap", ascending=False).head(top_n)
    cols = [c for c in ["Player", "Position", "Team", "Tier", "ADP", "Value Gap"] if c in avail.columns]
    return avail[cols].reset_index(drop=True)


def get_best_available(tracker: DraftTracker, keeper_names: set, top_n: int = 5) -> pd.DataFrame:
    """Simple top-N of the current available pool (keepers excluded), ranked as-is."""
    avail = filter_out_keepers(tracker.available_df(), keeper_names).head(top_n)
    cols = [c for c in ["Rank", "Player", "Position", "Team", "Tier", "VORP"] if c in avail.columns]
    return avail[cols].reset_index(drop=True)


# --- One-time setup, persisted across Streamlit reruns via session_state ---
if "tracker" not in st.session_state:
    st.session_state.tracker = DraftTracker(CSV_PATH, top_n=150)
    st.session_state.adp_lookup = load_adp_lookup(CSV_PATH)
    st.session_state.keeper_names = extract_keeper_names(CONTEXT_FOLDER, st.session_state.tracker)
    extra_context = load_context_folder(CONTEXT_FOLDER)
    st.session_state.system_instructions = build_system_instructions(extra_context)
    st.session_state.client = genai.Client()
    st.session_state.chat = create_chat_session(
        st.session_state.client, st.session_state.system_instructions
    )
    st.session_state.messages = []  # display history: {"role": "user"/"assistant"/"system", "content": str}
    st.session_state.turns_since_reset = 0
    st.session_state.board_key_counter = 0  # bumped after each draft click to reset table selection

    init_response = st.session_state.chat.send_message(
        build_filtered_turn_message(
            st.session_state.tracker,
            "Confirm that you have loaded my rankings. List my top 3 overall players.",
            st.session_state.keeper_names,
        )
    )
    st.session_state.messages.append({"role": "assistant", "content": init_response.text})

tracker = st.session_state.tracker

st.title("🏈 Fantasy Draft Assistant")

col_chat, col_side = st.columns([2, 1])

# --- Sidebar column: live available board (click to draft) + drafted list ---
with col_side:
    st.subheader("Available Players")
    avail_df = filter_out_keepers(tracker.available_df(), st.session_state.keeper_names).head(tracker.top_n)

    if st.session_state.keeper_names:
        with st.expander(f"🔒 {len(st.session_state.keeper_names)} keeper(s) excluded — click to verify"):
            keeper_rows = tracker.df[
                tracker.df["Player"].apply(lambda p: DraftTracker._normalize(p) in st.session_state.keeper_names)
            ]
            st.dataframe(keeper_rows[["Player"]], hide_index=True, width="stretch")
            st.caption("If a keeper is missing here, check the spelling/formatting in your keepers file.")

    selection_event = st.dataframe(
        avail_df,
        width="stretch",
        hide_index=True,
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        key=f"board_table_{st.session_state.board_key_counter}",
    )

    if selection_event and selection_event.selection and selection_event.selection.rows:
        selected_row = selection_event.selection.rows[0]
        selected_player = avail_df.iloc[selected_row]["Player"]
        result_msg = tracker.draft_player(selected_player)
        st.session_state.messages.append({"role": "system", "content": result_msg})
        # New key forces a fresh, unselected table on rerun so the same click
        # can't re-trigger a draft action repeatedly.
        st.session_state.board_key_counter += 1
        st.rerun()

    st.subheader("Drafted Players")
    st.text(tracker.drafted_summary())

# --- Main column: value/best-available panels above the chat, then chat itself ---
with col_chat:
    st.subheader("📈 Best Value Available")
    value_df = get_best_value_picks(tracker, st.session_state.adp_lookup, st.session_state.keeper_names, top_n=5)
    if value_df.empty:
        st.caption("No standout value picks yet — check back as more picks are made.")
    else:
        st.dataframe(value_df, hide_index=True, width="stretch")

    st.subheader("⭐ Best Available")
    best_df = get_best_available(tracker, st.session_state.keeper_names, top_n=5)
    if best_df.empty:
        st.caption("No players loaded.")
    else:
        st.dataframe(best_df, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Chat")
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            st.info(msg["content"])
        else:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Ask about matchups, tiers, who to target next...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if st.session_state.turns_since_reset >= RESET_EVERY_N_TURNS:
            with st.spinner("Summarizing context and resetting chat history..."):
                st.session_state.chat = reset_chat_with_recap(
                    st.session_state.client,
                    st.session_state.system_instructions,
                    tracker,
                    st.session_state.chat,
                )
            st.session_state.turns_since_reset = 0

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_text = ""
            response_stream = st.session_state.chat.send_message_stream(
                build_filtered_turn_message(tracker, user_input, st.session_state.keeper_names)
            )
            for chunk in response_stream:
                if chunk.text:
                    full_text += chunk.text
                    placeholder.markdown(full_text)

        st.session_state.messages.append({"role": "assistant", "content": full_text})
        st.session_state.turns_since_reset += 1

# --- Draft trends: positions taken per round, full width below the main layout ---
st.divider()
st.subheader("Draft Trends: Positions by Round")
trend_df = build_round_position_chart_data(tracker, NUM_TEAMS)
if trend_df.empty:
    st.caption("No picks yet — this chart fills in as players get drafted.")
else:
    st.bar_chart(trend_df)
