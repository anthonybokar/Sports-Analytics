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
import time
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors as genai_errors

from draft_agent import (
    DraftTracker,
    load_context_folder,
    parse_league_config,
    build_system_instructions,
    create_chat_session,
    reset_chat_with_recap,
    RESET_EVERY_N_TURNS,
)

load_dotenv()

CSV_PATH = "FantasyPros_2026_Overall_ADP_Rankings.csv"
# CONTEXT_FOLDER = "context_docs_espn"
CONTEXT_FOLDER = "context_docs_yahoo"
NUM_TEAMS = 12  # your league size — used to convert overall pick number into round number

# --- Quick prompt buttons: canned questions that reuse the exact same send path
# as manually typed chat, so they go through normal context-loading and the
# RESET_EVERY_N_TURNS logic without any special-casing. ---
QUICK_PROMPTS = {
    "💰 Value": (
        "Based on current ADP vs. my rankings, who represents the biggest value at my "
        "next pick — and is there a tier cliff coming up in the next 5-8 picks I should "
        "jump ahead of?"
    ),
    "🏃 Run Check": (
        "Which position is being drafted faster than ADP suggests right now, and does "
        "that change how I should prioritize my next two picks?"
    ),
    "🏗️ Roster": (
        "Given my roster so far and my league's roster requirements based on league settings, what "
        "positions am I at risk of punting, and what's my latest 'safe' round to wait "
        "on each?"
    ),
    "🔒 Keepers": (
        "Excluding all keepers, who are the top 3 players at each position of need "
        "still available, and how does that shift my next-pick strategy?"
    ),
    "📋 Recap": (
        "Give me a quick recap: my roster, my remaining needs, and the single best "
        "available player regardless of position."
    ),
}

# --- Injury/news check: appended to the turn message (not the static system
# instructions) so it can be toggled per-message without rebuilding/resetting
# the chat session. Relies on Google Search grounding already being enabled
# in create_chat_session(). ---
INJURY_NEWS_INSTRUCTION = """
### INJURY & NEWS CHECK (required for this turn)
For any player you recommend or mention by name, use search to check their latest
injury status and any notable non-injury news before including them.

Severity judgment:
- FLAG PROMINENTLY (real risk to the recommendation): torn ligament/muscle, surgery,
  IR designation, "out indefinitely," multi-week "week-to-week," fracture.
- NOTE BUT DON'T DOWNGRADE: "questionable," "day-to-day," soreness/tightness, veteran
  rest day, minor illness, limited practice participation.
- Non-injury news: surface anything materially relevant to role/opportunity
  (suspension, depth chart change, coaching change, contract/holdout). Don't let minor
  news override a rankings-based recommendation — just inform the pick.

Add a brief "Status:" line only when there's something worth noting. Omit it entirely
for players with nothing notable — don't clutter every recommendation with "no notable
news."
"""

st.set_page_config(page_title="Fantasy Draft Assistant", layout="wide")

# Streamlit headers don't expose a font-size parameter directly, so this trims
# h1/h2/h3 (title/header/subheader) sizes a bit via CSS rather than per-call.
st.markdown(
    """
    <style>
    h1 { font-size: 1.9rem !important; }
    h2 { font-size: 1.4rem !important; }
    h3 { font-size: 1.15rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY not found in environment variables. Check your .env file.")
    st.stop()


# --- Retry helpers: Gemini occasionally returns a transient 503 ("high demand")
# or 429 (rate limit) that clears up within seconds. Without retrying, one bad
# blip crashes the whole session — the worst possible time for that is mid-draft.
# These wrap the two places draft_app.py calls the model; draft_agent.py itself
# is untouched. ---
RETRYABLE_CODES = (429, 500, 502, 503, 504)


def _is_transient_genai_error(e) -> bool:
    return getattr(e, "code", None) in RETRYABLE_CODES


def call_with_retry(fn, *args, max_attempts=4, base_delay=2, on_retry=None, **kwargs):
    """Call a one-shot Gemini function (e.g. chat.send_message) with exponential
    backoff on transient server errors. Re-raises immediately on non-transient
    errors (bad request, auth, etc.) or once attempts are exhausted."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except (genai_errors.ServerError, genai_errors.ClientError) as e:
            if not _is_transient_genai_error(e) or attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            if on_retry:
                on_retry(attempt, max_attempts, delay)
            time.sleep(delay)


def stream_with_retry(send_fn, *args, max_attempts=4, base_delay=2, on_retry=None, on_reset=None, **kwargs):
    """Same idea for chat.send_message_stream. A stream can't be resumed after
    an error mid-generation, so on_reset() lets the caller clear any partial
    text already shown before the whole response is regenerated from scratch."""
    for attempt in range(1, max_attempts + 1):
        try:
            for chunk in send_fn(*args, **kwargs):
                yield chunk
            return
        except (genai_errors.ServerError, genai_errors.ClientError) as e:
            if not _is_transient_genai_error(e) or attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            if on_reset:
                on_reset()
            if on_retry:
                on_retry(attempt, max_attempts, delay)
            time.sleep(delay)


def build_round_position_chart_data(tracker: DraftTracker, num_teams: int) -> pd.DataFrame:
    """Derive a Round x Position pick-count table from the tracker's existing drafted
    data. Doesn't touch DraftTracker itself — just reads its public state."""
    if not tracker.drafted:
        return pd.DataFrame()

    rows = []
    for info in tracker.drafted.values():
        player = info["player"]
        if info.get("is_keeper"):
            round_no = info.get("keeper_round")
        elif info.get("round"):
            round_no = info["round"]  # accurate slot from the precomputed pick sequence
        else:
            round_no = (info["pick_no"] - 1) // num_teams + 1  # fallback: no sequence configured
        if round_no is None:
            continue
        match = tracker.df[tracker.df["Player"] == player]
        position = match["Position"].iloc[0] if not match.empty and "Position" in tracker.df.columns else "Unknown"
        rows.append({"Round": round_no, "Position": position})

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="Round", columns="Position", values="Position", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(range(1, pivot.index.max() + 1), fill_value=0)  # fill any skipped rounds
    return pivot


def build_turn_message_ui(tracker: DraftTracker, user_input: str, injury_check: bool = False) -> str:
    """Same shape/purpose as draft_agent.build_turn_message (roster, scarcity, bye
    collisions). Keepers no longer need a separate filter pass here — register_keepers()
    puts them directly into tracker.drafted, so tracker.available_df() already excludes
    them the same way it excludes any other drafted player. Kept local so
    draft_agent.py's own version stays the source of truth for the CLI's identical logic.

    injury_check=True appends INJURY_NEWS_INSTRUCTION so the assistant searches for
    and reports the latest injury/news status on any player it recommends this turn."""
    avail = tracker.available_df().head(tracker.top_n)
    table_md = avail.to_markdown(index=False) if not avail.empty else "No players loaded."

    position_counts = tracker.roster_position_counts()
    roster_line = ", ".join(f"{pos}: {count}" for pos, count in position_counts.items()) or "No players drafted to your team yet"

    scarcity_df = tracker.tier_scarcity()
    if not scarcity_df.empty:
        scarce = scarcity_df[scarcity_df["Scarce"]]
        scarcity_line = "; ".join(
            f"{row['Position']}: only {row['Players Left in Tier']} left in Tier {row['Best Remaining Tier']}"
            for _, row in scarce.iterrows()
        ) if not scarce.empty else "No immediate tier cliffs"
    else:
        scarcity_line = "No tier data available"

    bye_collisions = tracker.bye_week_collisions()
    bye_line = "; ".join(f"Week {wk}: {count} players" for wk, count in bye_collisions.items()) if bye_collisions else "None"

    keeper_entries = [d for d in tracker.drafted.values() if d.get("is_keeper")]
    keeper_line = (
        ", ".join(f"{d['player']} ({d.get('team', '?')})" for d in keeper_entries)
        if keeper_entries else "No keepers loaded this session"
    )

    injury_block = INJURY_NEWS_INSTRUCTION if injury_check else ""

    return f"""### CURRENT AVAILABLE PLAYERS (top {tracker.top_n}, already excludes drafted and keeper players)
{table_md}

### KEEPERS (excluded from the pool above — these are NOT available to draft, already held by their original owners)
{keeper_line}

### YOUR ROSTER SO FAR (by position)
{roster_line}

### POSITIONAL SCARCITY / TIER CLIFFS (available players only)
{scarcity_line}

### YOUR BYE WEEK COLLISIONS (3+ players sharing a bye)
{bye_line}
{injury_block}
### USER MESSAGE
{user_input}
"""


DOC_EXTENSIONS = {"pdf", "txt", "csv", "md", "xlsx", "docx"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}


def save_doc_attachment(uploaded_file, folder_path: str) -> str:
    """Write a PDF/text/CSV attachment to context_docs so it becomes part of the
    permanent, persistent context — picked up by load_context_folder() same as any
    file you'd dropped in manually, surviving future app restarts too."""
    os.makedirs(folder_path, exist_ok=True)
    dest_path = os.path.join(folder_path, uploaded_file.name)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return dest_path


def rebuild_context_and_chat(client, tracker: DraftTracker, old_chat):
    """Re-read context_docs (now including any newly-saved attachment), rebuild the
    system instructions, and reset the chat session with a recap — same mechanism
    RESET_EVERY_N_TURNS already uses, just triggered by a new file instead of a
    turn count."""
    extra_context = load_context_folder(CONTEXT_FOLDER)
    new_system_instructions = build_system_instructions(extra_context)
    new_chat = reset_chat_with_recap(client, new_system_instructions, tracker, old_chat)
    return new_system_instructions, new_chat


def build_image_part(uploaded_file):
    """Images can't go into a text system instruction — they're sent as actual
    image data in the message itself. This means an image persists for the rest
    of THIS running session (it's part of chat history, resent every turn like
    everything else) but won't survive a full app restart the way saved
    PDF/text/CSV context does."""
    from google.genai import types
    return types.Part.from_bytes(data=uploaded_file.getvalue(), mime_type=uploaded_file.type)


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


def get_best_value_picks(tracker: DraftTracker, adp_lookup: dict, top_n: int = 5) -> pd.DataFrame:
    """Players still available whose consensus ADP suggests they should already be
    gone, given how many picks have happened league-wide so far. Bigger gap = bigger
    value. Recomputed fresh every call, so it naturally updates as picks are made.

    Uses tracker.true_overall_pick_count() rather than the raw pick_counter, so keeper-
    forfeited draft slots (which the live draft silently skips) are counted as already
    "off the board" once the draft reaches that round — otherwise "picks made league-wide"
    undercounts and skews Value Gap low. Falls back to the plain pick_counter if
    DRAFT_ORDER/ROUNDS haven't been configured in league_config.txt.
    """
    avail = tracker.available_df().copy()
    if avail.empty or not adp_lookup:
        return pd.DataFrame()
    avail["ADP"] = avail["Player"].apply(lambda p: adp_lookup.get(DraftTracker._normalize(p)))
    avail = avail.dropna(subset=["ADP"])
    if avail.empty:
        return pd.DataFrame()
    avail["Value Gap"] = (tracker.true_overall_pick_count() - avail["ADP"]).round(1)
    avail = avail[avail["Value Gap"] > 0].sort_values("Value Gap", ascending=False).head(top_n)
    cols = [c for c in ["Player", "Position", "Team", "Tier", "ADP", "Value Gap"] if c in avail.columns]
    return avail[cols].reset_index(drop=True)


def get_best_available(tracker: DraftTracker, top_n: int = 5) -> pd.DataFrame:
    """Simple top-N of the current available pool (keepers excluded), ranked as-is."""
    avail = tracker.available_df().head(top_n)
    cols = [c for c in ["Rank", "Player", "Position", "Team", "Tier", "VORP"] if c in avail.columns]
    return avail[cols].reset_index(drop=True)


def build_draft_log_df(tracker: DraftTracker) -> pd.DataFrame:
    """Full draft history (both teams), in pick order, with position/NFL team/tier/bye
    joined in from the rankings — suitable for a CSV export at the end of a draft."""
    if not tracker.drafted:
        return pd.DataFrame(columns=["Pick", "Player", "Drafted By", "Position", "Team", "Tier", "Bye"])
    rows = []
    for info in sorted(tracker.drafted.values(), key=lambda d: d["pick_no"]):
        match = tracker.df[tracker.df["Player"] == info["player"]]
        row = {"Pick": info["pick_no"], "Player": info["player"], "Drafted By": info.get("team", "Opponent")}
        for col in ["Position", "Team", "Tier", "Bye"]:
            if col in match.columns and not match.empty:
                row[col] = match[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


# --- One-time setup, persisted across Streamlit reruns via session_state ---
if "tracker" not in st.session_state:
    league_config = parse_league_config(os.path.join(CONTEXT_FOLDER, "league_config.txt"))
    my_team_name = league_config["my_team_name"] or "My Team"
    if not league_config["my_team_name"]:
        st.warning(
            "No MY_TEAM_NAME found in context_docs/league_config.txt — defaulting to "
            "'My Team'. Add that file (or the MY_TEAM_NAME line) so keepers land on the right roster."
        )

    st.session_state.tracker = DraftTracker(CSV_PATH, top_n=150, my_team_name=my_team_name)
    st.session_state.adp_lookup = load_adp_lookup(CSV_PATH)
    st.session_state.keeper_warnings = st.session_state.tracker.register_keepers(
        league_config["keepers"], my_team_name
    )
    if league_config["draft_order"] and league_config["num_rounds"]:
        st.session_state.tracker.configure_pick_sequence(
            league_config["draft_order"], league_config["num_rounds"]
        )
    else:
        st.warning(
            "No DRAFT_ORDER/ROUNDS found in league_config.txt — the Round/Pick tracker and "
            "ADP value gap won't account for keeper-skipped slots until those are added."
        )
    extra_context = load_context_folder(CONTEXT_FOLDER)
    st.session_state.system_instructions = build_system_instructions(extra_context)
    st.session_state.client = genai.Client()
    st.session_state.chat = create_chat_session(
        st.session_state.client, st.session_state.system_instructions
    )
    st.session_state.messages = []  # display history: {"role": "user"/"assistant"/"system", "content": str}
    st.session_state.turns_since_reset = 0
    st.session_state.board_key_counter = 0  # bumped after each draft click to reset table selection
    st.session_state.injury_check_enabled = False  # toggle: search for latest injury/news on recommended players

    try:
        init_response = call_with_retry(
            st.session_state.chat.send_message,
            build_turn_message_ui(
                st.session_state.tracker,
                "Confirm that you have loaded my rankings. List my top 3 overall players.",
            ),
            on_retry=lambda attempt, total, delay: st.toast(
                f"Gemini is overloaded (attempt {attempt}/{total}) — retrying in {delay}s...", icon="⏳"
            ),
        )
        st.session_state.messages.append({"role": "assistant", "content": init_response.text})
    except (genai_errors.ServerError, genai_errors.ClientError) as e:
        st.error(
            f"Gemini is unavailable after several retries ({e}). This is usually temporary — "
            "refresh the page in a minute to try again."
        )
        st.stop()

for w in st.session_state.get("keeper_warnings", []):
    st.warning(w)

tracker = st.session_state.tracker

st.title("🏈 Fantasy Draft Assistant")

# --- Top container: header row, then Available Players, then Best Value/Best Available, then trends ---
with st.container(border=True):
    keeper_col, drafted_col, undo_col, pick_status_col = st.columns([2, 1, 1, 1.3])

    keeper_entries = [d for d in tracker.drafted.values() if d.get("is_keeper")]
    with keeper_col:
        if keeper_entries:
            with st.expander(f"🔒 {len(keeper_entries)} keeper(s) excluded — click to verify"):
                keeper_df = pd.DataFrame([
                    {"Player": d["player"], "Team": d.get("team"), "Keeper Cost": f"Rd {d['keeper_round']}" if d.get("keeper_round") else "?"}
                    for d in sorted(keeper_entries, key=lambda d: d["player"])
                ])
                st.dataframe(keeper_df, hide_index=True, width="stretch")
                st.caption("If a keeper is missing here, check league_config.txt spelling against the rankings CSV.")

    with drafted_col:
        with st.popover("📋 Edit Draft Log", width="stretch"):
            if not tracker.drafted:
                st.caption("No picks logged yet.")
            else:
                log_rows = [
                    {
                        "Pick": (f"Keeper (Rd {d['keeper_round']})" if d.get("keeper_round") else "Keeper")
                                if d.get("is_keeper") else d["pick_no"],
                        "Player": d["player"],
                        "Team": d.get("team", "Opponent"),
                        "🗑️ Remove": False,
                    }
                    for d in tracker.drafted.values()
                ]
                log_df = pd.DataFrame(log_rows).sort_values("Player").reset_index(drop=True)
                team_options = sorted(
                    {tracker.my_team_name, "Opponent"} | {d.get("team", "Opponent") for d in tracker.drafted.values()}
                )
                edited_log = st.data_editor(
                    log_df,
                    hide_index=True,
                    width="stretch",
                    disabled=["Pick", "Player"],
                    column_config={
                        "Team": st.column_config.SelectboxColumn(options=team_options),
                        "🗑️ Remove": st.column_config.CheckboxColumn(help="Check, then click Apply Changes below"),
                    },
                    key=f"draft_log_editor_{st.session_state.board_key_counter}",
                )
                if st.button("Apply Changes", width="stretch"):
                    changed = False
                    for i, row in edited_log.iterrows():
                        if row["🗑️ Remove"]:
                            st.toast(tracker.remove_pick(row["Player"]), icon="🗑️")
                            changed = True
                        elif row["Team"] != log_df.iloc[i]["Team"]:
                            st.toast(tracker.reassign_pick(row["Player"], row["Team"]), icon="✏️")
                            changed = True
                    if changed:
                        st.session_state.board_key_counter += 1
                        st.rerun()
                    else:
                        st.caption("No changes to apply.")

    with undo_col:
        if st.button("↩️ Undo Last Pick"):
            undo_msg = tracker.undo_last()
            st.toast(undo_msg, icon="↩️")
            st.session_state.board_key_counter += 1
            st.rerun()

    with pick_status_col:
        progress = tracker.draft_progress()
        if not progress:
            st.caption("⚠️ Add DRAFT_ORDER/ROUNDS to league_config.txt to enable pick tracking.")
        elif progress["next"] is None:
            st.metric("Draft", "Complete ✅", f"{progress['live_picks_made']} live picks made")
        else:
            nxt = progress["next"]
            you = " 🎯 YOU" if nxt["team"] == tracker.my_team_name else nxt["team"]
            st.metric(
                f"Round {nxt['round']} · Pick {nxt['pick_in_round']}",
                f"Overall #{nxt['overall_pick']}",
                you,
                delta_color="off",
            )

    st.subheader("Available Players")

    avail_df = tracker.available_df().head(tracker.top_n).copy()

    my_col = f"✅ {tracker.my_team_name}"
    opp_col = "✅ Opponent"
    avail_df[my_col] = False
    avail_df[opp_col] = False
    original_cols = [c for c in avail_df.columns if c not in (my_col, opp_col)]

    edited_df = st.data_editor(
        avail_df,
        width="stretch",
        hide_index=True,
        height=420,
        disabled=original_cols,  # only the two checkbox columns are actually editable
        column_config={
            my_col: st.column_config.CheckboxColumn(help=f"Check to draft this player to {tracker.my_team_name}"),
            opp_col: st.column_config.CheckboxColumn(help="Check to draft this player to an opponent"),
        },
        key=f"board_editor_{st.session_state.board_key_counter}",
    )

    my_picks = edited_df[edited_df[my_col]]
    opp_picks = edited_df[edited_df[opp_col]]

    picked_player, picked_team = None, None
    if not my_picks.empty:
        picked_player, picked_team = my_picks.iloc[0]["Player"], tracker.my_team_name
    elif not opp_picks.empty:
        picked_player, picked_team = opp_picks.iloc[0]["Player"], "Opponent"

    if picked_player:
        result_msg = tracker.draft_player(picked_player, team=picked_team)
        st.toast(result_msg, icon="🏈")
        # New key forces a fresh, unchecked table on rerun so the same click
        # can't re-trigger a draft action repeatedly.
        st.session_state.board_key_counter += 1
        st.rerun()

    val_col, best_col, cliffs_col = st.columns(3)

    with val_col:
        st.subheader("📈 Best Value")
        value_df = get_best_value_picks(tracker, st.session_state.adp_lookup, top_n=5)
        if value_df.empty:
            st.caption("No standout value picks yet.")
        else:
            st.dataframe(value_df, hide_index=True, width="content")

    with best_col:
        st.subheader("⭐ Best Available")
        best_df = get_best_available(tracker, top_n=5)
        if best_df.empty:
            st.caption("No players loaded.")
        else:
            st.dataframe(best_df, hide_index=True, width="content")

    with cliffs_col:
        st.subheader("⚠️ Tier Cliffs")
        scarcity_df = tracker.tier_scarcity()
        if scarcity_df.empty:
            st.caption("No tier data available.")
        else:
            scarce_only = scarcity_df[scarcity_df["Scarce"]]
            if scarce_only.empty:
                st.caption("No immediate tier cliffs.")
            else:
                st.dataframe(scarce_only.drop(columns=["Scarce"]), hide_index=True, width="content")

    with st.popover("📊 Draft Trends: Positions by Round", width="stretch"):
        trend_df = build_round_position_chart_data(tracker, NUM_TEAMS)
        if trend_df.empty:
            st.caption("No picks yet — this chart fills in as players get drafted.")
        else:
            st.bar_chart(trend_df)

# --- Bottom row: chat on the left, roster info on the right ---
col_chat, col_side = st.columns([2, 1])

with col_side:
    st.subheader("🧢 My Roster")
    my_roster_df = tracker.roster_df()
    if my_roster_df.empty:
        st.caption("No players drafted to your team yet.")
    else:
        st.dataframe(my_roster_df, hide_index=True, width="stretch")
        counts = tracker.roster_position_counts()
        if counts:
            st.caption("Positions: " + ", ".join(f"{pos} {n}" for pos, n in counts.items()))

        if "Bye" not in tracker.df.columns:
            st.caption("⚠️ No 'Bye' column found in your rankings CSV — check the exact column header name.")
        else:
            bye_collisions = tracker.bye_week_collisions()
            if bye_collisions:
                st.warning(
                    "Bye week collision: " + ", ".join(f"Week {wk} ({n} players)" for wk, n in bye_collisions.items())
                )
            else:
                st.caption("No bye week collisions yet (3+ players needed on the same bye to flag).")

    if tracker.drafted:
        st.download_button(
            "⬇️ Export Draft Log (CSV)",
            data=build_draft_log_df(tracker).to_csv(index=False),
            file_name="draft_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

with col_chat:
    st.subheader("Chat")

    # --- Quick prompts + injury/news toggle: compact, bordered, sits above the
    # input so it's always visible without disturbing any other component's layout. ---
    with st.container(border=True):
        st.session_state.injury_check_enabled = st.toggle(
            "🩹 Injury/News Check",
            value=st.session_state.injury_check_enabled,
            help=(
                "When on, the assistant searches for each recommended player's latest "
                "injury status and notable news before responding. Minor stuff "
                "(tightness, questionable, rest day) is noted but won't count against "
                "a recommendation; major stuff (tear, surgery, IR) is flagged clearly. "
                "Adds a bit of latency per message since it triggers a live search."
            ),
        )
        st.caption("Quick Prompts")
        qp_cols = st.columns(len(QUICK_PROMPTS))
        quick_prompt_text = None
        for col, (label, prompt_text) in zip(qp_cols, QUICK_PROMPTS.items()):
            if col.button(label, use_container_width=True, help=prompt_text, key=f"quick_{label}"):
                quick_prompt_text = prompt_text

    # --- Text input, always rendered here at the top of the column ---
    chat_submission = st.chat_input(
        "Ask about matchups, tiers, who to target next... (attach files with the + icon)",
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg", "pdf", "txt", "csv", "md", "xlsx", "docx"],
    )

    user_input, uploaded_files, process_turn = None, [], False
    if quick_prompt_text:
        user_input, uploaded_files, process_turn = quick_prompt_text, [], True
    elif chat_submission:
        user_input = chat_submission.text or ""
        uploaded_files = chat_submission.files or []
        process_turn = True

    if process_turn:
        doc_files = [f for f in uploaded_files if f.name.rsplit(".", 1)[-1].lower() in DOC_EXTENSIONS]
        image_files = [f for f in uploaded_files if f.name.rsplit(".", 1)[-1].lower() in IMAGE_EXTENSIONS]
        attachment_names = [f.name for f in uploaded_files]

        display_text = user_input if user_input else "(attachment only)"
        st.session_state.messages.append(
            {"role": "user", "content": display_text, "attachments": attachment_names}
        )
        with st.chat_message("user"):
            st.markdown(display_text)
            for name in attachment_names:
                st.caption(f"📎 {name}")

        # --- Permanent doc attachments: save to context_docs, then rebuild + reset ---
        if doc_files:
            with st.spinner(f"Saving {len(doc_files)} file(s) to permanent context and rebuilding..."):
                for f in doc_files:
                    save_doc_attachment(f, CONTEXT_FOLDER)
                # If a fresh league_config.txt came in, pick up any new/updated keepers.
                # register_keepers() skips anyone already in tracker.drafted, so this is
                # safe to re-run — it only adds keepers that weren't there before.
                if any(f.name == "league_config.txt" for f in doc_files):
                    refreshed_config = parse_league_config(os.path.join(CONTEXT_FOLDER, "league_config.txt"))
                    new_warnings = tracker.register_keepers(
                        refreshed_config["keepers"], tracker.my_team_name
                    )
                    st.session_state.keeper_warnings = new_warnings
                    if refreshed_config["draft_order"] and refreshed_config["num_rounds"]:
                        tracker.configure_pick_sequence(
                            refreshed_config["draft_order"], refreshed_config["num_rounds"]
                        )
                st.session_state.system_instructions, st.session_state.chat = rebuild_context_and_chat(
                    st.session_state.client, tracker, st.session_state.chat
                )
                st.session_state.turns_since_reset = 0
        elif st.session_state.turns_since_reset >= RESET_EVERY_N_TURNS:
            with st.spinner("Summarizing context and resetting chat history..."):
                st.session_state.chat = reset_chat_with_recap(
                    st.session_state.client,
                    st.session_state.system_instructions,
                    tracker,
                    st.session_state.chat,
                )
            st.session_state.turns_since_reset = 0

        # --- Image attachments: sent as actual image data alongside this turn's text ---
        message_parts = [
            build_turn_message_ui(
                tracker,
                user_input,
                injury_check=st.session_state.injury_check_enabled,
            )
        ]
        message_parts.extend(build_image_part(f) for f in image_files)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            stream_state = {"text": ""}
            stream_failed = False
            spinner_label = (
                "Searching for injury/news updates and thinking..."
                if st.session_state.injury_check_enabled
                else "Thinking..."
            )

            def _on_stream_reset():
                stream_state["text"] = ""
                placeholder.empty()

            with st.spinner(spinner_label):
                try:
                    for chunk in stream_with_retry(
                        st.session_state.chat.send_message_stream,
                        message_parts,
                        on_retry=lambda attempt, total, delay: st.toast(
                            f"Gemini is overloaded (attempt {attempt}/{total}) — retrying in {delay}s...",
                            icon="⏳",
                        ),
                        on_reset=_on_stream_reset,
                    ):
                        if chunk.text:
                            stream_state["text"] += chunk.text
                            placeholder.markdown(stream_state["text"])
                except (genai_errors.ServerError, genai_errors.ClientError) as e:
                    stream_failed = True
                    stream_state["text"] = (
                        f"⚠️ Gemini is unavailable right now ({e}). Try sending your message again in a moment."
                    )
                    placeholder.markdown(stream_state["text"])

        full_text = stream_state["text"]
        st.session_state.messages.append({"role": "assistant", "content": full_text})
        if not stream_failed:
            st.session_state.turns_since_reset += 1

    # --- Older history, newest-first, below the input/live turn above. This keeps
    # the input pinned at a consistent spot instead of drifting between responses
    # the way it did when history rendered above a bottom-anchored input. ---
    history_to_show = st.session_state.messages[:-2] if process_turn else st.session_state.messages
    for msg in reversed(history_to_show):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for name in msg.get("attachments", []):
                st.caption(f"📎 {name}")