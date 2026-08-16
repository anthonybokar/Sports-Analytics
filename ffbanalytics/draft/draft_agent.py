import os
import sys
import re
import difflib
import pandas as pd
from pypdf import PdfReader
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

if not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError("GEMINI_API_KEY not found in environment variables. Please set it before running the script.")
else:
    print("GEMINI_API_KEY found. Proceeding with Gemini client initialization.")


class DraftTracker:
    """Loads custom CSV rankings and tracks which players have been drafted."""

    REQUIRED_COLS = ["Rank", "Player", "Position", "Team", "Tier", "VORP", "Bye"]

    def __init__(self, csv_filepath: str, top_n: int = 150, my_team_name: str = "My Team"):
        self.df = self._load_csv(csv_filepath)
        self.top_n = top_n
        self.my_team_name = my_team_name
        self.drafted = {}  # normalized_name -> {"player": str, "pick_no": int, "team": str}
        self.pick_counter = 0

    def _load_csv(self, csv_filepath: str) -> pd.DataFrame:
        if not os.path.exists(csv_filepath):
            print(f"Warning: {csv_filepath} not found. Running with empty rankings.")
            return pd.DataFrame(columns=self.REQUIRED_COLS)
        df = pd.read_csv(csv_filepath)
        available_cols = [c for c in self.REQUIRED_COLS if c in df.columns]
        return df[available_cols].reset_index(drop=True)

    @staticmethod
    def _normalize(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def find_player(self, query: str, cutoff: float = 0.6):
        """Match a typed name against the full player pool. Returns the canonical 'Player' string or None."""
        if self.df.empty:
            return None
        norm_query = self._normalize(query)
        norm_pool = {self._normalize(p): p for p in self.df["Player"].tolist()}

        # exact match
        if norm_query in norm_pool:
            return norm_pool[norm_query]
        # substring match (handles "jefferson" -> "Justin Jefferson")
        substr_hits = [orig for norm, orig in norm_pool.items() if norm_query in norm]
        if len(substr_hits) == 1:
            return substr_hits[0]
        # fuzzy fallback
        matches = difflib.get_close_matches(norm_query, norm_pool.keys(), n=1, cutoff=cutoff)
        if matches:
            return norm_pool[matches[0]]
        return None

    def draft_player(self, query: str, team: str = "Opponent") -> str:
        """team is the FANTASY team that made the pick ("My Team" / self.my_team_name,
        or "Opponent" as a catch-all) — not to be confused with the player's NFL team
        in the 'Team' column."""
        player = self.find_player(query)
        if not player:
            return f"Couldn't find a player matching '{query}' in the rankings."
        norm = self._normalize(player)
        if norm in self.drafted:
            return f"{player} is already marked as drafted."
        self.pick_counter += 1
        self.drafted[norm] = {"player": player, "pick_no": self.pick_counter, "team": team}
        label = "your team" if team == self.my_team_name else team
        return f"Pick #{self.pick_counter}: {player} drafted by {label}."

    def undo_last(self) -> str:
        if not self.drafted:
            return "No picks to undo."
        last_norm = max(self.drafted, key=lambda k: self.drafted[k]["pick_no"])
        player = self.drafted.pop(last_norm)["player"]
        self.pick_counter -= 1
        return f"Undid pick: {player} is available again."

    def available_df(self) -> pd.DataFrame:
        if self.df.empty:
            return self.df
        drafted_norms = set(self.drafted.keys())
        mask = ~self.df["Player"].apply(lambda p: self._normalize(p) in drafted_norms)
        return self.df[mask]

    def available_markdown(self, top_n: int = None) -> str:
        n = top_n or self.top_n
        avail = self.available_df().head(n)
        if avail.empty:
            return "No players loaded."
        return avail.to_markdown(index=False)

    def drafted_summary(self) -> str:
        if not self.drafted:
            return "No players drafted yet."
        rows = sorted(self.drafted.values(), key=lambda d: d["pick_no"])
        return "\n".join(f"{d['pick_no']}. {d['player']} ({d.get('team', 'Opponent')})" for d in rows)

    def _roster_rows(self, team: str = None) -> list:
        team = team or self.my_team_name
        return sorted(
            (d for d in self.drafted.values() if d.get("team") == team),
            key=lambda d: d["pick_no"],
        )

    def roster_df(self, team: str = None) -> pd.DataFrame:
        """Full roster detail (position, NFL team, tier, bye) for one fantasy team,
        defaulting to your own (self.my_team_name)."""
        picks = self._roster_rows(team)
        if not picks:
            return pd.DataFrame(columns=["Pick", "Player"])
        rows = []
        for d in picks:
            match = self.df[self.df["Player"] == d["player"]]
            row = {"Pick": d["pick_no"], "Player": d["player"]}
            for col in ["Position", "Team", "Tier", "Bye"]:
                if col in match.columns and not match.empty:
                    row[col] = match[col].iloc[0]
            rows.append(row)
        return pd.DataFrame(rows)

    def roster_position_counts(self, team: str = None) -> dict:
        """Position -> count of drafted players, for spotting positional gaps."""
        picks = self._roster_rows(team)
        if not picks or "Position" not in self.df.columns:
            return {}
        positions = []
        for d in picks:
            match = self.df[self.df["Player"] == d["player"]]
            if not match.empty:
                positions.append(match["Position"].iloc[0])
        return dict(pd.Series(positions).value_counts()) if positions else {}

    def bye_week_collisions(self, team: str = None, threshold: int = 3) -> dict:
        """Bye week -> count, for any week where you've stacked threshold+ players
        on the same team (defaults to your own roster). Empty dict if the rankings
        CSV has no 'Bye' column, or no collisions exist yet."""
        if "Bye" not in self.df.columns:
            return {}
        picks = self._roster_rows(team)
        if not picks:
            return {}
        byes = []
        for d in picks:
            match = self.df[self.df["Player"] == d["player"]]
            if not match.empty and pd.notna(match["Bye"].iloc[0]):
                byes.append(match["Bye"].iloc[0])
        if not byes:
            return {}
        counts = pd.Series(byes).value_counts()
        return {week: int(count) for week, count in counts.items() if count >= threshold}

    def tier_scarcity(self, low_threshold: int = 3) -> pd.DataFrame:
        """For each position, the best (lowest-numbered) tier still available and how
        many players remain in it. Flags tiers at/below low_threshold as scarce — an
        early warning that a positional run could leave you with a real drop-off next
        time it's your pick."""
        avail = self.available_df()
        if avail.empty or "Tier" not in avail.columns or "Position" not in avail.columns:
            return pd.DataFrame()
        rows = []
        for position, group in avail.groupby("Position"):
            best_tier = group["Tier"].min()
            remaining = int((group["Tier"] == best_tier).sum())
            rows.append({
                "Position": position,
                "Best Remaining Tier": best_tier,
                "Players Left in Tier": remaining,
                "Scarce": remaining <= low_threshold,
            })
        return (
            pd.DataFrame(rows)
            .sort_values(["Scarce", "Players Left in Tier"], ascending=[False, True])
            .reset_index(drop=True)
        )


def load_context_folder(folder_path: str, max_chars: int = 1000000) -> str:
    """Load supplementary reference files (.txt, .md, .csv, .pdf, .xlsx) from a local folder and
    concatenate them for injection into the system prompt. Static, loaded once at startup —
    good for cheat sheets, league rules, sleeper notes, injury reports, etc.

    Note: this gets resent on every API call (Gemini chat sessions replay the full system
    instruction + history each turn), so max_chars trades off recurring token cost for more
    reference material. It won't cause billing charges on an unbilled project — going over
    free-tier quota just produces 429s, not a bill — but a larger budget does mean more
    tokens per message and a higher chance of hitting rate limits during a long session.
    """
    if not os.path.isdir(folder_path):
        return ""
    supported_ext = {".txt", ".md", ".csv", ".pdf", ".xlsx", ".rtf"}
    sections = []
    total_chars = 0
    for fname in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in supported_ext:
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            if ext == ".csv":
                content = pd.read_csv(fpath).to_markdown(index=False)
            elif ext == ".xlsx":
                sheets = pd.read_excel(fpath, sheet_name=None)  # dict of {sheet_name: DataFrame}
                sheet_blocks = [
                    f"[Sheet: {sheet_name}]\n{df.to_markdown(index=False)}"
                    for sheet_name, df in sheets.items()
                ]
                content = "\n\n".join(sheet_blocks)
            elif ext == ".pdf":
                reader = PdfReader(fpath)
                content = "\n".join(page.extract_text() or "" for page in reader.pages)
                if not content.strip():
                    print(f"Warning: no extractable text in {fname} (likely a scanned/image PDF); skipping.")
                    continue
            else:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
        except Exception as e:
            print(f"Warning: could not read {fname}: {e}")
            continue

        remaining = max_chars - total_chars
        if remaining <= 0:
            print(f"Warning: context budget ({max_chars} chars) exhausted; skipping {fname} and any later files.")
            break
        if len(content) > remaining:
            content = content[:remaining] + "\n...[truncated to fit context budget]"
            print(f"Warning: {fname} truncated to fit context budget.")
        sections.append(f"### FILE: {fname}\n{content}")
        total_chars += len(content)

    if sections:
        print(f"Loaded {len(sections)} context file(s) from '{folder_path}' ({total_chars} chars).")
    return "\n\n".join(sections)


def build_system_instructions(extra_context: str = "") -> str:
    """Static system prompt. The live available-players table is injected per-turn instead,
    since Gemini chat sessions can't have their system_instruction edited mid-conversation."""
    extra_block = f"""
    ### SUPPLEMENTARY REFERENCE MATERIAL
    Use the following user-provided reference files as additional context when relevant
    (e.g. injury notes, league rules, sleeper picks, your own strategy notes).
    If it conflicts with the live rankings/Tiers/VORP data, the live data wins.
    {extra_context}
    """ if extra_context else ""

    return f"""
    You are an elite Fantasy Football Draft Assistant.
    You MUST adhere strictly to the custom user rankings and projection data provided in each message,
    under the "CURRENT AVAILABLE PLAYERS" heading. That table reflects who is still on the board RIGHT NOW —
    treat any player not listed there as already drafted and unavailable, even if you suggested them earlier.
    {extra_block}
    ### DRAFT STRATEGY RULES:
    1. Base all player value comparisons and suggestions on the provided ranks, Tiers, and VORP.
    2. When asked for recommendations, highlight available players from the highest remaining Tiers.
    3. Never recommend a player who is not present in the current available-players table.
    4. Use the "YOUR ROSTER SO FAR" section to factor in positional need — don't keep recommending a
       position you're already deep at unless specifically asked.
    5. Use "POSITIONAL SCARCITY / TIER CLIFFS" to flag urgency — if a position you need is about to
       fall off a tier cliff, say so proactively, even if not directly asked.
    6. Use "YOUR BYE WEEK COLLISIONS" to note if a suggested pick would worsen an existing bye-week
       stack, or to prefer a similarly-ranked player with a different bye when it's close.
    7. Keep answers brief and concise, suitable for a real-time draft clock.
    """


def build_turn_message(tracker: DraftTracker, user_input: str) -> str:
    """Wrap the user's message with the current draft-board context, your roster
    composition, tier-scarcity signals, and bye-week collision warnings."""
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

    return f"""### CURRENT AVAILABLE PLAYERS (top {tracker.top_n}, already excludes drafted players)
{tracker.available_markdown()}

### YOUR ROSTER SO FAR (by position)
{roster_line}

### POSITIONAL SCARCITY / TIER CLIFFS (available players only)
{scarcity_line}

### YOUR BYE WEEK COLLISIONS (3+ players sharing a bye)
{bye_line}

### USER MESSAGE
{user_input}
"""


HELP_TEXT = """Commands:
  /draft <name>   Mark a player as drafted BY AN OPPONENT (fuzzy name matching supported)
  /mydraft <name> Mark a player as drafted BY YOUR TEAM
  /undo           Undo the most recent draft pick (either team)
  /board          Show the current top available players
  /drafted        List all players drafted so far, with which team took each
  /roster         Show your roster, position counts, and any bye-week collisions
  /scarcity       Show positional scarcity / tier-cliff warnings
  /help           Show this help message
  exit | quit     End the session
Anything else is sent to the assistant as a normal question."""

RESET_EVERY_N_TURNS = 15  # reset chat history after this many normal (non-command) turns


def create_chat_session(client, system_instructions: str):
    """Create a fresh chat session. Used both at startup and on periodic history resets."""
    return client.chats.create(
        model="gemini-3.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instructions,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
    )


def get_recap(chat) -> str:
    """Ask the current chat to summarize any strategic context worth carrying forward,
    right before its history gets wiped."""
    try:
        response = chat.send_message(
            "Before we continue, summarize in 2-3 short sentences any strategic preferences, "
            "patterns, or notes from our conversation so far that would help you keep advising me "
            "well after this point (e.g. a strategy I mentioned, positions I said I'm punting, "
            "players I said I like/dislike). Be concise, no filler."
        )
        return response.text.strip()
    except Exception as e:
        print(f"(Warning: couldn't generate recap, continuing without it: {e})")
        return "No recap available."


def reset_chat_with_recap(client, system_instructions: str, tracker: DraftTracker, chat):
    """Wipe chat history to control token growth, but seed the new session with a short
    recap plus the current draft state so continuity isn't fully lost."""
    print("\n(Resetting chat history to control token growth — summarizing context first...)")
    recap = get_recap(chat)
    new_chat = create_chat_session(client, system_instructions)
    seed_message = f"""### SESSION RECAP (context carried forward after a history reset)
Prior discussion summary: {recap}

Drafted so far:
{tracker.drafted_summary()}

Continue assisting with the draft using this context. Acknowledge briefly, then wait for my next question."""
    seed_response = new_chat.send_message(seed_message)
    print(f"Assistant: {seed_response.text}\n")
    return new_chat


def main():
    csv_path = "FantasyPros_2026_Overall_ADP_Rankings.csv"
    tracker = DraftTracker(csv_path, top_n=150)

    context_folder = "context_docs"
    extra_context = load_context_folder(context_folder)

    system_instructions = build_system_instructions(extra_context)

    client = genai.Client()
    try:
        chat = create_chat_session(client, system_instructions)
    except Exception as e:
        print(e)
        for model in client.models.list():
            print(f"Available model: {model.name}")
        return

    print("--- Fantasy Football Agent Loaded with Custom CSV Data ---")
    print("Type '/help' to see commands, 'exit' to quit.\n")

    init_response = chat.send_message(
        build_turn_message(tracker, "Confirm that you have loaded my rankings. List my top 3 overall players.")
    )
    print(f"Assistant: {init_response.text}\n")

    turns_since_reset = 0

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Ending draft session. Good luck!")
                break

            # --- Deterministic commands, handled locally (no LLM call, no ambiguity) ---
            if user_input.lower().startswith("/draft "):
                print(tracker.draft_player(user_input[len("/draft "):].strip(), team="Opponent"))
                continue
            if user_input.lower().startswith("/mydraft "):
                print(tracker.draft_player(user_input[len("/mydraft "):].strip(), team=tracker.my_team_name))
                continue
            if user_input.lower() == "/undo":
                print(tracker.undo_last())
                continue
            if user_input.lower() == "/board":
                print(tracker.available_markdown(top_n=20))
                continue
            if user_input.lower() == "/drafted":
                print(tracker.drafted_summary())
                continue
            if user_input.lower() == "/roster":
                roster = tracker.roster_df()
                print(roster.to_markdown(index=False) if not roster.empty else "No players drafted to your team yet.")
                counts = tracker.roster_position_counts()
                if counts:
                    print("Position counts: " + ", ".join(f"{pos}: {n}" for pos, n in counts.items()))
                byes = tracker.bye_week_collisions()
                if byes:
                    print("Bye week collisions: " + ", ".join(f"Week {wk}: {n} players" for wk, n in byes.items()))
                continue
            if user_input.lower() == "/scarcity":
                scarcity = tracker.tier_scarcity()
                print(scarcity.to_markdown(index=False) if not scarcity.empty else "No tier data available.")
                continue
            if user_input.lower() == "/help":
                print(HELP_TEXT)
                continue

            # --- Periodic history reset to keep token growth from compounding ---
            if turns_since_reset >= RESET_EVERY_N_TURNS:
                chat = reset_chat_with_recap(client, system_instructions, tracker, chat)
                turns_since_reset = 0

            # --- Normal chat turn: inject current board state alongside the question ---
            print("\nAssistant: ", end="", flush=True)
            response_stream = chat.send_message_stream(build_turn_message(tracker, user_input))
            for chunk in response_stream:
                if chunk.text:
                    sys.stdout.write(chunk.text)
                    sys.stdout.flush()
            print("\n")
            turns_since_reset += 1
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()