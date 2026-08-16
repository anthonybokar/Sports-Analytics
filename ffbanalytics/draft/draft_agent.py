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

    REQUIRED_COLS = ["Rank", "Player", "Position", "Team", "Tier", "VORP"]

    def __init__(self, csv_filepath: str, top_n: int = 150):
        self.df = self._load_csv(csv_filepath)
        self.top_n = top_n
        self.drafted = {}  # normalized_name -> {"player": str, "pick_no": int}
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

    def draft_player(self, query: str) -> str:
        player = self.find_player(query)
        if not player:
            return f"Couldn't find a player matching '{query}' in the rankings."
        norm = self._normalize(player)
        if norm in self.drafted:
            return f"{player} is already marked as drafted."
        self.pick_counter += 1
        self.drafted[norm] = {"player": player, "pick_no": self.pick_counter}
        return f"Pick #{self.pick_counter}: {player} marked as DRAFTED."

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
        return "\n".join(f"{d['pick_no']}. {d['player']}" for d in rows)


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
    4. Keep answers brief and concise, suitable for a real-time draft clock.
    """


def build_turn_message(tracker: DraftTracker, user_input: str) -> str:
    """Wrap the user's message with the current draft-board context."""
    return f"""### CURRENT AVAILABLE PLAYERS (top {tracker.top_n}, already excludes drafted players)
{tracker.available_markdown()}

### USER MESSAGE
{user_input}
"""


HELP_TEXT = """Commands:
  /draft <name>   Mark a player as drafted (fuzzy name matching supported)
  /undo           Undo the most recent draft pick
  /board          Show the current top available players
  /drafted        List all players drafted so far
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
                print(tracker.draft_player(user_input[len("/draft "):].strip()))
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