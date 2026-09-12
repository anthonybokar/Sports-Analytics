import os
import sys
import re
import difflib
import pandas as pd
from pypdf import PdfReader
from striprtf.striprtf import rtf_to_text
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
        self.drafted = {}  # normalized_name -> {"player": str, "pick_no": int, "team": str, ...}
        self.pick_counter = 0
        self._keeper_seq = 0  # counts down (0, -1, -2, ...) so keepers always sort before live pick #1

    def _load_csv(self, csv_filepath: str) -> pd.DataFrame:
        if not os.path.exists(csv_filepath):
            print(f"Warning: {csv_filepath} not found. Running with empty rankings.")
            return pd.DataFrame(columns=self.REQUIRED_COLS)
        df = pd.read_csv(csv_filepath)
        available_cols = [c for c in self.REQUIRED_COLS if c in df.columns]
        df = df[available_cols].reset_index(drop=True)
        if "Bye" in df.columns:
            # Coerce to numeric (NaN for anything unparseable/missing) instead of leaving
            # it as a raw string/mixed-type column — bye_week_collisions()'s pd.notna()
            # check and value_counts() grouping both depend on a clean numeric dtype.
            # Deliberately NOT filling NaN with 0: a missing bye should stay "unknown",
            # not silently become a fake "Week 0" that gets counted alongside real byes.
            df["Bye"] = pd.to_numeric(df["Bye"], errors="coerce")
        return df

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

    def configure_pick_sequence(self, draft_order: list, num_rounds: int):
        """Precompute the full-draft snake sequence and derive which slots are pre-filled
        by keepers from self.drafted (call this AFTER register_keepers()). Safe to call
        again later (e.g. after a league_config.txt re-upload adds/changes keepers) — it
        just rebuilds from scratch each time.

        Once configured, draft_player() attaches accurate round/overall-pick numbers to
        every live pick, draft_progress() can report what's on the clock, and
        true_overall_pick_count() gives an ADP-comparable pick count that isn't undercounted
        by keeper-forfeited slots. If draft_order/num_rounds aren't available (e.g. not yet
        added to league_config.txt), this is simply never called and everything downstream
        falls back to the old pick_counter-only behavior.
        """
        keeper_slots = {
            (d["team"], d["keeper_round"])
            for d in self.drafted.values()
            if d.get("is_keeper") and d.get("keeper_round")
        }
        self.draft_order = draft_order
        self.num_rounds = num_rounds
        self._pick_sequence = build_pick_sequence(draft_order, num_rounds, keeper_slots)
        self._live_queue = [s for s in self._pick_sequence if not s["is_keeper"]]

    def draft_progress(self) -> dict:
        """Snapshot for an always-visible 'what round/pick are we on' UI element. Returns {}
        if configure_pick_sequence() hasn't been called. 'next' is the live slot about to be
        picked (None if the draft is fully complete); 'last' is the live slot most recently
        filled (None if no live picks yet)."""
        if not getattr(self, "_live_queue", None):
            return {}
        idx = self.pick_counter  # live picks made so far == index of the next unfilled slot
        return {
            "next": self._live_queue[idx] if idx < len(self._live_queue) else None,
            "last": self._live_queue[idx - 1] if idx >= 1 else None,
            "live_picks_made": idx,
            "live_picks_total": len(self._live_queue),
            "num_rounds": self.num_rounds,
        }

    def true_overall_pick_count(self) -> int:
        """True number of draft slots (keeper + live) resolved so far — for ADP-based value
        comparisons. Corrects the plain pick_counter's undercount once keeper-forfeited slots
        start passing (a naive live-only count no longer matches 'how far into the draft we
        really are'). Falls back to pick_counter if the sequence isn't configured."""
        if not getattr(self, "_live_queue", None):
            return self.pick_counter
        idx = self.pick_counter
        if idx < len(self._live_queue):
            return self._live_queue[idx]["overall_pick"] - 1
        return len(self._pick_sequence)

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
        # Look up this pick's true slot BEFORE incrementing pick_counter, since pick_counter
        # doubles as the index of the next unfilled slot in the live queue.
        slot = None
        live_queue = getattr(self, "_live_queue", None)
        if live_queue and self.pick_counter < len(live_queue):
            slot = live_queue[self.pick_counter]
        self.pick_counter += 1
        entry = {"player": player, "pick_no": self.pick_counter, "team": team}
        if slot:
            entry["round"] = slot["round"]
            entry["overall_pick"] = slot["overall_pick"]
        self.drafted[norm] = entry
        label = "your team" if team == self.my_team_name else team
        slot_note = f" (Round {slot['round']}, Overall #{slot['overall_pick']})" if slot else ""
        return f"Pick #{self.pick_counter}: {player} drafted by {label}.{slot_note}"

    def draft_player_manual(self, name: str, position: str = "Unknown", team: str = "Opponent",
                             nfl_team: str = None, tier=None) -> str:
        """Log a pick for a player who ISN'T in the rankings CSV — e.g. a reach pick nobody
        had ranked. Exists because draft_player() requires a find_player() match against
        self.df; without this, an unranked pick simply couldn't be logged, which silently
        desyncs pick_counter/_live_queue from the real draft (every slot after it reports
        the wrong round/overall-pick number for the rest of the draft).

        Mirrors draft_player()'s slot-assignment logic exactly (same pick_counter/live_queue
        advance) so numbering stays correct, but stores position/nfl_team/tier directly on
        the entry (is_manual=True) since there's no CSV row to join against later in
        roster_df()/roster_position_counts().
        """
        name = name.strip()
        if not name:
            return "Player name can't be empty."
        norm = self._normalize(name)
        if norm in self.drafted:
            return f"{self.drafted[norm]['player']} is already marked as drafted."
        slot = None
        live_queue = getattr(self, "_live_queue", None)
        if live_queue and self.pick_counter < len(live_queue):
            slot = live_queue[self.pick_counter]
        self.pick_counter += 1
        entry = {
            "player": name,
            "pick_no": self.pick_counter,
            "team": team,
            "is_manual": True,
            "position": position or "Unknown",
            "nfl_team": nfl_team,
            "tier": tier,
        }
        if slot:
            entry["round"] = slot["round"]
            entry["overall_pick"] = slot["overall_pick"]
        self.drafted[norm] = entry
        label = "your team" if team == self.my_team_name else team
        slot_note = f" (Round {slot['round']}, Overall #{slot['overall_pick']})" if slot else ""
        return f"Pick #{self.pick_counter}: {name} (write-in, {position}) drafted by {label}.{slot_note}"

    def undo_last(self) -> str:
        if not self.drafted:
            return "No picks to undo."
        last_norm = max(self.drafted, key=lambda k: self.drafted[k]["pick_no"])
        entry = self.drafted.pop(last_norm)
        # Only decrement pick_counter for a real live pick — keepers use negative
        # pick_no and were never counted in pick_counter to begin with.
        if not entry.get("is_keeper"):
            self.pick_counter -= 1
        return f"Undid pick: {entry['player']} is available again."

    def _match_drafted(self, query: str):
        """Fuzzy-match a query against currently-drafted players only (not the full
        rankings pool) — used by remove_pick/reassign_pick, which by definition need
        to target something already on someone's roster."""
        if not self.drafted:
            return None
        norm_query = self._normalize(query)
        pool = {norm: d["player"] for norm, d in self.drafted.items()}
        if norm_query in pool:
            return norm_query
        substr_hits = [norm for norm, name in pool.items() if norm_query in norm]
        if len(substr_hits) == 1:
            return substr_hits[0]
        matches = difflib.get_close_matches(norm_query, pool.keys(), n=1, cutoff=0.6)
        return matches[0] if matches else None

    def remove_pick(self, query: str) -> str:
        """Remove any single pick from the log, not just the most recent one — for
        correcting a mistake that wasn't caught until several picks later. Leaves a
        gap in pick_no rather than renumbering everything after it; pick_no is just
        a chronological log id, not required to be contiguous."""
        norm = self._match_drafted(query)
        if not norm:
            return f"Couldn't find a drafted player matching '{query}'."
        entry = self.drafted.pop(norm)
        tag = " (was a keeper)" if entry.get("is_keeper") else ""
        return f"Removed {entry['player']}{tag} — available again."

    def reassign_pick(self, query: str, new_team: str) -> str:
        """Fix a pick logged to the wrong team without a remove+redraft round trip
        (which would also lose its original pick_no / draft-order position)."""
        norm = self._match_drafted(query)
        if not norm:
            return f"Couldn't find a drafted player matching '{query}'."
        old_team = self.drafted[norm].get("team", "Opponent")
        self.drafted[norm]["team"] = new_team
        return f"Reassigned {self.drafted[norm]['player']} from {old_team} to {new_team}."

    def register_keepers(self, keepers: list, my_team_name: str) -> list:
        """Register pre-draft keepers directly into self.drafted, same data structure
        as a live pick, so they automatically show up everywhere drafted players
        already do: excluded from available_df(), included in roster_df()/
        roster_position_counts()/bye_week_collisions() for whichever team holds them.

        keepers: list of {"team", "player", "position", "nfl_team", "round"} dicts,
        as produced by parse_league_config(). Uses negative pick_no values (via
        self._keeper_seq) so keepers always sort before real pick #1 in
        drafted_summary()/roster_df(), and so undo_last() never targets a keeper
        while any live pick exists.

        Returns a list of warning strings for any keeper name that couldn't be
        matched to the rankings CSV (e.g. a spelling mismatch) — surface these to
        the user, since a silently-unmatched keeper would otherwise still show up
        as "available" on the board.
        """
        warnings = []
        for k in keepers:
            resolved = self.find_player(k["player"])
            if not resolved:
                warnings.append(
                    f"Keeper '{k['player']}' ({k['team']}) not found in rankings CSV — check spelling/name match."
                )
                continue
            norm = self._normalize(resolved)
            if norm in self.drafted:
                continue  # already registered (e.g. Streamlit rerun) — don't double-count
            team_label = (
                my_team_name if k["team"].strip().lower() == my_team_name.strip().lower() else k["team"]
            )
            self._keeper_seq -= 1
            self.drafted[norm] = {
                "player": resolved,
                "pick_no": self._keeper_seq,
                "team": team_label,
                "is_keeper": True,
                "keeper_round": k.get("round"),
            }
        return warnings

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
        lines = []
        for d in rows:
            if d.get("is_keeper"):
                round_note = f", Rd {d['keeper_round']} cost" if d.get("keeper_round") else ""
                lines.append(f"(Keeper) {d['player']} ({d.get('team', 'Opponent')}{round_note})")
            else:
                tag = " [write-in]" if d.get("is_manual") else ""
                lines.append(f"{d['pick_no']}. {d['player']}{tag} ({d.get('team', 'Opponent')})")
        return "\n".join(lines)

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
            if d.get("is_keeper"):
                pick_label = f"Keeper (Rd {d['keeper_round']})" if d.get("keeper_round") else "Keeper"
            else:
                pick_label = d["pick_no"]
            row = {"Pick": pick_label, "Player": d["player"]}
            for col in ["Position", "Team", "Tier", "Bye"]:
                if col in match.columns and not match.empty:
                    row[col] = match[col].iloc[0]
            if d.get("is_manual"):
                # No CSV row to join against — use what was captured at write-in time instead.
                row.setdefault("Position", d.get("position", "Unknown"))
                if d.get("nfl_team"):
                    row.setdefault("Team", d["nfl_team"])
                if d.get("tier") is not None:
                    row.setdefault("Tier", d["tier"])
            rows.append(row)
        return pd.DataFrame(rows)

    def roster_position_counts(self, team: str = None) -> dict:
        """Position -> count of drafted players, for spotting positional gaps. Write-in
        picks (no CSV row) count using the position captured at write-in time, so a
        reach pick doesn't silently vanish from your positional totals."""
        picks = self._roster_rows(team)
        if not picks:
            return {}
        positions = []
        for d in picks:
            if d.get("is_manual"):
                positions.append(d.get("position") or "Unknown")
                continue
            if "Position" not in self.df.columns:
                continue
            match = self.df[self.df["Player"] == d["player"]]
            if not match.empty:
                positions.append(match["Position"].iloc[0])
        return dict(pd.Series(positions).value_counts()) if positions else {}

    def roster_summary_by_position(self, team: str = None) -> str:
        """'Position: Player, Player' breakdown of a roster, WITH NAMES — for direct
        injection into the LLM prompt every turn. roster_position_counts() alone (just
        numbers, e.g. "QB: 1") isn't enough context for the model to reliably answer
        questions like "who's my QB" or reason about a specific player's bye week; the
        model only otherwise "knows" a name from having seen it mentioned earlier in
        chat history, which is exactly what a RESET_EVERY_N_TURNS recap can lose.
        """
        picks = self._roster_rows(team)
        if not picks:
            return "No players drafted to your team yet"
        by_position = {}
        for d in picks:
            if d.get("is_manual") and d.get("position"):
                position = d["position"]
            else:
                match = self.df[self.df["Player"] == d["player"]]
                if not match.empty and "Position" in self.df.columns:
                    position = match["Position"].iloc[0]
                else:
                    position = "Unknown"
            by_position.setdefault(position, []).append(d["player"])
        return "; ".join(f"{pos}: {', '.join(players)}" for pos, players in by_position.items())

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
            elif ext == ".rtf":
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                content = rtf_to_text(raw)
                if not content.strip():
                    print(f"Warning: no extractable text in {fname} after RTF parsing; skipping.")
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


_LEAGUE_SETTING_RE = re.compile(r"^([A-Z_]+):\s*(.+)$")
_KEEPER_LINE_RE = re.compile(
    r"^-\s*(?P<player>[^,]+),\s*(?P<pos>[^,]+),\s*(?P<nfl>[^\u2014-]+?)\s*[\u2014-]+\s*"
    r"Keeper cost:\s*Rd\s*(?P<round>\d+)",
    re.IGNORECASE,
)


def parse_league_config(filepath: str) -> dict:
    """Parse a league_config.txt: a small settings header (KEY: value lines) followed
    by the same team-header / '- Player, POS, TEAM — Keeper cost: Rd N' keeper blocks
    used in the ESPN keepers screenshot export. Deterministic, regex-based — this is
    the actual source of truth for keeper team attribution and round cost, separate
    from the freeform prose that load_context_folder() feeds the model for general
    context. Missing file or missing fields degrade gracefully (None / empty list)
    rather than raising, since a lot of this is optional today and only fully used
    once the pick-counting work lands.
    """
    result = {
        "my_team_name": None,
        "num_teams": None,
        "draft_style": None,
        "my_draft_position": None,
        "num_rounds": None,
        "draft_order": None,  # list of team names in Round-1 pick order (position 1..NUM_TEAMS)
        "keepers": [],  # list of {"team", "player", "position", "nfl_team", "round"}
    }
    if not os.path.exists(filepath):
        return result

    settings = {}
    keepers = []
    current_team = None
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("(") or line.startswith("###"):
                continue
            m_keeper = _KEEPER_LINE_RE.match(line)
            if m_keeper:
                if current_team is None:
                    print(f"Warning: keeper line found before any team header, skipping: {line}")
                    continue
                keepers.append({
                    "team": current_team,
                    "player": m_keeper.group("player").strip(),
                    "position": m_keeper.group("pos").strip(),
                    "nfl_team": m_keeper.group("nfl").strip(),
                    "round": int(m_keeper.group("round")),
                })
                continue
            if line.startswith("-"):
                print(f"Warning: couldn't parse keeper line, skipping: {line}")
                continue
            m_setting = _LEAGUE_SETTING_RE.match(line)
            if m_setting:
                settings[m_setting.group(1)] = m_setting.group(2).strip()
                continue
            # Anything else non-blank, non-"-", non-"KEY: value" is treated as a team header
            current_team = line

    result["my_team_name"] = settings.get("MY_TEAM_NAME") or None
    result["draft_style"] = settings.get("DRAFT_STYLE") or None
    if "NUM_TEAMS" in settings:
        try:
            result["num_teams"] = int(settings["NUM_TEAMS"])
        except ValueError:
            pass
    if "MY_DRAFT_POSITION" in settings:
        try:
            result["my_draft_position"] = int(settings["MY_DRAFT_POSITION"])
        except ValueError:
            pass
    if "ROUNDS" in settings:
        try:
            result["num_rounds"] = int(settings["ROUNDS"])
        except ValueError:
            pass
    if "DRAFT_ORDER" in settings:
        # Comma-separated Round-1 team order, position 1..NUM_TEAMS. Team names must match
        # the keeper-block headers above exactly (case-insensitive) so keeper slots can be
        # placed correctly in build_pick_sequence().
        result["draft_order"] = [t.strip() for t in settings["DRAFT_ORDER"].split(",") if t.strip()]
    result["keepers"] = keepers
    return result


def build_pick_sequence(draft_order: list, num_rounds: int, keeper_slots: set) -> list:
    """Precompute every draft slot for the whole draft, in true overall order, standard snake
    (Round 1 order, then reversed, alternating). Each slot is tagged with which team picks
    there and whether that slot is pre-filled by a keeper.

    draft_order: list of team names in Round-1 order (position 1..N).
    keeper_slots: set of (team_name, round_number) tuples — a keeper occupies that team's
    slot in that specific round, so it's never a live pick.

    This is what makes pick numbering match Yahoo exactly even when keeper-forfeited slots
    are silently skipped in the live draft: a live pick's true round/overall-pick number is
    just wherever it falls in this precomputed sequence, not a naive running count of clicks.
    """
    sequence = []
    overall = 0
    for r in range(1, num_rounds + 1):
        order = draft_order if r % 2 == 1 else list(reversed(draft_order))
        for pick_in_round, team in enumerate(order, start=1):
            overall += 1
            sequence.append({
                "round": r,
                "pick_in_round": pick_in_round,
                "overall_pick": overall,
                "team": team,
                "is_keeper": (team, r) in keeper_slots,
            })
    return sequence


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
    roster_line = tracker.roster_summary_by_position()

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

### YOUR ROSTER SO FAR (by position, with player names)
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
  /remove <n>     Remove any drafted player from the log (not just the most recent)
  /reassign <n> | <team>   Fix a pick logged to the wrong team, e.g. /reassign Bijan | My Team
  /board          Show the current top available players
  /drafted        List all players drafted so far, with which team took each
  /roster         Show your roster, position counts, and any bye-week collisions
  /scarcity       Show positional scarcity / tier-cliff warnings
  /pick           Show what round/pick we're on and who's on the clock
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

    # context_folder = "context_docs_espn"
    context_folder = "context_docs_yahoo"
    league_config = parse_league_config(os.path.join(context_folder, "league_config.txt"))
    tracker = DraftTracker(csv_path, top_n=150, my_team_name=league_config["my_team_name"] or "My Team")

    if league_config["keepers"]:
        warnings = tracker.register_keepers(league_config["keepers"], tracker.my_team_name)
        print(f"Registered {len(league_config['keepers']) - len(warnings)} keeper(s) from league_config.txt.")
        for w in warnings:
            print(f"Warning: {w}")

    if league_config["draft_order"] and league_config["num_rounds"]:
        tracker.configure_pick_sequence(league_config["draft_order"], league_config["num_rounds"])
    else:
        print("Note: DRAFT_ORDER/ROUNDS not set in league_config.txt — pick numbering won't account for keeper-skipped slots.")

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
            if user_input.lower().startswith("/remove "):
                print(tracker.remove_pick(user_input[len("/remove "):].strip()))
                continue
            if user_input.lower().startswith("/reassign "):
                # Usage: /reassign <player> | <team>
                payload = user_input[len("/reassign "):].strip()
                if "|" not in payload:
                    print("Usage: /reassign <player> | <team>")
                    continue
                player_part, team_part = payload.split("|", 1)
                print(tracker.reassign_pick(player_part.strip(), team_part.strip()))
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
            if user_input.lower() == "/pick":
                progress = tracker.draft_progress()
                if not progress:
                    print("Pick sequence not configured — add DRAFT_ORDER and ROUNDS to league_config.txt.")
                elif progress["next"] is None:
                    print(f"Draft complete — {progress['live_picks_made']} live picks made.")
                else:
                    nxt = progress["next"]
                    you = " (YOU)" if nxt["team"] == tracker.my_team_name else ""
                    print(f"On the clock: Round {nxt['round']}, Pick {nxt['pick_in_round']} (Overall #{nxt['overall_pick']}) — {nxt['team']}{you}")
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