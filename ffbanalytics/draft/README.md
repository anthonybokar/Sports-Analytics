# 🏈 Fantasy Football Draft Assistant

An AI-powered draft-day assistant for a 12-team PPR **keeper** league on Yahoo Fantasy Sports. It tracks every pick (including keepers), flags value and tier-cliff opportunities against your own rankings, and answers strategy questions live during the draft via Google Gemini with search grounding — all through a Streamlit UI built for speed and reliability on draft day.

Built for and by a single team owner ("Cellar Dweller," pick 6 of 12, 15 rounds) but generalizable to any snake-draft keeper league given the right config files.

---

## Table of Contents

- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Setup](#setup)
- [Configuration](#configuration)
  - [Rankings CSV](#rankings-csv)
  - [league_config.txt](#league_configtxt)
  - [context_docs_yahoo/ folder](#context_docs_yahoo-folder)
- [Running the App](#running-the-app)
- [Streamlit UI Guide](#streamlit-ui-guide)
- [CLI Commands (draft_agent.py standalone)](#cli-commands-draft_agentpy-standalone)
- [Keeper Cost Rule](#keeper-cost-rule)
- [Yahoo League History Tool](#yahoo-league-history-tool-yahoo_history_pullpy)
- [Known Limitations & Architecture Notes](#known-limitations--architecture-notes)
- [Troubleshooting](#troubleshooting)

---

## Project Structure

```
.
├── draft_agent.py                          # Core logic — source of truth. CLI-runnable standalone.
├── draft_app.py                            # Streamlit UI layer. Imports from draft_agent.py, never modifies it.
├── yahoo_history_pull.py                   # Standalone script to pull full Yahoo league history to JSON.
├── requirements.txt
├── FantasyPros_2026_Overall_ADP_Rankings.csv   # Your rankings CSV (not included — see below).
└── context_docs_yahoo/                     # Drop-in folder for reference material fed to the model.
    ├── league_config.txt                   # Settings + keeper list (source of truth for keeper logic).
    ├── live_notes.txt                      # Auto-generated — your live notes, persisted between sessions.
    └── (any .txt/.md/.csv/.pdf/.xlsx/.rtf you want the assistant to reference)
```

**Design principle:** `draft_agent.py` is treated as the stable core and is only changed when genuinely necessary (e.g. correctness fixes). `draft_app.py` is the Streamlit front end and should absorb UI-only changes whenever possible, importing everything it needs from `draft_agent.py` rather than duplicating logic.

---

## How It Works

1. **`DraftTracker`** (in `draft_agent.py`) loads your rankings CSV and is the single source of truth for who's been drafted, by whom, and in what round.
2. **Keepers** are registered at startup from `league_config.txt` directly into the same "drafted" data structure as a live pick, so they're automatically excluded from the available-players pool and appear on the correct team's roster.
3. **The full draft sequence** (all 180 slots for a 12-team/15-round draft) is precomputed once keepers and draft order are known, so live picks get accurate Round/Pick/Overall numbers even with keeper-forfeited slots sprinkled throughout.
4. **Every chat turn** re-injects the current board state (available players, your roster, tier scarcity, bye-week collisions, live notes, and optionally injury/news search instructions) into the message sent to Gemini — Gemini's chat API is stateless per session, so this context has to be resent every turn rather than relying on the model "remembering" the board.
5. **Chat history resets** every 15 turns (configurable via `RESET_EVERY_N_TURNS`) to control token growth, with a short AI-generated recap carried forward so continuity isn't fully lost.

---

## Setup

### Requirements

- Python 3.10+
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))

### Install

```bash
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your-gemini-api-key-here

# Only needed if you use yahoo_history_pull.py:
YAHOO_CLIENT_ID=your-yahoo-client-id
YAHOO_CLIENT_SECRET=your-yahoo-client-secret
YAHOO_LEAGUE_ID=29604
```

---

## Configuration

### Rankings CSV

Place a CSV at the path set by `CSV_PATH` (default: `FantasyPros_2026_Overall_ADP_Rankings.csv`) in the project root. Recognized columns:

| Column     | Required | Notes                                                        |
|------------|----------|---------------------------------------------------------------|
| `Rank`     | Yes      | Your overall rank.                                            |
| `Player`   | Yes      | Full player name — used for all name matching.                 |
| `Position` | Yes      | e.g. `QB`, `RB`, `WR`, `TE`, `K`, `DST`.                        |
| `Team`     | Yes      | Player's **NFL** team (not your fantasy team).                 |
| `Tier`     | Yes      | Lower number = better tier. Drives tier-cliff/scarcity logic.  |
| `VORP`     | Yes      | Value over replacement.                                       |
| `Bye`      | Yes      | Bye week — coerced to numeric; missing values stay "unknown" rather than defaulting to 0. |
| `AVG`      | Optional | Consensus ADP — powers the "Best Value" panel in the UI.       |

### `league_config.txt`

Lives in `context_docs_yahoo/`. It's a small settings header followed by team blocks listing that team's keepers. This file is the **deterministic source of truth** for keeper attribution, draft order, and pick-sequence numbering — separate from any freeform prose you also drop in the folder for the model to read casually.

```txt
MY_TEAM_NAME: Cellar Dweller
NUM_TEAMS: 12
DRAFT_STYLE: snake
MY_DRAFT_POSITION: 6
ROUNDS: 15
DRAFT_ORDER: Team A, Team B, Cellar Dweller, Team D, Team E, Team F, Team G, Team H, Team I, Team J, Team K, Team L

Team A
- Justin Jefferson, WR, MIN — Keeper cost: Rd 2
- Christian McCaffrey, RB, SF — Keeper cost: Rd 1

Cellar Dweller
- Lamar Jackson, QB, BAL — Keeper cost: Rd 4
```

Notes:
- `DRAFT_ORDER` must list teams in **Round 1** order, position 1..`NUM_TEAMS`.
- Team names in `DRAFT_ORDER` must match the team header lines **exactly** (case-insensitive) — this is a known fragile point; keep files consistent.
- `DRAFT_ORDER` and `ROUNDS` are optional but strongly recommended — without them, pick numbering falls back to a naive running count that doesn't account for keeper-skipped slots.
- Keeper cost is always the round **before** the player's original draft round, regardless of who currently owns them (see [Keeper Cost Rule](#keeper-cost-rule)).

### `context_docs_yahoo/` folder

Any `.txt`, `.md`, `.csv`, `.pdf`, `.xlsx`, or `.rtf` file dropped in this folder is loaded as supplementary reference material at startup (league rules, sleeper notes, injury reports, etc.) and resent as part of the system prompt on every turn. Budget is controlled by `CONTEXT_CHAR_BUDGET` in `load_context_folder()`. Keeper files must have "keeper" somewhere in the filename to be auto-detected by `find_keeper_files()` in some workflows.

`live_notes.txt` in this folder is managed automatically by the Streamlit app's Live Notes box (see below) — you generally don't need to touch it directly.

---

## Running the App

### Streamlit (recommended — full UI)

```bash
streamlit run draft_app.py
```

> ⚠️ **Must be run with `streamlit run`, not `python draft_app.py`.** Running it as plain Python suppresses all UI feedback and will look like nothing is happening.

### CLI (terminal-only, standalone)

```bash
python draft_agent.py
```

The CLI is fully independent of Streamlit and useful for quick testing or a lightweight fallback.

---

## Streamlit UI Guide

**Header row**
- 🔒 **Keepers** — expandable list of registered keepers, for verifying names matched correctly against your CSV.
- 📋 **Edit Draft Log** — edit *any* pick (not just the most recent), reassign a pick to a different team, or remove it. Write-in picks are marked with ✍️.
- ✍️ **Write-in Pick** — log a pick for a player **not in your rankings CSV** (e.g. someone got reached for). Captures name, position, NFL team (optional), and drafting team, and slots correctly into the precomputed pick sequence so Round/Pick numbering stays accurate for the rest of the draft.
- ↩️ **Undo Last Pick** — undo only the most recent pick (use Edit Draft Log for anything further back).
- **Round/Pick metric** — always-visible on-the-clock indicator, including whether it's your pick.

**Available Players table** — check a box in the "✅ [Your Team]" or "✅ Opponent" column to log a pick directly from the board.

**Best Value / Best Available / Tier Cliffs** — three side-by-side panels: biggest ADP-vs-pick-count value gaps, top remaining players regardless of value, and positions at risk of falling off a tier.

**📊 Draft Trends** — a Round × Position bar chart of what's been drafted so far.

**Chat column**
- 🩹 **Injury/News Check** toggle — when on, the assistant searches for the latest injury/news status on any player it recommends that turn.
- 🔍 **Auto Insights** toggle + "Every N picks" control — automatically fires a full board-scan prompt into chat after every *N* live picks (default 5; adjustable 1–15; toggle off to disable entirely). Appears in chat labeled "🔍 Auto Insight."
- 📝 **Live Notes** — a text box for freeform notes ("Team 4 said they're punting RB," "targeting a QB rounds 4–6"). Autosaves to `context_docs_yahoo/live_notes.txt` and is included in every message **without requiring a chat reset**.
- **Quick Prompts** — one-click canned questions (Value, Run Check, Roster, Keepers, Recap) that go through the exact same send path as manually typed chat.
- Chat input supports file attachments (images, PDFs, CSVs, etc.) — documents get saved permanently to `context_docs_yahoo/` and trigger a context rebuild + chat reset; images are sent as part of that turn only.

**My Roster (sidebar)** — your roster with position/tier/bye detail, position counts, bye-week collision warnings, and a CSV export of the full draft log.

---

## CLI Commands (`draft_agent.py` standalone)

| Command | Description |
|---|---|
| `/draft <name>` | Mark a player drafted by an opponent (fuzzy name matching). |
| `/mydraft <name>` | Mark a player drafted by your team. |
| `/undo` | Undo the most recent pick. |
| `/remove <name>` | Remove any drafted player from the log. |
| `/reassign <name> \| <team>` | Fix a pick logged to the wrong team. |
| `/board` | Show current top available players. |
| `/drafted` | List all picks so far, with team. |
| `/roster` | Show your roster, position counts, bye collisions. |
| `/scarcity` | Show tier-cliff warnings. |
| `/pick` | Show what round/pick we're on and who's on the clock. |
| `/help` | Show this help text. |
| `exit` / `quit` | End the session. |

Anything else is sent to the assistant as a normal question.

---

## Keeper Cost Rule

A keeper costs the draft pick **one round earlier than the round in which the player was originally drafted** — regardless of who currently owns them. This is encoded per-keeper in `league_config.txt` (`Keeper cost: Rd N`) rather than computed automatically, since original draft round isn't always reconstructable from current league data alone.

---

## Yahoo League History Tool (`yahoo_history_pull.py`)

A standalone script (doesn't import from or affect `draft_agent.py`/`draft_app.py`) that pulls **every season** your Yahoo league has existed, walking Yahoo's season-to-season "renew" chain automatically, and flattens it into one JSON file — useful as extra context for the assistant (trade history, past standings, etc.) or just as an archive.

### One-time setup

1. Register an app at [developer.yahoo.com/apps](https://developer.yahoo.com/apps/) with **Fantasy Sports (read)** access. Redirect URI can be `https://localhost:8080` or `oob`.
2. Add `YAHOO_CLIENT_ID`, `YAHOO_CLIENT_SECRET`, and `YAHOO_LEAGUE_ID` to `.env` (see [Setup](#setup)).
3. First run opens a browser for you to approve access; paste the verification code back into the terminal. Tokens cache to `yahoo_tokens.json` and auto-refresh after that.

### Usage

```bash
python yahoo_history_pull.py
python yahoo_history_pull.py --include-weekly-rosters      # slower, much bigger
python yahoo_history_pull.py --dump-raw raw_yahoo_json/     # save every raw API response for debugging
python yahoo_history_pull.py --max-seasons 5                # cap how far back to pull
python yahoo_history_pull.py --output my_history.json
```

### Known risk

This script hasn't been run against live Yahoo API responses yet. The riskiest assumptions — the `renew` field format and Yahoo's inconsistent nested JSON shapes — are called out in detail in the script's own docstring. Every per-item extraction is wrapped in try/except, and any mismatch will surface in the `warnings` array of the output JSON. Run with `--dump-raw` first if something looks wrong.

---

## Known Limitations & Architecture Notes

- **Yahoo Fantasy API access**: requested but pending approval as of this writing. No self-serve status dashboard exists — attempting the OAuth flow in `yahoo_history_pull.py` is currently the best way to check.
- **Google Search grounding is session-bound, not per-message** — the Injury/News Check toggle controls whether the *instruction* to search is included in a given turn, but the underlying grounding capability is enabled for the whole chat session, not switched on/off per call. This is an open architectural constraint, not a bug.
- **Team name matching** between `DRAFT_ORDER` and keeper file headers is exact-string (case-insensitive) — a known fragile point, managed by keeping files consistent rather than solved with fuzzy matching.
- **Keeper filenames** must contain the word "keeper" to be auto-detected in some workflows.
- **Keeper exclusions require a full app restart** to take effect — a browser refresh alone won't pick up changes.
- **RTF files** require `striprtf` for parsing; reading them as plain text lets raw markup bleed into the model's context.
- **`remove_pick()` vs. write-in/keeper handling**: removing a pick leaves a numbering gap (a real-world clock slot was consumed); marking as a keeper or logging a write-in instead keeps downstream numbering fully in sync.
- Gemini's chat is stateless — full context is resent every turn, and history is periodically reset with a recap (`RESET_EVERY_N_TURNS`) to control token growth. A recap is a lossy 2–3 sentence summary; anything that must persist reliably (like your actual roster) is sent explicitly by name every turn rather than relied upon from memory.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Streamlit app shows nothing / seems frozen | You ran `python draft_app.py` instead of `streamlit run draft_app.py`. |
| A keeper doesn't show up as excluded | Check the spelling in `league_config.txt` against the exact `Player` string in your rankings CSV — matching is fuzzy but not magic. |
| Keeper changes aren't taking effect | Fully restart the app; a browser refresh isn't enough. |
| Round/Pick numbering looks off after a reach pick | Use **✍️ Write-in Pick** instead of skipping the pick entirely — this keeps the precomputed sequence in sync. |
| `GEMINI_API_KEY not found` error | Confirm `.env` is in the project root and contains a valid key; both scripts call `load_dotenv()` at startup. |
| Model gives outdated info about your own roster | Check the "YOUR ROSTER SO FAR" section of the debug/turn message — it now sends actual player names, not just position counts; if a player's missing there, check `/roster` or the Edit Draft Log to confirm they're logged to the right team. |
| Gemini returns repeated 429/503 errors | Built-in exponential backoff (`call_with_retry`/`stream_with_retry`) retries automatically up to 4 attempts; if it still fails, you've likely hit a real quota limit — wait a bit and try again. |
| RTF context file not loading | Confirm `striprtf` is installed (`pip install striprtf`) and the file actually contains extractable text. |
