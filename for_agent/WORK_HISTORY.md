# Work History

## b5df9ef — Add vLLM SkyPilot config and entity extraction pipeline
- Created `skypilot.yaml`: GCP us-central1, A10G spot, serves Qwen3-8B-FP8 via vLLM, then runs extraction
- Created `extract.py`: full extraction pipeline — chunks → LLM (OpenAI-compatible API, thinking disabled) → entity resolution → graph upsert → checkpoint
- Graph model: JSON with versioned properties and relationship assertions, each carrying source chunk provenance
- Entity resolution: exact match → alias match → create new; merges properties and aliases on upsert
- Checkpoints after every chunk for spot interruption safety; final output is `graph.json`

## fcecac9 — Add scene-based chunker and scene word count stats to explore.py
- Added `scene_word_count_stats()` to `explore.py`: reports min/median/mean/max/p90/p95 word counts per scene across all seasons (8,356 scenes, median 29 words, max 733 words)
- Created `chunk.py`: scene-based chunker for season 8 (626 chunks; 617 mixed, 6 narration-only, 3 dialogue-only)
- Chunking strategy (Option 3): process per-episode, merge consecutive stage directions into a single block as scene header, attach to following dialogue; trailing narration becomes its own chunk
- Supports `--output/-o` flag to write all chunks to a text file with `season/episode/scene_index` headers
- Season 8 chunk stats: median 30 words, max 388 words, p95 114 words — all well within LLM extraction context limits

## ff9b39a — Add explore.py with comprehensive GoT dataset statistics
- Created `explore.py` to summarize `got.csv` (33,198 rows of GoT dialogue)
- **Basic stats**: row counts, missing values, dtypes, speaker/season/episode/text distributions
- **Graph-relevant**: conversational adjacency (4,617 directed pairs), episode co-occurrence (40,231 undirected pairs)
- **Character-level**: verbosity, vocabulary richness (TTR), character arcs across seasons, longest monologues
- **Temporal/structural**: pacing trends, stage direction ratios, word frequency with per-character TF-IDF
- Identified noise speakers to filter: `CUT TO`, `MAN`, `MAN #2`, `EXT`, `INT`, `ALL`, `CROWD`, etc.
- Updated `.gitignore` to use `*.csv` wildcard, added `for_agent/commit.md` and `for_agent/NOTES.md`