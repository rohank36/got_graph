# Work History

## ff9b39a — Add explore.py with comprehensive GoT dataset statistics
- Created `explore.py` to summarize `got.csv` (33,198 rows of GoT dialogue)
- **Basic stats**: row counts, missing values, dtypes, speaker/season/episode/text distributions
- **Graph-relevant**: conversational adjacency (4,617 directed pairs), episode co-occurrence (40,231 undirected pairs)
- **Character-level**: verbosity, vocabulary richness (TTR), character arcs across seasons, longest monologues
- **Temporal/structural**: pacing trends, stage direction ratios, word frequency with per-character TF-IDF
- Identified noise speakers to filter: `CUT TO`, `MAN`, `MAN #2`, `EXT`, `INT`, `ALL`, `CROWD`, etc.
- Updated `.gitignore` to use `*.csv` wildcard, added `for_agent/commit.md` and `for_agent/NOTES.md`