"""Explore and summarize the Game of Thrones dialogue dataset (got.csv)."""

from collections import Counter

import pandas as pd


def load_data(path: str = "got.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def basic_stats(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("BASIC OVERVIEW")
    print("=" * 60)
    print(f"  Total rows        : {len(df):,}")
    print(f"  Columns           : {list(df.columns)}")
    print()
    print("  Missing values per column:")
    for col in df.columns:
        n_miss = df[col].isna().sum()
        pct = n_miss / len(df) * 100
        print(f"    {col:<15s} : {n_miss:>6,}  ({pct:.1f}%)")
    print()
    print("  Dtypes:")
    for col in df.columns:
        print(f"    {col:<15s} : {df[col].dtype}")


def speaker_stats(df: pd.DataFrame) -> None:
    speakers = df["Speaker"].dropna()
    counts = speakers.value_counts()

    print()
    print("=" * 60)
    print("SPEAKER STATISTICS")
    print("=" * 60)
    print(f"  Unique speakers   : {counts.shape[0]:,}")
    print(f"  Rows with speaker : {len(speakers):,}")
    print(f"  Rows without      : {df['Speaker'].isna().sum():,}  (stage directions / narration)")
    print()
    print("  Top 20 speakers by line count:")
    for rank, (name, count) in enumerate(counts.head(20).items(), 1):
        pct = count / len(speakers) * 100
        print(f"    {rank:>2}. {name:<25s}  {count:>5,} lines  ({pct:.1f}%)")


def season_stats(df: pd.DataFrame) -> None:
    print()
    print("=" * 60)
    print("SEASON STATISTICS")
    print("=" * 60)
    season_counts = df["Season"].value_counts().sort_index()
    for season, count in season_counts.items():
        n_episodes = df.loc[df["Season"] == season, "Episode"].nunique()
        n_speakers = df.loc[df["Season"] == season, "Speaker"].nunique()
        print(f"  {season:<12s} : {count:>5,} rows | {n_episodes:>2} episodes | {n_speakers:>3} speakers")


def episode_stats(df: pd.DataFrame) -> None:
    print()
    print("=" * 60)
    print("EPISODE STATISTICS")
    print("=" * 60)
    ep_counts = df.groupby(["Season", "Episode"]).size()
    print(f"  Total unique episodes : {df['Episode'].nunique()}")
    print(f"  Rows per episode:")
    print(f"    min   : {ep_counts.min():>5,}")
    print(f"    median: {int(ep_counts.median()):>5,}")
    print(f"    max   : {ep_counts.max():>5,}")
    print()

    biggest = ep_counts.nlargest(5)
    print("  Top 5 episodes by row count:")
    for (season, episode), count in biggest.items():
        print(f"    {season} / {episode:<30s} : {count:>5,} rows")


def text_stats(df: pd.DataFrame) -> None:
    print()
    print("=" * 60)
    print("TEXT STATISTICS")
    print("=" * 60)
    texts = df["Text"].dropna()
    lengths = texts.str.len()
    word_counts = texts.str.split().str.len()

    print(f"  Non-empty text rows : {len(texts):,}")
    print()
    print("  Character length per line:")
    print(f"    min   : {lengths.min():>6,}")
    print(f"    median: {int(lengths.median()):>6,}")
    print(f"    mean  : {lengths.mean():>9,.1f}")
    print(f"    max   : {lengths.max():>6,}")
    print()
    print("  Word count per line:")
    print(f"    min   : {word_counts.min():>6,}")
    print(f"    median: {int(word_counts.median()):>6,}")
    print(f"    mean  : {word_counts.mean():>9,.1f}")
    print(f"    max   : {word_counts.max():>6,}")


NON_CHARACTERS = {"CUT TO", "ALL", "CROWD", "MEN", "SOLDIERS", "BOYS", "WOMEN"}


def dialogue_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows that are actual character dialogue."""
    mask = df["Speaker"].notna() & ~df["Speaker"].isin(NON_CHARACTERS)
    return df[mask].copy()


# ── Graph-relevant statistics ────────────────────────────────


def conversational_adjacency(df: pd.DataFrame) -> None:
    """Who speaks right after whom? Directed edge weights for a character graph."""
    print()
    print("=" * 60)
    print("CONVERSATIONAL ADJACENCY  (speaker A -> speaker B)")
    print("=" * 60)

    dial = dialogue_only(df)
    pairs: Counter[tuple[str, str]] = Counter()

    for (season, episode), group in dial.groupby(["Season", "Episode"]):
        speakers = group["Speaker"].tolist()
        for a, b in zip(speakers, speakers[1:]):
            if a != b:
                pairs[(a, b)] += 1

    top = pairs.most_common(25)
    print(f"  Unique directed pairs : {len(pairs):,}")
    print()
    print("  Top 25 conversational pairs (A speaks, then B):")
    for rank, ((a, b), count) in enumerate(top, 1):
        print(f"    {rank:>2}. {a:<18s} -> {b:<18s}  {count:>4} transitions")


def scene_cooccurrence(df: pd.DataFrame) -> None:
    """Which characters appear together in the same episode?"""
    print()
    print("=" * 60)
    print("EPISODE CO-OCCURRENCE  (undirected)")
    print("=" * 60)

    dial = dialogue_only(df)
    pair_counts: Counter[tuple[str, str]] = Counter()

    for _, group in dial.groupby(["Season", "Episode"]):
        chars = sorted(group["Speaker"].unique())
        for i, a in enumerate(chars):
            for b in chars[i + 1 :]:
                pair_counts[(a, b)] += 1

    top = pair_counts.most_common(25)
    print(f"  Unique undirected pairs : {len(pair_counts):,}")
    print()
    print("  Top 25 co-occurring character pairs (by shared episodes):")
    for rank, ((a, b), count) in enumerate(top, 1):
        print(f"    {rank:>2}. {a:<18s} & {b:<18s}  {count:>3} episodes")


# ── Character-level statistics ───────────────────────────────


def verbosity_stats(df: pd.DataFrame) -> None:
    """Average words per line — who monologues vs. who's terse?"""
    print()
    print("=" * 60)
    print("VERBOSITY  (avg words per line, min 20 lines)")
    print("=" * 60)

    dial = dialogue_only(df)
    dial["word_count"] = dial["Text"].str.split().str.len()

    speaker_wc = dial.groupby("Speaker")["word_count"]
    stats = pd.DataFrame({
        "lines": speaker_wc.count(),
        "avg_words": speaker_wc.mean(),
        "total_words": speaker_wc.sum(),
    })
    stats = stats[stats["lines"] >= 20]

    print()
    most_verbose = stats.nlargest(15, "avg_words")
    print("  Most verbose (highest avg words/line):")
    for rank, (name, row) in enumerate(most_verbose.iterrows(), 1):
        print(
            f"    {rank:>2}. {name:<22s}  {row['avg_words']:>5.1f} words/line"
            f"  ({int(row['lines']):>4} lines, {int(row['total_words']):>6,} total words)"
        )

    print()
    most_terse = stats.nsmallest(15, "avg_words")
    print("  Most terse (lowest avg words/line):")
    for rank, (name, row) in enumerate(most_terse.iterrows(), 1):
        print(
            f"    {rank:>2}. {name:<22s}  {row['avg_words']:>5.1f} words/line"
            f"  ({int(row['lines']):>4} lines, {int(row['total_words']):>6,} total words)"
        )


def vocabulary_richness(df: pd.DataFrame) -> None:
    """Type-token ratio — unique words / total words per character."""
    print()
    print("=" * 60)
    print("VOCABULARY RICHNESS  (type-token ratio, min 50 lines)")
    print("=" * 60)

    dial = dialogue_only(df)
    results = []

    for speaker, group in dial.groupby("Speaker"):
        if len(group) < 50:
            continue
        all_words = " ".join(group["Text"].tolist()).lower().split()
        total = len(all_words)
        unique = len(set(all_words))
        ttr = unique / total if total else 0
        results.append((speaker, ttr, unique, total, len(group)))

    results.sort(key=lambda x: x[1], reverse=True)

    print()
    print("  Richest vocabulary (highest type-token ratio):")
    for rank, (name, ttr, uniq, total, lines) in enumerate(results[:15], 1):
        print(
            f"    {rank:>2}. {name:<22s}  TTR={ttr:.3f}"
            f"  ({uniq:>4,} unique / {total:>5,} total, {lines} lines)"
        )

    print()
    print("  Most repetitive vocabulary (lowest type-token ratio):")
    for rank, (name, ttr, uniq, total, lines) in enumerate(results[-15:], 1):
        print(
            f"    {rank:>2}. {name:<22s}  TTR={ttr:.3f}"
            f"  ({uniq:>4,} unique / {total:>5,} total, {lines} lines)"
        )


def character_arcs(df: pd.DataFrame) -> None:
    """First/last appearance and line count per season for major characters."""
    print()
    print("=" * 60)
    print("CHARACTER ARCS  (top 20 characters by total lines)")
    print("=" * 60)

    dial = dialogue_only(df)
    top_chars = dial["Speaker"].value_counts().head(20).index.tolist()
    seasons = sorted(dial["Season"].unique())

    print()
    header = f"  {'Character':<18s}  {'First':<12s} {'Last':<12s} | " + " ".join(
        f"S{s[-2:]}" for s in seasons
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for char in top_chars:
        char_df = dial[dial["Speaker"] == char]
        first = char_df["Season"].min()
        last = char_df["Season"].max()
        per_season = char_df.groupby("Season").size()
        cols = [f"{per_season.get(s, 0):>4}" for s in seasons]
        print(f"  {char:<18s}  {first:<12s} {last:<12s} | {' '.join(cols)}")


def longest_monologues(df: pd.DataFrame) -> None:
    """Longest uninterrupted speaking stretches by the same character."""
    print()
    print("=" * 60)
    print("LONGEST MONOLOGUES  (consecutive lines by same speaker)")
    print("=" * 60)

    dial = dialogue_only(df)
    monologues: list[tuple[str, int, str, str, str]] = []

    for (season, episode), group in dial.groupby(["Season", "Episode"]):
        speakers = group["Speaker"].tolist()
        texts = group["Text"].tolist()
        i = 0
        while i < len(speakers):
            j = i + 1
            while j < len(speakers) and speakers[j] == speakers[i]:
                j += 1
            run_len = j - i
            if run_len >= 3:
                preview = texts[i][:80].replace("\n", " ")
                monologues.append((speakers[i], run_len, season, episode, preview))
            i = j

    monologues.sort(key=lambda x: x[1], reverse=True)

    print()
    print("  Top 20 longest monologues:")
    for rank, (speaker, length, season, episode, preview) in enumerate(
        monologues[:20], 1
    ):
        print(f"    {rank:>2}. {speaker:<18s}  {length:>3} lines  ({season} / {episode})")
        print(f"        \"{preview}...\"")


# ── Temporal / structural statistics ─────────────────────────


def pacing_trends(df: pd.DataFrame) -> None:
    """Dialogue density per season — lines per episode trend."""
    print()
    print("=" * 60)
    print("PACING TRENDS  (dialogue lines per episode by season)")
    print("=" * 60)

    dial = dialogue_only(df)
    ep_lines = dial.groupby(["Season", "Episode"]).size().reset_index(name="lines")
    season_avg = ep_lines.groupby("Season")["lines"].agg(["mean", "std", "min", "max"])
    season_avg = season_avg.sort_index()

    print()
    print(f"  {'Season':<12s}  {'Avg':>6s}  {'Std':>6s}  {'Min':>5s}  {'Max':>5s}")
    print("  " + "-" * 42)
    for season, row in season_avg.iterrows():
        print(
            f"  {season:<12s}  {row['mean']:>6.0f}  {row['std']:>6.1f}"
            f"  {row['min']:>5.0f}  {row['max']:>5.0f}"
        )


def stage_direction_ratio(df: pd.DataFrame) -> None:
    """Narration vs. spoken dialogue per season."""
    print()
    print("=" * 60)
    print("STAGE DIRECTION RATIO  (narration vs dialogue per season)")
    print("=" * 60)

    print()
    print(f"  {'Season':<12s}  {'Dialogue':>8s}  {'Narration':>9s}  {'Ratio':>7s}")
    print("  " + "-" * 42)

    for season in sorted(df["Season"].unique()):
        sdf = df[df["Season"] == season]
        n_dial = sdf["Speaker"].notna().sum()
        n_narr = sdf["Speaker"].isna().sum()
        ratio = n_dial / n_narr if n_narr else float("inf")
        print(f"  {season:<12s}  {n_dial:>8,}  {n_narr:>9,}  {ratio:>7.2f}")


def word_frequency(df: pd.DataFrame) -> None:
    """Most common words overall and most distinctive per top character (TF-IDF style)."""
    print()
    print("=" * 60)
    print("WORD FREQUENCY")
    print("=" * 60)

    dial = dialogue_only(df)
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "is", "it", "i", "you", "he", "she", "we", "they", "my", "your",
        "his", "her", "our", "that", "this", "was", "are", "be", "have", "has",
        "had", "do", "did", "will", "would", "could", "should", "not", "no",
        "with", "from", "as", "if", "me", "him", "them", "what", "who", "how",
        "when", "where", "there", "been", "were", "am", "so", "than", "then",
        "just", "can", "don't", "all", "about", "up", "out", "one", "know",
        "it's", "i'm", "don't", "didn't", "won't", "can't", "isn't", "that's",
        "here", "get", "got", "go", "going", "come", "us", "by",
    }

    all_words = " ".join(dial["Text"].tolist()).lower().split()
    filtered = [w.strip(".,!?;:\"'()[]") for w in all_words]
    filtered = [w for w in filtered if w and w not in stopwords and len(w) > 1]

    overall = Counter(filtered).most_common(25)
    print()
    print("  Top 25 words (excluding stopwords):")
    for rank, (word, count) in enumerate(overall, 1):
        print(f"    {rank:>2}. {word:<20s}  {count:>5,}")

    # Per-character distinctive words (simple TF-IDF approximation)
    print()
    print("  Most distinctive words per major character (vs corpus):")
    corpus_freq = Counter(filtered)
    corpus_total = len(filtered)
    top_chars = dial["Speaker"].value_counts().head(10).index.tolist()

    for char in top_chars:
        char_words = " ".join(dial[dial["Speaker"] == char]["Text"].tolist()).lower().split()
        char_words = [w.strip(".,!?;:\"'()[]") for w in char_words]
        char_words = [w for w in char_words if w and w not in stopwords and len(w) > 1]
        char_freq = Counter(char_words)
        char_total = len(char_words)

        scored = {}
        for word, count in char_freq.items():
            if count < 3:
                continue
            tf = count / char_total
            cf = corpus_freq[word] / corpus_total
            scored[word] = tf / cf if cf else 0

        top_distinctive = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:5]
        words_str = ", ".join(f"{w} ({char_freq[w]}x)" for w, _ in top_distinctive)
        print(f"    {char:<18s}: {words_str}")


def show_stats(df: pd.DataFrame) -> None:
    shows = df["Show"].value_counts()
    if len(shows) > 1:
        print()
        print("=" * 60)
        print("SHOW BREAKDOWN")
        print("=" * 60)
        for show, count in shows.items():
            print(f"  {show:<30s} : {count:>6,} rows")


def main() -> None:
    df = load_data()

    print()
    print("*" * 60)
    print("  GAME OF THRONES DATASET SUMMARY")
    print("*" * 60)

    basic_stats(df)
    speaker_stats(df)
    season_stats(df)
    episode_stats(df)
    text_stats(df)
    show_stats(df)

    # Graph-relevant
    conversational_adjacency(df)
    scene_cooccurrence(df)

    # Character-level
    verbosity_stats(df)
    vocabulary_richness(df)
    character_arcs(df)
    longest_monologues(df)

    # Temporal / structural
    pacing_trends(df)
    stage_direction_ratio(df)
    word_frequency(df)

    print()
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
