"""Chunk the Game of Thrones screenplay into scene-based chunks.

Strategy (Option 3):
- Process per episode to avoid cross-episode contamination.
- Within each episode, merge consecutive stage directions into a single block,
  then attach that block as the header of the scene that follows.
- Trailing narration with no following dialogue becomes a narration-only chunk.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Chunk:
    season: str
    episode: str
    scene_index: int
    narration: list[str]
    dialogue: list[tuple[str, str]]  # (speaker, text)

    def word_count(self) -> int:
        narration_words = sum(len(t.split()) for t in self.narration)
        dialogue_words = sum(len(text.split()) for _, text in self.dialogue)
        return narration_words + dialogue_words

    def to_text(self) -> str:
        parts = []
        if self.narration:
            parts.append(" ".join(self.narration))
        for speaker, text in self.dialogue:
            parts.append(f"{speaker}: {text.strip()}")
        return "\n".join(parts)


def load_season(path: str, season: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[df["Season"] == season].copy()
    df["Text"] = df["Text"].fillna("").str.strip()
    df["Speaker"] = df["Speaker"].fillna("").str.strip()
    return df


def chunk_episode(season: str, episode: str, rows: pd.DataFrame) -> list[Chunk]:
    chunks: list[Chunk] = []
    scene_index = 0

    pending_narration: list[str] = []
    pending_dialogue: list[tuple[str, str]] = []

    def flush() -> None:
        nonlocal scene_index
        if not pending_narration and not pending_dialogue:
            return
        chunks.append(Chunk(
            season=season,
            episode=episode,
            scene_index=scene_index,
            narration=list(pending_narration),
            dialogue=list(pending_dialogue),
        ))
        scene_index += 1
        pending_narration.clear()
        pending_dialogue.clear()

    for _, row in rows.iterrows():
        is_stage = row["Speaker"] == ""
        text = row["Text"]

        if is_stage:
            # If we have accumulated dialogue, this narration starts a new scene
            if pending_dialogue:
                flush()
            if text:
                pending_narration.append(text)
        else:
            pending_dialogue.append((row["Speaker"], text))

    flush()
    return chunks


def chunk_season(path: str = "got.csv", season: str = "season-08") -> list[Chunk]:
    df = load_season(path, season)
    all_chunks: list[Chunk] = []

    for episode, ep_df in df.groupby("Episode", sort=True):
        chunks = chunk_episode(season, episode, ep_df)
        all_chunks.extend(chunks)

    return all_chunks


def print_stats(chunks: list[Chunk]) -> None:
    word_counts = pd.Series([c.word_count() for c in chunks])
    narration_only = sum(1 for c in chunks if c.dialogue == [])
    dialogue_only = sum(1 for c in chunks if c.narration == [])
    mixed = len(chunks) - narration_only - dialogue_only

    print(f"Total chunks     : {len(chunks):,}")
    print(f"  narration-only : {narration_only:,}")
    print(f"  dialogue-only  : {dialogue_only:,}")
    print(f"  mixed          : {mixed:,}")
    print()
    print("Word count per chunk:")
    print(f"  min    : {word_counts.min():>6,.0f}")
    print(f"  median : {word_counts.median():>6,.0f}")
    print(f"  mean   : {word_counts.mean():>6,.1f}")
    print(f"  max    : {word_counts.max():>6,.0f}")
    print(f"  p90    : {word_counts.quantile(0.90):>6,.0f}")
    print(f"  p95    : {word_counts.quantile(0.95):>6,.0f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk GoT screenplay into scenes.")
    parser.add_argument("--output", "-o", help="Write chunks to this text file.")
    args = parser.parse_args()

    chunks = chunk_season()
    print_stats(chunks)

    if args.output:
        with open(args.output, "w") as f:
            for chunk in chunks:
                f.write(f"=== {chunk.season} / {chunk.episode} / scene {chunk.scene_index} ===\n")
                f.write(chunk.to_text())
                f.write("\n\n")
        print(f"\nWrote {len(chunks)} chunks to {args.output}")
    else:
        print()
        print("--- Sample chunk (index 2) ---")
        print(chunks[2].to_text())
