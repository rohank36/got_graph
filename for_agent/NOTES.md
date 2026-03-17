# Instructions

This is the place to write important findings, especially ones that will be useful for reference in future work and in new chats wit new context. Its organized and managed by you. Its completely yours.

# Notes

## Chunking
- Scene boundaries are defined by stage directions (rows with no Speaker). Consecutive stage directions are merged into a single block that becomes the header of the following scene.
- Process per-episode to avoid cross-episode contamination.
- Season 8: 626 chunks, median ~30 words, max 388 words — no size-cap splitting needed.
- Stage directions often contain character names in ALL CAPS and location cues — valuable for entity extraction even in narration-only chunks.