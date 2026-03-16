This codebase is for implementing graphs and having AI search over it.

See `./for_agent/WORK_HISTORY.md` for the history of work done and decisions made per commit. After making a commit, take the commit hash as entry ID and update the file. Keep entries concise and uniformly structured.

See `./for_agent/NOTES.md` for your important findings that could be useful for future reference. Best to check before implementing something.

To run local LLMs we'll use SkyPilot (see `./skypilot.yaml`). Make sure to include frequent checkpointing where necessary to handle spot instance interruptions gracefully.

Always ensure that `.venv` is activated before installing any packages or running any code.

Try and abide by the following principles of Python:
- Beautiful is better than ugly.
- Explicit is better than implicit.
- Simple is better than complex.
- Complex is better than complicated.
- Flat is better than nested.
- Sparse is better than dense.
- Readability counts.
- Special cases aren't special enough to break the rules.
- Although practicality beats purity.
- Errors should never pass silently, unless explicitly silenced.
- In the face of ambiguity, refuse the temptation to guess.
- There should be one-- and preferably only one --obvious way to do it.
- Although that way may not be obvious at first unless you're Dutch.
- Now is better than never.
- Although never is often better than right now.
- If the implementation is hard to explain, it's a bad idea.
- If the implementation is easy to explain, it may be a good idea.