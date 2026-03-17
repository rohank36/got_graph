"""Extract entities, relationships, and facts from GoT screenplay chunks.

Pipeline:
  chunks → LLM extraction → entity resolution → upsert into graph JSON

Checkpointing is done after every chunk so spot interruptions are safe.
"""

import argparse
import json
import re
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from chunk import Chunk, chunk_season


VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3-8B-FP8"

EXTRACTION_SYSTEM_PROMPT = """\
You are extracting structured knowledge from a Game of Thrones screenplay chunk.
Think carefully, then return ONLY valid JSON — no markdown, no explanation.
/no_think
"""

EXTRACTION_USER_TEMPLATE = """\
Known entities (match to these before creating new ones):
{known_entities}

Extract from the chunk below:
1. ENTITIES — characters, locations, factions, objects, creatures, concepts
2. RELATIONSHIPS — connections between entities stated or clearly implied
3. FACTS — important standalone statements that don't fit a relationship

Return this exact JSON structure:
{{
  "entities": [
    {{
      "name": "canonical name (e.g. 'Tyrion Lannister')",
      "type": "Person | Place | Faction | Object | Creature | Concept",
      "aliases": ["other names used in this chunk"],
      "properties": {{"key": "value"}}
    }}
  ],
  "relationships": [
    {{
      "from": "canonical entity name",
      "to": "canonical entity name",
      "type": "SHORT_VERB_PHRASE",
      "context": "brief quote or note"
    }}
  ],
  "facts": [
    {{
      "entity": "canonical entity name",
      "fact": "atomic statement"
    }}
  ]
}}

Rules:
- Use full canonical names (e.g. "Jon Snow", not "Jon")
- Only extract what is stated or clearly implied — do not speculate
- Relationship types should be short, consistent verb phrases in SCREAMING_SNAKE_CASE
- Stage directions are as informative as dialogue — mine them
- If unsure of canonical name, use the most complete name from the chunk

Chunk ({chunk_id}):
{chunk_text}
"""


def format_known_entities(registry: dict) -> str:
    if not registry:
        return "(none yet)"
    lines = []
    for name, entity in registry.items():
        aliases = entity.get("aliases", [])
        alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
        lines.append(f"- {name}{alias_str}")
    return "\n".join(lines)


def chunk_id(chunk: Chunk) -> str:
    return f"{chunk.season}/{chunk.episode}/scene-{chunk.scene_index}"


def extract_from_chunk(client: OpenAI, chunk: Chunk, registry: dict) -> dict:
    user_prompt = EXTRACTION_USER_TEMPLATE.format(
        known_entities=format_known_entities(registry),
        chunk_id=chunk_id(chunk),
        chunk_text=chunk.to_text(),
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


def resolve_entity_name(name: str, registry: dict) -> str:
    """Return the canonical name for a given name string.

    Checks: exact match, alias match, then returns name as-is.
    """
    if name in registry:
        return name
    name_lower = name.lower()
    for canonical, entity in registry.items():
        if name_lower == canonical.lower():
            return canonical
        for alias in entity.get("aliases", []):
            if name_lower == alias.lower():
                return canonical
    return name


def upsert_entity(entity: dict, source_chunk: str, registry: dict) -> str:
    """Merge extracted entity into registry. Returns canonical name."""
    name = resolve_entity_name(entity["name"], registry)

    if name not in registry:
        registry[name] = {
            "type": entity.get("type", "Unknown"),
            "aliases": [],
            "properties": {},
        }

    stored = registry[name]

    # Merge aliases
    for alias in entity.get("aliases", []):
        if alias != name and alias not in stored["aliases"]:
            stored["aliases"].append(alias)

    # Merge properties — versioned with source chunk
    for key, value in entity.get("properties", {}).items():
        if key not in stored["properties"]:
            stored["properties"][key] = []
        stored["properties"][key].append({"value": value, "chunk": source_chunk})

    return name


def upsert_relationship(rel: dict, source_chunk: str, graph: dict, registry: dict) -> None:
    from_name = resolve_entity_name(rel["from"], registry)
    to_name = resolve_entity_name(rel["to"], registry)
    rel_type = rel["type"]
    context = rel.get("context", "")

    # Find existing edge or create new one
    for edge in graph["edges"]:
        if edge["from"] == from_name and edge["to"] == to_name and edge["type"] == rel_type:
            edge["assertions"].append({"chunk": source_chunk, "context": context})
            return

    graph["edges"].append({
        "from": from_name,
        "to": to_name,
        "type": rel_type,
        "assertions": [{"chunk": source_chunk, "context": context}],
    })


def upsert_fact(fact: dict, source_chunk: str, graph: dict, registry: dict) -> None:
    entity = resolve_entity_name(fact["entity"], registry)
    graph["facts"].append({
        "entity": entity,
        "fact": fact["fact"],
        "chunk": source_chunk,
    })


def upsert_extraction(extraction: dict, source_chunk: str, graph: dict, registry: dict) -> None:
    for entity in extraction.get("entities", []):
        upsert_entity(entity, source_chunk, registry)

    # Sync registry into graph nodes
    graph["nodes"] = registry

    for rel in extraction.get("relationships", []):
        upsert_relationship(rel, source_chunk, graph, registry)

    for fact in extraction.get("facts", []):
        upsert_fact(fact, source_chunk, graph, registry)


def load_checkpoint(path: Path) -> tuple[dict, dict, set]:
    """Load graph, registry, and set of already-processed chunk IDs."""
    if path.exists():
        data = json.loads(path.read_text())
        graph = data["graph"]
        registry = data["registry"]
        processed = set(data["processed"])
        print(f"Resumed from checkpoint: {len(processed)} chunks already done.")
    else:
        graph = {"nodes": {}, "edges": [], "facts": []}
        registry = {}
        processed = set()
    return graph, registry, processed


def save_checkpoint(path: Path, graph: dict, registry: dict, processed: set) -> None:
    path.write_text(json.dumps({
        "graph": graph,
        "registry": registry,
        "processed": list(processed),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract knowledge graph from GoT chunks.")
    parser.add_argument("--output", "-o", default="graph.json", help="Final graph output path.")
    parser.add_argument("--checkpoint", "-c", default="graph_checkpoint.json", help="Checkpoint file path.")
    parser.add_argument("--season", default="season-08", help="Season to process.")
    args = parser.parse_args()

    client = OpenAI(base_url=VLLM_BASE_URL, api_key="placeholder")
    chunks = chunk_season(season=args.season)

    checkpoint_path = Path(args.checkpoint)
    graph, registry, processed = load_checkpoint(checkpoint_path)

    for chunk in tqdm(chunks, desc="Extracting"):
        cid = chunk_id(chunk)
        if cid in processed:
            continue

        try:
            extraction = extract_from_chunk(client, chunk, registry)
            upsert_extraction(extraction, cid, graph, registry)
            processed.add(cid)
            save_checkpoint(checkpoint_path, graph, registry, processed)
        except Exception as e:
            print(f"\nFailed on {cid}: {e}")
            continue

    Path(args.output).write_text(json.dumps(graph, indent=2))
    print(f"\nDone. Graph written to {args.output}")
    print(f"  Nodes : {len(graph['nodes'])}")
    print(f"  Edges : {len(graph['edges'])}")
    print(f"  Facts : {len(graph['facts'])}")


if __name__ == "__main__":
    main()
