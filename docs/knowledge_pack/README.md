---
id: synthkit-knowledge-pack
type: pack-manifest
name: synthkit knowledge pack for Mnemo
part_of: synthkit
updated: 2026-07-28
files: [entity_synthkit.md, synthkit_architecture.md,
        synthkit_methodology.md, synthkit_studies.md,
        synthkit_operations.md, synthkit_glossary.md]
---

# Ingesting synthkit into Mnemo

Six entity documents in the WorkGraph pattern (YAML frontmatter
+ markdown body, chunk-friendly ## sections, atomic facts,
explicit part_of relations). Ingestion options:

1. **RAG intake**: point the FAISS/BM25 indexer at this
   directory (or copy the files into Mnemo's notes root). Each
   ## section is a natural chunk; headers carry entity + topic
   so BM25 hits stay precise.
2. **WorkGraph entities**: consolidate frontmatter as six typed
   entities related via part_of -> synthkit.
3. **Partition note**: synthkit is Keck-demo-relevant but lives
   in a personal repo and was built on personal + work
   machines. If Mnemo's employer-partition rule applies, place
   under data/work/ for clean excludability; otherwise general.

Freshness: numbers herein are canonical as of 2026-07-28 (324
checks, capstone 0.822/0.724/0.604, atlas v2). When synthkit
changes materially, regenerate by updating these files — the
studies file lists canonical numbers explicitly so drift is
visible.
