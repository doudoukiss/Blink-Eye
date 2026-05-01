# Temporal Continuity Graph / Active Context Compiler QA

This QA note focuses on the next-phase memory/context frontier:

- temporal continuity graph
- dossiers and compiled summaries
- graph-aware context packets
- contradiction / freshness reporting
- planning packets that use richer continuity state

## Operator surfaces to inspect first

When validating this slice, start with the derived operator artifacts instead of
ad hoc screenshots:

- audit JSON top-level:
  - `continuity_graph`
  - `continuity_graph_digest`
  - `continuity_dossiers`
  - `continuity_governance_report`
  - `packet_traces`
  - `context_packet_digest`
- audit Markdown:
  - `## Graph Review`
  - `## Governance Review`
  - `## Context Packet Review`
- replay fixtures with explicit `context_queries.reply` / `context_queries.planning`

Use raw event / SQLite inspection only to verify the evidence trail under those
derived reports.

Recommended live validation ladder:

1. `uv run --python 3.12 blink-local-chat --once "Explain Blink in one sentence."`
2. `uv run --python 3.12 blink-local-brain-chat --once "Blink 现在记得我什么？"`
3. `uv run --python 3.12 blink-local-brain-audit --reply-query "Blink 现在记得我什么？" --planning-query "当前有哪些挂起规划？"`

The plain `blink-local-chat` path is only a baseline Ollama sanity check. Use
`blink-local-brain-chat` and query-aware `blink-local-brain-audit` when you
need to validate the Phase 10 continuity graph, dossiers, and active packet
compiler in real use.

## Useful focused tests to add or run

Use existing continuity / planning suites as the starting point, then add new
targeted coverage for graph and context behavior.

When the new code exists, the suite should eventually include checks for:

- temporal current-vs-historical recall
- multi-hop continuity retrieval
- dossier compilation determinism
- contradiction/freshness surfacing
- context compilation trace correctness
- runtime-backed planning packet quality

## Scenario 1: current fact vs prior fact

1. Tell Blink one fact, for example where the user lives.
2. Later correct it with a newer fact.
3. Ask:
   - where the user lives now
   - where the user lived before
4. Inspect:
   - `continuity_graph_digest`
   - `continuity_governance_report`
   - `context_packet_digest.reply`
   - raw `packet_traces.reply` if the summary looks wrong

Expected:
- current answer reflects the new fact
- historical answer reflects the older fact
- dossier or graph artifact shows both with explicit temporal status
- supersession remains visible
- the current-query replay fixture and historical-query replay fixture stay
  deterministic but differ in `context_packet_digest.reply.temporal_mode`

## Scenario 2: relationship / project multi-hop recall

1. Create a small project arc involving:
   - a project goal
   - a relationship-scoped preference or constraint
   - a later blocker or wake condition
2. Ask a question that does not directly quote the original wording but clearly
   depends on those connected facts.

Expected:
- the context compiler surfaces connected evidence, not only direct lexical hits
- the answer references the right project or relationship context
- the compilation trace shows anchor selection and bounded expansion
- `context_packet_digest.reply.selected_anchor_types` and selected backing ids
  show what actually survived budgeting

## Scenario 3: planning packet quality

1. Create a commitment that needs planning or revision.
2. Ensure relevant continuity facts exist:
   - project status
   - prior blocker
   - recent user preference or constraint
3. Trigger bounded planning.

Expected:
- the planning packet includes the right continuity evidence
- it does not dump all memory indiscriminately
- the planning trace remains inspectable
- the downstream proposal is consistent with the packet
- `context_packet_digest.planning` remains stable on replay for the same query

## Scenario 4: contradiction / freshness visibility

1. Seed a dossier-worthy topic with a claim.
2. Add later evidence that weakens, contradicts, or supersedes the earlier claim.
3. Run the maintenance or dossier refresh lane.

Expected:
- the relevant dossier updates
- contradiction or freshness status is visible
- evidence links remain intact
- stale summaries are not silently treated as fresh truth
- `continuity_governance_report.open_issue_rows` include explicit evidence ids

## Scenario 5: operator dossier / governance inspection

Inspect the new artifact surface for:

- relationship dossier
- project dossier
- graph digest / governance report
- context packet digest

Expected:
- each dossier answers what Blink currently thinks
- each dossier exposes at least some direct evidence ids or source links
- the dossier says what changed recently
- contradictions or uncertainty are legible
- non-action outcomes are legible, not only adopted/current summaries

## Scenario 6: replay determinism

1. Produce a small but non-trivial continuity graph:
   - multiple entities
   - superseded claims
   - one autobiographical arc
   - one commitment / plan object
2. Export or inspect artifacts.
3. Rebuild from replay.

Expected:
- the same dossiers and graph-aware digests rebuild
- the same current-vs-historical state remains visible
- the same context packet trace can be reproduced for the same query/task

## Query-aware replay fixtures

Use the checked-in replay fixtures as the canonical deterministic QA lane:

- current-truth corrected fact reply
- historical/change-focused corrected fact reply
- associative project/relationship recall reply

The fixture should carry explicit `context_queries.reply` so the packet summary
and trace are query-stable during replay.
