# Domain Docs

This repo does not use `CONTEXT.md` or `docs/adr/`. Domain-modeling skills
(`/domain-modeling`, `/grill-with-docs`, `/wayfinder`, and anything else that
reads or writes domain docs) should use the existing project documents
instead:

- **Architecture decisions** → `DECISIONS.md`, an append-only ledger. Never
  create `docs/adr/`.
- **Operational terms and contracts** → `HANDOFF.md`, which defines how the
  system works today. Never create `CONTEXT.md`.
- **Future scope and priorities** → `ROADMAP.md`.

Use the terms as `HANDOFF.md` defines them. Don't invent synonyms for
existing concepts.

Don't write to these files directly. When a skill would otherwise create or
update `CONTEXT.md` or `docs/adr/`, propose the addition in chat and name
the target file. Decisions and contract changes are recorded during unit
closeout. If your output contradicts an existing `DECISIONS.md` entry,
surface it explicitly rather than silently overriding.
