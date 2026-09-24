# Close each unit completely

After implementation and validation:
- remove temporary migration and diagnostic scripts;
- update affected operational and architectural documentation;
- condense the completed `PLAN.md` unit to exactly these headings:
  `Completed`, `Goal`, `Files Added/Removed/Changed`, `Tests`, and
  `Acceptance`;
- list every committed file added, removed, or changed, grouped by category
  with a concise description of its lasting responsibility or modification;
- explicitly state `None` when an Added, Removed, or Changed category has no
  entries;
- include the completed `PLAN.md` update in the same commit as the unit's
  implementation;
- record durable architectural choices in `DECISIONS.md`;
- record shipped behavior in `CHANGELOG.md`;
- update `HANDOFF.md` only when the current operational contract changes;
- update `ROADMAP.md` when future scope, sequencing, or priority changes;
- verify that the staged file list agrees with the
  `Files Added/Removed/Changed` section;
- inspect the staged diff before committing.

Use this completed-unit structure:

```markdown
#### Completed

Concise description of the implemented behavior and resulting contract.

#### Goal

The lasting purpose of the unit.

#### Files Added/Removed/Changed

Added:
- `path/to/new_file.py` - Lasting responsibility of the new file.
- None.

Changed:
- `path/to/existing_file.py` - Behavioral or contract change.
- `tests/path/test_file.py` - Regression or acceptance coverage.
- None.

Removed:
- `path/to/retired_file.py` - Superseded responsibility that was removed.
- None.

#### Tests

Focused tests, quality gates, integration checks, and real-data validation
performed for the unit.

#### Acceptance

Concise statement proving the unit's intended contract is implemented,
validated, documented, and ready for downstream use.
```

Include only categories and files that reflect the committed unit scope.
Do not list files that were merely inspected. Temporary scripts removed
before the commit are not committed files and should not appear in the
file-change inventory.

Before committing, run:

```bash
git diff --cached --name-status
git diff --cached --stat
git diff --cached --check
```
