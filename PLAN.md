# Migration Plan to New State

## 1. Analyze the Relevant Architecture

### Goals:
- Understand the current architecture and identify key components.
- Identify dependencies between different modules and files.

### Steps:
1. Review the repository layout described in `HANDOFF.md`.
2. Map out the relationships between different directories and files.
3. Identify the main entry points (CLI, legacy scripts).
4. Document the data flow from ingestion to output generation.

## 2. Identify Affected Files and Dependencies

### Goals:
- Determine which files need changes or updates.
- Understand dependencies between affected files.

### Steps:
1. Review `HANDOFF.md` for any specific mentions of files that need attention.
2. Identify files that are part of the legacy pipeline (`PFR_data_pipeline_run.py`, `PFR_model_pipeline_run.py`) and their dependencies.
3. Identify files in the new CLI (`src/gridiron_edge/cli.py`) and their dependencies.
4. Document any shared utilities or configurations.

## 3. Create a Step-by-Step Implementation Plan

### Goals:
- Develop a detailed plan for migrating to the new state.
- Ensure minimal changes and preserve existing architecture.

### Steps:
1. **Update Configuration Management:**
   - Replace `configs/config.yaml` with environment variables where possible.
   - Update documentation in `HANDOFF.md`.

2. **Refactor Legacy Scripts:**
   - Deprecate legacy scripts (`PFR_data_pipeline_run.py`, `PFR_model_pipeline_run.py`) by marking them as deprecated.
   - Update `README.md` to recommend using the new CLI.

3. **Enhance Docker Configuration:**
   - Generate a real `requirements.txt` from Poetry.
   - Update the `Dockerfile` to use Poetry for dependency management.
   - Update `HANDOFF.md` with instructions on building and running Docker containers.

4. **Improve Data Directory Management:**
   - Ensure that necessary directories (`data/raw`, `data/cleaned`, `data/output`) are created automatically if they do not exist.
   - Update relevant scripts to handle missing directories gracefully.

5. **Update Documentation:**
   - Review and update all documentation files (`README.md`, `HANDOFF.md`) to reflect the new state.
   - Ensure that all workflows, commands, and configurations are up-to-date.

6. **Testing and Validation:**
   - Develop a testing strategy for both legacy and new CLI components.
   - Write tests alongside `src/gridiron_edge/...` as part of refactoring efforts.

## 4. Identify Risks, Assumptions, and Edge Cases

### Goals:
- Identify potential issues that could arise during migration.
- Document assumptions made during the planning process.

### Steps:
1. **Risks:**
   - Breaking changes in legacy scripts due to deprecation.
   - Incompatibilities between new CLI and existing data structures.
   - Issues with Docker builds if `requirements.txt` is not correctly generated.

2. **Assumptions:**
   - All dependencies are correctly listed in Poetry.
   - Environment variables will be set correctly for all required configurations.
   - Existing data structures will remain compatible with the new pipeline.

3. **Edge Cases:**
   - Handling missing configuration files (`configs/config.yaml`).
   - Managing data directory permissions and access issues.
   - Ensuring compatibility between different Python versions (>=3.12,<4).

## 5. Suggest Validation/Testing Strategy

### Goals:
- Ensure that the migration is successful and stable.
- Validate all components of the new state.

### Steps:
1. **Unit Tests:**
   - Write unit tests for each component in `src/gridiron_edge/...`.
   - Use `pytest` to run tests and ensure coverage.

2. **Integration Tests:**
   - Develop integration tests that simulate end-to-end workflows.
   - Test both the new CLI and legacy scripts (if still used) against known data sets.

3. **Regression Testing:**
   - Run existing test suites for legacy scripts to ensure no regressions.
   - Compare outputs from legacy scripts with new CLI to verify consistency.

4. **Continuous Integration (CI):**
   - Set up CI pipelines to automatically run tests on code changes.
   - Ensure that all tests pass before merging any changes.

5. **User Acceptance Testing (UAT):**
   - Conduct UAT sessions with stakeholders to validate the new state.
   - Gather feedback and make necessary adjustments.

---

## Conclusion

By following this plan, we can ensure a smooth transition to the new state while minimizing disruptions and preserving existing architecture. Each step is designed to be incremental, allowing for easy rollback if needed.

