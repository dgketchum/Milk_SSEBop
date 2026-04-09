# Repository Working Notes

- Treat Google Docs as the source of truth for manuscript text.
- Do **not** edit `notes/draft.md` directly unless explicitly requested by the user.
- Default to updating planning/reconciliation notes (for example `notes/UPDATE_PLAN.md` and `notes/CONSISTENCY.md`) instead of manuscript prose files.
- Use the Conda environment `milk` for project commands, figures, and analysis runs.
- Do **not** install project dependencies into the system Python; prefer `conda run -n milk ...`.
