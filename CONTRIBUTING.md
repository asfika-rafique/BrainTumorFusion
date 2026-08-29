# Contributing

- Keep patient-sensitive, restricted, and large data outside Git.
- Add or update a configuration when changing an experiment; do not hard-code local paths.
- Do not present generated metrics as research claims without recording the dataset split and checkpoint.
- Run `python -m pytest` and `python -m compileall -q src scripts ui` before sharing changes.
- Keep reusable code under `src/brain_tumor_fusion`; keep command-line orchestration under `scripts`.
