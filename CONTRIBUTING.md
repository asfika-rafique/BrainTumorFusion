# Contributing

## Development setup

Use Python 3.11 and the supplied environment specification:

```powershell
conda env create -f environment.yml
conda activate brain-tumor-fusion
python -m pip install -e ".[test,ui]"
```

## Project conventions

- Keep reusable code under `src/brain_tumor_fusion` and command-line orchestration under `scripts`.
- Add or update a configuration when changing an experiment; do not hard-code local paths.
- Keep patient-sensitive, restricted, and large data outside Git.
- Record the dataset split, seed, configuration, checkpoint, and environment for research results.
- Do not present historical or generated metrics as verified claims without supporting provenance.

## Tests

Run the lightweight checks before opening a pull request:

```powershell
python -m pytest -q
python -m compileall -q src scripts ui tests
```

## Issues and pull requests

Open an issue with a concise description, reproduction steps, configuration, and environment details. Pull requests should explain the scope of the change, preserve historical research artifacts, and include relevant test results. Do not include raw medical data, credentials, model weights, or generated experiment outputs.
