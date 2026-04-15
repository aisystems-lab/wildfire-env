# Release Checklist

Use this checklist for each new release.

## Versioning

- Decide the next Semantic Version:
  - `PATCH` for bug fixes only
  - `MINOR` for backward-compatible features
  - `MAJOR` for breaking API changes
- Update `project.version` in `pyproject.toml`
- Add a new entry to `CHANGELOG.md`

## Verification

- Run the package setup flow:
  - `uv sync`
- Verify basic imports:
  - `uv run python -c "import firecastrl_env"`
- Verify example scripts:
  - `uv run python scripts/random_agent_human.py --help`
  - `uv run python scripts/random_agent_3d.py --help`
- If the 3D viewer changed, rebuild it:
  - `cd web && npm install && npm run build`

## Git Release

- Commit release changes
- Create an annotated tag:
  - `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
- Push the branch and tag:
  - `git push`
  - `git push --tags`

## Build And Publish

- Build distributions:
  - `uv build`
- Publish when ready:
  - `uv publish`

## Post Release

- Create a GitHub Release for tag `vX.Y.Z`
- Paste the relevant `CHANGELOG.md` notes into the release description
