# ImageProcessor — Claude Code guidance

The full guidance for this component lives in `AGENTS.md` and is shared with every agent tool. It is
imported here in full:

@AGENTS.md

## Local development

- **`--dep-source local`** (the default): `requirements.txt` already points at the sibling
  `edgecommons` checkout — `pip install -r requirements.txt` from this directory resolves it.
- **`--dep-source registry` / `pinned-rev`**: `requirements.txt` pins a git revision of the library.
  To iterate against a local monorepo checkout instead, run `pip install -e ../core/libs/python`
  after the initial install — it overlays the editable sibling on top of the pinned line.
