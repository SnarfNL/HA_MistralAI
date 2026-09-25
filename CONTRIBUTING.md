# Contributing

Thanks for helping out! The rules are short:

1. **Never commit straight to `main`.** Create a branch for every change
   (for example `fix/tts-voice` or `ci/ma-08`).
2. **Open a pull request** to `main` and describe in plain language what
   changed and why.
3. **CI must be green** before merging. Every pull request runs:
   - *Validate*: hassfest and HACS validation
   - *Tests*: ruff (lint) and pytest
4. Add a line to [CHANGELOG.md](CHANGELOG.md) for changes users will notice.
5. Releases: the `version` in
   `custom_components/mistral_conversation/manifest.json` must equal the
   release tag (for example `2026.09.03`). The Release workflow fails otherwise.

## Running the checks locally

```bash
pip install ruff pytest
ruff check .
pytest tests
```
