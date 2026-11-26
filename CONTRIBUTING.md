# Contributing

Everything merges through a pull request. The suite runs in GitHub Actions on every
push and PR (Ubuntu + macOS, Python 3.10/3.12) plus a web job (typecheck, lint, build).

## Before opening a PR

1. Run the full gate locally — all four must pass:

   ```bash
   make test      # pytest, no network, no models
   make lint      # ruff + eslint
   cd web && ./node_modules/.bin/tsc --noEmit && ./node_modules/.bin/eslint .
   cd web && npm run build
   ```

2. Every behavioral change needs a test. Durability and validation logic must not
   be "obviously fine" — `tests/` is the place that proves it.

3. Small modules. The largest file is ~1070 lines (`pipeline.py`); if yours would
   be bigger, split it.

4. One turn, one implementation. `pipeline.run_turn()` is the single turn
   iterator for the CLI and the Flask API. Do not add a second code path that
   answers a turn — add an event.

## Shape of a PR

- Imperative title, then a why-heavy body with measured specifics.
- Backend changes come with tests near the seam — `tests/test_durability.py`,
  `tests/test_server.py`, `tests/test_skills.py`.
- Screenshots or frame samples for UI changes; the frontend gates are typecheck,
  eslint and build, not tests.
- Keep the offline promise: nothing may fetch at runtime.

## How to review

- The frontend is the stricter half: any change that touches chat flow, the
  composer, or `lib/api.ts` gets re-read after merge.
- CI is authoritative. Do not "re-run the tests and push" — keep the PR green.
