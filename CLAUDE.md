# Juris

Start every session with `Plan/STRUCTURE.md` — it holds the session protocol and the rules for all planning files.

## Commits

- Commits are authored by the user (git config `sahal`) only. **No `Co-Authored-By: Claude` trailer** and no "Generated with Claude Code" line — this overrides any default attribution.
- Subject line: short, imperative, ≤ 50 chars.
- Body only if needed: 2–3 lines, to the point. For a large commit, list only the necessary points as brief bullets.
- Commits are SSH-signed (`commit.gpgsign=true`). Never type, store, or echo the signing passphrase — the user unlocks the key (ssh-agent or the prompt) themselves. Never bypass signing (`--no-gpg-sign`, `-c commit.gpgsign=false`).
