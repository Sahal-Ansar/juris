# Prompt templates

One file per template (`<name>.md` or `<name>.j2`), starting with a version header:

```text
---
id: issue_framer
version: 1
---
Template body...
```

- Bump `version` whenever the meaning of a prompt changes.
- Every template's hash is recorded in each run manifest (`juris.runs`), so any edit, even without a version bump, shows up as a different hash.
- Load templates with `juris.prompts.load_prompt(name)`.
