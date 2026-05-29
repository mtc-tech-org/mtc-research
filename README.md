# mtc-research

A drop-in knowledge base for research notes, organized **one directory per topic**. Add Markdown files under the relevant topic folder, commit, and ask questions about them through Claude Code.

## Layout

```
offline-inference/   # running LLMs yourself: batch/offline inference deep dive + B200 runbook
README.md
.gitignore
```

Each research topic is its own top-level folder. To start a new area, add a folder (e.g. `vector-databases/`) and drop notes in it.

## How to use

1. **Add research** — pick or create a topic folder and drop any `.md` file in it.
2. **Save it** — commit and push:
   ```bash
   git add <topic>/ && git commit -m "<topic>: <note>" && git push
   ```
3. **Query it** — open Claude Code in this repo and just ask, e.g. *"What do my notes say about KV-cache sizing?"* Claude searches across the topic folders, reads the relevant files, and answers with citations.

## Tips for better answers (all optional)

- Descriptive filenames (`2026-05-29-offline-inference-running-llms-yourself.md`).
- A few lines of YAML frontmatter help Claude filter and cite:
  ```yaml
  ---
  title: ...
  date: 2026-05-29
  tags: [...]
  source: https://example.com
  ---
  ```
- Plain Markdown works fine — none of this is enforced.
