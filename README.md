# mtc-research

A drop-in knowledge base for research notes. Put Markdown files in `research/`, commit, and ask questions about them through Claude Code.

## How to use

1. **Add research** — drop any `.md` file into `research/` (subfolders are fine). No required format, no tooling to run.
2. **Save it** — commit and push so it's backed up and shared:
   ```bash
   git add research && git commit -m "research: <topic>" && git push
   ```
3. **Query it** — open Claude Code in this repo and just ask, e.g.:
   - "What do my notes say about X?"
   - "Summarize everything I have on Y."
   - "Which files mention Z, and what do they conclude?"

   Claude searches across `research/`, reads the relevant files, and answers with citations to the source files.

## Tips for better answers (all optional)

- Give files descriptive names (`2026-05-vector-db-comparison.md` beats `notes.md`).
- A few lines of YAML frontmatter at the top help Claude filter and cite:
  ```yaml
  ---
  title: Vector DB comparison
  date: 2026-05-29
  tags: [infra, rag]
  source: https://example.com
  ---
  ```
- None of this is enforced — plain Markdown works fine.
