Create a pull request for the current branch using the `gh` CLI. Steps:

1. **Resolve the upstream remote.** Determine which remote points to the
   canonical repo (`sweet/sweet`). Prefer a remote named `upstream`; if
   none exists, fall back to `origin`.
   ```
   UPSTREAM_REMOTE=$(git remote -v | grep -q '[u]pstream' && echo upstream || echo origin)
   ```
   All `fetch`, `log`, and `diff` commands below use `$UPSTREAM_REMOTE`,
   never a bare `origin` — so the command works identically for fork-based
   contributors (who have both `origin`=fork + `upstream`=canonical) and
   for maintainers working directly on the upstream repo (where `origin`
   *is* the canonical repo).

2. **Gather context.** First refresh the remote base so the comparison is
   against the *latest* master, not a stale local copy:
   - `git fetch "$UPSTREAM_REMOTE" master` — update the remote's master
     (does not touch your working tree or local `master`)

   Then run these commands and study the output (note: compare against
   `$UPSTREAM_REMOTE/master`, never the local `master` branch, which may
   be behind):
   - `git branch --show-current` — the branch name
   - `git log "$UPSTREAM_REMOTE/master"..HEAD --oneline` — commits on this branch
   - `git diff "$UPSTREAM_REMOTE/master"...HEAD --stat` — file-level change summary
   - `git diff "$UPSTREAM_REMOTE/master"...HEAD` — the full diff

3. **Draft the PR.** Based on the commits and diff:
   - **Title:** imperative mood, ≤72 chars, summarizing the branch's purpose.
   - **Body:** group the changes into logical sections with Markdown headings. Explain *why*, not just what. Reference relevant issue numbers if present in commit messages.
   - **Base branch:** `master` unless the branch name or commits suggest otherwise. (The PR's base is the remote branch name `master`; `gh` resolves it against the remote regardless of your local `master`.)

4. **Create the PR.** Run:
   ```
   gh pr create --title "<title>" --body "<body>" --base <base>
   ```
   Use a heredoc or temp file for the body to avoid shell escaping issues.
   If the current remote (`UPSTREAM_REMOTE`) is not `origin`, also pass
   `--repo sweet/sweet` so `gh` targets the correct repository.

5. **Report.** Print the PR URL from the output.

**Hard rules — do NOT:**
- Push the branch or any refs. If the branch is not pushed, tell the user to push first.
- Commit, stage, or amend anything.
- Modify any code or files. This command is read-only except for the `gh pr create` call.
