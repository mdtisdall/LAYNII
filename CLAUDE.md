# LayNii

Each command is one self-contained C++11 file in `src/`, built by the
`Makefile` with the pinned GCC and zlib from `flake.nix`
(`nix develop --command make <NAME>`). A new command needs its file in `src/`
plus an entry in a group list and a build rule in the `Makefile`.

## Branch and PR workflow

Every change to a tracked file gets to `master` through its own
branch and a reviewed pull request. There are no exceptions for documentation
or process changes: `CLAUDE.md`, `README.md`, `.envrc`, `flake.nix`,
`.gitignore`, CI configuration, and hooks also go through a branch and a PR.

One branch, one concern. Do not add an unrelated fix, chore, or doc edit to a
branch that already has work in progress. Start a new branch from
`master` for it.

The dev-workflow Claude Code plugin has the procedures: `start-task`, `ship`,
`finish-task`, `parallel-agents`, `github-token`, `project-setup`. Specific to
this project:

- Default branch: `master`. Merge method: squash.
- Checks, before each PR (CI runs the same checks):
  `nix develop --command scripts/check`
- Worktrees: one for each task, in `.worktrees/<short-name>` of the main
  checkout (git-ignored), made with the `start-task` skill. Do not create a
  worktree outside the project directory, inside another worktree, or with a
  different tool. The hook blocks `git worktree add` and `git worktree move`
  to any other path.
- Dependency sync, one time in each new worktree, before sub-agents start:
  `nix develop --command true` (builds the devShell; there are no other
  dependencies)
- GitHub CLI: `direnv exec . gh ...`. It uses this repository's own
  fine-grained token in the git-ignored `.envrc.local`. The permissions that
  the token needs are in `.claude/gh-token-permissions`. `git` uses SSH.
- CI status: the Actions API (`ci-status.sh`, `gh run list`), not
  `gh pr checks`.
- `git commit` and `git push` to `master` are blocked for Claude
  Code by `.claude/hooks/block-main-writes.sh`. To run the git command
  yourself in a terminal is a break-glass action for a stuck state, not a
  shortcut for routine edits.
- Show the commit message and wait for approval before `git commit`. Merge
  only when told to.
