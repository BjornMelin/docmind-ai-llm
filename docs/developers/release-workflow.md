# Automated Release Workflow

This repository uses [Release Please](https://github.com/googleapis/release-please) to automate the release process, including Semantic Versioning, Changelog generation, and GitHub Releases.

## Configuration

The workflow is defined in [`.github/workflows/release.yml`](../../.github/workflows/release.yml) and is configured for a single Python package.

### Key Settings

- **Trigger**: Runs on every push to the `main` branch.
- **Release Type**: `python` (Handles `pyproject.toml` version bumping and `CHANGELOG.md` updates).
- **SemVer Strategy**: `bump-minor-pre-major: true`.
  - This ensures that breaking changes ( `feat!:` ) increment the **minor** version (e.g., `0.1.0` -> `0.2.0`) rather than the **major** version, protecting the `1.0.0` milestone for the actual stable release.

## How It Works

1. **Commit Messages**: Developers must use [Conventional Commits](https://www.conventionalcommits.org/).
    - `fix:` -> Patch bump (0.1.0 -> 0.1.1)
    - `feat:` -> Minor bump (0.1.0 -> 0.2.0)
    - `feat!:` or `BREAKING CHANGE:` -> Minor bump (0.1.0 -> 0.2.0) (due to pre-major mode).
    - `chore:` -> No release (usually).

2. **Release PR**:
    - When commits are merged to `main`, `release-please` analyzes them.
    - It creates or updates a special "Release PR".
    - This PR contains the updated `CHANGELOG.md` and the version bump in `pyproject.toml`.
    - If `uv.lock` is out of date (`uv lock --check`), the workflow regenerates and commits `uv.lock` on the release PR branch.

3. **Publishing**:
    - When a maintainer **merges** the Release PR, `release-please`:
        - Creates a new GitHub Release.
        - Tags the commit with the new version (e.g., `v0.2.0`).

## Source of Truth

- **Version**: The `version` field in `pyproject.toml` and the latest GitHub Tag.
- **Changelog**: `CHANGELOG.md` in the root directory.
