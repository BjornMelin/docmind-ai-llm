# Automated Release Workflow

This repository uses [Release Please](https://github.com/googleapis/release-please) to automate the release process, including Semantic Versioning, Changelog generation, and GitHub Releases.

## Configuration

The workflow is defined in [`.github/workflows/release.yml`](../../.github/workflows/release.yml) and is configured for a single Python package.

### Key Settings

- **Trigger**: Runs on every push to the `main` branch.
- **Release Type**: `python` (Handles `pyproject.toml` version bumping and `CHANGELOG.md` updates).
- **Manifest Config**: Advanced Release Please settings live in
  [`release-please-config.json`](../../release-please-config.json) and
  [`.release-please-manifest.json`](../../.release-please-manifest.json).
- **Release Token**: The workflow requires the repository secret
  `RELEASE_PLEASE_TOKEN`, a fine-grained personal access token owned by the
  maintainer whose release PR commits should be credited. The token needs
  repository contents and pull request write access.
- **Release Commit Identity**: The workflow requires repository variables
  `RELEASE_COMMIT_NAME` and `RELEASE_COMMIT_EMAIL` for release PR lockfile
  commits. Use the maintainer's GitHub noreply email if the email should remain
  private.
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
    - Release PR commits are created with `RELEASE_PLEASE_TOKEN`. Lockfile
      follow-up commits use `RELEASE_COMMIT_NAME` and `RELEASE_COMMIT_EMAIL`.
      Do not use `GITHUB_TOKEN` here; GitHub automatically adds commit authors
      as squash-merge coauthors, so bot-authored release commits create
      unwanted bot coauthor trailers.
    - Release PR descriptions must keep the Release Please parseable version
      heading: `## [x.y.z]`, followed by the compare URL and release date.
      Manual note curation can edit bullets and sections inside that block, but
      must not replace the version heading with a custom title.

3. **Publishing**:
    - When a maintainer **merges** the Release PR, `release-please`:
        - Creates a new GitHub Release.
        - Tags the commit with the new version (e.g., `v0.2.0`).
    - The release workflow validates the published GitHub Release body so empty
      or unparsable notes fail the run instead of silently shipping bad release
      notes.

## Release Note Guardrails

- The `Semantic Pull Request` workflow uses
  `amannn/action-semantic-pull-request` to enforce Conventional Commit PR
  titles. This is required because squash merges use PR titles as release
  commits.
  It runs on `pull_request_target` without checking out pull request code so the
  required title check works for forks while keeping the token read-only.
- Allowed release-driving PR title types are `feat`, `fix`, `chore`, `docs`,
  `style`, `refactor`, `perf`, `test`, and `deps`.
- The release workflow fails after publishing if the GitHub Release body is
  blank, which catches Release Please parse failures before they go unnoticed.
- GitHub Releases should use the curated `CHANGELOG.md` release block as their
  source of truth. Do not use GitHub's auto-generated notes for Release Please
  releases; they summarize pull request titles and contributors instead of the
  semantic release content.

## Source of Truth

- **Version**: The `version` field in `pyproject.toml` and the latest GitHub Tag.
- **Changelog**: `CHANGELOG.md` in the root directory.
