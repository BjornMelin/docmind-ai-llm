# Automated Release Workflow

This repository uses [Release Please](https://github.com/googleapis/release-please) to automate the release process, including Semantic Versioning, Changelog generation, and GitHub Releases.

## Configuration

The workflow is defined in [`.github/workflows/release.yml`](../../.github/workflows/release.yml) and is configured for a single Python package.

### Key Settings

- **Triggers**: Release Please runs only after the `CI` workflow succeeds for a
  `main` push. Releases publish the changelog and source tag; DocMind does not
  build or attach a Python wheel.
- **Release Type**: `python` (Handles `pyproject.toml` version bumping and `CHANGELOG.md` updates).
- **Manifest Config**: Advanced Release Please settings live in
  [`release-please-config.json`](../../release-please-config.json) and
  [`.release-please-manifest.json`](../../.release-please-manifest.json).
- **Tag and Release Name Format**: Releases use plain SemVer tags and titles,
  such as `v0.8.2`, not component-prefixed names like
  `docmind-ai-llm-v0.8.2`.
- **Release Token**: The workflow requires the repository secret
  `RELEASE_PLEASE_TOKEN`, a fine-grained personal access token owned by the
  maintainer whose release PR commits should be credited. The token needs
  repository contents and pull request write access.
- **SemVer Strategy**: After 1.0, `fix:` creates a patch release, `feat:` creates
  a minor release, and `feat!:` or a `BREAKING CHANGE:` footer creates a major
  release. This rule is independent of the current major version.

## How It Works

1. **Commit Messages**: Developers must use [Conventional Commits](https://www.conventionalcommits.org/).
    - `fix:` -> Patch bump (`1.0.0` -> `1.0.1`).
    - `feat:` -> Minor bump (`1.0.0` -> `1.1.0`).
    - `feat!:` or `BREAKING CHANGE:` -> Major bump (`1.0.0` -> `2.0.0`).
    - `chore:` -> No release (usually).

2. **Release PR**:
    - After a `main` commit passes CI, `release-please` analyzes it.
    - It creates or updates a special "Release PR".
    - This PR contains the updated `CHANGELOG.md`, the version bump in
      `pyproject.toml`, and the matching root package version in `uv.lock`.
    - Release PR commits are created with `RELEASE_PLEASE_TOKEN`. Do not use
      `GITHUB_TOKEN` here; GitHub automatically adds commit authors as
      squash-merge coauthors, so bot-authored release commits create unwanted
      bot coauthor trailers.
    - Release PRs must stay single-commit. Do not add follow-up lockfile commits
      from the workflow; `release-please-config.json` updates `uv.lock` through
      Release Please `extra-files` so the version files remain in one commit.
    - Release PR descriptions must keep the Release Please parseable version
      heading: `## [x.y.z]`, followed by the compare URL and release date.
      Manual note curation can edit bullets and sections inside that block, but
      must not replace the version heading with a custom title.

3. **Publishing**:
    - When a maintainer **merges** the Release PR, `release-please`:
        - Creates a new GitHub Release.
        - Tags the commit with the new version (for example, `v2.0.0`).
    - The publisher validates the GitHub Release body so empty or unparsable
      notes fail the release workflow.
    - GitHub source archives and the production container recipe are the release
      distribution contract. No package artifact is attached.

## Release Note Guardrails

- The `Semantic Pull Request` workflow uses
  `amannn/action-semantic-pull-request` to enforce Conventional Commit PR
  titles. This is required because squash merges use PR titles as release
  commits.
  It runs on `pull_request_target` without checking out pull request code so the
  required title check works for forks while keeping the token read-only.
- Allowed release-driving PR title types are `feat`, `fix`, `chore`, `docs`,
  `style`, `refactor`, `perf`, `test`, and `deps`.
- Release publication refuses a blank GitHub Release body.
- Generated Release Please PRs run the normal CI plus a release-contract job that
  runs `scripts/check_release_contract.py` and `uv lock --check`. The script
  verifies the root `pyproject.toml` project version, root package version in
  `uv.lock`, `.release-please-manifest.json`, and newest released `CHANGELOG.md`
  heading agree. Documentation CI skips generated
  release PRs because their content was already validated before merge; the
  semantic PR title check remains required.
- GitHub Releases should use the curated `CHANGELOG.md` release block as their
  source of truth. Do not use GitHub's auto-generated notes for Release Please
  releases; they summarize pull request titles and contributors instead of the
  semantic release content.

## Source of Truth

- **Release metadata**: The root `pyproject.toml`, root package entry in
  `uv.lock`, `.release-please-manifest.json`, and newest released
  `CHANGELOG.md` heading must agree.
- **Published version**: The latest published GitHub tag must be
  `v<release metadata version>`.
- **Tag Format**: `v<major>.<minor>.<patch>`.
- **Changelog**: `CHANGELOG.md` in the root directory.

Verify a release PR locally:

```bash
uv run python scripts/check_release_contract.py
uv lock --check
```
