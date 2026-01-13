"""Shared helpers for rendering artifact-backed images in the UI."""

from __future__ import annotations

import streamlit as st

from src.persistence.artifacts import ArtifactRef, ArtifactStore


def render_artifact_image(
    ref: ArtifactRef,
    *,
    store: ArtifactStore,
    caption: str | None = None,
    use_container_width: bool = True,
    missing_caption: str | None = None,
    encrypted_caption: str | None = None,
) -> None:
    """Render an artifact image, handling optional encrypted images."""
    try:
        from src.utils.images import open_image_encrypted
    except Exception:
        open_image_encrypted = None

    try:
        img_path = store.resolve_path(ref)
        is_encrypted = img_path.suffix == ".enc"
        if is_encrypted:
            if open_image_encrypted is None:
                if encrypted_caption:
                    st.caption(encrypted_caption)
                elif missing_caption:
                    st.caption(missing_caption)
                return
            with open_image_encrypted(str(img_path)) as im:
                st.image(im, caption=caption, use_container_width=use_container_width)
            return
        st.image(str(img_path), caption=caption, use_container_width=use_container_width)
    except Exception as exc:
        if missing_caption:
            st.caption(missing_caption)
        if missing_caption is None:
            st.caption(f"Image artifact unavailable ({type(exc).__name__}).")
