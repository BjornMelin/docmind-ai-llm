"""Shared helpers for rendering artifact-backed images in the UI."""

from __future__ import annotations

import streamlit as st
from loguru import logger

from src.persistence.artifacts import ArtifactRef, ArtifactStore
from src.utils.log_safety import build_pii_log_entry


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
    except (ImportError, ModuleNotFoundError):
        open_image_encrypted = None

    try:
        img_path = store.resolve_path(ref)
        is_encrypted = img_path.suffix == ".enc"

        # Guard: encrypted image without decryption support
        if is_encrypted:
            if open_image_encrypted is None:
                st.caption(
                    encrypted_caption
                    or missing_caption
                    or "Encrypted image unavailable"
                )
                return
            with open_image_encrypted(str(img_path)) as im:
                st.image(im, caption=caption, use_container_width=use_container_width)
            return

        # Display regular image
        st.image(
            str(img_path), caption=caption, use_container_width=use_container_width
        )
    except Exception as exc:
        redaction = build_pii_log_entry(str(exc), key_id="ui.artifacts.render_image")
        logger.error(
            "Failed to render image artifact (artifact_id={}, error_type={}, error={})",
            ref.sha256,
            type(exc).__name__,
            redaction.redacted,
        )
        st.caption(
            missing_caption or f"Image artifact unavailable ({type(exc).__name__})."
        )
