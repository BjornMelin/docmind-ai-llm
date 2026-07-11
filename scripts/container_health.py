"""Fail unless the local Streamlit port accepts a TCP connection."""

from __future__ import annotations

import socket


def main() -> None:
    """Run the recurring local TCP liveness check."""
    with socket.create_connection(("127.0.0.1", 8501), timeout=3):
        pass


if __name__ == "__main__":
    main()
