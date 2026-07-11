#!/usr/bin/env bash
set -euo pipefail

name="${DOCMIND_QDRANT_CONTAINER:-docmind-qdrant-local}"
image="${DOCMIND_QDRANT_IMAGE:-qdrant/qdrant:v1.18.2}"
port="${DOCMIND_QDRANT_PORT:-6333}"
grpc_port="${DOCMIND_QDRANT_GRPC_PORT:-6334}"
storage="${DOCMIND_QDRANT_STORAGE:-$PWD/data/qdrant-local}"

mkdir -p "$storage"

container_ports_are_safe() {
  local rest_binding grpc_binding
  if ! rest_binding="$(docker port "$name" 6333/tcp 2>/dev/null)"; then
    return 1
  fi
  if ! grpc_binding="$(docker port "$name" 6334/tcp 2>/dev/null)"; then
    return 1
  fi
  [[ "$rest_binding" == "127.0.0.1:${port}" ]] &&
    [[ "$grpc_binding" == "127.0.0.1:${grpc_port}" ]]
}

refuse_unsafe_reuse() {
  echo "Refusing to reuse existing container '${name}': Qdrant ports are missing or not bound exactly to 127.0.0.1:${port} and 127.0.0.1:${grpc_port}." >&2
  echo "Inspect it with: docker port ${name}" >&2
  echo "Preserve any existing data, then remove/rename the container manually or choose another DOCMIND_QDRANT_CONTAINER name." >&2
}

container_state=""
if docker ps --format '{{.Names}}' | grep -Fxq "$name"; then
  container_state="running"
elif docker ps -a --format '{{.Names}}' | grep -Fxq "$name"; then
  container_state="stopped"
fi

if [[ -n "$container_state" ]]; then
  if ! container_ports_are_safe; then
    refuse_unsafe_reuse
    exit 1
  fi
  if [[ "$container_state" == "running" ]]; then
    echo "Qdrant already running: http://localhost:${port} grpc://localhost:${grpc_port}"
  else
    docker start "$name" >/dev/null
    echo "Qdrant running: http://localhost:${port} grpc://localhost:${grpc_port}"
  fi
  exit 0
fi

docker run -d \
  --name "$name" \
  -p "127.0.0.1:${port}:6333" \
  -p "127.0.0.1:${grpc_port}:6334" \
  -v "${storage}:/qdrant/storage" \
  "$image" >/dev/null

echo "Qdrant running: http://localhost:${port} grpc://localhost:${grpc_port}"
