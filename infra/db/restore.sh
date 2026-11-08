#!/bin/sh
# Runs once, when the data directory is created (docker-entrypoint-initdb.d).
set -e
createdb -U "$POSTGRES_USER" "$RAG_DB"
pg_restore -U "$POSTGRES_USER" -d "$RAG_DB" --no-owner /docker-entrypoint-initdb.d/index.dump
