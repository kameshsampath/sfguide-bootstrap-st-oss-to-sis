#!/usr/bin/env bash

set -o pipefail
set -u

trap exit 1 SIGINT

streamlit run streamlit_app.py \
  --server.address="${SERVICE_HOST:-0.0.0.0}" \
  --server.port="${SERVER_PORT}" \
  --server.headless=true \
  --browser.gatherUsageStats=false

