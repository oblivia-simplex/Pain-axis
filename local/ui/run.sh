#!/usr/bin/env bash
# Start the explorer on localhost only (the probe tab can run GPU jobs) with telemetry off.
cd "$(dirname "$0")/../.." || exit 1
exec .venv-Pain-axis/bin/streamlit run local/ui/app.py \
  --server.address 127.0.0.1 --server.port "${PORT:-8501}" --server.headless true \
  --browser.gatherUsageStats false "$@"
