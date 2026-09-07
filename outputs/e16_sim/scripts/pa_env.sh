#!/usr/bin/env bash
export PRIME_AGENT_CODING_AGENT_DIR=/root/nld/.prime-agent-e16-sim
key=$(python3 -c "import json;print(json.load(open('/root/.prime/config.json'))['api_key'])")
export PI_API_KEY="$key" PRIME_API_KEY="$key"
unset key
export PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0
export E16_MODEL="z-ai/glm-5.2"
export E16_PROVIDER="prime-inference"
