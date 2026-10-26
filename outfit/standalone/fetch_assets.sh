#!/bin/sh
# Download the MediaPipe files the standalone page loads from its own folder.
# The model goes out as base64 text because the artifact host only serves web file types.
set -eu
cd "$(dirname "$0")"
MP=https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14
MODEL=https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
mkdir -p wasm
curl -sSfo vision_bundle.mjs "$MP/vision_bundle.mjs"
curl -sSfo wasm/vision_wasm_internal.js "$MP/wasm/vision_wasm_internal.js"
curl -sSfo wasm/vision_wasm_internal.wasm "$MP/wasm/vision_wasm_internal.wasm"
curl -sSf "$MODEL" | base64 | tr -d '\n' > pose_landmarker_full.b64.txt
echo "done: $(du -sh . | cut -f1)"
