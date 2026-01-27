#!/bin/bash
# Query Qwen3-TTS Gradio API using curl
# Usage: ./query_tts_curl.sh "Your text here" [speaker] [language] [instruction]

# Configuration
HOST="localhost:8000"
TEXT="${1:-Hello, this is a test of the Qwen3 TTS system.}"
SPEAKER="${2:-Vivian}"
LANGUAGE="${3:-Auto}"
INSTRUCTION="${4:-}"

echo "========================================"
echo "Qwen3-TTS API Query"
echo "========================================"
echo "Text: ${TEXT}"
echo "Speaker: ${SPEAKER}"
echo "Language: ${LANGUAGE}"
if [ -n "${INSTRUCTION}" ]; then
    echo "Instruction: ${INSTRUCTION}"
fi
echo "========================================"
echo ""

# Step 1: Call the API endpoint
echo "Step 1: Calling API..."
RESPONSE=$(curl -s -X POST "http://${HOST}/gradio_api/run/run_instruct" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      "'"${TEXT}"'",
      "'"${LANGUAGE}"'",
      "'"${SPEAKER}"'",
      "'"${INSTRUCTION}"'"
    ]
  }')

echo "Response: ${RESPONSE}"
echo ""

# Check if response contains data directly (fast response) or event_id (SSE streaming)
HAS_DATA=$(echo "${RESPONSE}" | jq -r '.data // empty')
EVENT_ID=$(echo "${RESPONSE}" | jq -r '.event_id // empty')

if [ -n "${HAS_DATA}" ] && [ "${HAS_DATA}" != "null" ]; then
    # Direct response - data is already available
    echo "✓ Generation complete (direct response)"
    AUDIO_INFO="${HAS_DATA}"
elif [ -n "${EVENT_ID}" ]; then
    # SSE streaming - need to poll for results
    echo "Event ID: ${EVENT_ID}"
    echo "Step 2: Polling for results..."
    
    RESULT=$(curl -s -N "http://${HOST}/gradio_api/run/run_instruct/${EVENT_ID}")
    
    # Parse SSE stream
    AUDIO_INFO=$(echo "${RESULT}" | grep "^data:" | tail -1 | sed 's/^data: //')
    
    if [ -z "${AUDIO_INFO}" ]; then
        echo "Error: No audio data in response"
        exit 1
    fi
else
    echo "Error: Invalid response format"
    echo "Full response: ${RESPONSE}"
    exit 1
fi

# Extract the file path from the audio info
AUDIO_FILE=$(echo "${AUDIO_INFO}" | jq -r '.[0].path // .[0].name // .[0] // empty' 2>/dev/null)

if [ -z "${AUDIO_FILE}" ] || [ "${AUDIO_FILE}" = "null" ]; then
    echo "Error: Could not extract audio file path"
    echo "Audio info: ${AUDIO_INFO}"
    exit 1
fi

echo "Audio file generated: ${AUDIO_FILE}"

# Step 3: Download the audio file
OUTPUT_FILE="output_${SPEAKER}_$(date +%s).wav"
echo ""
echo "Downloading audio to ${OUTPUT_FILE}..."

# The file path is relative to the Gradio server
curl -s -o "${OUTPUT_FILE}" "http://${HOST}/gradio_api/file=${AUDIO_FILE}"

if [ -f "${OUTPUT_FILE}" ] && [ -s "${OUTPUT_FILE}" ]; then
    FILE_SIZE=$(stat -c%s "${OUTPUT_FILE}" 2>/dev/null || stat -f%z "${OUTPUT_FILE}" 2>/dev/null)
    echo "✓ Success! Audio saved to: ${OUTPUT_FILE} (${FILE_SIZE} bytes)"
else
    echo "✗ Error: Failed to download audio file"
    exit 1
fi

