#!/bin/bash
# TracheaAI — Start local server
cd "$(dirname "$0")"
echo ""
echo "🫁 TracheaAI Server"
echo "   Open in browser: http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""
python3 -m uvicorn server.app:app --host 127.0.0.1 --port 8000
