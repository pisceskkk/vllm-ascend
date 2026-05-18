curl -X POST 'http://localhost:8091/v1/chat/completions' \
  -H 'content-type: application/json' \
  -d '{
    "messages": [
      {
        "content": "你是谁？",
        "role": "user"
      }
    ],
    "max_tokens": 100,
    "stop": null,
    "temperature": 0.0,
    "top_p": 0.95
  }'
