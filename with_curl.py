
import requests, base64

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = False

def read_b64(path):
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode()

headers = {
  "Authorization": "Bearer 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",
  "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
  "model": "google/gemma-4-31b-it",
  "messages": [{"role":"user","content":""}],
  "max_tokens": 16384,
  "temperature": 1.00,
  "top_p": 0.95,
  "stream": stream,
  "chat_template_kwargs": {"enable_thinking":True},
}

response = requests.post(invoke_url, headers=headers, json=payload, stream=stream)
if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())
