# Meridian Chatbot

A small Gradio app that lets users chat with an OpenAI Agents SDK agent connected
to an MCP server over Streamable HTTP.

## Run

```bash
MCP_SERVER_URL=https://order-mcp-74afyau24q-uc.a.run.app/mcp uv run app.py
```

Open the local URL printed by Gradio.

## Configure

Set the MCP server URL before launch:

- `MCP_SERVER_URL`: required Streamable HTTP MCP endpoint.
- `OPENAI_API_KEY`: required by the OpenAI Agents SDK.

The app exits during startup if `MCP_SERVER_URL` is not set. Once running, the
agent discovers and calls the MCP server's tools through the OpenAI Agents SDK.
