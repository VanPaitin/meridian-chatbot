import os
from collections.abc import AsyncIterator

import gradio as gr
from agents import Agent, Runner, trace
from agents.mcp import MCPServerStreamableHttp
from agents.model_settings import ModelSettings
from dotenv import load_dotenv

load_dotenv()


def get_mcp_server_url() -> str:
    server_url = os.getenv("MCP_SERVER_URL", "").strip()
    if not server_url:
        raise RuntimeError("MCP_SERVER_URL environment variable is required.")
    return server_url


def _conversation_input(
    message: str,
    history: list[dict[str, str]],
) -> list[dict[str, object]]:
    input_items: list[dict[str, object]] = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"user", "assistant"} or not content:
            continue

        content = content[0]["text"] if isinstance(content, list) else str(content)
        input_items.append({"role": role, "content": content})

    input_items.append({"role": "user", "content": message})
    return input_items


def build_agent(server: MCPServerStreamableHttp) -> Agent:
    return Agent(
        name="Meridian Assistant",
        instructions=(
            "Use the MCP server tools to answer the user's question. "
            "Prefer tool results over general knowledge. "
            "If the question is not answerable by the tools, say you don't know "
            "and avoid making up an answer. You can nudge the user to ask "
            "questions that the tools can answer. "
            "Please respond with only the answer, without any additional commentary or formatting."
        ),
        model="gpt-4.1-mini",
        mcp_servers=[server],
        model_settings=ModelSettings(
            tool_choice="required",
        ),
    )


async def _stream_agent(
    user_message: str,
    history: list[dict[str, str]],
) -> AsyncIterator[str]:
    server_url = get_mcp_server_url()
    async with MCPServerStreamableHttp(
        name="Meridian MCP Server",
        params={"url": server_url},
        cache_tools_list=True,
        max_retry_attempts=3,
    ) as server:
        agent = build_agent(server)

        with trace(
            "Meridian MCP Chatbot",
            metadata={
                "mcp_server_url": server_url,
                "history_turns": str(len(history)),
                "message_length": str(len(user_message)),
            },
        ):
            result = Runner.run_streamed(
                agent,
                _conversation_input(user_message, history),
            )
            streamed_text = ""
            async for event in result.stream_events():
                if event.type != "raw_response_event":
                    continue

                data = event.data
                if data.type != "response.output_text.delta":
                    continue

                streamed_text += data.delta
                yield streamed_text

            final_output = result.final_output
            if final_output and final_output != streamed_text:
                yield final_output


# chat function that gradio will call, it will stream the response from the agent to the UI
async def respond(
    message: str,
    history: list[dict[str, str]],
) -> AsyncIterator[str]:
    history = history or []
    if not message.strip():
        yield ""
        return

    try:
        async for chunk in _stream_agent(message, history):
            yield chunk
    except Exception as exc:
        yield f"Agent request failed: {exc}"


def main() -> None:
    get_mcp_server_url()
    gr.ChatInterface(
        fn=respond,
        title="Meridian MCP Chatbot",
        autoscroll=True,
    ).launch()


if __name__ == "__main__":
    main()
