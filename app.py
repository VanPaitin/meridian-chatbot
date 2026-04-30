import asyncio
import os

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

        content_type = "input_text" if role == "user" else "output_text"
        input_items.append(
            {
                "role": role,
                "content": [{"type": content_type, "text": str(content)}],
            }
        )

    input_items.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": message}],
        }
    )
    return input_items


async def _run_agent(user_message: str, history: list[dict[str, str]]) -> str:
    async with MCPServerStreamableHttp(
        name="Meridian MCP Server",
        params={"url": get_mcp_server_url()},
        cache_tools_list=True,
        max_retry_attempts=3,
    ) as server:
        agent = Agent(
            name="Meridian Assistant",
            instructions=(
                "Use the MCP server tools to answer the user's question. "
                "Prefer tool results over general knowledge. "
                "If the question is not answerable by the tools, say you don't know "
                "and avoid making up an answer. You can nudge the user to ask "
                "questions that the tools can answer."
            ),
            mcp_servers=[server],
            model_settings=ModelSettings(
                tool_choice="required",
            ),
        )
        with trace(
            "Meridian MCP Chatbot",
            metadata={
                "mcp_server_url": get_mcp_server_url(),
                "history_turns": str(len(history)),
                "message_length": str(len(user_message)),
            },
        ):
            result = await Runner.run(
                agent,
                _conversation_input(user_message, history),
            )

    return str(result.final_output)


def respond(
    message: str,
    history: list[dict[str, str]],
):
    history = history or []
    if not message.strip():
        return history, ""

    try:
        response = asyncio.run(_run_agent(message, history))
    except Exception as exc:
        response = f"Agent request failed: {exc}"

    history = history + [{"role": "user", "content": message}]
    history.append({"role": "assistant", "content": response})
    return history, ""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Meridian MCP Chatbot") as demo:
        gr.Markdown("# Meridian MCP Chatbot")

        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(height=520)
                message = gr.Textbox(
                    label="Message",
                    placeholder="Ask something the MCP server can answer...",
                    lines=2,
                )
                send = gr.Button("Send", variant="primary")

        send.click(
            respond,
            inputs=[message, chatbot],
            outputs=[chatbot, message],
        )
        message.submit(
            respond,
            inputs=[message, chatbot],
            outputs=[chatbot, message],
        )

    return demo


def main() -> None:
    get_mcp_server_url()
    build_app().launch()


if __name__ == "__main__":
    main()
