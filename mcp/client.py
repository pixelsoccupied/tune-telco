#!/usr/bin/env python3
"""
MCP Demo: Fine-tuned model calling MCP tools.

This demonstrates the complete integration:
1. Fine-tuned model generates function calls (Glaive format)
2. Parser extracts the call
3. MCP client executes the tool
4. Results displayed to user

Run the MCP server first in another terminal:
    uv run mcp/server.py

Then run this demo:
    uv run mcp/client.py
"""

import asyncio
import json
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import mlx_lm
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_NAME = "mlx-community/Qwen3-0.6B-4bit"
ADAPTER_PATH = "./train/adapters"


# ============================================================================
# UTILITIES
# ============================================================================

def build_system_prompt(functions):
    """Build system prompt matching Glaive training format."""
    functions_str = "\n".join(json.dumps(f, indent=4) for f in functions)
    return f"""You are a helpful assistant with access to the following functions. Use them if required -
{functions_str}
"""


def parse_function_call(response: str) -> dict:
    """
    Parse function call from model response.

    Glaive format: <functioncall> {"name": "func", "arguments": '{"param": value}'}
    Note: arguments is a JSON string with single quotes around it.
    """
    # Look for <functioncall> tags
    match = re.search(r'<functioncall>\s*(.+?)(?:</functioncall>|$)', response, re.DOTALL | re.IGNORECASE)

    if match:
        json_str = match.group(1).strip()

        # Extract function name and arguments
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', json_str)
        args_match = re.search(r'"arguments"\s*:\s*\'([^\']+)\'', json_str)

        if name_match and args_match:
            try:
                return {
                    "name": name_match.group(1),
                    "arguments": json.loads(args_match.group(1))
                }
            except (json.JSONDecodeError, ValueError):
                pass

    return None


# ============================================================================
# MCP CLIENT
# ============================================================================

class MCPClient:
    """Client for interacting with MCP servers."""

    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.tools = []

    async def connect(self, server_script: str):
        """Connect to an MCP server."""
        server_params = StdioServerParameters(
            command="uv",
            args=["run", server_script],
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self.session.initialize()

        # List available tools
        await self.refresh_tools()

    async def refresh_tools(self):
        """Fetch available tools from the server."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        response = await self.session.list_tools()
        self.tools = []

        for item in response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    self.tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    })

    async def call_tool(self, name: str, arguments: dict):
        """Execute a tool via MCP."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        try:
            result = await self.session.call_tool(name, arguments)
            # Extract text content from result
            if hasattr(result, 'content') and len(result.content) > 0:
                return result.content[0].text
            return str(result)
        except Exception as e:
            return f"Error calling tool: {e}"

    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()


# ============================================================================
# AGENT LOGIC
# ============================================================================

async def run_mcp_agent(model, tokenizer, mcp_client, user_query: str, model_name: str = "Model", show_prompt: bool = False):
    """Run agent with MCP tool calling."""
    import time

    console.print(f"[dim]→ {model_name}: Generating response...[/dim]")

    # Build system prompt with MCP tools
    tools_json = mcp_client.tools
    system_prompt = build_system_prompt(tools_json)

    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        add_generation_prompt=True,
        tokenize=False
    )

    # Show the full prompt if requested
    if show_prompt:
        console.print(Panel(
            formatted_prompt,
            title=f"{model_name} - Full Prompt Sent to LLM",
            border_style="blue",
            expand=False
        ))

    # Generate response from model (with timing)
    start_time = time.time()
    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=200,
        verbose=False
    )
    generation_time = time.time() - start_time

    # Parse function call (Glaive format)
    function_call = parse_function_call(response)

    result = {
        "response": response,
        "function_call": function_call,
        "tool_result": None,
    }

    # Display generation time
    console.print(f"[dim]  ⏱️  Generation time: {generation_time:.2f}s[/dim]")

    # If function call was detected, execute via MCP
    if function_call:
        console.print(f"[cyan]→ {model_name}: Detected function call → {function_call['name']}()[/cyan]")
        console.print(f"[dim]  Arguments: {function_call['arguments']}[/dim]")
        console.print(f"[cyan]→ {model_name}: Executing via MCP...[/cyan]")

        tool_result = await mcp_client.call_tool(
            function_call['name'],
            function_call['arguments']
        )
        result["tool_result"] = tool_result
        console.print(f"[green]→ {model_name}: ✓ Tool execution complete[/green]")
    else:
        console.print(f"[yellow]→ {model_name}: No function call detected[/yellow]")

    result["generation_time"] = generation_time
    return result


# ============================================================================
# DEMO
# ============================================================================

async def main():
    console.print("\n[bold cyan]Loading models...[/bold cyan]")

    # Load base model
    console.print(f"  • Loading base model ({MODEL_NAME})...")
    base_model, base_tokenizer = mlx_lm.load(MODEL_NAME)

    # Load fine-tuned model
    console.print("  • Loading fine-tuned model (with adapters)...")
    finetuned_model, finetuned_tokenizer = mlx_lm.load(
        MODEL_NAME,
        adapter_path=ADAPTER_PATH
    )
    console.print("[bold green]✓ Models loaded![/bold green]\n")

    # Connect to MCP server
    console.print("[bold cyan]Connecting to MCP server...[/bold cyan]")
    mcp_client = MCPClient()

    try:
        await mcp_client.connect("mcp/server.py")
        console.print(f"[bold green]✓ Connected! Found {len(mcp_client.tools)} tools[/bold green]")

        # Show available tools
        console.print("\n[bold yellow]Available MCP Tools:[/bold yellow]")
        for tool in mcp_client.tools:
            console.print(f"  • [cyan]{tool['name']}[/cyan]: {tool['description']}")

        console.print("\n" + "=" * 100)

        # Test queries - real API calls only
        test_queries = [
            "i live in canada, i want to travel to europe. how much local currency will i get for $100",
            "What's the weather like in Boston?",
            "Convert 100 USD to EUR",
            "What's the weather in Tokyo?",
            "How much is 50 GBP in USD?",
            "what's the weather like in Raleigh?",
        ]

        for i, query in enumerate(test_queries, 1):
            console.print(f"\n[bold cyan]Test {i}/{len(test_queries)}[/bold cyan]")
            console.print(Panel(query, title="User Query", border_style="blue"))

            # Show full prompt only for first query
            show_full_prompt = (i == 1)

            # Run both models
            console.print("\n[bold yellow]🔸 Base Model[/bold yellow]")
            base_result = await run_mcp_agent(
                base_model, base_tokenizer, mcp_client, query,
                model_name="Base Model", show_prompt=show_full_prompt
            )

            console.print("\n[bold green]🔹 Fine-tuned Model[/bold green]")
            finetuned_result = await run_mcp_agent(
                finetuned_model, finetuned_tokenizer, mcp_client, query,
                model_name="Fine-tuned Model", show_prompt=show_full_prompt
            )

            # Display side-by-side comparison - raw output only
            console.print("\n[bold magenta]Raw Model Outputs:[/bold magenta]")
            table = Table(show_header=True, header_style="bold magenta", expand=True)
            table.add_column("Base Model", style="yellow", width=40)
            table.add_column("Fine-tuned Model", style="green", width=40)

            table.add_row(
                base_result['response'],
                finetuned_result['response']
            )

            console.print(table)

            # Show the MCP tool result
            console.print("\n[bold cyan]MCP Tool Result:[/bold cyan]")

            if finetuned_result['tool_result']:
                console.print(Panel(
                    finetuned_result['tool_result'],
                    title="✓ Fine-tuned Model Result",
                    border_style="green"
                ))
            elif base_result['tool_result']:
                console.print(Panel(
                    base_result['tool_result'],
                    title="Base Model Result",
                    border_style="yellow"
                ))
            else:
                console.print("[red]Neither model generated a valid function call[/red]")

            console.print("\n" + "-" * 100)

        console.print("\n[bold green]Demo complete![/bold green]\n")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
    finally:
        await mcp_client.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise