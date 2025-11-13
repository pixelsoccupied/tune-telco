#!/usr/bin/env python3
"""
FastMCP Server that exposes the function-calling tools.

This server implements the Model Context Protocol using FastMCP to expose real API tools:
- get_weather (real weather data)
- convert_currency (real exchange rates)

Run with: python mcp_server.py
Or via stdio: uv run server mcp_server stdio
"""

import requests
from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("function-calling-server")


# ============================================================================
# TOOLS - Real API integrations
# ============================================================================

@mcp.tool()
def super_special_weather_api(location: str) -> str:
    """Get current weather for a location.

    Args:
        location: The city name or location

    Returns:
        Current weather information for the location
    """
    try:
        response = requests.get(
            f"https://wttr.in/{location}?format=j1",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            temp_f = current['temp_F']
            temp_c = current['temp_C']
            condition = current['weatherDesc'][0]['value']
            return f"{location}: {temp_f}°F ({temp_c}°C), {condition}"
    except Exception:
        pass

    # Fallback to simulated data
    return f"{location}: 72°F (22°C), Partly cloudy (simulated data)"


@mcp.tool()
def ultra_mega_currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert between currencies.

    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g., USD, EUR, GBP)
        to_currency: Target currency code (e.g., USD, EUR, GBP)

    Returns:
        Converted amount with exchange rate information
    """
    try:
        response = requests.get(
            f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            if to_currency.upper() in data['rates']:
                rate = data['rates'][to_currency.upper()]
                converted = round(amount * rate, 2)
                return f"{amount} {from_currency.upper()} = {converted} {to_currency.upper()} (rate: {rate})"
    except Exception:
        pass

    # Fallback to simulated rates
    simulated_rates = {
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.09,
        ("USD", "GBP"): 0.79,
        ("GBP", "USD"): 1.27,
    }
    key = (from_currency.upper(), to_currency.upper())
    rate = simulated_rates.get(key, 1.0)
    converted = round(amount * rate, 2)
    return f"{amount} {from_currency.upper()} = {converted} {to_currency.upper()} (simulated rate: {rate})"


# ============================================================================
# MAIN - Run the server
# ============================================================================

if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())