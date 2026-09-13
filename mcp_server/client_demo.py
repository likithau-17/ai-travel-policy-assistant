import asyncio

from mcp import Client
from mcp_server.server import server


async def main():
    async with Client(server) as client:
        tools = await client.list_tools()

        print("Available tools:")
        for tool in tools:
            print("-", tool)

        result = await client.call_tool(
            "check_employee_eligibility",
            {"employee_id": "EMP001"},
        )

        print("\nTool result:")
        for item in result.content:
            print(item.text)


if __name__ == "__main__":
    asyncio.run(main())
