import asyncio

from mcp.server import MCPServer

from src.tools import check_employee_eligibility


server = MCPServer(
    name="AI Travel Policy Assistant",
    version="1.0.0",
)


@server.tool(
    name="check_employee_eligibility",
    description="Check whether an employee is eligible for company travel.",
)
async def employee_eligibility(employee_id: str) -> dict:
    return check_employee_eligibility(employee_id)


async def main():
    await server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())