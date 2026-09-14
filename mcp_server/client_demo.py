import asyncio

from mcp import Client
from mcp_server.server import server


async def main():
    async with Client(server) as client:
        tools = await client.list_tools()

        print("Available tools:")
        for tool in tools:
            print("-", tool)

        print("\nEligibility result:")
        result = await client.call_tool(
            "check_employee_eligibility",
            {"employee_id": "EMP001"},
        )
        for item in result.content:
            print(item.text)

        print("\nTrip validation result:")
        result = await client.call_tool(
            "validate_trip",
            {
                "employee_id": "EMP001",
                "country": "India",
                "trip_amount": 1500,
                "business_purpose": True,
            },
        )
        for item in result.content:
            print(item.text)

        print("\nReimbursement result:")
        result = await client.call_tool(
            "calculate_reimbursement",
            {
                "country": "India",
                "trip_amount": 2500,
            },
        )
        for item in result.content:
            print(item.text)


if __name__ == "__main__":
    asyncio.run(main())