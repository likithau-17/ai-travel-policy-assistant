import asyncio

from mcp.server import MCPServer

from src.tools import (
    check_employee_eligibility,
    validate_trip,
    calculate_reimbursement,
)


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


@server.tool(
    name="validate_trip",
    description="Validate an employee business trip against company travel policy.",
)
async def trip_validation(
    employee_id: str,
    country: str,
    trip_amount: float,
    business_purpose: bool,
) -> dict:
    return validate_trip(
        employee_id=employee_id,
        country=country,
        trip_amount=trip_amount,
        business_purpose=business_purpose,
    )


@server.tool(
    name="calculate_reimbursement",
    description="Calculate the reimbursable amount for a business trip.",
)
async def reimbursement_calculation(
    country: str,
    trip_amount: float,
) -> dict:
    return calculate_reimbursement(
        country=country,
        trip_amount=trip_amount,
    )


async def main():
    await server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())