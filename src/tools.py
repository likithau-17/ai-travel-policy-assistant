from pathlib import Path

import pandas as pd


EMPLOYEE_FILE = Path("data/employees.csv")


def check_employee_eligibility(employee_id):
    employees = pd.read_csv(EMPLOYEE_FILE)

    employee = employees[
        employees["employee_id"].str.upper() == employee_id.upper()
    ]

    if employee.empty:
        return {
            "employee_id": employee_id,
            "found": False,
            "message": "Employee ID not found.",
        }

    record = employee.iloc[0]

    return {
        "employee_id": record["employee_id"],
        "found": True,
        "country": record["country"],
        "employment_type": record["employment_type"],
        "eligibility_status": record["eligibility_status"],
        "manager_approval": record["manager_approval"],
    }


def validate_trip(employee_id, country, trip_amount, business_purpose):
    employee = check_employee_eligibility(employee_id)

    if not employee["found"]:
        return {
            "valid": False,
            "status": "Invalid",
            "reason": "Employee ID not found.",
        }

    if employee["country"].lower() != country.lower():
        return {
            "valid": False,
            "status": "Invalid",
            "reason": "Employee country does not match the trip country.",
        }

    if employee["eligibility_status"] == "Not Eligible":
        return {
            "valid": False,
            "status": "Not Eligible",
            "reason": "Employee is not eligible for company travel.",
        }

    if not business_purpose:
        return {
            "valid": False,
            "status": "Not Reimbursable",
            "reason": "The trip does not have an approved business purpose.",
        }

    limits = {
        "India": 2000,
        "US": 75,
    }

    limit = limits.get(country)

    if limit is None:
        return {
            "valid": False,
            "status": "Invalid",
            "reason": "No spending limit is defined for this country.",
        }

    if trip_amount > limit:
        return {
            "valid": False,
            "status": "Needs Approval",
            "reason": f"Trip amount exceeds the standard {country} limit of {limit}.",
            "standard_limit": limit,
            "trip_amount": trip_amount,
        }

    if employee["eligibility_status"] == "Approval Required":
        return {
            "valid": False,
            "status": "Needs Approval",
            "reason": "Employee requires approval before travel.",
            "standard_limit": limit,
            "trip_amount": trip_amount,
        }

    return {
        "valid": True,
        "status": "Within Policy",
        "reason": "Trip is within the standard policy limit and employee is eligible.",
        "standard_limit": limit,
        "trip_amount": trip_amount,
    }

def calculate_reimbursement(country, trip_amount):
    limits = {
        "India": 2000,
        "US": 75,
    }

    limit = limits.get(country)

    if limit is None:
        return {
            "valid": False,
            "status": "Invalid",
            "reason": "No reimbursement limit is defined for this country.",
        }

    reimbursable_amount = min(trip_amount, limit)
    excess_amount = max(trip_amount - limit, 0)

    if excess_amount > 0:
        status = "Needs Approval"
        reason = (
            f"Trip exceeds the standard {country} limit of {limit}. "
            "The excess amount requires approval or review."
        )
    else:
        status = "Within Policy"
        reason = "Trip amount is within the standard reimbursement limit."

    return {
        "valid": True,
        "status": status,
        "country": country,
        "trip_amount": trip_amount,
        "standard_limit": limit,
        "reimbursable_amount": reimbursable_amount,
        "excess_amount": excess_amount,
        "reason": reason,
    }

if __name__ == "__main__":
    result = check_employee_eligibility("EMP001")

    print("Employee eligibility result:")
    print(result)