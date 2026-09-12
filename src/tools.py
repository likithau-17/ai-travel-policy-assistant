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


if __name__ == "__main__":
    result = check_employee_eligibility("EMP001")

    print("Employee eligibility result:")
    print(result)