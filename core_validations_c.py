import re
import numpy as np
import pandas as pd
from datetime import datetime

def check_C61_missing_facility_id(df, facility_id_column):
    """Check for missing Facility ID and return results in table format."""

    missing_count = df[facility_id_column].isna().sum()  # Count missing values

    result_df = pd.DataFrame([{
        "Check ID": 61,
        "Variable checked": facility_id_column,
        "Description": "Missing check for Facility ID.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C62_facility_borrower_mapping(df, facility_id_column, borrower_id_column):
    """Ensure each Facility ID maps to a valid Borrower ID."""

    # Identify Facility IDs where Borrower ID is missing or invalid
    failed_count = df[borrower_id_column].isna().sum()  # Count facilities without valid borrower IDs

    result_df = pd.DataFrame([{
        "Check ID": 62,
        "Variable checked": f"{facility_id_column}, {borrower_id_column}",
        "Description": "Each Facility ID must map to a valid Borrower ID (referential integrity).",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C63_missing_facility_start_date(df, start_date_column):
    """Check for missing Facility Start Date and return results in table format."""

    missing_count = df[start_date_column].isna().sum()  # Count missing values

    result_df = pd.DataFrame([{
        "Check ID": 63,
        "Variable checked": start_date_column,
        "Description": "Missing check for Facility Start Date. Must have a start date.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C64_facility_end_vs_start(df, start_date_column, end_date_column):
    """Ensure Facility End Date (if present) is >= Facility Start Date."""

    # Convert columns to datetime format
    df[start_date_column] = pd.to_datetime(df[start_date_column], errors='coerce')
    df[end_date_column] = pd.to_datetime(df[end_date_column], errors='coerce')

    # Identify cases where Facility End Date is before the Facility Start Date
    failed_count = (df[end_date_column] < df[start_date_column]).sum()

    result_df = pd.DataFrame([{
        "Check ID": 64,
        "Variable checked": f"{start_date_column}, {end_date_column}",
        "Description": "If Facility End Date is not null, it must be >= Facility Start Date. No negative durations.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C65_valid_facility_type(df, facility_type_column, valid_types = ['Term Loan', 'Revolver', 'Overdraft', 'Credit Line']):
    """Check if Facility Type is within the valid categories."""

    invalid_count = (~df[facility_type_column].isin(valid_types)).sum()  # Count invalid values

    result_df = pd.DataFrame([{
        "Check ID": 65,
        "Variable checked": facility_type_column,
        "Description": "Facility Type must be in a valid category (term loan, revolver, overdraft, etc.).",
        "Failed Count": invalid_count,
        "% of overall data": f"{(invalid_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_C66_valid_facility_currency(df, currency_column, valid_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF']):
    """Check if Facility Currency is a valid ISO 4217 currency code."""

    invalid_count = (~df[currency_column].isin(valid_currencies)).sum()  # Count invalid values

    result_df = pd.DataFrame([{
        "Check ID": 66,
        "Variable checked": currency_column,
        "Description": "Facility Currency must be a valid ISO 4217 currency code (e.g., USD, EUR).",
        "Failed Count": invalid_count,
        "% of overall data": f"{(invalid_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_C67_missing_original_facility_amount(df, amount_column):
    """Check for missing Original Facility Amount and return results in table format."""

    missing_count = df[amount_column].isna().sum()  # Count missing values

    result_df = pd.DataFrame([{
        "Check ID": 67,
        "Variable checked": amount_column,
        "Description": "Missing check for Original Facility Amount. Must have a value.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C68_non_negative_original_facility_amount(df, amount_column):
    """Ensure Original Facility Amount is non-negative (≥ 0)."""

    df[amount_column] = pd.to_numeric(df[amount_column], errors='coerce')  # Convert to numeric
    failed_count = (df[amount_column] < 0).sum()

    result_df = pd.DataFrame([{
        "Check ID": 68,
        "Variable checked": amount_column,
        "Description": "Numeric range check (≥ 0). No negative Original Facility Amount.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C69_missing_credit_limit(df, credit_limit_column):
    """Check for missing Credit Limit (Month-End) values."""

    missing_count = df[credit_limit_column].isna().sum()

    result_df = pd.DataFrame([{
        "Check ID": 69,
        "Variable checked": credit_limit_column,
        "Description": "Missing check for Credit Limit (Month-End). Must have a value.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_C70_no_credit_limit_after_closure(df, end_date_column, snapshot_date_column, credit_limit_column):
    """Ensure Credit Limit is zero if the facility is closed (End Date < Snapshot Date)."""

    df[end_date_column] = pd.to_datetime(df[end_date_column], errors='coerce')
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    failed_count = ((df[end_date_column] < df[snapshot_date_column]) & (df[credit_limit_column] > 0)).sum()

    result_df = pd.DataFrame([{
        "Check ID": 70,
        "Variable checked": f"{end_date_column}, {credit_limit_column}",
        "Description": "If facility is closed (End Date < Snapshot Date), Credit Limit should be zero.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_C71_exposure_vs_credit_limit(df, exposure_column, credit_limit_column):
    """Ensure Outstanding Exposure is ≤ Credit Limit (Month-End)."""

    failed_count = (df[exposure_column] > df[credit_limit_column]).sum()

    result_df = pd.DataFrame([{
        "Check ID": 71,
        "Variable checked": f"{exposure_column}, {credit_limit_column}",
        "Description": "Outstanding Exposure must be ≤ Credit Limit (Month-End), unless system override is in place.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C72_missing_outstanding_exposure(df, exposure_column):
    """Check for missing Outstanding Exposure values."""

    missing_count = df[exposure_column].isna().sum()

    result_df = pd.DataFrame([{
        "Check ID": 72,
        "Variable checked": exposure_column,
        "Description": "Missing check for Outstanding Exposure (Month-End). Must have a value for active facilities.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_C73_utilization_rate(df, exposure_column, credit_limit_column, utilization_column):
    """Ensure Utilization Rate is consistent with Outstanding Exposure / Credit Limit."""

    failed_count = (df[utilization_column] != (df[exposure_column] / df[credit_limit_column])).sum()

    result_df = pd.DataFrame([{
        "Check ID": 73,
        "Variable checked": f"{utilization_column}, {exposure_column}, {credit_limit_column}",
        "Description": "Utilization Rate must match actual calculated ratio (Outstanding Exposure / Credit Limit).",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C74_interest_rate_range(df, interest_rate_column, min_rate=-5, max_rate=100):
    """Ensure Interest Rate is within a valid range (e.g., -5% to 100%)."""

    df[interest_rate_column] = pd.to_numeric(df[interest_rate_column], errors='coerce')  # Convert to numeric
    failed_count = ((df[interest_rate_column] < min_rate) | (df[interest_rate_column] > max_rate)).sum()

    result_df = pd.DataFrame([{
        "Check ID": 74,
        "Variable checked": interest_rate_column,
        "Description": f"Interest Rate must be within a valid range ({min_rate}% to {max_rate}%).",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C75_valid_interest_rate_type(df, interest_rate_type_column):
    """Check if Interest Rate Type is 'Fixed' or 'Floating'."""

    valid_types = ['Fixed', 'Floating']
    failed_count = (~df[interest_rate_type_column].isin(valid_types)).sum()

    result_df = pd.DataFrame([{
        "Check ID": 75,
        "Variable checked": interest_rate_type_column,
        "Description": "Interest Rate Type must be either 'Fixed' or 'Floating'.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C76_valid_repayment_schedule(df, repayment_schedule_column):
    """Ensure Repayment Schedule is in a valid category."""

    valid_schedules = ['Monthly', 'Quarterly', 'Semi-Annual', 'Annual', 'Bullet']
    failed_count = (~df[repayment_schedule_column].isin(valid_schedules)).sum()

    result_df = pd.DataFrame([{
        "Check ID": 76,
        "Variable checked": repayment_schedule_column,
        "Description": "Repayment Schedule must be in a recognized set (e.g., Monthly, Quarterly, Bullet).",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_C77_valid_amortization_type(df, amortization_type_column):
    """Ensure Amortization Type is valid (recognized category)."""

    valid_types = ['Straight-Line', 'Balloon', 'Bullet']
    failed_count = (~df[amortization_type_column].isin(valid_types)).sum()

    result_df = pd.DataFrame([{
        "Check ID": 77,
        "Variable checked": amortization_type_column,
        "Description": "Amortization Type must be in a recognized set (e.g., Straight-Line, Balloon, Bullet).",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C78_missing_facility_purpose(df, purpose_column, max_missing_ratio=0.10):
    """Check for missing Facility Purpose / Loan Purpose values and ensure missing is below 10%."""

    missing_count = df[purpose_column].isna().sum()
    missing_ratio = missing_count / len(df)

    result_df = pd.DataFrame([{
        "Check ID": 78,
        "Variable checked": purpose_column,
        "Description": "Facility Purpose / Loan Purpose must not be missing (>10% missing is flagged).",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_ratio * 100):.2f}%"
    }])

    return result_df

def check_C79_working_capital_maturity(df, purpose_column, maturity_date_column, snapshot_column, max_maturity_years=5):
    """Ensure that if Facility Purpose is 'Working Capital', the Maturity Date is within a reasonable range."""

    df[maturity_date_column] = pd.to_datetime(df[maturity_date_column], errors='coerce')
    df[snapshot_column] = pd.to_datetime(df[snapshot_column], errors='coerce')

    # Calculate the difference in years
    failed_count = ((df[purpose_column] == 'Working Capital') &
                    ((df[maturity_date_column] - df[snapshot_column]).dt.days / 365 > max_maturity_years)).sum()

    result_df = pd.DataFrame([{
        "Check ID": 79,
        "Variable checked": f"{purpose_column}, {maturity_date_column}",
        "Description": f"If 'Working Capital', ensure maturity is within {max_maturity_years} years.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C80_valid_last_drawdown_date(df, drawdown_date_column, start_date_column, snapshot_date_column):
    """Ensure Date of Last Drawdown is >= Facility Start Date and <= Snapshot Date."""

    df[drawdown_date_column] = pd.to_datetime(df[drawdown_date_column], errors='coerce')
    df[start_date_column] = pd.to_datetime(df[start_date_column], errors='coerce')
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    failed_count = ((df[drawdown_date_column] < df[start_date_column]) |
                    (df[drawdown_date_column] > df[snapshot_date_column])).sum()

    result_df = pd.DataFrame([{
        "Check ID": 80,
        "Variable checked": f"{drawdown_date_column}, {start_date_column}, {snapshot_date_column}",
        "Description": "Date of Last Drawdown must be >= Facility Start Date and <= Snapshot Date. No future drawdowns.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_C81_valid_last_payment_date(df, payment_date_column, start_date_column, snapshot_date_column):
    """Ensure Date of Last Payment is <= Snapshot Date and >= Facility Start Date if payment is made."""

    # Convert columns to datetime format
    df[payment_date_column] = pd.to_datetime(df[payment_date_column], errors='coerce')
    df[start_date_column] = pd.to_datetime(df[start_date_column], errors='coerce')
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    # Identify cases where Date of Last Payment is outside the valid range
    failed_count = ((df[payment_date_column] < df[start_date_column]) |
                    (df[payment_date_column] > df[snapshot_date_column])).sum()

    result_df = pd.DataFrame([{
        "Check ID": 81,
        "Variable checked": f"{payment_date_column}, {start_date_column}, {snapshot_date_column}",
        "Description": "Date of Last Payment must be ≤ Snapshot Date and ≥ Facility Start Date if payment is made. No future or pre-start payments.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_C82_closed_facility_exposure(df, end_date_column, snapshot_date_column, exposure_column):
    """Ensure that if Facility End Date < Snapshot Date, Outstanding Exposure should be zero or missing."""

    # Convert columns to datetime format
    df[end_date_column] = pd.to_datetime(df[end_date_column], errors='coerce')
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    # Identify cases where Facility End Date is before Snapshot Date but Exposure is non-zero
    failed_count = ((df[end_date_column] < df[snapshot_date_column]) & (df[exposure_column] > 0)).sum()

    result_df = pd.DataFrame([{
        "Check ID": 82,
        "Variable checked": f"{end_date_column}, {snapshot_date_column}, {exposure_column}",
        "Description": "If End Date < Snapshot, facility should have zero or no outstanding exposure.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_C83_unique_facility_per_borrower(df, facility_id_column, borrower_id_column):
    """Ensure that each Facility ID is associated with only one Borrower ID."""

    duplicate_count = df.duplicated(subset=[facility_id_column, borrower_id_column], keep=False).sum()

    result_df = pd.DataFrame([{
        "Check ID": 83,
        "Variable checked": f"{facility_id_column}, {borrower_id_column}",
        "Description": "Confirm no duplicate Facility IDs across different borrowers. Each Facility ID should belong to only one Borrower.",
        "Failed Count": duplicate_count,
        "% of overall data": f"{(duplicate_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C84_local_currency_vs_country(df, currency_column, country_column, local_currency_mapping = {
    "USD": "United States",
    "EUR": "European Union",
    "GBP": "United Kingdom",
    "JPY": "Japan",
    "CAD": "Canada"
}):
    """Ensure if currency is local, the country of borrower matches the requirement (policy dependent)."""

    failed_count = df[df[currency_column].map(local_currency_mapping) != df[country_column]].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 84,
        "Variable checked": f"{currency_column}, {country_column}",
        "Description": "If currency is local, borrower country must match as per bank policy.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C85_credit_limit_vs_original(df, original_amount_column, credit_limit_column):
    """Ensure current Credit Limit is typically greater than or equal to the Original Facility Amount."""

    failed_count = (df[credit_limit_column] < df[original_amount_column]).sum()

    result_df = pd.DataFrame([{
        "Check ID": 85,
        "Variable checked": f"{original_amount_column}, {credit_limit_column}",
        "Description": "Typically, current Credit Limit should be ≥ Original Facility Amount. Flag if limit is lower.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_C86_credit_limit_fluctuations(df, credit_limit_column):
    """Identify abrupt fluctuations in Credit Limit (e.g., 0 -> large jump -> 0)."""

    failed_count = ((df[credit_limit_column].diff().abs() > (df[credit_limit_column].median() * 2)) &
                    (df[credit_limit_column].shift(1) == 0)).sum()

    result_df = pd.DataFrame([{
        "Check ID": 86,
        "Variable checked": credit_limit_column,
        "Description": "Check for illogical Credit Limit fluctuations (e.g., 0 -> large jump -> 0). Investigate abrupt changes.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C87_other_classification_distribution(df, purpose_column, max_other_percentage=10):
    """Ensure 'Other' classification does not exceed the allowed percentage."""

    failed_count = (df[purpose_column] == 'Other').sum()
    percentage_other = (failed_count / len(df)) * 100

    result_df = pd.DataFrame([{
        "Check ID": 87,
        "Variable checked": purpose_column,
        "Description": f"Ensure 'Other' classification is less than {max_other_percentage}% of total.",
        "Failed Count": failed_count,
        "% of overall data": f"{percentage_other:.2f}%"
    }])

    return result_df


def check_C88_active_facility_status(df, facility_status_column, end_date_column, snapshot_date_column):
    """Ensure that active facilities have an End Date that is null or in the future."""

    df[end_date_column] = pd.to_datetime(df[end_date_column], errors='coerce')
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    failed_count = ((df[facility_status_column] == "Active") & (df[end_date_column] < df[snapshot_date_column])).sum()

    result_df = pd.DataFrame([{
        "Check ID": 88,
        "Variable checked": f"{facility_status_column}, {end_date_column}, {snapshot_date_column}",
        "Description": "If the facility is 'Active,' confirm End Date is null or in the future.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_C89_valid_last_drawdown_date(df, last_drawdown_column, usage_column):
    """Ensure Date of Last Drawdown exists if the facility is new and has usage."""

    # Convert to datetime format
    df[last_drawdown_column] = pd.to_datetime(df[last_drawdown_column], errors='coerce')

    # Identify cases where a facility has usage but no last drawdown date
    failed_count = ((df[usage_column] > 0) & df[last_drawdown_column].isna()).sum()

    result_df = pd.DataFrame([{
        "Check ID": 89,
        "Variable checked": f"{last_drawdown_column}, {usage_column}",
        "Description": "If facility is brand new but has usage, Date of Last Drawdown must exist.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_C90_overdraft_repayment_schedule(df, facility_type_column, repayment_schedule_column):
    """Ensure Overdraft facilities do not have a repayment schedule (typically no amortization)."""

    # Identify cases where an overdraft has a repayment schedule
    failed_count = ((df[facility_type_column] == "Overdraft") & df[repayment_schedule_column].notna()).sum()

    result_df = pd.DataFrame([{
        "Check ID": 90,
        "Variable checked": f"{facility_type_column}, {repayment_schedule_column}",
        "Description": "If Facility Type = Overdraft, typically no amortization schedule. Flag mismatch if schedule is provided.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def section_c_checks(df):
    """Run all Section C (Facility Info) checks and return a compiled results table."""

    checks = [
        check_C61_missing_facility_id(df, 'Facility ID'),
        check_C62_facility_borrower_mapping(df, 'Facility ID', 'Borrower ID'),
        check_C63_missing_facility_start_date(df, 'Facility Start Date'),
        check_C64_facility_end_vs_start(df, 'Facility Start Date', 'Facility End Date'),
        check_C65_valid_facility_type(df, 'Facility Type', valid_types = ['Term Loan', 'Revolver', 'Overdraft', 'Credit Line']),
        check_C66_valid_facility_currency(df, 'Facility Currency'),
        check_C67_missing_original_facility_amount(df, 'Original Facility Amount'),
        check_C68_non_negative_original_facility_amount(df, 'Original Facility Amount'),
        check_C69_missing_credit_limit(df, 'Credit Limit (Month-End)'),
        check_C70_no_credit_limit_after_closure(df, 'Facility End Date', 'Snapshot Date', 'Credit Limit (Month-End)'),
        check_C71_exposure_vs_credit_limit(df, 'Outstanding Exposure (Month-End)', 'Credit Limit (Month-End)'),
        check_C72_missing_outstanding_exposure(df, 'Outstanding Exposure (Month-End)'),
        check_C73_utilization_rate(df, 'Outstanding Exposure (Month-End)', 'Credit Limit (Month-End)', 'Utilization Rate'),
        check_C74_interest_rate_range(df, 'Interest Rate / Pricing'),
        check_C75_valid_interest_rate_type(df, 'Interest Rate Type'),
        check_C76_valid_repayment_schedule(df, 'Repayment Schedule / Frequency'),
        check_C77_valid_amortization_type(df, 'Amortization Type'),
        check_C78_missing_facility_purpose(df, 'Facility Purpose / Loan Purpose'),
        check_C79_working_capital_maturity(df, 'Facility Purpose / Loan Purpose', 'Maturity Date', 'Snapshot Date'),
        check_C80_valid_last_drawdown_date(df, 'Date of Last Drawdown', 'Facility Start Date', 'Snapshot Date'),
        check_C81_valid_last_payment_date(df, 'Date of Last Payment', 'Facility Start Date', 'Snapshot Date'),
        check_C82_closed_facility_exposure(df, 'Facility End Date', 'Snapshot Date', 'Outstanding Exposure (Month-End)'),
        check_C83_unique_facility_per_borrower(df, 'Facility ID', 'Borrower ID'),
        check_C84_local_currency_vs_country(df, 'Facility Currency', 'Borrower Country'),
        check_C85_credit_limit_vs_original(df, 'Original Facility Amount', 'Credit Limit (Month-End)'),
        check_C86_credit_limit_fluctuations(df, 'Credit Limit (Month-End)'),
        check_C87_other_classification_distribution(df, 'Facility Purpose / Loan Purpose'),
        check_C88_active_facility_status(df, 'Facility Status', 'Facility End Date', 'Snapshot Date'),
        check_C89_valid_last_drawdown_date(df, 'Date of Last Drawdown', 'Outstanding Exposure (Month-End)'),
        check_C90_overdraft_repayment_schedule(df, 'Facility Type', 'Repayment Schedule / Frequency')
    ]

    return pd.concat(checks, ignore_index=True)