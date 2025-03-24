import re
import numpy as np
import pandas as pd
from datetime import datetime


def check_G181_missing_covenant_details(df, covenant_column):
    """Ensure Covenant Details are not missing if covenants exist."""

    missing_count = df[covenant_column].isna().sum()

    result_df = pd.DataFrame([{
        "Check ID": 181,
        "Variable checked": covenant_column,
        "Description": "Missing check. Must not be null if covenants exist.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_G182_financial_ratio_fields(df, covenant_column, ratio_columns):
    """Ensure financial ratio fields are not missing if the covenant requires them."""

    # Check if ratio columns exist in the DataFrame
    ratio_columns = [col for col in ratio_columns if col in df.columns]
    #If any of the ratio_columns are not present, use empty list for subset to avoid KeyError
    if not ratio_columns:
        failed_count = 0
    else:
        failed_count = df[df[covenant_column].str.contains('ratio', case=False, na=False)][ratio_columns].isna().sum().sum()

    result_df = pd.DataFrame([{
        "Check ID": 182,
        "Variable checked": f"{covenant_column}, {', '.join(ratio_columns)}",
        "Description": "If covenant is a financial ratio threshold, verify ratio fields are not missing.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_G183_covenant_breach_vs_default(df, covenant_column, default_flag_column):
    """Ensure policy compliance: if covenant is breached, check if default was triggered."""

    failed_count = ((df[covenant_column].str.contains('breach', case=False, na=False)) & (df[default_flag_column] == 'No')).sum()

    result_df = pd.DataFrame([{
        "Check ID": 183,
        "Variable checked": f"{covenant_column}, {default_flag_column}",
        "Description": "If covenant is breached, check if default triggered depends on policy. Flag mismatch.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_G184_covenant_breach_date(df, breach_date_column, snapshot_date_column):
    """Ensure covenant breach date is not in the future (must be ≤ Snapshot Date)."""

    df[breach_date_column] = pd.to_datetime(df[breach_date_column], errors='coerce')
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    failed_count = (df[breach_date_column] > df[snapshot_date_column]).sum()

    result_df = pd.DataFrame([{
        "Check ID": 184,
        "Variable checked": f"{breach_date_column}, {snapshot_date_column}",
        "Description": "If there's a breach date, it must be ≤ Snapshot Date. No future breach dates allowed.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_G185_breach_vs_relationship_start(df, breach_date_column, relationship_start_column, facility_start_column):
    """Ensure no covenant breach occurs before the relationship or facility start date."""

    df[breach_date_column] = pd.to_datetime(df[breach_date_column], errors='coerce')
    df[relationship_start_column] = pd.to_datetime(df[relationship_start_column], errors='coerce')
    df[facility_start_column] = pd.to_datetime(df[facility_start_column], errors='coerce')

    failed_count = ((df[breach_date_column] < df[relationship_start_column]) |
                    (df[breach_date_column] < df[facility_start_column])).sum()

    result_df = pd.DataFrame([{
        "Check ID": 185,
        "Variable checked": f"{breach_date_column}, {relationship_start_column}, {facility_start_column}",
        "Description": "No covenant breach date before the relationship or facility start date.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%"
    }])

    return result_df

def check_G186_covenant_distribution(df, covenant_column):
    """Ensure that not all loans have exactly the same covenant details."""

    unique_values = df[covenant_column].nunique()
    total_values = len(df)

    failed_count = total_values if unique_values == 1 else 0

    result_df = pd.DataFrame([{
        "Check ID": 186,
        "Variable checked": covenant_column,
        "Description": "Distribution check: ensure not all loans have exactly the same covenant. Some variation expected.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%"
    }])

    return result_df

def check_G187_flag_na_in_covenant_details(df, covenant_column):
    """If text includes 'NA,' confirm if it truly means no covenants or is just incomplete."""

    failed_count = df[covenant_column].str.contains('NA', case=False, na=False).sum()

    result_df = pd.DataFrame([{
        "Check ID": 187,
        "Variable checked": covenant_column,
        "Description": "If text includes 'NA,' confirm it truly has no covenants or is just incomplete.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%"
    }])

    return result_df

def check_G188_min_turnover_covenant(df, turnover_covenant_column, actual_turnover_column):
    """Ensure actual turnover is above the required covenant threshold."""
    # Convert columns to numeric, handling errors by coercing non-numeric values to NaN
    df[actual_turnover_column] = pd.to_numeric(df[actual_turnover_column], errors='coerce')
    df[turnover_covenant_column] = pd.to_numeric(df[turnover_covenant_column], errors='coerce')

    failed_count = (df[actual_turnover_column] < df[turnover_covenant_column]).sum()

    result_df = pd.DataFrame([{
        "Check ID": 188,
        "Variable checked": f"{turnover_covenant_column}, {actual_turnover_column}",
        "Description": "If required covenant = min turnover, cross-check actual turnover is above. Violation flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%"
    }])

    return result_df

def check_G189_new_covenants_effective_date(df, covenant_column, effective_date_column):
    """Ensure new covenants added mid-relationship have an effective date recorded."""

    df[effective_date_column] = pd.to_datetime(df[effective_date_column], errors='coerce')
    failed_count = (df[covenant_column].notna() & df[effective_date_column].isna()).sum()

    result_df = pd.DataFrame([{
        "Check ID": 189,
        "Variable checked": f"{covenant_column}, {effective_date_column}",
        "Description": "If new covenants are added mid-relationship, ensure effective date is recorded.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%"
    }])

    return result_df

def check_G190_stage_shift_on_breach(df, covenant_column, stage_column):
    """Ensure that if a major covenant is breached, IFRS 9 stage might shift to Stage 2."""

    failed_count = ((df[covenant_column].str.contains('breach', case=False, na=False)) & (df[stage_column] == 1)).sum()

    result_df = pd.DataFrame([{
        "Check ID": 190,
        "Variable checked": f"{covenant_column}, {stage_column}",
        "Description": "If major covenant is breached, IFRS 9 Stage might shift to 2. Cross-check logic.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%"
    }])

    return result_df

def section_g_checks(df):
    """Run all Section G (Operational / Covenants) checks and return a compiled results table."""

    checks = [
        check_G181_missing_covenant_details(df, 'Covenant Details'),
        check_G182_financial_ratio_fields(df, 'Covenant Details', ['DSCR', 'Leverage Ratio']),  # Update ratio fields if necessary
        check_G183_covenant_breach_vs_default(df, 'Covenant Details', 'Default Flag'),
        check_G184_covenant_breach_date(df, 'Breach Date', 'Snapshot Date'),
        check_G185_breach_vs_relationship_start(df, 'Breach Date', 'Relationship Start Date', 'Facility Start Date'),
        check_G186_covenant_distribution(df, 'Covenant Details'),
        check_G187_flag_na_in_covenant_details(df, 'Covenant Details'),
        check_G188_min_turnover_covenant(df, 'Min Turnover Covenant', 'Actual Turnover'),
        check_G189_new_covenants_effective_date(df, 'Covenant Details', 'Covenant Effective Date'),
        check_G190_stage_shift_on_breach(df, 'Covenant Details', 'IFRS 9 Stage')
    ]

    return pd.concat(checks, ignore_index=True)