import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def check_I221_audit_opinion_validity(df, audit_opinion_column):
    """
    Ensure Audit Opinion is valid (must be one of {Unqualified, Qualified, Adverse, Disclaimer}).

    Parameters:
        df: DataFrame containing financial data
        audit_opinion_column: Column name for Audit Opinion

    Returns:
        DataFrame summarizing invalid audit opinions.
    """
    valid_opinions = {"Unqualified", "Qualified", "Adverse", "Disclaimer"}
    invalid_count = df[~df[audit_opinion_column].isin(valid_opinions)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 221,
        "Variable checked": audit_opinion_column,
        "Description": "Audit Opinion must be one of {Unqualified, Qualified, Adverse, Disclaimer}.",
        "Failed Count": invalid_count,
        "% of overall data": f"{(invalid_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_I222_audit_opinion_vs_ifrs9_stage(df, audit_opinion_column, ifrs9_stage_column):
    high_risk_opinions = {"Adverse", "Disclaimer"}

    df[ifrs9_stage_column] = pd.to_numeric(df[ifrs9_stage_column], errors='coerce')

    failed_count = df[(df[audit_opinion_column].isin(high_risk_opinions)) & (df[ifrs9_stage_column] < 2)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 222,
        "Variable checked": f"{audit_opinion_column}, {ifrs9_stage_column}",
        "Description": "If Audit Opinion is Adverse or Disclaimer, IFRS 9 Stage should be ≥ 2.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_I223_missing_management_quality_score(df, governance_score_column):
    missing_count = df[governance_score_column].isna().sum()

    result_df = pd.DataFrame([{
        "Check ID": 223,
        "Variable checked": governance_score_column,
        "Description": "If Management Quality / Governance Score is used, it must not be null.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_I224_governance_score_range(df, governance_score_column, min_value=1, max_value=10):
    """
    Ensure Management Quality / Governance Score is within the expected range (e.g., 1–10 or 1–5).

    Parameters:
        df: DataFrame containing financial data
        governance_score_column: Column name for Governance Score
        min_value: Minimum allowed score (default 1)
        max_value: Maximum allowed score (default 10)

    Returns:
        DataFrame summarizing out-of-range scores.
    """
    df[governance_score_column] = pd.to_numeric(df[governance_score_column], errors='coerce')

    failed_count = df[(df[governance_score_column] < min_value) | (df[governance_score_column] > max_value)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 224,
        "Variable checked": governance_score_column,
        "Description": f"Governance Score must be between {min_value} and {max_value}.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_I225_external_credit_rating(df, rating_column):
    valid_ratings = {
        "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
        "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D",
        "Aaa", "Aa1", "Aa2", "Aa3", "A1", "A2", "A3", "Baa1", "Baa2", "Baa3",
        "Ba1", "Ba2", "Ba3", "B1", "B2", "B3", "Caa1", "Caa2", "Caa3", "Ca", "C"
    }

    failed_count = df[~df[rating_column].isin(valid_ratings)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 225,
        "Variable checked": rating_column,
        "Description": "External Credit Rating must be a recognized rating (S&P, Moody’s, Fitch).",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_I226_external_vs_internal_credit_rating(df, external_rating_column, internal_rating_column):
    worst_external_ratings = {"D", "C", "CC", "CCC-", "Caa3", "Ca"}

    failed_count = df[(df[external_rating_column].isin(worst_external_ratings)) & (df[internal_rating_column] != "High Risk")].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 226,
        "Variable checked": f"{external_rating_column}, {internal_rating_column}",
        "Description": "If External Credit Rating is worst grade, Internal Credit Rating should reflect high risk.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_I227_fraud_indicators_validity(df, fraud_column):
    valid_values = {True, False, "Yes", "No"}

    failed_count = df[~df[fraud_column].isin(valid_values)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 227,
        "Variable checked": fraud_column,
        "Description": "Fraud Indicators / Flags must be boolean (True/False) or {Yes, No}.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_I228_fraud_vs_default_flag(df, fraud_column, default_column):
    failed_count = df[(df[fraud_column].isin(["Yes", True])) & (df[default_column] != "Yes")].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 228,
        "Variable checked": f"{fraud_column}, {default_column}",
        "Description": "If Fraud Indicator = Yes, Default Flag should likely be Yes (or forced closure).",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_I229_missing_esg_score(df, esg_score_column, threshold=5):
    missing_count = df[esg_score_column].isna().sum()
    missing_percentage = (missing_count / len(df)) * 100 if len(df) > 0 else 0

    result_df = pd.DataFrame([{
        "Check ID": 229,
        "Variable checked": esg_score_column,
        "Description": f"ESG Score should not be missing if the bank collects it (Flagged if missing > {threshold}%).",
        "Failed Count": missing_count,
        "% of overall data": f"{missing_percentage:.2f}%"
    }])

    return result_df

def check_I230_esg_score_range(df, esg_score_column, min_value=1, max_value=100, valid_categories={"A", "B", "C"}):
    df[esg_score_column] = pd.to_numeric(df[esg_score_column], errors='coerce')

    out_of_range_count = df[(~df[esg_score_column].between(min_value, max_value)) & (~df[esg_score_column].isin(valid_categories))].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 230,
        "Variable checked": esg_score_column,
        "Description": f"ESG Score should be between {min_value}-{max_value} or in {valid_categories}.",
        "Failed Count": out_of_range_count,
        "% of overall data": f"{(out_of_range_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def section_i_checks(df):
    """Run all Section I (Qualitative / ESG Validation) checks and return a compiled results table."""

    checks = [
        check_I221_audit_opinion_validity(df, 'Audit Opinion'),
        check_I222_audit_opinion_vs_ifrs9_stage(df, 'Audit Opinion', 'IFRS 9 Stage'),
        check_I223_missing_management_quality_score(df, 'Management Quality / Governance Score'),
        check_I224_governance_score_range(df, 'Management Quality / Governance Score', min_value=1, max_value=10),
        check_I225_external_credit_rating(df, 'External Credit Rating'),
        check_I226_external_vs_internal_credit_rating(df, 'External Credit Rating', 'Internal Credit Rating'),
        check_I227_fraud_indicators_validity(df, 'Fraud Indicators / Flags'),
        check_I228_fraud_vs_default_flag(df, 'Fraud Indicators / Flags', 'Default Flag'),
        check_I229_missing_esg_score(df, 'ESG Score', threshold=5),
        check_I230_esg_score_range(df, 'ESG Score', min_value=1, max_value=100, valid_categories={"A", "B", "C"})
    ]

    return pd.concat(checks, ignore_index=True)