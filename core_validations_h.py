import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def check_H191_missing_financial_statement_date(df, column_name):
    """Check for missing values and return results in table format."""
    failed_count = df[column_name].isna().sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 191,
        "Variable checked": column_name,
        "Description": "Missing check.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_H192_financial_statement_date_validity(df, statement_date_column, snapshot_date_column):
    """Ensure Financial Statement Date is not in the future (must be <= Snapshot Date)."""
    df[statement_date_column] = pd.to_datetime(df[statement_date_column], errors='coerce')
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    failed_count = (df[statement_date_column] > df[snapshot_date_column]).sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 192,
        "Variable checked": f"{statement_date_column}, {snapshot_date_column}",
        "Description": "Must be <= Snapshot Date. No future-dated financials allowed.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_H193_missing_total_assets(df, assets_column):
    """Ensure Total Assets are not missing (No missing > 5%)."""
    missing_count = df[assets_column].isna().sum()

    result_df = pd.DataFrame([{
        "Check ID": 193,
        "Variable checked": assets_column,
        "Description": "Missing check. No missing > 5%.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H194_missing_total_liabilities(df, liabilities_column):
    """Ensure Total Liabilities are not missing (No missing > 5%)."""
    missing_count = df[liabilities_column].isna().sum()

    result_df = pd.DataFrame([{
        "Check ID": 194,
        "Variable checked": liabilities_column,
        "Description": "Missing check. No missing > 5%.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H195_total_equity(df, equity_column, net_worth_column):
    """Ensure Total Equity is >= 0 if an entity has positive Net Worth."""
    failed_count = df[(df[net_worth_column] > 0) & (df[equity_column] < 0)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 195,
        "Variable checked": equity_column,
        "Description": "Total Equity must be >= 0 if entity has positive Net Worth.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H196_assets_liabilities_equity(df, assets_column, liabilities_column, equity_column, threshold=0.01):
    """
    Ensure Total Assets ≈ Total Liabilities + Equity within a given threshold.

    Parameters:
        df: DataFrame containing financial data
        assets_column: Column name for Total Assets
        liabilities_column: Column name for Total Liabilities
        equity_column: Column name for Total Equity
        threshold: Acceptable deviation percentage (default 1%)
    """
    df["Computed Assets"] = df[liabilities_column] + df[equity_column]
    df["Deviation"] = abs(df[assets_column] - df["Computed Assets"]) / df["Computed Assets"]

    failed_count = df[df["Deviation"] > threshold].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 196,
        "Variable checked": f"{assets_column}, {liabilities_column}, {equity_column}",
        "Description": "Cross-check: Assets ≈ Liabilities + Equity (± threshold).",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H197_missing_current_assets_liabilities(df, assets_column, liabilities_column):
    """Ensure Current Assets and Current Liabilities are not missing (No missing > 5%)."""
    missing_assets = df[assets_column].isna().sum()
    missing_liabilities = df[liabilities_column].isna().sum()

    result_df = pd.DataFrame([
        {
            "Check ID": 197,
            "Variable checked": assets_column,
            "Description": "Missing check. No missing > 5%.",
            "Failed Count": missing_assets,
            "% of overall data": f"{(missing_assets / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
        },
        {
            "Check ID": 197,
            "Variable checked": liabilities_column,
            "Description": "Missing check. No missing > 5%.",
            "Failed Count": missing_liabilities,
            "% of overall data": f"{(missing_liabilities / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
        }
    ])

    return result_df

def check_H198_non_negative_revenue(df, revenue_column):
    """Ensure Revenue/Sales is ≥ 0. Flag negative values."""
    failed_count = df[df[revenue_column] < 0].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 198,
        "Variable checked": revenue_column,
        "Description": "Revenue/Sales must be ≥ 0.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H199_ebitda_vs_net_income(df, ebitda_column, net_income_column):
    """Ensure EBITDA is not less than Net Income if provided."""
    failed_count = df[(df[ebitda_column].notna()) & (df[ebitda_column] < df[net_income_column])].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 199,
        "Variable checked": f"{ebitda_column}, {net_income_column}",
        "Description": "EBITDA must not be < Net Income if provided.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H200_net_income_vs_assets(df, net_income_column, assets_column, threshold=3):
    """
    Ensure absolute Net Income is not greater than X times Total Assets.

    Parameters:
        df: DataFrame containing financial data
        net_income_column: Column name for Net Income
        assets_column: Column name for Total Assets
        threshold: Maximum allowable multiple of Total Assets for absolute Net Income (default 3x)
    """
    failed_count = df[abs(df[net_income_column]) > (threshold * df[assets_column])].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 200,
        "Variable checked": f"{net_income_column}, {assets_column}",
        "Description": f"Absolute Net Income should not exceed {threshold} times Total Assets.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H201_non_negative_interest_expense(df, interest_expense_column):
    """Ensure Interest Expense is typically ≥ 0. Flag negative values."""
    failed_count = df[df[interest_expense_column] < 0].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 201,
        "Variable checked": interest_expense_column,
        "Description": "Interest Expense should be ≥ 0.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H202_non_negative_depreciation(df, depreciation_column):
    """Ensure Depreciation & Amortization is typically ≥ 0. Flag negative values."""
    failed_count = df[df[depreciation_column] < 0].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 202,
        "Variable checked": depreciation_column,
        "Description": "Depreciation & Amortization should be ≥ 0.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H203_off_balance_sheet_exposure(df, exposure_column):
    """
    Ensure Off-Balance Sheet Exposure is valid:
    - If numeric, it must be ≥ 0.
    - If text, ensure it references a valid exposure type.
    """
    # Identify numeric and text values
    numeric_issues = df[pd.to_numeric(df[exposure_column], errors='coerce') < 0].shape[0]
    text_issues = df[df[exposure_column].apply(lambda x: isinstance(x, str) and not x.strip())].shape[0]

    failed_count = numeric_issues + text_issues

    result_df = pd.DataFrame([{
        "Check ID": 203,
        "Variable checked": exposure_column,
        "Description": "Off-Balance Sheet Exposure must be ≥ 0 if numeric, valid text otherwise.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H204_financial_statement_date(df, date_column):
    """
    Ensure Financial Statement Date follows annual or quarterly intervals only.
    Flags any random dates that do not align with typical financial reporting periods.
    """
    # Convert to datetime format
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Define valid quarter-end and year-end months
    valid_months = [3, 6, 9, 12]  # Quarterly or annual reporting periods
    invalid_dates = df[~df[date_column].dt.month.isin(valid_months)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 204,
        "Variable checked": date_column,
        "Description": "Ensure annual or quarterly intervals only.",
        "Failed Count": invalid_dates,
        "% of overall data": f"{(invalid_dates / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H205_assets_growth(df, date_column, assets_column, threshold=50):
    """
    Ensure Total Assets growth from prior period is within ±X%.

    Parameters:
        df: DataFrame containing financial data
        date_column: Column name for Financial Statement Date
        assets_column: Column name for Total Assets
        threshold: Maximum allowable year-over-year growth percentage (default 50%)
    """
    df = df.sort_values(by=[date_column])  # Ensure chronological order
    df["Prior Period Assets"] = df[assets_column].shift(1)

    df["YoY Growth %"] = ((df[assets_column] - df["Prior Period Assets"]) / df["Prior Period Assets"]) * 100
    failed_count = df[(df["YoY Growth %"].abs() > threshold)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 205,
        "Variable checked": assets_column,
        "Description": f"Total Assets growth should be within ±{threshold}%.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H206_revenue_growth(df, date_column, revenue_column, upper_threshold=200, lower_threshold=-50):
    """
    Ensure Revenue/Sales growth from prior period is within ±X%.

    Parameters:
        df: DataFrame containing financial data
        date_column: Column name for Financial Statement Date
        revenue_column: Column name for Revenue/Sales
        upper_threshold: Maximum allowable YoY growth percentage (default 200%)
        lower_threshold: Minimum allowable YoY decline percentage (default -50%)
    """
    df = df.sort_values(by=[date_column])  # Ensure chronological order
    df["Prior Period Revenue"] = df[revenue_column].shift(1)

    df["YoY Growth %"] = ((df[revenue_column] - df["Prior Period Revenue"]) / df["Prior Period Revenue"]) * 100
    failed_count = df[(df["YoY Growth %"] > upper_threshold) | (df["YoY Growth %"] < lower_threshold)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 206,
        "Variable checked": revenue_column,
        "Description": f"Revenue growth should be within -50% to 200%.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H207_net_income_flips(df, date_column, net_income_column, threshold=200):
    """
    Check if Net Income flips from a large negative to a large positive (or vice versa).

    Parameters:
        df: DataFrame containing financial data
        date_column: Column name for Financial Statement Date
        net_income_column: Column name for Net Income
        threshold: Significant change percentage to flag (default 200%)
    """
    df = df.sort_values(by=[date_column])  # Ensure chronological order
    df["Prior Period Net Income"] = df[net_income_column].shift(1)

    df["YoY Growth %"] = ((df[net_income_column] - df["Prior Period Net Income"]) / abs(df["Prior Period Net Income"])) * 100
    failed_count = df[(df["Prior Period Net Income"] < 0) & (df[net_income_column] > 0) & (df["YoY Growth %"] > threshold) |
                      (df["Prior Period Net Income"] > 0) & (df[net_income_column] < 0) & (df["YoY Growth %"] < -threshold)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 207,
        "Variable checked": net_income_column,
        "Description": "Check if Net Income flips from large negative to large positive (or vice versa).",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H208_current_ratio(df, assets_column, liabilities_column, threshold=1.0):
    """
    Ensure Current Assets / Current Liabilities ratio is greater than X.

    Parameters:
        df: DataFrame containing financial data
        assets_column: Column name for Current Assets
        liabilities_column: Column name for Current Liabilities
        threshold: Minimum acceptable current ratio (default 1.0)
    """
    df["Current Ratio"] = df[assets_column] / df[liabilities_column]

    failed_count = df[df["Current Ratio"] < threshold].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 208,
        "Variable checked": f"{assets_column}, {liabilities_column}",
        "Description": f"Current Ratio (Assets/Liabilities) should be > {threshold}.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H209_interest_coverage(df, ebitda_column, interest_expense_column, stage_column):
    """
    Ensure EBITDA / Interest Expense is ≥ 1.

    Parameters:
        df: DataFrame containing financial data
        ebitda_column: Column name for EBITDA
        interest_expense_column: Column name for Interest Expense
        stage_column: Column indicating financial stage (e.g., Stage 1)
    """
    df["Coverage Ratio"] = df[ebitda_column] / df[interest_expense_column]

    # Flag cases where Coverage Ratio < 1 AND Stage = 1
    failed_count = df[(df["Coverage Ratio"] < 1) & (df[stage_column] == 1)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 209,
        "Variable checked": f"{ebitda_column}, {interest_expense_column}, {stage_column}",
        "Description": "If coverage ratio (EBITDA / Interest Expense) < 1, potential risk.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H210_depreciation_vs_fixed_assets(df, depreciation_column, fixed_assets_column, lower_threshold=1, upper_threshold=20):
    """
    Ensure Depreciation & Amortization (D&A) relative to Fixed Assets is within reasonable range.

    Parameters:
        df: DataFrame containing financial data
        depreciation_column: Column name for Depreciation & Amortization
        fixed_assets_column: Column name for Fixed Assets
        lower_threshold: Minimum acceptable D&A to Fixed Assets ratio (default 1%)
        upper_threshold: Maximum acceptable D&A to Fixed Assets ratio (default 20%)
    """
    df["D&A to Fixed Assets Ratio"] = (df[depreciation_column] / df[fixed_assets_column]) * 100

    failed_count = df[(df["D&A to Fixed Assets Ratio"] < lower_threshold) |
                      (df["D&A to Fixed Assets Ratio"] > upper_threshold)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 210,
        "Variable checked": f"{depreciation_column}, {fixed_assets_column}",
        "Description": f"D&A to Fixed Assets Ratio should be between {lower_threshold}% and {upper_threshold}%.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H211_off_balance_exposure(df, exposure_column, threshold=5000000):
    """
    Ensure Off-Balance Sheet Exposure is reviewed when very high.

    Parameters:
        df: DataFrame containing financial data
        exposure_column: Column name for Off-Balance Sheet Exposure
        threshold: Value above which exposure should be flagged (default 5,000,000)
    """
    failed_count = df[df[exposure_column] > threshold].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 211,
        "Variable checked": exposure_column,
        "Description": f"Off-Balance Sheet Exposure above {threshold} should be reviewed for default risk.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H212_revenue_vs_borrower_segment(df, revenue_column, segment_column, threshold=10000000):
    """
    Ensure Revenue/Sales aligns with Borrower Segment expectations.

    Parameters:
        df: DataFrame containing financial data
        revenue_column: Column name for Revenue/Sales
        segment_column: Column name for Borrower Segment
        threshold: Minimum expected revenue for Large Corporate (default 10,000,000)
    """
    # Flag cases where segment is "Large Corporate" but revenue is below threshold
    failed_count = df[(df[segment_column] == "Large Corporate") & (df[revenue_column] < threshold)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 212,
        "Variable checked": f"{revenue_column}, {segment_column}",
        "Description": f"Large Corporate borrowers should have Revenue > {threshold}.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H213_current_vs_total_liabilities(df, current_liabilities_column, total_liabilities_column, lower_threshold=1, upper_threshold=90):
    """
    Ensure Current Liabilities are within a reasonable proportion of Total Liabilities.

    Parameters:
        df: DataFrame containing financial data
        current_liabilities_column: Column name for Current Liabilities
        total_liabilities_column: Column name for Total Liabilities
        lower_threshold: Minimum acceptable % of Total Liabilities as Current (default 1%)
        upper_threshold: Maximum acceptable % of Total Liabilities as Current (default 90%)
    """
    df["Current Liabilities %"] = (df[current_liabilities_column] / df[total_liabilities_column]) * 100

    failed_count = df[(df["Current Liabilities %"] < lower_threshold) |
                      (df["Current Liabilities %"] > upper_threshold)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 213,
        "Variable checked": f"{current_liabilities_column}, {total_liabilities_column}",
        "Description": f"Current Liabilities should be between {lower_threshold}% and {upper_threshold}% of Total Liabilities.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H214_financial_statement_staleness(df, date_column, months_threshold=18):
    """
    Ensure Financial Statement Date is not older than a defined threshold (default 18 months).

    Parameters:
        df: DataFrame containing financial data
        date_column: Column name for Financial Statement Date
        months_threshold: Maximum allowed months before considering the statement stale (default 18 months)
    """
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    cutoff_date = datetime.today() - timedelta(days=months_threshold * 30)  # Approximate 18 months

    failed_count = df[df[date_column] < cutoff_date].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 214,
        "Variable checked": date_column,
        "Description": f"Financial statements older than {months_threshold} months are flagged as stale.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H215_dividends_vs_net_income(df, net_income_column, dividends_column):
    """
    Ensure Dividends do not exceed Net Income (if known).

    Parameters:
        df: DataFrame containing financial data
        net_income_column: Column name for Net Income
        dividends_column: Column name for Dividends
    """
    failed_count = df[df[dividends_column] > df[net_income_column]].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 215,
        "Variable checked": f"{net_income_column}, {dividends_column}",
        "Description": "Dividends should not be greater than Net Income.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H216_depreciation_vs_revenue(df, revenue_column, depreciation_column):
    """
    Ensure Depreciation & Amortization (D&A) is not greater than Revenue/Sales.

    Parameters:
        df: DataFrame containing financial data
        revenue_column: Column name for Revenue/Sales
        depreciation_column: Column name for Depreciation & Amortization
    """
    failed_count = df[df[depreciation_column] > df[revenue_column]].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 216,
        "Variable checked": f"{revenue_column}, {depreciation_column}",
        "Description": "If D&A > Revenue, flagged as an outlier unless justified.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H217_equity_vs_ifrs9_stage(df, equity_column, ifrs9_stage_column, stage_value=3, equity_threshold=1000000):
    """
    Ensure Total Equity aligns with IFRS 9 Stage classification.

    Parameters:
        df: DataFrame containing financial data
        equity_column: Column name for Total Equity
        ifrs9_stage_column: Column name for IFRS 9 Stage
        stage_value: The stage to check for inconsistencies (default: Stage 3)
        equity_threshold: Minimum equity value considered "huge positive" (default: 1,000,000)
    """
    failed_count = df[(df[ifrs9_stage_column] == stage_value) & (df[equity_column] > equity_threshold)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 217,
        "Variable checked": f"{equity_column}, {ifrs9_stage_column}",
        "Description": f"If IFRS 9 Stage = {stage_value} but Equity > {equity_threshold}, investigate potential mismatch.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H218_interest_expense_vs_borrower_segment(df, interest_expense_column, segment_column, threshold=1000000):
    """
    Ensure Interest Expense aligns with Borrower Segment expectations.

    Parameters:
        df: DataFrame containing financial data
        interest_expense_column: Column name for Interest Expense
        segment_column: Column name for Borrower Segment
        threshold: Maximum expected Interest Expense for SMEs (default 1,000,000)
    """
    # Flag cases where segment is "SME" but Interest Expense is extremely high
    failed_count = df[(df[segment_column] == "SME") & (df[interest_expense_column] > threshold)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 218,
        "Variable checked": f"{interest_expense_column}, {segment_column}",
        "Description": f"If SME, Interest Expense > {threshold} may indicate an outlier.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H219_inventory_turnover_vs_assets(df, inventory_column, turnover_column, threshold=1000000):
    """
    Ensure Inventory Turnover aligns with Inventory levels.

    Parameters:
        df: DataFrame containing financial data
        inventory_column: Column name for Inventory (part of Current Assets)
        turnover_column: Column name for Inventory Turnover
        threshold: Minimum inventory level to flag when turnover is zero (default 1,000,000)
    """
    # Flag cases where Inventory Turnover is 0 but Inventory is significantly high
    failed_count = df[(df[turnover_column] == 0) & (df[inventory_column] > threshold)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 219,
        "Variable checked": f"{inventory_column}, {turnover_column}",
        "Description": f"If Inventory Turnover = 0 but Inventory > {threshold}, potential mismatch.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_H220_net_income_vs_ifrs9_stage(df, net_income_column, ifrs9_stage_column, stage_2_threshold=2):
    """
    Ensure Net Income < 0 does not automatically mean Stage 3 (IFRS 9), but multiple losses may indicate Stage 2.

    Parameters:
        df: DataFrame containing financial data
        net_income_column: Column name for Net Income
        ifrs9_stage_column: Column name for IFRS 9 Stage
        stage_2_threshold: Number of consecutive periods of negative income before Stage 2 is expected (default 2)
    """
    df = df.sort_values(by="Financial Statement Date")  # Ensure chronological order
    df["Negative Income Streak"] = df[net_income_column] < 0
    df["Negative Income Streak"] = df["Negative Income Streak"].groupby((df["Negative Income Streak"] != df["Negative Income Streak"].shift()).cumsum()).cumsum()

    # Flag cases where Net Income < 0 multiple times but not Stage 2
    failed_count = df[(df["Negative Income Streak"] >= stage_2_threshold) & (df[ifrs9_stage_column] != 2)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 220,
        "Variable checked": f"{net_income_column}, {ifrs9_stage_column}",
        "Description": f"If multiple consecutive losses (≥ {stage_2_threshold} periods) occur, Stage 2 should be considered.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def section_h_checks(df):
    """Run all Section H (Financial Statement Validation) checks and return a compiled results table."""

    checks = [
        check_H191_missing_financial_statement_date(df, 'Financial Statement Date'),
        check_H192_financial_statement_date_validity(df, 'Financial Statement Date', 'Snapshot Date'),
        check_H193_missing_total_assets(df, 'Total Assets'),
        check_H194_missing_total_liabilities(df, 'Total Liabilities'),
        check_H195_total_equity(df, 'Total Equity', 'Net Worth'),
        check_H196_assets_liabilities_equity(df, 'Total Assets', 'Total Liabilities', 'Total Equity', threshold=0.01),
        check_H197_missing_current_assets_liabilities(df, 'Current Assets', 'Current Liabilities'),
        check_H198_non_negative_revenue(df, 'Revenue'),
        check_H199_ebitda_vs_net_income(df, 'EBITDA', 'Net Income'),
        check_H200_net_income_vs_assets(df, 'Net Income', 'Total Assets', threshold=3),
        check_H201_non_negative_interest_expense(df, 'Interest Expense'),
        check_H202_non_negative_depreciation(df, 'Depreciation & Amortization'),
        check_H203_off_balance_sheet_exposure(df, 'Off-Balance Sheet Exposure'),
        check_H204_financial_statement_date(df, 'Financial Statement Date'),
        check_H205_assets_growth(df, 'Financial Statement Date', 'Total Assets', threshold=50),
        check_H206_revenue_growth(df, 'Financial Statement Date', 'Revenue', upper_threshold=200, lower_threshold=-50),
        check_H207_net_income_flips(df, 'Financial Statement Date', 'Net Income', threshold=200),
        check_H208_current_ratio(df, 'Current Assets', 'Current Liabilities', threshold=1.0),
        check_H209_interest_coverage(df, 'EBITDA', 'Interest Expense', 'IFRS 9 Stage'),
        check_H210_depreciation_vs_fixed_assets(df, 'Depreciation & Amortization', 'Fixed Assets', lower_threshold=1, upper_threshold=20),
        check_H211_off_balance_exposure(df, 'Off-Balance Sheet Exposure', threshold=5000000),
        check_H212_revenue_vs_borrower_segment(df, 'Revenue', 'Borrower Segment', threshold=10000000),
        check_H213_current_vs_total_liabilities(df, 'Current Liabilities', 'Total Liabilities', lower_threshold=1, upper_threshold=90),
        check_H214_financial_statement_staleness(df, 'Financial Statement Date', months_threshold=18),
        check_H215_dividends_vs_net_income(df, 'Net Income', 'Dividends'),
        check_H216_depreciation_vs_revenue(df, 'Revenue', 'Depreciation & Amortization'),
        check_H217_equity_vs_ifrs9_stage(df, 'Total Equity', 'IFRS 9 Stage', stage_value=3, equity_threshold=1000000),
        check_H218_interest_expense_vs_borrower_segment(df, 'Interest Expense', 'Borrower Segment', threshold=1000000),
        check_H219_inventory_turnover_vs_assets(df, 'Inventory', 'Inventory Turnover', threshold=1000000),
        check_H220_net_income_vs_ifrs9_stage(df, 'Net Income', 'IFRS 9 Stage', stage_2_threshold=2)
    ]

    return pd.concat(checks, ignore_index=True)