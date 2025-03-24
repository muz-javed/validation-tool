import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def check_J231_missing_roa(df, roa_column):
    missing_count = df[roa_column].isna().sum()

    result_df = pd.DataFrame([{
        "Check ID": 231,
        "Variable checked": roa_column,
        "Description": "Return on Assets (ROA) must not be null if computed.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_J232_roa_range(df, roa_column, min_value=-100, max_value=100):
    df[roa_column] = pd.to_numeric(df[roa_column], errors='coerce')

    out_of_range_count = df[(df[roa_column] < min_value) | (df[roa_column] > max_value)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 232,
        "Variable checked": roa_column,
        "Description": f"Return on Assets (ROA) should be between {min_value}% and {max_value}%. Outliers flagged.",
        "Failed Count": out_of_range_count,
        "% of overall data": f"{(out_of_range_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_J233_roe_range(df, roe_column, min_value=-500, max_value=500):
    df[roe_column] = pd.to_numeric(df[roe_column], errors='coerce')

    out_of_range_count = df[(df[roe_column] < min_value) | (df[roe_column] > max_value)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 233,
        "Variable checked": roe_column,
        "Description": f"Return on Equity (ROE) should be between {min_value}% and {max_value}%. Outliers flagged.",
        "Failed Count": out_of_range_count,
        "% of overall data": f"{(out_of_range_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_J234_net_profit_margin_range(df, npm_column, min_value=-100, max_value=100):
    df[npm_column] = pd.to_numeric(df[npm_column], errors='coerce')

    out_of_range_count = df[(df[npm_column] < min_value) | (df[npm_column] > max_value)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 234,
        "Variable checked": npm_column,
        "Description": f"Net Profit Margin should be between {min_value}% and {max_value}%. Outliers flagged.",
        "Failed Count": out_of_range_count,
        "% of overall data": f"{(out_of_range_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_J235_operating_margin_range(df, operating_margin_column, max_value=100):
    df[operating_margin_column] = pd.to_numeric(df[operating_margin_column], errors='coerce')

    out_of_range_count = df[df[operating_margin_column] > max_value].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 235,
        "Variable checked": operating_margin_column,
        "Description": f"Operating Margin should usually be <{max_value}%. If >{max_value}%, investigate.",
        "Failed Count": out_of_range_count,
        "% of overall data": f"{(out_of_range_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_J236_ebitda_vs_operating_margin(df, ebitda_margin_column, operating_margin_column, depreciation_column):
    df[ebitda_margin_column] = pd.to_numeric(df[ebitda_margin_column], errors='coerce')
    df[operating_margin_column] = pd.to_numeric(df[operating_margin_column], errors='coerce')
    df[depreciation_column] = pd.to_numeric(df[depreciation_column], errors='coerce')

    failed_count = df[(df[depreciation_column] > 0) & (df[ebitda_margin_column] < df[operating_margin_column])].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 236,
        "Variable checked": f"{ebitda_margin_column}, {operating_margin_column}, {depreciation_column}",
        "Description": "EBITDA Margin must be ≥ Operating Margin if D&A > 0. Flag cases where EBITDA Margin < Operating Margin.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_J237_negative_gross_profit_margin(df, gross_profit_margin_column):
    df[gross_profit_margin_column] = pd.to_numeric(df[gross_profit_margin_column], errors='coerce')

    failed_count = df[df[gross_profit_margin_column] < 0].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 237,
        "Variable checked": gross_profit_margin_column,
        "Description": "If Gross Profit Margin is negative, confirm negative revenue or huge COGS. Possibly outlier.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_J238_net_profit_vs_operating_margin(df, net_profit_margin_column, operating_margin_column):
    df[net_profit_margin_column] = pd.to_numeric(df[net_profit_margin_column], errors='coerce')
    df[operating_margin_column] = pd.to_numeric(df[operating_margin_column], errors='coerce')

    failed_count = df[df[net_profit_margin_column] > df[operating_margin_column]].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 238,
        "Variable checked": f"{net_profit_margin_column}, {operating_margin_column}",
        "Description": "Net Profit Margin should be ≤ Operating Margin. Flagged if reversed.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_J239_yoy_profitability_changes(df, date_column, ratio_column, threshold=200):
    df = df.sort_values(by=[date_column])

    df[ratio_column] = pd.to_numeric(df[ratio_column], errors='coerce')  # Convert to numeric
    df["Prior Period Ratio"] = df[ratio_column].shift(1)  # Shift to get prior period values

    df["YoY Change %"] = ((df[ratio_column] - df["Prior Period Ratio"]) / abs(df["Prior Period Ratio"])) * 100

    failed_count = df[abs(df["YoY Change %"]) > threshold].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 239,
        "Variable checked": ratio_column,
        "Description": f"YoY changes in profitability ratios should not exceed ±{threshold}%. Large swings flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_J240_net_profit_margin_vs_ifrs9_stage(df, net_profit_margin_column, ifrs9_stage_column, positive_threshold=20):
    df[net_profit_margin_column] = pd.to_numeric(df[net_profit_margin_column], errors='coerce')
    df[ifrs9_stage_column] = pd.to_numeric(df[ifrs9_stage_column], errors='coerce')

    failed_count = df[(df[net_profit_margin_column] > positive_threshold) & (df[ifrs9_stage_column] == 3)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 240,
        "Variable checked": f"{net_profit_margin_column}, {ifrs9_stage_column}",
        "Description": f"If Net Profit Margin is highly positive (> {positive_threshold}%), it should not be Stage 3.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def section_j_checks(df):
    """Run all Section J1 (Profitability Ratios) checks and return a compiled results table."""

    checks = [
        check_J231_missing_roa(df, 'Return on Assets (ROA)'),
        check_J232_roa_range(df, 'Return on Assets (ROA)', min_value=-100, max_value=100),
        check_J233_roe_range(df, 'Return on Equity (ROE)', min_value=-500, max_value=500),
        check_J234_net_profit_margin_range(df, 'Net Profit Margin', min_value=-100, max_value=100),
        check_J235_operating_margin_range(df, 'Operating Margin', max_value=100),
        check_J236_ebitda_vs_operating_margin(df, 'EBITDA Margin', 'Operating Margin', 'Depreciation & Amortization'),
        check_J237_negative_gross_profit_margin(df, 'Gross Profit Margin'),
        check_J238_net_profit_vs_operating_margin(df, 'Net Profit Margin', 'Operating Margin'),
        check_J239_yoy_profitability_changes(df, 'Financial Statement Date', 'Profitability Ratio', threshold=200),
        check_J240_net_profit_margin_vs_ifrs9_stage(df, 'Net Profit Margin', 'IFRS 9 Stage', positive_threshold=20)
    ]

    return pd.concat(checks, ignore_index=True)