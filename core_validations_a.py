import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def check_1_SnapshotDateMV(df, column_name):
    """Check for missing values and return results in table format."""
    failed_count = df[column_name].isna().sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 1,
        "Variable checked": column_name,
        "Description": "Missing check.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df


def check_2_SnapshotDate_format(df, column_name):
    """Validate date format (YYYY-MM-DD) and return results in table format."""
    failed_count = sum(pd.to_datetime(df[column_name], errors='coerce').isna())
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 2,
        "Variable checked": column_name,
        "Description": "Validate date format (YYYY-MM-DD).",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_3_SnapshotDate_future(df, column_name):
    """Ensure date is not in the future and return results in table format."""
    today = datetime.today().date()
    failed_count = sum(pd.to_datetime(df[column_name], errors='coerce').dt.date > today)
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 3,
        "Variable checked": column_name,
        "Description": "Ensure date is not in the future.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_4_SnapshotDate_duplicates(df, column_name):
    """Check for duplicate Snapshot Dates and return results in table format."""
    duplicate_count = df.duplicated(subset=[column_name]).sum()
    total_count = len(df)
    failed_percent = (duplicate_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 4,
        "Variable checked": column_name,
        "Description": "Check for duplicates if each file should have a unique Snapshot Date (if applicable).",
        "Failed Count": duplicate_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_5_SnapshotDate_sequence(df, column_name):
    """Check if Snapshot Dates are in chronological order and return results in table format."""
    sorted_dates = pd.to_datetime(df[column_name], errors='coerce').dropna()
    failed_count = sum(sorted_dates.diff().dropna() < pd.Timedelta(0))
    total_count = len(sorted_dates)
    failed_percent = (failed_count / total_count) * 100 if total_count > 0 else 0

    result_df = pd.DataFrame([{
        "Check ID": 5,
        "Variable checked": column_name,
        "Description": "Check for chronological sequence between multiple Snapshot Dates.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_6_row_count(df, expected_min_rows):
    """Check overall row count vs. expected (sanity check) and return results in table format."""
    total_rows = len(df)
    failed_count = 1 if total_rows < expected_min_rows else 0
    failed_percent = (failed_count / total_rows) * 100 if total_rows > 0 else 0

    result_df = pd.DataFrame([{
        "Check ID": 6,
        "Variable checked": "All fields",
        "Description": "Check overall row count vs. expected (sanity check).",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_7_empty_rows(df):
    """Ensure no row is completely empty and return results in table format."""
    failed_count = (df.isna().all(axis=1)).sum()
    total_rows = len(df)
    failed_percent = (failed_count / total_rows) * 100 if total_rows > 0 else 0

    result_df = pd.DataFrame([{
        "Check ID": 7,
        "Variable checked": "All fields",
        "Description": "Ensure no row is completely empty.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_8_missing_ratio(df, max_missing_ratio=0.10):
    """Check ratio of missing fields per row and return results in table format."""
    total_columns = df.shape[1]
    max_missing_allowed = total_columns * max_missing_ratio
    failed_count = (df.isna().sum(axis=1) > max_missing_allowed).sum()
    total_rows = len(df)
    failed_percent = (failed_count / total_rows) * 100 if total_rows > 0 else 0

    result_df = pd.DataFrame([{
        "Check ID": 8,
        "Variable checked": "All fields",
        "Description": f"Check ratio of missing fields per row (Max {max_missing_ratio*100:.0f}%).",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_9_column_count(df, expected_columns):
    """Validate correct number of columns (schema compliance) and return results in table format."""
    actual_columns = df.shape[1]
    expected_count = len(expected_columns)
    failed_count = 1 if actual_columns != expected_count else 0

    result_df = pd.DataFrame([{
        "Check ID": 9,
        "Variable checked": "All fields",
        "Description": "Validate correct number of columns (schema compliance).",
        "Failed Count": failed_count,
        "% of overall data": "N/A"  # Not row-based, applies to structure
    }])

    return result_df

def check_10_unexpected_columns(df, expected_columns):
    """Check that no unexpected columns are present and return results in table format."""
    unexpected_cols = [col for col in df.columns if col not in expected_columns]
    failed_count = len(unexpected_cols)

    result_df = pd.DataFrame([{
        "Check ID": 10,
        "Variable checked": ", ".join(unexpected_cols) if unexpected_cols else "None",
        "Description": "Check that no unexpected columns are present.",
        "Failed Count": failed_count,
        "% of overall data": "N/A"  # Not row-based, applies to structure
    }])

    return result_df

def check_11_column_data_types(df, expected_dtypes):
    """Validate column data types against expected schema and return results in table format."""
    mismatched_columns = [
        col for col in expected_dtypes.keys() if col in df.columns and df[col].dtype != expected_dtypes[col]
    ]
    failed_count = len(mismatched_columns)

    result_df = pd.DataFrame([{
        "Check ID": 11,
        "Variable checked": ", ".join(mismatched_columns) if mismatched_columns else "None",
        "Description": "Validate column data types against expected schema.",
        "Failed Count": failed_count,
        "% of overall data": "N/A"  # Not row-based, applies to structure
    }])

    return result_df

import re

def check_12_special_characters(df, text_columns):
    """Validate no unprintable/special characters in text fields and return results in table format."""
    unprintable_pattern = re.compile(r'[\x00-\x1F\x7F-\x9F]')  # Matches unprintable control characters

    failed_columns = []
    failed_count = 0

    for col in text_columns:
        if col in df.columns:
            special_chars_found = df[col].astype(str).apply(lambda x: bool(unprintable_pattern.search(x)))
            count = special_chars_found.sum()
            if count > 0:
                failed_columns.append(col)
                failed_count += count

    result_df = pd.DataFrame([{
        "Check ID": 12,
        "Variable checked": ", ".join(failed_columns) if failed_columns else "None",
        "Description": "Validate no unprintable/special characters (e.g., \\0) in text fields.",
        "Failed Count": failed_count,
        "% of overall data": "N/A"  # Not row-based, applies to text fields
    }])

    return result_df

def check_13_encoding_issues(df, text_columns, expected_encoding="utf-8"):
    """Check encoding or language issues in text fields and return results in table format."""
    failed_columns = []
    failed_count = 0

    for col in text_columns:
        if col in df.columns:
            encoding_issues = df[col].astype(str).apply(lambda x: not x.encode(expected_encoding, errors='ignore').decode(expected_encoding) == x)
            count = encoding_issues.sum()
            if count > 0:
                failed_columns.append(col)
                failed_count += count

    result_df = pd.DataFrame([{
        "Check ID": 13,
        "Variable checked": ", ".join(failed_columns) if failed_columns else "None",
        "Description": "Check encoding or language issues (e.g., name fields must use correct encoding).",
        "Failed Count": failed_count,
        "% of overall data": "N/A"  # Not row-based, applies to text fields
    }])

    return result_df

def check_14_dataset_size(df, max_rows=1000000, max_file_size_mb=500, dataset_path=None):
    """Verify overall dataset size doesn’t exceed system limits and return results in table format."""

    total_rows = len(df)
    row_fail = 1 if total_rows > max_rows else 0

    file_size_fail = 0
    if dataset_path:
        import os
        file_size = os.path.getsize(dataset_path) / (1024 * 1024)  # Convert bytes to MB
        file_size_fail = 1 if file_size > max_file_size_mb else 0
    else:
        file_size = "N/A"

    failed_count = row_fail + file_size_fail

    result_df = pd.DataFrame([{
        "Check ID": 14,
        "Variable checked": "All fields",
        "Description": "Verify overall dataset size doesn’t exceed system limits.",
        "Failed Count": failed_count,
        "% of overall data": "N/A"  # Not row-based, applies to file constraints
    }])

    return result_df

def check_15_primary_keys(df, primary_key_columns):
    """Validate presence of primary keys and return results in table format."""

    missing_count = df[primary_key_columns].isna().sum().sum()  # Count missing values in primary key columns
    duplicate_count = df.duplicated(subset=primary_key_columns).sum()  # Count duplicate primary keys

    failed_count = missing_count + duplicate_count

    result_df = pd.DataFrame([{
        "Check ID": 15,
        "Variable checked": ", ".join(primary_key_columns),
        "Description": "Validate presence of primary keys (Borrower ID, Facility ID, etc.).",
        "Failed Count": failed_count,
        "% of overall data": "N/A"  # Not row-based, applies to key constraints
    }])

    return result_df

def check_16_duplicate_entire_rows(df):
    """Check for any obviously duplicated entire rows and return results in table format."""

    duplicate_count = df.duplicated().sum()  # Count fully duplicated rows

    result_df = pd.DataFrame([{
        "Check ID": 16,
        "Variable checked": "All fields",
        "Description": "Check for any obviously duplicated entire rows.",
        "Failed Count": duplicate_count,
        "% of overall data": f"{(duplicate_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_17_reference_code_match(df, column_code_map):
    """Validate references to external dictionaries or code sets and return results in table format."""

    failed_columns = []
    failed_count = 0

    for col, valid_codes in column_code_map.items():
        if col in df.columns:
            invalid_entries = ~df[col].isin(valid_codes)
            count = invalid_entries.sum()
            if count > 0:
                failed_columns.append(col)
                failed_count += count

    result_df = pd.DataFrame([{
        "Check ID": 17,
        "Variable checked": ", ".join(failed_columns) if failed_columns else "None",
        "Description": "Validate references to external dictionaries or code sets.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_18_text_length(df, text_column_limits):
    """Ensure that text fields do not exceed maximum length and return results in table format."""

    failed_columns = []
    failed_count = 0

    for col, max_length in text_column_limits.items():
        if col in df.columns:
            exceeded_length = df[col].astype(str).apply(lambda x: len(x) > max_length).sum()
            if exceeded_length > 0:
                failed_columns.append(col)
                failed_count += exceeded_length

    result_df = pd.DataFrame([{
        "Check ID": 18,
        "Variable checked": ", ".join(failed_columns) if failed_columns else "None",
        "Description": "Ensure that text fields do not exceed maximum length.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_19_numeric_fields(df, numeric_columns):
    """Check that numeric fields do not contain text or special characters and return results in table format."""

    failed_columns = []
    failed_count = 0

    for col in numeric_columns:
        if col in df.columns:
            non_numeric_values = df[col].apply(lambda x: not str(x).replace('.', '', 1).isdigit() if pd.notna(x) else False)
            count = non_numeric_values.sum()
            if count > 0:
                failed_columns.append(col)
                failed_count += count

    result_df = pd.DataFrame([{
        "Check ID": 19,
        "Variable checked": ", ".join(failed_columns) if failed_columns else "None",
        "Description": "Check that numeric fields do not contain text/special characters.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

import numpy as np

def check_20_random_row_plausibility(df, num_samples=5):
    """Spot check random rows for plausibility and return results in table format."""

    # Randomly select `num_samples` rows
    random_sample = df.sample(n=min(num_samples, len(df)), random_state=42)

    # Identify suspicious rows where all values are missing
    blank_rows = random_sample.isna().all(axis=1).sum()

    # Identify extreme outliers (Numeric fields only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_rows = 0
    if not numeric_cols.empty:
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_rows += random_sample[col].apply(lambda x: x < lower_bound or x > upper_bound).sum()

    failed_count = blank_rows + outlier_rows

    result_df = pd.DataFrame([{
        "Check ID": 20,
        "Variable checked": "All fields",
        "Description": "Spot check random rows for plausibility (manual or automated review).",
        "Failed Count": failed_count,
        "% of overall data": "N/A"  # Manual check, so percentage not applicable
    }])

    return result_df

def check_21_placeholder_values(df, placeholder_values=["N/A", "XXXX", "0", "NULL"]):
    """Ensure no leftover placeholder values are present and return results in table format."""

    failed_columns = []
    failed_count = 0

    for col in df.columns:
        placeholder_found = df[col].astype(str).isin(placeholder_values).sum()
        if placeholder_found > 0:
            failed_columns.append(col)
            failed_count += placeholder_found

    result_df = pd.DataFrame([{
        "Check ID": 21,
        "Variable checked": ", ".join(failed_columns) if failed_columns else "None",
        "Description": "Ensure no leftover placeholder values (e.g., 'N/A', 'XXXX', '0').",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_22_cross_field_logic(df, validation_rules):
    """Validate any relationships not specifically in schema and return results in table format."""

    failed_columns = []
    failed_count = 0

    for rule_name, condition in validation_rules.items():
        try:
            failed_rows = df[~df.eval(condition)].shape[0]  # Count rows that fail the condition
            if failed_rows > 0:
                failed_columns.append(rule_name)
                failed_count += failed_rows
        except Exception as e:
            print(f"Error in rule '{rule_name}': {e}")

    result_df = pd.DataFrame([{
        "Check ID": 22,
        "Variable checked": ", ".join(failed_columns) if failed_columns else "None",
        "Description": "Validate any relationships not specifically in schema (cross-field logic).",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_23_snapshot_frequency(df, column_name, expected_frequency="1 per month"):
    """Check frequency of snapshots if multiple monthly files exist and return results in table format."""

    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    snapshot_counts = df[column_name].dt.to_period("M").value_counts()

    # Identify months where the snapshot count is incorrect
    failed_months = snapshot_counts[snapshot_counts > 1].index.astype(str).tolist()
    failed_count = len(failed_months)

    result_df = pd.DataFrame([{
        "Check ID": 23,
        "Variable checked": column_name,
        "Description": "Check frequency of snapshots if multiple monthly files exist.",
        "Failed Count": failed_count,
        "% of overall data": "N/A"  # Applies to monthly frequency, not individual rows
    }])

    return result_df

def check_24_snapshot_date_range(df, column_name, min_date, max_date):
    """Confirm data extraction period is within expected date range and return results in table format."""

    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')

    # Count records outside the expected date range
    failed_count = df[(df[column_name] < min_date) | (df[column_name] > max_date)].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 24,
        "Variable checked": column_name,
        "Description": "Confirm data extraction period is within expected date range.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

from datetime import datetime, timedelta

def check_25_future_snapshot_dates(df, column_name, max_future_days=30):
    """Check for accidental future-dated snapshots and return results in table format."""

    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')

    today = datetime.today().date()
    max_allowed_date = today + timedelta(days=max_future_days)

    # Count records beyond the allowed future date
    failed_count = df[df[column_name].dt.date > max_allowed_date].shape[0]

    result_df = pd.DataFrame([{
        "Check ID": 25,
        "Variable checked": column_name,
        "Description": "Check for accidental future-dated snapshots beyond the allowed threshold.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_26_missing_data_ratio(df, critical_fields, max_missing_ratio=0.05):
    """Evaluate ratio of missing vs. present data for critical fields and return results in table format."""

    failed_columns = []
    failed_count = 0

    for col in critical_fields:
        if col in df.columns:
            missing_ratio = df[col].isna().sum() / len(df)
            if missing_ratio > max_missing_ratio:
                failed_columns.append(col)
                failed_count += df[col].isna().sum()

    result_df = pd.DataFrame([{
        "Check ID": 26,
        "Variable checked": ", ".join(failed_columns) if failed_columns else "None",
        "Description": "Evaluate ratio of missing vs. present data for critical fields.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / (len(df) * len(critical_fields))) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_27_schema_version(df, metadata_column, expected_version="v1.0"):
    """Check data version or schema version metadata matches expected version and return results in table format."""

    if metadata_column not in df.columns:
        failed_count = len(df)  # If the column is missing, all rows fail
    else:
        failed_count = df[df[metadata_column] != expected_version].shape[0]  # Count rows where the version does not match

    result_df = pd.DataFrame([{
        "Check ID": 27,
        "Variable checked": metadata_column if metadata_column in df.columns else "Schema Version (Missing)",
        "Description": "Check data version or schema version metadata matches expected version.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_28_snapshot_date_consistency(df, related_df, column_name):
    """Validate that each row’s Snapshot Date is consistent across related tables and return results in table format."""

    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    related_df[column_name] = pd.to_datetime(related_df[column_name], errors='coerce')

    # Merge both datasets on a common key and check Snapshot Date consistency
    merged_df = df.merge(related_df, on=['Borrower ID', 'Facility ID'], suffixes=('_main', '_related'))

    failed_count = (merged_df[f"{column_name}_main"] != merged_df[f"{column_name}_related"]).sum()

    result_df = pd.DataFrame([{
        "Check ID": 28,
        "Variable checked": column_name,
        "Description": "Validate that each row’s Snapshot Date is consistent across related tables.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(merged_df)) * 100:.2f}%" if len(merged_df) > 0 else "0%"
    }])

    return result_df

def check_30_decimal_separator(df, numeric_columns):
    """Confirm consistent decimal separators in numeric fields and return results in table format."""

    failed_columns = []
    failed_count = 0

    decimal_pattern = re.compile(r'^\d+,\d+$')  # Pattern to detect incorrect decimal separator (comma instead of dot)

    for col in numeric_columns:
        if col in df.columns:
            invalid_format = df[col].astype(str).apply(lambda x: bool(decimal_pattern.search(x)) if pd.notna(x) else False)
            count = invalid_format.sum()
            if count > 0:
                failed_columns.append(col)
                failed_count += count

    result_df = pd.DataFrame([{
        "Check ID": 30,
        "Variable checked": ", ".join(failed_columns) if failed_columns else "None",
        "Description": "Confirm consistent decimal separators (Only '.' used for decimals).",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_29_crosstab_outliers(df, category_column, numeric_column, zscore_threshold=3):
    """Perform basic cross-tab comparison for outliers and return results in table format."""

    if category_column not in df.columns or numeric_column not in df.columns:
        return pd.DataFrame([{
            "Check ID": 29,
            "Variable checked": f"{category_column}, {numeric_column}",
            "Description": "Basic cross-tab comparison for outliers.",
            "Failed Count": "Error: Column not found",
            "% of overall data": "N/A"
        }])

    # Compute mean and standard deviation per category
    summary_stats = df.groupby(category_column)[numeric_column].agg(['mean', 'std']).reset_index()

    # Merge summary statistics back to main dataset
    df = df.merge(summary_stats, on=category_column, suffixes=('', '_stats'))

    # Compute Z-scores
    df['z_score'] = (df[numeric_column] - df['mean']) / df['std']
    df['z_score'] = df['z_score'].abs()  # Take absolute Z-score

    # Identify outliers beyond the threshold
    failed_count = (df['z_score'] > zscore_threshold).sum()

    result_df = pd.DataFrame([{
        "Check ID": 29,
        "Variable checked": f"{category_column}, {numeric_column}",
        "Description": "Basic cross-tab comparison for outliers.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

# Define category and numeric column to check for anomalies
category_column = 'Legal Form / Ownership Type'  # Replace with a relevant categorical variable
numeric_column = 'Loan Amount'  # Replace with the numeric c

def section_a_checks(df):
    """Run all 30 checks for Section A and return a combined results table."""

    # Run each check individually (Ensure all check functions are defined)
    checks = [
        check_1_SnapshotDateMV(df, 'Snapshot Date'),
        check_2_SnapshotDate_format(df, 'Snapshot Date'),
        check_3_SnapshotDate_future(df, 'Snapshot Date'),
        check_4_SnapshotDate_duplicates(df, 'Snapshot Date'),
        check_5_SnapshotDate_sequence(df, 'Snapshot Date'),
        check_6_row_count(df, 1000),
        check_7_empty_rows(df),
        check_8_missing_ratio(df),
        check_9_column_count(df, ['Snapshot Date', 'Borrower ID', 'Facility ID', 'Loan Amount', 'Balance']),
        check_10_unexpected_columns(df, ['Snapshot Date', 'Borrower ID', 'Facility ID', 'Loan Amount', 'Balance']),
        check_11_column_data_types(df, {'Snapshot Date': 'datetime64[ns]', 'Loan Amount': 'float64'}),
        check_12_special_characters(df, ['Customer Name', 'Address']),
        check_13_encoding_issues(df, ['Customer Name', 'Address']),
        check_14_dataset_size(df),
        check_15_primary_keys(df, ['Borrower ID', 'Facility ID']),
        check_16_duplicate_entire_rows(df),
        check_17_reference_code_match(df, {'Loan Type': ['Personal', 'Mortgage', 'Auto']}),
        check_18_text_length(df, {'Customer Name': 255, 'Address': 500}),
        check_19_numeric_fields(df, ['Loan Amount', 'Balance']),
        check_20_random_row_plausibility(df),
        check_21_placeholder_values(df),
        check_22_cross_field_logic(df, {
            "Loan Amount vs. Balance": "`Loan Amount` <= `Balance`",
            "Interest Rate Positive": "`Return on Assets` > 0",
            "Maturity Date after Snapshot Date": "`Maturity Date` > `Snapshot Date`"
        }),
        check_23_snapshot_frequency(df, 'Snapshot Date'),
        check_24_snapshot_date_range(df, "Snapshot Date", "2023-01-01", "2025-12-31"),
        check_25_future_snapshot_dates(df, "Snapshot Date"),
        check_26_missing_data_ratio(df, ['Snapshot Date', 'Borrower ID', 'Facility ID']),
        check_27_schema_version(df, 'Schema Version', "v1.0"),
        check_28_snapshot_date_consistency(df, df, "Snapshot Date"),
        check_29_crosstab_outliers(df, 'Legal Form / Ownership Type', 'Loan Amount'),
        check_30_decimal_separator(df, ['Loan Amount', 'Balance'])
    ]

    processed_checks = []
    for check in checks:
        if isinstance(check, dict):  # Convert dict to DataFrame
            processed_checks.append(pd.DataFrame([check]))
        elif isinstance(check, pd.DataFrame):
            processed_checks.append(check)

    # Combine all check results into a single DataFrame
    combined_results = pd.concat(processed_checks, ignore_index=True)

    return combined_results
