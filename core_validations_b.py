import re
import numpy as np
import pandas as pd
from datetime import datetime

def check_B31_missing_borrower_id(df, borrower_column):
    """Check for missing Borrower ID and return results in table format."""

    missing_count = df[borrower_column].isna().sum()  # Count missing values

    result_df = pd.DataFrame([{
        "Check ID": 31,
        "Variable checked": borrower_column,
        "Description": "Check for missing Borrower ID (Primary Key).",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B32_unique_borrower_id(df, borrower_column):
    """Check uniqueness of Borrower ID and return results in table format."""

    # Count duplicate Borrower IDs (if Borrower ID should be unique)
    duplicate_count = df[borrower_column].duplicated(keep=False).sum()

    result_df = pd.DataFrame([{
        "Check ID": 32,
        "Variable checked": borrower_column,
        "Description": "Check uniqueness if needed (or consistent usage of Borrower ID).",
        "Failed Count": duplicate_count,
        "% of overall data": f"{(duplicate_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B33_missing_borrower_name(df, borrower_name_column):
    """Check for missing Borrower Name and return results in table format."""

    missing_count = df[borrower_name_column].isna().sum()  # Count missing values

    result_df = pd.DataFrame([{
        "Check ID": 33,
        "Variable checked": borrower_name_column,
        "Description": "Check for missing or null Borrower Name.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B34_borrower_name_suspicious(df, borrower_name_column):
    """Check for suspicious patterns in Borrower Name and return results in table format."""

    suspicious_pattern = re.compile(r'^\d+$')  # Matches names that contain only numbers

    failed_count = df[borrower_name_column].astype(str).apply(lambda x: bool(suspicious_pattern.match(x))).sum()

    result_df = pd.DataFrame([{
        "Check ID": 34,
        "Variable checked": borrower_name_column,
        "Description": "Check for suspicious strings (only digits, or special patterns) in Borrower Name.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B35_missing_legal_form(df, legal_form_column):
    """Check for missing Legal Form / Ownership Type and return results in table format."""

    missing_count = df[legal_form_column].isna().sum()  # Count missing values

    result_df = pd.DataFrame([{
        "Check ID": 35,
        "Variable checked": legal_form_column,
        "Description": "Check for missing Legal Form / Ownership Type.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B36_valid_legal_form(df, legal_form_column, valid_values = ['Public', 'Private', 'Partnership', 'Government', 'Non-Profit']):
    """Check if Legal Form / Ownership Type is within the allowed categories and return results in table format."""

    invalid_count = (~df[legal_form_column].isin(valid_values)).sum()  # Count invalid values

    result_df = pd.DataFrame([{
        "Check ID": 36,
        "Variable checked": legal_form_column,
        "Description": "Check if Legal Form / Ownership Type is in valid list.",
        "Failed Count": invalid_count,
        "% of overall data": f"{(invalid_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B37_missing_industry_classification(df, industry_column, max_missing_ratio=0.05):
    """Check for missing Industry Classification and ensure missing ratio is below 5%."""

    missing_count = df[industry_column].isna().sum()  # Count missing values
    missing_ratio = missing_count / len(df)

    result_df = pd.DataFrame([{
        "Check ID": 37,
        "Variable checked": industry_column,
        "Description": "Check for missing Industry Classification (Missing ratio < 5%).",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_ratio * 100):.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B38_valid_industry_classification(df, industry_column, valid_codes = ['1111', '2362', '5313', '6221', '8111']):  # Replace with actual valid codes):
    """Check if Industry Classification is within the allowed code list."""

    invalid_count = (~df[industry_column].isin(valid_codes)).sum()  # Count invalid values

    result_df = pd.DataFrame([{
        "Check ID": 38,
        "Variable checked": industry_column,
        "Description": "Confirm code is from valid NAICS/SIC or internal reference list.",
        "Failed Count": invalid_count,
        "% of overall data": f"{(invalid_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B39_missing_country(df, country_column, max_missing_ratio=0.02):
    """Check for missing Country of Incorporation and ensure missing ratio is below 2%."""

    missing_count = df[country_column].isna().sum()  # Count missing values
    missing_ratio = missing_count / len(df)

    result_df = pd.DataFrame([{
        "Check ID": 39,
        "Variable checked": country_column,
        "Description": "Check for missing Country of Incorporation (Missing ratio < 2%).",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_ratio * 100):.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B40_valid_country(df, country_column, valid_countries = ['United States', 'Canada', 'United Kingdom', 'France', 'Germany', 'UAE', 'Saudi Arabia', 'IN', 'US', 'GB', 'FR', 'DE']):
    """Check if Country of Incorporation matches the approved country list."""

    invalid_count = (~df[country_column].isin(valid_countries)).sum()  # Count invalid values

    result_df = pd.DataFrame([{
        "Check ID": 40,
        "Variable checked": country_column,
        "Description": "Must be valid country name or ISO code (100% match to country list).",
        "Failed Count": invalid_count,
        "% of overall data": f"{(invalid_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B41_country_operations(df, incorporation_column, operations_column, valid_countries = ['United States', 'Canada', 'United Kingdom', 'France', 'Germany', 'UAE', 'Saudi Arabia', 'IN', 'US', 'GB', 'FR', 'DE']):
    """Cross-check Country of Main Operations with Country of Incorporation and validate against the approved list."""

    mismatch_count = (df[incorporation_column] != df[operations_column]).sum()  # Count mismatches
    invalid_count = (~df[operations_column].isin(valid_countries)).sum()  # Count invalid country names

    result_df = pd.DataFrame([
        {
            "Check ID": 41,
            "Variable checked": operations_column,
            "Description": "Cross-check if it differs from incorporation, must also be in valid country list.",
            "Failed Count": mismatch_count,
            "% of overall data": f"{(mismatch_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
        },
        {
            "Check ID": 41,
            "Variable checked": operations_column,
            "Description": "Country of Main Operations must be a valid country.",
            "Failed Count": invalid_count,
            "% of overall data": f"{(invalid_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
        }
    ])

    return result_df


def check_B42_years_in_operation(df, years_column, min_years=0, max_years=200, max_out_of_range_ratio=0.01):
    """Check if Years in Operation / Incorporation Date is within the valid range and ensure out-of-range records are below 1%."""

    out_of_range_count = ((df[years_column] < min_years) | (df[years_column] > max_years)).sum()
    out_of_range_ratio = out_of_range_count / len(df)

    result_df = pd.DataFrame([{
        "Check ID": 42,
        "Variable checked": years_column,
        "Description": f"Years in Operation must be in range [{min_years}..{max_years}], Out-of-range < 1% of records.",
        "Failed Count": out_of_range_count,
        "% of overall data": f"{(out_of_range_ratio * 100):.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_B43_valid_incorporation_date(df, date_column, min_year=1850):
    """Check if Incorporation Date is within a valid range (1850 - current year) and not in the future."""

    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')  # Convert to datetime
    current_year = datetime.today().year

    out_of_range_count = ((df[date_column].dt.year < min_year) | (df[date_column].dt.year > current_year)).sum()

    result_df = pd.DataFrame([{
        "Check ID": 43,
        "Variable checked": date_column,
        "Description": f"Incorporation Date must be between {min_year} and {current_year}, with no future dates.",
        "Failed Count": out_of_range_count,
        "% of overall data": f"{(out_of_range_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B44_missing_parent_company(df, parent_id_column, max_missing_ratio=0.50):
    """Check missing ratio for Group / Parent Company ID and ensure it is within acceptable limits."""

    missing_count = df[parent_id_column].isna().sum()  # Count missing values
    missing_ratio = missing_count / len(df)

    result_df = pd.DataFrame([{
        "Check ID": 44,
        "Variable checked": parent_id_column,
        "Description": "Check missing ratio for Group / Parent Company ID.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_ratio * 100):.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_B45_valid_parent_company(df, parent_id_column, borrower_id_column):
    """Ensure Group / Parent Company ID, if present, references a valid Borrower ID."""

    if parent_id_column not in df.columns or borrower_id_column not in df.columns:
        return pd.DataFrame([{
            "Check ID": 45,
            "Variable checked": f"{parent_id_column}, {borrower_id_column}",
            "Description": "Check that Parent Company ID references a valid Borrower ID.",
            "Failed Count": "Error: Column not found",
            "% of overall data": "N/A"
        }])

    # Identify Parent Company IDs that do not exist in Borrower ID column
    invalid_references = (~df[parent_id_column].isin(df[borrower_id_column])) & df[parent_id_column].notna()
    failed_count = invalid_references.sum()

    result_df = pd.DataFrame([{
        "Check ID": 45,
        "Variable checked": parent_id_column,
        "Description": "If present, confirm it references a valid Borrower ID.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_B46_missing_relationship_start_date(df, start_date_column):
    """Check for missing Relationship Start Date and return results in table format."""

    missing_count = df[start_date_column].isna().sum()  # Count missing values

    result_df = pd.DataFrame([{
        "Check ID": 46,
        "Variable checked": start_date_column,
        "Description": "Must have a start date for the relationship.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B47_relationship_vs_incorporation(df, start_date_column, incorporation_date_column):
    """Check that Relationship Start Date is >= Incorporation Date."""

    df[start_date_column] = pd.to_datetime(df[start_date_column], errors='coerce')
    df[incorporation_date_column] = pd.to_datetime(df[incorporation_date_column], errors='coerce')

    failed_count = (df[start_date_column] < df[incorporation_date_column]).sum()

    result_df = pd.DataFrame([{
        "Check ID": 47,
        "Variable checked": f"{start_date_column}, {incorporation_date_column}",
        "Description": "Relationship Start Date must be >= Borrower’s Incorporation Date.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B48_valid_relationship_type(df, relationship_column, valid_types = ['Subsidiary', 'Direct Corporate', 'Affiliate', 'Joint Venture']):
    """Check if Relationship Type is within the valid categories."""

    invalid_count = (~df[relationship_column].isin(valid_types)).sum()  # Count invalid values

    result_df = pd.DataFrame([{
        "Check ID": 48,
        "Variable checked": relationship_column,
        "Description": "Check if in valid set (Subsidiary, Direct Corporate, etc.).",
        "Failed Count": invalid_count,
        "% of overall data": f"{(invalid_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B49_missing_borrower_segment(df, segment_column):
    """Check for missing Borrower Segment / Classification and return results in table format."""

    missing_count = df[segment_column].isna().sum()  # Count missing values

    result_df = pd.DataFrame([{
        "Check ID": 49,
        "Variable checked": segment_column,
        "Description": "Must have a segment/classification assigned.",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B50_large_corporate_assets(df, segment_column, assets_column, min_large_corporate_assets=10000000):
    """Ensure Borrower Segment 'Large Corporate' has Total Assets exceeding the required threshold."""

    # Identify Large Corporate borrowers with assets below the threshold
    failed_count = ((df[segment_column] == "Large Corporate") & (df[assets_column] < min_large_corporate_assets)).sum()

    result_df = pd.DataFrame([{
        "Check ID": 50,
        "Variable checked": f"{segment_column}, {assets_column}",
        "Description": "If Large Corporate, check that Total Assets exceed defined threshold.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B51_missing_related_parties_exposure(df, exposure_column, max_missing_ratio=0.10):
    """Check for missing Connected / Related Parties Exposure and ensure missing ratio is below 10%."""

    missing_count = df[exposure_column].isna().sum()  # Count missing values
    missing_ratio = missing_count / len(df)

    result_df = pd.DataFrame([{
        "Check ID": 51,
        "Variable checked": exposure_column,
        "Description": "Missing check for Connected / Related Parties Exposure (Allowed missing < 10%).",
        "Failed Count": missing_count,
        "% of overall data": f"{(missing_ratio * 100):.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B52_non_negative_related_exposure(df, exposure_column):
    """Ensure Connected / Related Parties Exposure is non-negative (≥ 0)."""

    # Convert column to numeric (force errors='coerce' to handle non-numeric values)
    df[exposure_column] = pd.to_numeric(df[exposure_column], errors='coerce')

    failed_count = (df[exposure_column] < 0).sum()  # Count negative values

    result_df = pd.DataFrame([{
        "Check ID": 52,
        "Variable checked": exposure_column,
        "Description": "Numeric range check (≥ 0), no negative exposures.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B53_borrower_id_name_consistency(df, borrower_id_column, borrower_name_column):
    """Ensure that each Borrower ID is consistently mapped to the same Borrower Name."""

    # Identify Borrower IDs that map to multiple different names
    inconsistent_borrowers = df.groupby(borrower_id_column)[borrower_name_column].nunique()
    failed_count = (inconsistent_borrowers > 1).sum()

    result_df = pd.DataFrame([{
        "Check ID": 53,
        "Variable checked": f"{borrower_id_column}, {borrower_name_column}",
        "Description": "Ensure Borrower ID is consistently associated with the same Borrower Name.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B54_borrower_data_consistency(df, borrower_id_column, fields_to_check = ['Country of Incorporation', 'Legal Form / Ownership Type', 'Industry Classification']):
    """Ensure no contradictory data for a single Borrower across multiple records."""

    # Identify Borrowers with inconsistent values in the specified fields
    inconsistent_borrowers = df.groupby(borrower_id_column)[fields_to_check].nunique()
    failed_count = (inconsistent_borrowers > 1).sum().sum()  # Count cases where there is more than 1 unique value for a borrower

    result_df = pd.DataFrame([{
        "Check ID": 54,
        "Variable checked": ", ".join(fields_to_check),
        "Description": "Validate no contradictory data for a single Borrower (e.g., country, legal form rarely changes).",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_B55_segment_vs_industry_logic(df, segment_column, industry_column, restricted_combos = ['Large-Scale Manufacturing', 'Investment Banking', 'Multinational Corporate']):
    """Check logical consistency: If segment=SME, ensure the industry aligns with SME rules."""

    # Identify cases where an SME is in a restricted industry
    failed_count = df[df[segment_column] == "SME"][industry_column].isin(restricted_combos).sum()

    result_df = pd.DataFrame([{
        "Check ID": 55,
        "Variable checked": f"{segment_column}, {industry_column}",
        "Description": "Logic check: If segment=SME, ensure it belongs to an acceptable industry.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df

def check_B56_parent_not_same_as_borrower(df, borrower_id_column, parent_id_column):
    """Ensure that Group / Parent Company ID is not the same as the Borrower ID."""

    failed_count = (df[borrower_id_column] == df[parent_id_column]).sum()

    result_df = pd.DataFrame([{
        "Check ID": 56,
        "Variable checked": f"{borrower_id_column}, {parent_id_column}",
        "Description": "If parent is given, confirm parent is not the same as the Borrower ID.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_B57_relationship_vs_snapshot_date(df, relationship_date_column, snapshot_date_column):
    """Check that Relationship Start Date is <= Snapshot Date and not in the future."""

    # Convert columns to datetime format
    df[relationship_date_column] = pd.to_datetime(df[relationship_date_column], errors='coerce')
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    # Identify cases where Relationship Start Date is after the Snapshot Date
    failed_count = (df[relationship_date_column] > df[snapshot_date_column]).sum()

    result_df = pd.DataFrame([{
        "Check ID": 57,
        "Variable checked": f"{relationship_date_column}, {snapshot_date_column}",
        "Description": "Relationship Start Date must be <= Snapshot Date. No future start date relative to snapshot.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_B58_borrower_name_whitespace(df, borrower_name_column):
    """Check for leading/trailing whitespace in Borrower Name and return results in table format."""

    # Identify cases where Borrower Name has leading or trailing spaces
    failed_count = df[borrower_name_column].astype(str).apply(lambda x: x != x.strip()).sum()

    result_df = pd.DataFrame([{
        "Check ID": 58,
        "Variable checked": borrower_name_column,
        "Description": "Check for leading/trailing whitespace in Borrower Name. Trim & confirm no extraneous spaces.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def check_B59_borrower_segment_distribution(df, segment_column, max_other_percentage=10):
    """Evaluate distribution of Borrower Segments and ensure 'Other' category is below a defined threshold."""

    # Count occurrences of each segment
    segment_counts = df[segment_column].value_counts(normalize=True) * 100

    # Identify if 'Other' exceeds the defined threshold
    failed_count = segment_counts.get('Other', 0)  # Get % of 'Other' segment, default to 0 if not found
    # failed_status = "Fail" if failed_count > max_other_percentage else "Pass"

    result_df = pd.DataFrame([{
        "Check ID": 59,
        "Variable checked": segment_column,
        "Description": f"Evaluate distribution: Ensure 'Other' segment < {max_other_percentage}% of total.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_count:.2f}%"
    }])

    return result_df


def check_B60_borrower_id_format(df, borrower_id_column, min_length=6, max_length=10):
    """Check that Borrower ID follows a consistent numeric format with the required length."""

    # Ensure Borrower ID is a string and check length constraints
    failed_count = df[borrower_id_column].astype(str).apply(
        lambda x: not re.fullmatch(r"\d{" + str(min_length) + "," + str(max_length) + r"}", x)
    ).sum()

    result_df = pd.DataFrame([{
        "Check ID": 60,
        "Variable checked": borrower_id_column,
        "Description": f"Check Borrower ID format: Must be numeric and {min_length}-{max_length} digits long.",
        "Failed Count": failed_count,
        "% of overall data": f"{(failed_count / len(df)) * 100:.2f}%" if len(df) > 0 else "0%"
    }])

    return result_df


def section_b_checks(df):
    """Run all 30 checks for Section B and return a combined results table."""

    # Run each check individually (Ensure all check functions are defined)
    checks = [
        check_B31_missing_borrower_id(df, 'Borrower ID'),
        check_B32_unique_borrower_id(df, 'Borrower ID'),
        check_B33_missing_borrower_name(df, 'Borrower Name'),
        check_B34_borrower_name_suspicious(df, 'Borrower Name'),
        check_B35_missing_legal_form(df, 'Legal Form / Ownership Type'),
        check_B36_valid_legal_form(df, 'Legal Form / Ownership Type', ['Public', 'Private', 'Partnership', 'Government', 'Non-Profit']),
        check_B37_missing_industry_classification(df, 'Industry Classification'),
        check_B38_valid_industry_classification(df, 'Industry Classification', ['1111', '2362', '5313', '6221', '8111']),  # Update with actual codes
        check_B39_missing_country(df, 'Country of Incorporation'),
        check_B40_valid_country(df, 'Country of Incorporation', ['United States', 'Canada', 'United Kingdom', 'France', 'Germany', 'UAE', 'Saudi Arabia']),
        check_B41_country_operations(df, 'Country of Incorporation', 'Country of Main Operations', ['United States', 'Canada', 'United Kingdom', 'France', 'Germany', 'UAE', 'Saudi Arabia']),
        check_B42_years_in_operation(df, 'Years in Operation'),
        check_B43_valid_incorporation_date(df, 'Incorporation Date'),
        check_B44_missing_parent_company(df, 'Group / Parent Company ID'),
        check_B45_valid_parent_company(df, 'Group / Parent Company ID', 'Borrower ID'),
        check_B46_missing_relationship_start_date(df, 'Relationship Start Date'),
        check_B47_relationship_vs_incorporation(df, 'Relationship Start Date', 'Incorporation Date'),
        check_B48_valid_relationship_type(df, 'Relationship Type', ['Subsidiary', 'Direct Corporate', 'Affiliate', 'Joint Venture']),
        check_B49_missing_borrower_segment(df, 'Borrower Segment / Classification'),
        check_B50_large_corporate_assets(df, 'Borrower Segment / Classification', 'Total Assets'),
        check_B51_missing_related_parties_exposure(df, 'Connected / Related Parties Exposure'),
        check_B52_non_negative_related_exposure(df, 'Connected / Related Parties Exposure'),
        check_B53_borrower_id_name_consistency(df, 'Borrower ID', 'Borrower Name'),
        check_B54_borrower_data_consistency(df, 'Borrower ID', ['Country of Incorporation', 'Legal Form / Ownership Type', 'Industry Classification']),
        check_B55_segment_vs_industry_logic(df, 'Borrower Segment / Classification', 'Industry Classification', ['Large-Scale Manufacturing', 'Investment Banking', 'Multinational Corporate']),
        check_B56_parent_not_same_as_borrower(df, 'Borrower ID', 'Group / Parent Company ID'),
        check_B57_relationship_vs_snapshot_date(df, 'Relationship Start Date', 'Snapshot Date'),
        check_B58_borrower_name_whitespace(df, 'Borrower Name'),
        check_B59_borrower_segment_distribution(df, 'Borrower Segment / Classification'),
        check_B60_borrower_id_format(df, 'Borrower ID')
    ]

    # Ensure each check returns a DataFrame, convert dict results to DataFrame where needed
    processed_checks = []
    for check in checks:
        if isinstance(check, dict):  # Convert dict to DataFrame
            processed_checks.append(pd.DataFrame([check]))
        elif isinstance(check, pd.DataFrame):
            processed_checks.append(check)

    # Combine all check results into a single DataFrame
    combined_results = pd.concat(processed_checks, ignore_index=True)

    return combined_results