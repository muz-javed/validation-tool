import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def check_91_CollateralID(df, column_name):
    """Check for missing Collateral ID when collateral is indicated."""
    failed_count = df[column_name].isna().sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 91,
        "Variable checked": column_name,
        "Description": "Missing check for pledged collateral records.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_92_CollateralType(df, column_name, valid_types):
    """
    Validation Check 92: Collateral Type must be in a valid set.
    Check allowed categories.
    """
    failed_count = (~df[column_name].isin(valid_types)).sum()  # Count invalid collateral types
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 92,
        "Variable checked": column_name,
        "Description": "Collateral Type must be in a valid set.",
        "Threshold / Condition": "Check allowed categories.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_93_OriginalCollateralValue(df, column_name):
    """
    Validation Check 93: Original Collateral Value must be ≥ 0.
    No negative collateral value allowed.
    """
    failed_count = (df[column_name].astype(float) < 0).sum()  # Count negative values
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 93,
        "Variable checked": column_name,
        "Description": "Original Collateral Value must be ≥ 0.",
        "Threshold / Condition": "No negative collateral value.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_94_CollateralValueMonthEnd(df, column_name):
    """
    Validation Check 94: Collateral Value (Month-End) must be ≥ 0.
    No negative current values allowed.
    """
    failed_count = (df[column_name].astype(float) < 0).sum()  # Count negative values
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 94,
        "Variable checked": column_name,
        "Description": "Collateral Value (Month-End) must be ≥ 0.",
        "Threshold / Condition": "No negative current values.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_95_CollateralValueFacilityStatus(df, collateral_column, facility_status_column):
    """
    Validation Check 95: If the facility is closed, collateral value might be 0 or no entry.
    Cross-check with facility status.
    """
    # Identify rows where facility is closed but collateral value is not 0 or missing
    closed_facility = df[facility_status_column].notnull()  # Facility is considered closed if End Date is not null
    invalid_collateral = df[collateral_column].astype(float) > 0  # Collateral value should be 0 or missing
    
    failed_count = (closed_facility & invalid_collateral).sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 95,
        "Variable checked": collateral_column,
        "Description": "If the facility is closed, collateral value might be 0 or no entry.",
        "Threshold / Condition": "Cross-check with facility status.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_96_GuaranteeType(df, column_name, valid_types):
    """
    Validation Check 96: Guarantee Type must be from a valid list.
    Check category validity.
    """
    failed_count = (~df[column_name].isin(valid_types)).sum()  # Count invalid guarantee types
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 96,
        "Variable checked": column_name,
        "Description": "Guarantee Type must be from a valid list.",
        "Threshold / Condition": "Check category validity.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df


def check_97_GuaranteeAmount(df, guarantee_column, exposure_column):
    """
    Validation Check 97: Guarantee Amount should be ≥ 0 and should not exceed facility exposure in typical scenarios.
    Guarantee ≤ Outstanding Exposure.
    """
    # Identify rows where Guarantee Amount is negative or exceeds Outstanding Exposure
    negative_guarantee = df[guarantee_column].astype(float) < 0
    exceeds_exposure = df[guarantee_column].astype(float) > df[exposure_column].astype(float)
    
    failed_count = (negative_guarantee | exceeds_exposure).sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 97,
        "Variable checked": guarantee_column,
        "Description": "Guarantee Amount must be ≥ 0 and should not exceed facility exposure in typical scenarios.",
        "Threshold / Condition": "Guarantee ≤ Outstanding Exposure.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df



def check_98_LienPosition(df, column_name, valid_positions):
    """
    Validation Check 98: Lien Position must be in {Senior, Junior, Subordinated}.
    Invalid lien positions should be flagged.
    """
    failed_count = (~df[column_name].isin(valid_positions)).sum()  # Count invalid lien positions
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 98,
        "Variable checked": column_name,
        "Description": "Lien Position must be in {Senior, Junior, Subordinated}.",
        "Threshold / Condition": "Invalid lien positions flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df



def check_99_CollateralIDUniqueness(df, column_name):
    """
    Validation Check 99: If multiple collaterals exist, each ID must be unique.
    Uniqueness check.
    """
    failed_count = df.duplicated(subset=[column_name], keep=False).sum()  # Count duplicate collateral IDs
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 99,
        "Variable checked": column_name,
        "Description": "If multiple collaterals exist, each ID must be unique.",
        "Threshold / Condition": "Uniqueness check.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df


def check_100_CollateralAndGuarantee(df, collateral_column, guarantee_column):
    """
    Validation Check 100: Check if both collateral and guarantee exist.
    Possibly both are valid or only one, depending on bank policy.
    """
    failed_count = df[collateral_column].isna() & df[guarantee_column].isna()  # Both should not be null
    failed_count = failed_count.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 100,
        "Variable checked": f"{collateral_column}, {guarantee_column}",
        "Description": "Check if both collateral and guarantee exist.",
        "Threshold / Condition": "Possibly both are valid or only one. Bank policy dependent.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df




def check_101_CollateralValueTrend(df, column_name):
    """
    Validation Check 101: Trend check for Collateral Value (Month-End).
    If it changes monthly, confirm no abnormal large jumps.
    > 200% or < 50% changes in 1 month is flagged.
    """
    df_sorted = df.sort_values(by=['Snapshot Date'])  # Ensure data is sorted by date
    df['Collateral Change %'] = df[column_name].pct_change() * 100  # Calculate percentage change

    failed_count = ((df['Collateral Change %'] > 200) | (df['Collateral Change %'] < -50)).sum()
    total_count = len(df) - 1  # pct_change reduces one row
    failed_percent = (failed_count / total_count) * 100 if total_count > 0 else 0

    result_df = pd.DataFrame([{
        "Check ID": 101,
        "Variable checked": column_name,
        "Description": "Trend check: if it changes monthly, confirm no abnormal large jumps.",
        "Threshold / Condition": "> 200% or < 50% changes in 1 month is flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df



def check_102_CollateralValueConsistency(df, original_value_column, month_end_value_column):
    """
    Validation Check 102: If Collateral Value (Month-End) is consistently below original for real estate, flag it.
    Variation within X% is normal.
    """
    df['Value Difference %'] = ((df[month_end_value_column] - df[original_value_column]) / df[original_value_column]) * 100

    failed_count = (df['Value Difference %'] < -X).sum()  # Flagged if below allowed variation (set X as per policy)
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 102,
        "Variable checked": f"{original_value_column}, {month_end_value_column}",
        "Description": "If Collateral Value (Month-End) is consistently below original for real estate, might flag.",
        "Threshold / Condition": "Variation within X% is normal.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df



def check_103_GuaranteeTypePartial(df, guarantee_type_column, guarantee_amount_column, facility_limit_column):
    """
    Validation Check 103: If Guarantee Type is personal, check if personal guarantee is partial or full.
    If partial, Guarantee < facility limit.
    """
    # Filtering rows where Guarantee Type is 'Personal'
    personal_guarantee = df[guarantee_type_column] == 'Personal'
    
    # Checking if Guarantee Amount exceeds Facility Limit
    partial_guarantee_violation = personal_guarantee & (df[guarantee_amount_column].astype(float) >= df[facility_limit_column].astype(float))

    failed_count = partial_guarantee_violation.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 103,
        "Variable checked": f"{guarantee_type_column}, {guarantee_amount_column}, {facility_limit_column}",
        "Description": "If Guarantee Type is personal, check if personal guarantee is partial or full.",
        "Threshold / Condition": "If partial, Guarantee < facility limit.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df




def check_104_CollateralAppraisalID(df, collateral_type_column, appraisal_id_column, external_list):
    """
    Validation Check 104: If Collateral Type = real estate, ensure appraisal ID is in external list.
    Cross-ref with appraisal registry.
    """
    # Filtering rows where Collateral Type is 'Real Estate'
    real_estate_collateral = df[collateral_type_column] == 'Real Estate'
    
    # Checking if Appraisal ID is in the external list
    invalid_appraisal_id = real_estate_collateral & (~df[appraisal_id_column].isin(external_list))

    failed_count = invalid_appraisal_id.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 104,
        "Variable checked": f"{collateral_type_column}, {appraisal_id_column}",
        "Description": "If Collateral Type = real estate, ensure appraisal ID is in external list.",
        "Threshold / Condition": "Cross-ref with appraisal registry.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df



def check_105_LienPositionRanking(df, lien_position_column, borrower_facility_column):
    """
    Validation Check 105: If lien is senior, check facility ranking among borrower's other facilities.
    Only 1 senior lien per asset as policy states.
    """
    # Filtering rows where Lien Position is 'Senior'
    senior_lien = df[df[lien_position_column] == 'Senior']

    # Count the number of senior liens per borrower facility
    senior_lien_count = senior_lien[borrower_facility_column].value_counts()

    # Identify borrower facilities with more than one senior lien
    invalid_senior_lien = senior_lien_count[senior_lien_count > 1].index

    failed_count = df[borrower_facility_column].isin(invalid_senior_lien).sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 105,
        "Variable checked": f"{lien_position_column}, {borrower_facility_column}",
        "Description": "If lien is senior, check facility ranking among borrower's other facilities.",
        "Threshold / Condition": "Only 1 senior lien per asset as policy states.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df



def check_106_CollateralCoverageRatio(df, collateral_value_column, exposure_column, coverage_threshold):
    """
    Validation Check 106: Collateral coverage ratio = Collateral Value / Outstanding Exposure.
    If coverage < X%, flagged as under-secured.
    """
    # Calculate Collateral Coverage Ratio
    df['Collateral Coverage Ratio'] = (df[collateral_value_column].astype(float) / df[exposure_column].astype(float)) * 100
    
    # Identify under-secured cases
    under_secured = df['Collateral Coverage Ratio'] < coverage_threshold

    failed_count = under_secured.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 106,
        "Variable checked": f"{collateral_value_column}, {exposure_column}",
        "Description": "Collateral coverage ratio = Collateral Value / Outstanding Exposure.",
        "Threshold / Condition": f"If coverage < {coverage_threshold}%, flagged as under-secured.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df


def check_107_CollateralReappraisal(df, collateral_value_column, stage_column, reappraisal_required_stage=3):
    """
    Validation Check 107: If Stage=3, check if collateral is reappraised recently.
    Must have updated appraisal if default.
    """
    # Identify records where Stage = 3 (default scenario)
    default_stage = df[stage_column] == reappraisal_required_stage
    
    # Identify records where collateral value is missing or unchanged (assumed not reappraised)
    no_recent_reappraisal = df[collateral_value_column].isna() | (df[collateral_value_column].astype(float) == df[collateral_value_column].astype(float).shift(1))
    
    # Flagged cases
    failed_records = default_stage & no_recent_reappraisal
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 107,
        "Variable checked": f"{collateral_value_column}, {stage_column}",
        "Description": "If Stage=3, check if collateral is reappraised recently.",
        "Threshold / Condition": "Must have updated appraisal if default.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df


def check_108_GuaranteeActivation(df, guarantee_amount_column, default_flag_column):
    """
    Validation Check 108: If facility defaulted, confirm guarantee activation steps.
    For internal risk review (some banks track activation events).
    """
    # Identify records where the facility has defaulted (Default Flag > 0)
    defaulted_facilities = df[default_flag_column] > 0
    
    # Identify cases where Guarantee Amount is still zero or missing after default
    guarantee_not_activated = df[guarantee_amount_column].isna() | (df[guarantee_amount_column].astype(float) == 0)
    
    # Flagged cases
    failed_records = defaulted_facilities & guarantee_not_activated
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 108,
        "Variable checked": f"{guarantee_amount_column}, {default_flag_column}",
        "Description": "If facility defaulted, confirm guarantee activation steps.",
        "Threshold / Condition": "For internal risk review (some banks track activation events).",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_109_CollateralTypeDistribution(df, collateral_type_column, none_threshold):
    """
    Validation Check 109: Distribution check for Collateral Type.
    Ensure not all are 'None'.
    """
    # Count occurrences of 'None' in the Collateral Type column
    none_count = (df[collateral_type_column] == "None").sum()
    total_count = len(df)
    none_percent = (none_count / total_count) * 100

    # Flag if 'None' exceeds the defined threshold
    failed_check = none_percent > none_threshold
    failed_count = none_count if failed_check else 0

    result_df = pd.DataFrame([{
        "Check ID": 109,
        "Variable checked": collateral_type_column,
        "Description": "Distribution check: ensure not all are 'None'.",
        "Threshold / Condition": f"'None' < {none_threshold}% if bank requires security.",
        "Failed Count": failed_count,
        "% of overall data": f"{none_percent:.2f}%"
    }])

    return result_df


def check_110_CollateralIDUniquenessAcrossFacilities(df, collateral_id_column, facility_id_column):
    """
    Validation Check 110: Check that no duplicates exist across distinct facilities unless it's truly shared collateral.
    If shared, it must be documented.
    """
    # Count occurrences of each Collateral ID in different Facility IDs
    duplicate_collateral = df.groupby(collateral_id_column)[facility_id_column].nunique() > 1

    # Identify Collateral IDs that appear in multiple facilities
    duplicated_ids = duplicate_collateral[duplicate_collateral].index

    # Count failed cases
    failed_count = df[df[collateral_id_column].isin(duplicated_ids)].shape[0]
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 110,
        "Variable checked": f"{collateral_id_column}, {facility_id_column}",
        "Description": "Check that no duplicates across distinct facilities unless it’s truly shared collateral.",
        "Threshold / Condition": "If shared, must be documented.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_111_ReceivablesUpdateFrequency(df, collateral_type_column, collateral_value_column, snapshot_date_column, max_allowed_staleness_days):
    """
    Validation Check 111: If Collateral Type = receivables, confirm value is updated frequently.
    If stale > 6 months, flagged.
    """
    # Convert date columns to datetime format
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    # Identify cases where collateral type is receivables
    receivables_records = df[collateral_type_column] == "Receivables"

    # Identify cases where the collateral value has not changed in the given period
    df['Collateral Change'] = df[collateral_value_column].astype(float).diff().fillna(0)
    stale_values = receivables_records & (df['Collateral Change'] == 0)

    failed_count = stale_values.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 111,
        "Variable checked": f"{collateral_type_column}, {collateral_value_column}",
        "Description": "If Collateral Type=receivables, confirm value is updated frequently.",
        "Threshold / Condition": f"If stale > {max_allowed_staleness_days // 30} months, flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_112_PersonalGuaranteeReference(df, guarantee_type_column, borrower_relationship_column):
    """
    Validation Check 112: If Guarantee Type = personal, confirm link to an individual (director or owner).
    Must have reference in the borrower group.
    """
    # Identify cases where Guarantee Type is 'Personal'
    personal_guarantee = df[guarantee_type_column] == "Personal"

    # Identify cases where Borrower Relationship is missing
    missing_reference = df[borrower_relationship_column].isna()

    # Flagged cases
    failed_records = personal_guarantee & missing_reference
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 112,
        "Variable checked": f"{guarantee_type_column}, {borrower_relationship_column}",
        "Description": "If Guarantee Type = personal, confirm link to an individual (director or owner).",
        "Threshold / Condition": "Must have reference in the borrower group.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df


def check_113_SubordinatedGuaranteeValidation(df, lien_position_column, guarantee_type_column):
    """
    Validation Check 113: If Lien Position = Subordinated, check that Guarantee Type is properly subordinated if relevant.
    Ensures consistent legal structure.
    """
    # Identify cases where Lien Position is 'Subordinated'
    subordinated_lien = df[lien_position_column] == "Subordinated"

    # Identify cases where Guarantee Type is missing or not properly structured
    missing_or_invalid_guarantee = df[guarantee_type_column].isna()

    # Flagged cases
    failed_records = subordinated_lien & missing_or_invalid_guarantee
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 113,
        "Variable checked": f"{lien_position_column}, {guarantee_type_column}",
        "Description": "If Lien Position = Subordinated, check that Guarantee Type is properly subordinated if relevant.",
        "Threshold / Condition": "Ensures consistent legal structure.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_114_InventoryCoverageRatio(df, collateral_type_column, coverage_ratio_column, max_coverage_threshold):
    """
    Validation Check 114: If Collateral Type = inventory, check normal coverage ratio vs. total.
    Inventory coverage typically < X%.
    """
    # Identify cases where Collateral Type is 'Inventory'
    inventory_collateral = df[collateral_type_column] == "Inventory"

    # Identify cases where the coverage ratio exceeds the expected maximum threshold
    high_coverage_ratio = df[coverage_ratio_column].astype(float) > max_coverage_threshold

    # Flagged cases
    failed_records = inventory_collateral & high_coverage_ratio
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 114,
        "Variable checked": f"{collateral_type_column}, {coverage_ratio_column}",
        "Description": "If Collateral Type = inventory, check normal coverage ratio vs. total.",
        "Threshold / Condition": f"Inventory coverage typically < {max_coverage_threshold}%.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df


def check_115_CollateralValueNotZero(df, collateral_type_column, collateral_value_column):
    """
    Validation Check 115: Collateral Value must not be zero if Collateral Type is not "None".
    Ensures valid collateral reporting.
    """
    # Identify cases where Collateral Type is not "None"
    valid_collateral = df[collateral_type_column] != "None"

    # Identify cases where Collateral Value is zero or missing
    zero_or_missing_value = df[collateral_value_column].astype(float) == 0

    # Flagged cases
    failed_records = valid_collateral & zero_or_missing_value
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 115,
        "Variable checked": f"{collateral_type_column}, {collateral_value_column}",
        "Description": "Must not be zero if Collateral Type is not 'None'.",
        "Threshold / Condition": "If Type != 'None,' Value > 0.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df


def check_116_GovernmentGuaranteeCoverage(df, guarantee_type_column, guarantee_amount_column, expected_coverage_range):
    """
    Validation Check 116: If Guarantee Type = government, check coverage is typically 80-100%.
    Ensures alignment with bank policy.
    """
    # Identify cases where Guarantee Type is 'Government'
    government_guarantee = df[guarantee_type_column] == "Government"

    # Identify cases where the Guarantee Amount is outside the expected range
    outside_expected_coverage = ~df[guarantee_amount_column].between(expected_coverage_range[0], expected_coverage_range[1])

    # Flagged cases
    failed_records = government_guarantee & outside_expected_coverage
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 116,
        "Variable checked": f"{guarantee_type_column}, {guarantee_amount_column}",
        "Description": "If Guarantee Type = government, check coverage is typically 80-100%.",
        "Threshold / Condition": "Bank policy.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df


def check_117_CollateralGuaranteeSynergy(df, collateral_id_column, guarantee_column):
    """
    Validation Check 117: Check for missing synergy between collateral and guarantee.
    If Collateral ID is null, check if Guarantee is also null for an unsecured loan.
    """
    # Identify cases where Collateral ID is missing
    missing_collateral = df[collateral_id_column].isna()

    # Identify cases where Guarantee is also missing
    missing_guarantee = df[guarantee_column].isna()

    # Flagged cases where both collateral and guarantee are missing (potentially unsecured loan)
    failed_records = missing_collateral & missing_guarantee
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 117,
        "Variable checked": f"{collateral_id_column}, {guarantee_column}",
        "Description": "Check for missing synergy: if Collateral ID is null, Guarantee is also null for an unsecured loan?",
        "Threshold / Condition": "Possibly 'unsecured' if both are null.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df


def check_118_CollateralRevaluationConsistency(df, original_collateral_value_column, month_end_collateral_value_column, variation_threshold):
    """
    Validation Check 118: Compare the difference between Original Collateral Value and Month-End Collateral Value.
    If variation exceeds ±X% year-over-year, flag for review.
    """
    # Calculate the percentage difference between the original and month-end collateral value
    df['Collateral Value Change %'] = ((df[month_end_collateral_value_column].astype(float) - df[original_collateral_value_column].astype(float)) 
                                       / df[original_collateral_value_column].astype(float)) * 100

    # Identify cases where the change exceeds the threshold in either direction
    excessive_variation = df['Collateral Value Change %'].abs() > variation_threshold

    failed_count = excessive_variation.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 118,
        "Variable checked": f"{original_collateral_value_column}, {month_end_collateral_value_column}",
        "Description": "Compare the difference to see if revaluation is consistent.",
        "Threshold / Condition": f"Variation < ±{variation_threshold}% YoY.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_119_RealEstatePlausibility(df, collateral_type_column, collateral_value_column, market_value_column, max_deviation_threshold):
    """
    Validation Check 119: If Collateral Type = real estate, check standard market indices or external references for plausibility.
    Should not exceed typical local property values by a huge margin.
    """
    # Identify cases where Collateral Type is 'Real Estate'
    real_estate_collateral = df[collateral_type_column] == "Real Estate"

    # Calculate the percentage deviation from the market value
    df['Market Value Deviation %'] = ((df[collateral_value_column].astype(float) - df[market_value_column].astype(float)) 
                                      / df[market_value_column].astype(float)) * 100

    # Identify cases where the deviation is beyond the acceptable threshold
    excessive_deviation = real_estate_collateral & (df['Market Value Deviation %'].abs() > max_deviation_threshold)

    failed_count = excessive_deviation.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 119,
        "Variable checked": f"{collateral_type_column}, {collateral_value_column}, {market_value_column}",
        "Description": "If real estate, check standard market indices or external references for plausibility.",
        "Threshold / Condition": f"Should not exceed typical local property values by > {max_deviation_threshold}%.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df


def check_120_CorporateGuaranteeTrackability(df, guarantee_type_column, corporate_id_column):
    """
    Validation Check 120: If Guarantee Type = corporate, ensure corporate ID is trackable.
    Must reference the guaranteeing entity.
    """
    # Identify cases where Guarantee Type is 'Corporate'
    corporate_guarantee = df[guarantee_type_column] == "Corporate"

    # Identify cases where Corporate ID is missing
    missing_corporate_id = df[corporate_id_column].isna()

    # Flagged cases where corporate guarantee exists but no corporate ID is provided
    failed_records = corporate_guarantee & missing_corporate_id
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 120,
        "Variable checked": f"{guarantee_type_column}, {corporate_id_column}",
        "Description": "If Guarantee Type = corporate, ensure corporate ID is trackable.",
        "Threshold / Condition": "Must reference the guaranteeing entity.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df


def section_d_checks(core_data):
    """Run all 30 checks for Section A and return a combined results table."""

    # Run each check individually (Ensure all check functions are defined)
    checks = [
        check_91_CollateralID(core_data,'Collateral ID / Reference'),
        check_92_CollateralType(core_data,'Collateral Type', 
                                {"Real Estate", "Inventory", "Receivables", "Equipment", "Vehicles", "Cash", "Securities", "Guarantees", "Intellectual Property"}),
        check_93_OriginalCollateralValue(core_data, 'Original Collateral Value'),
        check_94_CollateralValueMonthEnd(core_data, 'Collateral Value (Month-End)'),
        check_95_CollateralValueFacilityStatus(core_data, 'Collateral Value (Month-End)', 'Facility End Date'),
        check_96_GuaranteeType(core_data, 'Guarantee Type', {"Corporate Guarantee", "Bank Guarantee", "Personal Guarantee", "Performance Guarantee", "Financial Guarantee", "Payment Guarantee", "Bid Bond Guarantee", "Advance Payment Guarantee", "Standby Letter of Credit (SBLC)", "Letter of Credit (LC)", "Surety Bond", "Parent Company Guarantee", "Supplier Guarantee", "Export Credit Guarantee", "Import Guarantee", "Retention Money Guarantee", "Customs Bond Guarantee", "Credit Guarantee", "Mortgage Guarantee", "Government Guarantee", "Third-Party Guarantee", "Completion Guarantee", "Deferred Payment Guarantee", "Trade Credit Guarantee", "Lease Guarantee", "Loan Guarantee", "Tax Guarantee", "Contract Performance Bond", "Insurance Guarantee", "Factoring Guarantee"}),
        check_97_GuaranteeAmount(core_data, 'Guarantee Amount', 'Outstanding Exposure'),
        check_98_LienPosition(core_data, 'Lien Position', {"Senior", "Junior", "Subordinated"}),
        check_99_CollateralIDUniqueness(core_data, 'Collateral ID / Reference'),
        check_100_CollateralAndGuarantee(core_data, 'Collateral Type', 'Guarantee Type'),
        check_101_CollateralValueTrend(core_data, 'Collateral Value (Month-End)'),
        # check_102_CollateralValueConsistency(core_data, 'Original Collateral Value', 'Collateral Value (Month-End)'),
        check_103_GuaranteeTypePartial(core_data, 'Guarantee Type', 'Guarantee Amount', 'Facility Limit'),
        check_104_CollateralAppraisalID(core_data, 'Collateral Type', 'Appraisal ID', {"APP123", "APP456", "APP789"}),
        check_105_LienPositionRanking(core_data, 'Lien Position', 'Facility ID'),
        check_106_CollateralCoverageRatio(core_data, 'Collateral Value (Month-End)', 'Outstanding Exposure', 50),
        check_107_CollateralReappraisal(core_data, 'Collateral Value (Month-End)', 'Stage (IFRS 9)'),
        # check_108_GuaranteeActivation(core_data, 'Guarantee Amount', 'Default Flag'),
        check_109_CollateralTypeDistribution(core_data, 'Collateral Type', none_threshold=50),
        check_110_CollateralIDUniquenessAcrossFacilities(core_data, 'Collateral ID / Reference', 'Facility ID'),
        check_111_ReceivablesUpdateFrequency(core_data, 'Collateral Type', 'Collateral Value (Month-End)', 'Snapshot Date', max_allowed_staleness_days=60*30),
        check_112_PersonalGuaranteeReference(core_data, 'Guarantee Type', 'Borrower Relationship'),
        check_113_SubordinatedGuaranteeValidation(core_data, 'Lien Position', 'Guarantee Type'),
        check_114_InventoryCoverageRatio(core_data, 'Collateral Type', 'Coverage Ratio', max_coverage_threshold = 50),
        check_115_CollateralValueNotZero(core_data, 'Collateral Type', 'Collateral Value (Month-End)'),
        check_116_GovernmentGuaranteeCoverage(core_data, 'Guarantee Type', 'Guarantee Amount', expected_coverage_range = (80, 100)),
        check_117_CollateralGuaranteeSynergy(core_data, 'Collateral ID / Reference', 'Guarantee Type'),
        check_118_CollateralRevaluationConsistency(core_data, 'Original Collateral Value', 'Collateral Value (Month-End)', variation_threshold = 10),
        check_119_RealEstatePlausibility(core_data, 'Collateral Type', 'Collateral Value (Month-End)', 'Market Value', max_deviation_threshold = 20),
        check_120_CorporateGuaranteeTrackability(core_data, 'Guarantee Type', 'Corporate ID')
    ]

    processed_checks = []
    for check in checks:
        if isinstance(check, dict):  # Convert dict to DataFrame
            processed_checks.append(pd.DataFrame([check]))
        elif isinstance(check, pd.DataFrame):
            processed_checks.append(check)

    # Combine all check results into a single DataFrame
    combined_results = pd.concat(processed_checks, ignore_index=True)

    combined_results = combined_results.drop('Threshold / Condition', axis = 1)
    

    return combined_results
