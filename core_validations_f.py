import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def check_151_MissingDaysPastDue(df, days_past_due_column, facility_status_column):
    """
    Validation Check 151: Check missing values in Days Past Due (Month-End).
    Ensures that active facilities have non-null values.
    """
    # Identify active facilities (assuming non-null Facility Status means active)
    active_facilities = df[facility_status_column].notna()

    # Identify cases where Days Past Due is missing
    missing_days_past_due = df[days_past_due_column].isna()

    # Flagged cases where active facilities have missing Days Past Due
    failed_records = active_facilities & missing_days_past_due
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 151,
        "Variable checked": f"{days_past_due_column}, {facility_status_column}",
        "Description": "Check missing values in Days Past Due (Month-End).",
        "Threshold / Condition": "Must not be null for active facilities.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_152_NonNegativeDaysPastDue(df, days_past_due_column):
    """
    Validation Check 152: Ensure Days Past Due (Month-End) is ≥ 0.
    Flags negative values as invalid.
    """
    # Identify cases where Days Past Due is negative
    negative_days_past_due = df[days_past_due_column].astype(float) < 0

    failed_count = negative_days_past_due.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 152,
        "Variable checked": days_past_due_column,
        "Description": "Ensure Days Past Due (Month-End) is ≥ 0.",
        "Threshold / Condition": "Negative days flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_153_MissingLatePayments(df, late_payments_column, missing_threshold):
    """
    Validation Check 153: Check missing values in Number of Late Payments (Last 12 Months).
    Flags if more than a certain percentage of records are missing.
    """
    # Identify cases where Number of Late Payments is missing
    missing_late_payments = df[late_payments_column].isna()

    failed_count = missing_late_payments.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    # Check if the missing percentage exceeds the threshold
    fail_flag = failed_percent > missing_threshold

    result_df = pd.DataFrame([{
        "Check ID": 153,
        "Variable checked": late_payments_column,
        "Description": "Check missing values in Number of Late Payments (Last 12 Months).",
        "Threshold / Condition": f"No missing > {missing_threshold}%.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_154_LatePaymentsWithDefault(df, late_payments_column, default_flag_column):
    """
    Validation Check 154: If Default Flag = 1, must have ≥ 1 late payment.
    Ensures consistency between default status and late payment records.
    """
    # Identify cases where Default Flag is set to 1 (indicating default)
    defaulted_accounts = df[default_flag_column] == 1

    # Identify cases where the number of late payments is zero or missing
    no_late_payments = (df[late_payments_column].isna()) | (df[late_payments_column].astype(float) < 1)

    # Flagged cases where default exists, but no late payments are recorded
    failed_records = defaulted_accounts & no_late_payments
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 154,
        "Variable checked": f"{late_payments_column}, {default_flag_column}",
        "Description": "If Default Flag=1, must have ≥ 1 late payment.",
        "Threshold / Condition": "Consistency rule.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_155_InterestSuspenseVsPerformingFacility(df, interest_suspense_column, stage_column):
    """
    Validation Check 155: If facility is performing (Stage=1), interest in suspense must be 0.
    Flags contradictions where a performing facility has interest in suspense.
    """
    # Identify cases where the facility is performing (Stage = 1)
    performing_facilities = df[stage_column] == 1

    # Identify cases where Interest in Suspense is non-zero or not null
    interest_in_suspense = df[interest_suspense_column].notna() & (df[interest_suspense_column].astype(float) > 0)

    # Flagged cases where performing facilities have interest in suspense
    failed_records = performing_facilities & interest_in_suspense
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 155,
        "Variable checked": f"{interest_suspense_column}, {stage_column}",
        "Description": "If facility is performing (Stage=1), interest in suspense must be 0.",
        "Threshold / Condition": "Contradiction flagged otherwise.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_156_RestructuringFlagValidity(df, restructuring_flag_column):
    """
    Validation Check 156: Ensure Restructuring/Forbearance Flag contains only True/False values.
    Flags cases where the values are not boolean.
    """
    # Identify cases where the Restructuring Flag is not True or False
    invalid_values = ~df[restructuring_flag_column].isin([True, False, 1, 0])  # Accepts boolean or binary (0/1)

    failed_count = invalid_values.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 156,
        "Variable checked": restructuring_flag_column,
        "Description": "Restructuring / Forbearance Flag must be True/False.",
        "Threshold / Condition": "Must be True/False.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_157_RestructuringFlagVsIFRS9Stage(df, restructuring_flag_column, stage_column):
    """
    Validation Check 157: If Restructuring / Forbearance Flag is True, confirm IFRS 9 Stage is ≥ 2.
    Ensures that restructured loans are not classified as Stage 1.
    """
    # Identify cases where the Restructuring Flag is set (assuming 1 or True indicates restructuring)
    restructured_accounts = df[restructuring_flag_column].isin([True, 1])

    # Identify cases where IFRS 9 Stage is incorrectly set to 1
    incorrect_stage = df[stage_column] == 1

    # Flagged cases where restructuring exists but IFRS 9 Stage remains at 1
    failed_records = restructured_accounts & incorrect_stage
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 157,
        "Variable checked": f"{restructuring_flag_column}, {stage_column}",
        "Description": "If Restructuring / Forbearance Flag is True, confirm IFRS 9 Stage is ≥ 2.",
        "Threshold / Condition": "IFRS 9 logic.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_158_DefaultFlagValidity(df, default_flag_column):
    """
    Validation Check 158: Default Flag must be in {0, 1}.
    Ensures only valid binary values are present in the Default Flag column.
    """
    # Identify cases where Default Flag is not 0 or 1
    invalid_values = ~df[default_flag_column].isin([0, 1])

    failed_count = invalid_values.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 158,
        "Variable checked": default_flag_column,
        "Description": "Default Flag must be in {0, 1}.",
        "Threshold / Condition": "No other values allowed.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_159_FinalDefaultFlagConsistency(df, final_default_flag_column, default_date_column, months_threshold=12):
    """
    Validation Check 159: If default is triggered, future default flags might remain 1 up to 12 months.
    Ensures consistency of the default flag with the timeline.
    """
    # Convert date columns to datetime format
    df[default_date_column] = pd.to_datetime(df[default_date_column], errors='coerce')

    # Calculate the number of months between Default Date and today
    df['Months Since Default'] = ((pd.to_datetime('today') - df[default_date_column]).dt.days / 30).astype(float)

    # Identify cases where Final Default Flag is still 1 beyond the allowed timeframe
    inconsistent_default_flag = (df[final_default_flag_column] == 1) & (df['Months Since Default'] > months_threshold)

    failed_count = inconsistent_default_flag.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 159,
        "Variable checked": f"{final_default_flag_column}, {default_date_column}",
        "Description": "If default is triggered, future default flags might remain 1 up to 12 months.",
        "Threshold / Condition": "Validate consistency with timeline.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_160_DefaultDateRequired(df, default_flag_column, default_date_column):
    """
    Validation Check 160: If Default Flag = 1, Default Date cannot be null.
    Ensures that every defaulted account has a corresponding default date.
    """
    # Identify cases where Default Flag is set to 1 (indicating default)
    defaulted_accounts = df[default_flag_column] == 1

    # Identify cases where Default Date is missing
    missing_default_date = df[default_date_column].isna()

    # Flagged cases where Default Flag = 1 but Default Date is missing
    failed_records = defaulted_accounts & missing_default_date
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 160,
        "Variable checked": f"{default_flag_column}, {default_date_column}",
        "Description": "If Default Flag=1, Default Date can’t be null.",
        "Threshold / Condition": "Must have default date.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_161_ReasonForDefaultRequired(df, default_flag_column, reason_for_default_column):
    """
    Validation Check 161: If Default Flag = 1, Reason for Default cannot be null.
    Ensures that every defaulted account has a corresponding reason.
    """
    # Identify cases where Default Flag is set to 1 (indicating default)
    defaulted_accounts = df[default_flag_column] == 1

    # Identify cases where Reason for Default is missing
    missing_reason = df[reason_for_default_column].isna()

    # Flagged cases where Default Flag = 1 but Reason for Default is missing
    failed_records = defaulted_accounts & missing_reason
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 161,
        "Variable checked": f"{default_flag_column}, {reason_for_default_column}",
        "Description": "If Default Flag=1, Reason for Default can’t be null.",
        "Threshold / Condition": "Must provide reason.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_162_CureDateVsDefaultDate(df, cure_date_column, default_date_column):
    """
    Validation Check 162: Cure Date must be >= Default Date if present.
    Ensures that the cure date does not occur before the recorded default date.
    """
    # Convert date columns to datetime format
    df[cure_date_column] = pd.to_datetime(df[cure_date_column], errors='coerce')
    df[default_date_column] = pd.to_datetime(df[default_date_column], errors='coerce')

    # Identify cases where Cure Date is earlier than Default Date
    invalid_cure_date = df[cure_date_column] < df[default_date_column]

    failed_count = invalid_cure_date.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 162,
        "Variable checked": f"{cure_date_column}, {default_date_column}",
        "Description": "Cure Date must be >= Default Date if present.",
        "Threshold / Condition": "No negative durations.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_163_RecoveryRateRange(df, recovery_rate_column):
    """
    Validation Check 163: If facility defaulted and workout complete, Recovery Rate must be between 0 and 1.
    Flags cases where the recovery rate is outside the valid range.
    """
    # Identify cases where Recovery Rate is outside the valid range [0, 1]
    invalid_recovery_rate = (df[recovery_rate_column] < 0) | (df[recovery_rate_column] > 1)

    failed_count = invalid_recovery_rate.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 163,
        "Variable checked": recovery_rate_column,
        "Description": "If facility defaulted and workout complete, Recovery Rate must be between 0 and 1.",
        "Threshold / Condition": "0 ≤ Recovery Rate ≤ 1.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_164_WorkoutResolutionTypeRequired(df, default_flag_column, resolution_type_column):
    """
    Validation Check 164: If facility is defaulted, it must have a resolution type once resolved.
    Flags cases where a defaulted facility does not have a resolution type.
    """
    # Identify cases where Default Flag is set to 1 (indicating default)
    defaulted_accounts = df[default_flag_column] == 1

    # Identify cases where Workout/Resolution Type is missing
    missing_resolution_type = df[resolution_type_column].isna()

    # Flagged cases where Default Flag = 1 but Workout/Resolution Type is missing
    failed_records = defaulted_accounts & missing_resolution_type
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 164,
        "Variable checked": f"{default_flag_column}, {resolution_type_column}",
        "Description": "If facility is defaulted, it must have a resolution type once resolved.",
        "Threshold / Condition": "No missing if default resolved.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_165_AccountTurnoverPattern(df, account_turnover_column, facility_status_column):
    """
    Validation Check 165: If the facility is operational, it must have some account turnover/payment pattern data.
    Flags cases where an active facility has missing or zero account turnover.
    """
    # Identify active facilities (assuming non-null Facility Status means operational)
    active_facilities = df[facility_status_column].notna()

    # Identify cases where Account Turnover is missing or zero
    missing_or_zero_turnover = df[account_turnover_column].isna() | (df[account_turnover_column].astype(float) == 0)

    # Flagged cases where active facilities have no account turnover
    failed_records = active_facilities & missing_or_zero_turnover
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 165,
        "Variable checked": f"{account_turnover_column}, {facility_status_column}",
        "Description": "If the facility is operational, it must have some account turnover/payment pattern data.",
        "Threshold / Condition": "Must have some pattern data.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_166_DaysPastDueVsIFRS9Stage(df, days_past_due_column, stage_column, overdue_threshold):
    """
    Validation Check 166: If Days Past Due > 90, typically IFRS 9 Stage should be 3.
    Flags cases where past due days exceed the threshold but stage is not correctly classified.
    """
    # Identify cases where Days Past Due exceeds the threshold
    overdue_accounts = df[days_past_due_column].astype(float) > overdue_threshold

    # Identify cases where IFRS 9 Stage is not 3 despite overdue status
    incorrect_stage = df[stage_column] != 3

    # Flagged cases where past due days exceed 90 but stage is incorrect
    failed_records = overdue_accounts & incorrect_stage
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 166,
        "Variable checked": f"{days_past_due_column}, {stage_column}",
        "Description": "If Days Past Due > 90, typically Stage=3.",
        "Threshold / Condition": "Flag mismatch.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_167_DefaultFlagVsIFRS9Stage(df, default_flag_column, stage_column):
    """
    Validation Check 167: If Default Flag = 1, IFRS 9 Stage must be 3.
    Ensures consistency between default status and IFRS 9 classification.
    """
    # Identify cases where Default Flag is set to 1 (indicating default)
    defaulted_accounts = df[default_flag_column] == 1

    # Identify cases where IFRS 9 Stage is not 3 despite default flag
    incorrect_stage = df[stage_column] != 3

    # Flagged cases where Default Flag = 1 but Stage is incorrect
    failed_records = defaulted_accounts & incorrect_stage
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 167,
        "Variable checked": f"{default_flag_column}, {stage_column}",
        "Description": "If Default Flag=1, Stage=3.",
        "Threshold / Condition": "Hard rule.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_168_LatePaymentsVsIFRS9Stage(df, late_payments_column, stage_column, late_payment_threshold):
    """
    Validation Check 168: If Number of Late Payments exceeds the threshold (e.g., > 3), 
    then Stage should be at least 2 unless bank policy differs.
    Flags cases where excessive late payments exist, but the IFRS 9 Stage remains 1.
    """
    # Identify cases where Number of Late Payments exceeds the threshold
    excessive_late_payments = df[late_payments_column].astype(float) > late_payment_threshold

    # Identify cases where IFRS 9 Stage is still 1 despite multiple late payments
    incorrect_stage = df[stage_column] == 1

    # Flagged cases where late payments exceed the threshold but stage remains incorrect
    failed_records = excessive_late_payments & incorrect_stage
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 168,
        "Variable checked": f"{late_payments_column}, {stage_column}",
        "Description": f"If > {late_payment_threshold} late payments, consider Stage=2 unless bank policy differs.",
        "Threshold / Condition": "Potential mismatch flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_169_RestructuringVsReasonForDefault(df, restructuring_flag_column, reason_for_default_column):
    """
    Validation Check 169: If Reason for Default is 'Restructured,' then the Forbearance/Restructuring Flag should be True.
    Ensures consistency between restructuring classification and default reason.
    """
    # Identify cases where Reason for Default is "Restructured"
    restructured_reason = df[reason_for_default_column].str.lower() == "restructured"

    # Identify cases where Restructuring/Forbearance Flag is not set to True
    incorrect_flag = ~df[restructuring_flag_column].isin([True, 1])

    # Flagged cases where reason is 'Restructured' but flag is incorrect
    failed_records = restructured_reason & incorrect_flag
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 169,
        "Variable checked": f"{restructuring_flag_column}, {reason_for_default_column}",
        "Description": "If reason='Restructured,' then forbearance should be True.",
        "Threshold / Condition": "Cross-check.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_170_DefaultDateVsFacilityEndDate(df, default_date_column, facility_end_date_column):
    """
    Validation Check 170: If Facility End Date < Default Date, it is contradictory.
    Ensures that the facility end date does not precede the recorded default date.
    """
    # Convert date columns to datetime format
    df[default_date_column] = pd.to_datetime(df[default_date_column], errors='coerce')
    df[facility_end_date_column] = pd.to_datetime(df[facility_end_date_column], errors='coerce')

    # Identify cases where Facility End Date is earlier than Default Date
    invalid_end_date = df[facility_end_date_column] < df[default_date_column]

    failed_count = invalid_end_date.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 170,
        "Variable checked": f"{default_date_column}, {facility_end_date_column}",
        "Description": "If End Date < Default Date, contradictory.",
        "Threshold / Condition": "End date should not precede default date.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_171_DefaultFlagVsCureDate(df, default_flag_column, cure_date_column, months_threshold=12):
    """
    Validation Check 171: If Cure Date is present, eventually Default Flag should revert to 0 in subsequent months.
    Ensures that the default flag is updated after a cure date.
    """
    # Convert date columns to datetime format
    df[cure_date_column] = pd.to_datetime(df[cure_date_column], errors='coerce')

    # Calculate the number of months since the cure date
    df['Months Since Cure'] = ((pd.to_datetime('today') - df[cure_date_column]).dt.days / 30).astype(float)

    # Identify cases where Default Flag is still 1 beyond the allowed timeframe after cure
    inconsistent_default_flag = (df[default_flag_column] == 1) & (df['Months Since Cure'] > months_threshold)

    failed_count = inconsistent_default_flag.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 171,
        "Variable checked": f"{default_flag_column}, {cure_date_column}",
        "Description": "If Cure Date is present, eventually Default Flag should revert to 0 in subsequent months.",
        "Threshold / Condition": "Timeline check.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_172_RecoveryRateVsCollateralValue(df, recovery_rate_column, collateral_value_column, high_collateral_threshold, expected_recovery_threshold):
    """
    Validation Check 172: If Collateral Coverage is high, expect a higher Recovery Rate.
    Flags cases where high collateral does not correspond with expected recovery.
    """
    # Identify cases where Collateral Value is high
    high_collateral = df[collateral_value_column].astype(float) > high_collateral_threshold

    # Identify cases where Recovery Rate is lower than expected despite high collateral
    low_recovery_rate = df[recovery_rate_column].astype(float) < expected_recovery_threshold

    # Flagged cases where collateral is high but recovery rate is unexpectedly low
    failed_records = high_collateral & low_recovery_rate
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 172,
        "Variable checked": f"{recovery_rate_column}, {collateral_value_column}",
        "Description": "If Collateral Coverage is high, expect a higher Recovery Rate.",
        "Threshold / Condition": "Check for mismatch.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_173_DefaultFlagVsLastPaymentDate(df, default_flag_column, last_payment_date_column, days_past_due_column, dpd_threshold):
    """
    Validation Check 173: If the last payment was well before default, check if Days Past Due (DPD) exceeds the threshold.
    Ensures logical consistency between last payment date and default status.
    """
    # Convert date columns to datetime format
    df[last_payment_date_column] = pd.to_datetime(df[last_payment_date_column], errors='coerce')

    # Identify cases where Default Flag is set to 1 (indicating default)
    defaulted_accounts = df[default_flag_column] == 1

    # Identify cases where Days Past Due is below the expected threshold
    dpd_below_threshold = df[days_past_due_column].astype(float) < dpd_threshold

    # Flagged cases where default is recorded, but DPD does not support the default claim
    failed_records = defaulted_accounts & dpd_below_threshold
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 173,
        "Variable checked": f"{default_flag_column}, {last_payment_date_column}, {days_past_due_column}",
        "Description": "If last payment was well before default, check DPD > threshold.",
        "Threshold / Condition": "Logical consistency.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_174_AbnormalDropInTurnover(df, account_turnover_column, threshold_drop_percentage):
    """
    Validation Check 174: If there is an abnormal drop in account turnover, it might indicate looming default.
    Flags cases where turnover decreases significantly beyond a threshold.
    """
    # Calculate percentage change in account turnover
    df['Turnover Change (%)'] = df[account_turnover_column].pct_change() * 100

    # Identify cases where turnover has dropped beyond the threshold
    abnormal_drop = df['Turnover Change (%)'] < -threshold_drop_percentage

    failed_count = abnormal_drop.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 174,
        "Variable checked": account_turnover_column,
        "Description": "If abnormal drop in turnover, might indicate looming default.",
        "Threshold / Condition": f"Investigate outliers if drop > {threshold_drop_percentage}%.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_175_RestructuringVsInterestSuspense(df, restructuring_flag_column, interest_suspense_column):
    """
    Validation Check 175: If loan is restructured, interest might be partially in suspense.
    Flags cases where restructuring is marked, but interest in suspense is missing or inconsistent.
    """
    # Identify cases where the loan is marked as restructured (Restructuring Flag = True or 1)
    restructured_accounts = df[restructuring_flag_column].isin([True, 1])

    # Identify cases where Interest in Suspense is zero or missing despite restructuring
    no_interest_suspense = df[interest_suspense_column].isna() | (df[interest_suspense_column].astype(float) == 0)

    # Flagged cases where restructuring exists but interest in suspense is not recorded
    failed_records = restructured_accounts & no_interest_suspense
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 175,
        "Variable checked": f"{restructuring_flag_column}, {interest_suspense_column}",
        "Description": "If restructured, possibly partial suspense.",
        "Threshold / Condition": "Check consistency.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_176_DaysPastDueVsReasonForDefault(df, days_past_due_column, reason_for_default_column, dpd_threshold):
    """
    Validation Check 176: If reason for default is 'Missed Payment,' ensure Days Past Due (DPD) is at least the threshold.
    Flags cases where the reason indicates missed payments, but DPD is too low.
    """
    # Identify cases where Reason for Default is "Missed Payment"
    missed_payment_reason = df[reason_for_default_column].str.lower() == "missed payment"

    # Identify cases where Days Past Due is below the expected threshold
    dpd_below_threshold = df[days_past_due_column].astype(float) < dpd_threshold

    # Flagged cases where reason is 'Missed Payment' but DPD does not meet the threshold
    failed_records = missed_payment_reason & dpd_below_threshold
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 176,
        "Variable checked": f"{days_past_due_column}, {reason_for_default_column}",
        "Description": "If reason='Missed Payment,' ensure DPD ≥ threshold.",
        "Threshold / Condition": f"Mismatch flagged if DPD < {dpd_threshold}.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_177_DefaultDateVsCureDate(df, default_date_column, cure_date_column, max_gap_months):
    """
    Validation Check 177: If Cure Date is present, the maximum gap allowed between Default Date and Cure Date should be within the business rule.
    Flags cases where the cure period exceeds the allowed threshold.
    """
    # Convert date columns to datetime format
    df[default_date_column] = pd.to_datetime(df[default_date_column], errors='coerce')
    df[cure_date_column] = pd.to_datetime(df[cure_date_column], errors='coerce')

    # Calculate the number of months between Default Date and Cure Date
    df['Months Between Default and Cure'] = ((df[cure_date_column] - df[default_date_column]).dt.days / 30).astype(float)

    # Identify cases where the gap exceeds the maximum allowed months
    excessive_gap = df['Months Between Default and Cure'] > max_gap_months

    failed_count = excessive_gap.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 177,
        "Variable checked": f"{default_date_column}, {cure_date_column}",
        "Description": "Maximum gap allowed for standard cure period? (business rule).",
        "Threshold / Condition": f"If gap > {max_gap_months} months, investigate.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_178_RecoveryRateVsWorkoutType(df, recovery_rate_column, workout_type_column):
    """
    Validation Check 178: If the workout type is liquidation, ensure the recovery rate is within the expected range (0 < RR < 1) or final.
    Flags cases where recovery rate is outside this range unless it's an over-recovery scenario.
    """
    # Identify cases where the Workout Type is "Liquidation"
    liquidation_cases = df[workout_type_column].str.lower() == "liquidation"

    # Identify cases where Recovery Rate is outside the valid range (0 < RR < 1)
    invalid_recovery_rate = (df[recovery_rate_column] <= 0) | (df[recovery_rate_column] > 1)

    # Flagged cases where workout type is liquidation but recovery rate is invalid
    failed_records = liquidation_cases & invalid_recovery_rate
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 178,
        "Variable checked": f"{recovery_rate_column}, {workout_type_column}",
        "Description": "If workout type=liquidation, ensure 0 < RR < 1 or final.",
        "Threshold / Condition": "No >1 unless it’s an over-recovery scenario.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_179_FinalDefaultFlagVsCureDate(df, final_default_flag_column, cure_date_column, months_threshold=12):
    """
    Validation Check 179: If Cure Date occurs, the final default flag in the next 12 months might revert to 0.
    Ensures that the final default flag is updated correctly post-cure.
    """
    # Convert date columns to datetime format
    df[cure_date_column] = pd.to_datetime(df[cure_date_column], errors='coerce')

    # Calculate the number of months since the cure date
    df['Months Since Cure'] = ((pd.to_datetime('today') - df[cure_date_column]).dt.days / 30).astype(float)

    # Identify cases where Final Default Flag is still 1 beyond the allowed timeframe after cure
    inconsistent_default_flag = (df[final_default_flag_column] == 1) & (df['Months Since Cure'] > months_threshold)

    failed_count = inconsistent_default_flag.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 179,
        "Variable checked": f"{final_default_flag_column}, {cure_date_column}",
        "Description": "If Cure Date occurs, final default flag in next 12 months might revert to 0.",
        "Threshold / Condition": "Cross-check timeline.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_180_RestructuredLoanConcentration(df, restructuring_flag_column, segment_column, threshold_percentage):
    """
    Validation Check 180: Compare the volume of restructured loans across segments.
    Flags segments with an unusually high concentration of restructured loans.
    """
    # Calculate the total number of loans
    total_loans = len(df)

    # Count the number of restructured loans per segment
    restructured_loans_per_segment = df[df[restructuring_flag_column].isin([True, 1])].groupby(segment_column).size()

    # Calculate the percentage of restructured loans per segment
    restructured_percentage = (restructured_loans_per_segment / total_loans) * 100

    # Identify segments where restructured loans exceed the threshold
    unusual_concentration = restructured_percentage > threshold_percentage

    # Prepare the results
    result_df = pd.DataFrame({
        "Segment": restructured_percentage.index,
        "Restructured Loan Percentage": restructured_percentage.values,
        "Flagged": unusual_concentration.values
    })

    # Count failed cases
    failed_count = unusual_concentration.sum()
    total_segments = len(restructured_percentage)
    failed_percent = (failed_count / total_segments) * 100

    summary_df = pd.DataFrame([{
        "Check ID": 180,
        "Variable checked": f"{restructuring_flag_column}, {segment_column}",
        "Description": "Compare volume of restructured loans across segments.",
        "Threshold / Condition": "No unusual concentration.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return summary_df

def section_f_checks(core_data):
    """Run all 30 checks for Section A and return a combined results table."""

    # Run each check individually (Ensure all check functions are defined)
    checks = [
        check_151_MissingDaysPastDue(core_data, 'Days Past Due (Month-End)', 'Facility Status'),
        check_152_NonNegativeDaysPastDue(core_data, 'Days Past Due (Month-End)'),
        check_153_MissingLatePayments(core_data, 'Number of Late Payments (Last 12 Months)', missing_threshold = 5),
        check_154_LatePaymentsWithDefault(core_data, 'Number of Late Payments (Last 12 Months)', 'Default Flag'),
        check_155_InterestSuspenseVsPerformingFacility(core_data, 'Interest in Suspense', 'Stage (IFRS 9)'),
        check_156_RestructuringFlagValidity(core_data, 'Restructuring / Forbearance Flag'),
        check_157_RestructuringFlagVsIFRS9Stage(core_data, 'Restructuring / Forbearance Flag', 'Stage (IFRS 9)'),
        check_158_DefaultFlagValidity(core_data, 'Default Flag'),
        check_159_FinalDefaultFlagConsistency(core_data, 'Final Default Flag (12 months)', 'Default Date', months_threshold = 12),
        check_160_DefaultDateRequired(core_data, 'Default Flag', 'Default Date'),
        check_161_ReasonForDefaultRequired(core_data, 'Default Flag', 'Reason for Default'),
        check_162_CureDateVsDefaultDate(core_data, 'Cure Date', 'Default Date'),
        check_163_RecoveryRateRange(core_data, 'Recovery Rate'),
        check_164_WorkoutResolutionTypeRequired(core_data, 'Default Flag', 'Workout / Resolution Type'),
        # check_165_AccountTurnoverPattern(core_data, 'Account Turnover / Payment Patterns', 'Facility Status'),
        check_166_DaysPastDueVsIFRS9Stage(core_data, 'Days Past Due (Month-End)', 'Stage (IFRS 9)', overdue_threshold = 90),
        check_167_DefaultFlagVsIFRS9Stage(core_data, 'Default Flag', 'Stage (IFRS 9)'),
        check_168_LatePaymentsVsIFRS9Stage(core_data, 'Number of Late Payments (Last 12 Months)', 'Stage (IFRS 9)', late_payment_threshold = 3),
        # check_169_RestructuringVsReasonForDefault(core_data, 'Restructuring / Forbearance Flag', 'Reason for Default'),
        check_170_DefaultDateVsFacilityEndDate(core_data, 'Default Date', 'Facility End Date'),
        check_171_DefaultFlagVsCureDate(core_data, 'Default Flag', 'Cure Date', months_threshold = 12),
        check_172_RecoveryRateVsCollateralValue(core_data, 'Recovery Rate', 'Collateral Value (Month-End)', high_collateral_threshold = 100000, expected_recovery_threshold = 0.5),
        check_173_DefaultFlagVsLastPaymentDate(core_data, 'Default Flag', 'Date of Last Payment', 'Days Past Due (Month-End)', dpd_threshold = 90),
        # check_174_AbnormalDropInTurnover(core_data, 'Account Turnover / Payment Patterns', threshold_drop_percentage = 50),
        check_175_RestructuringVsInterestSuspense(core_data, 'Restructuring / Forbearance Flag', 'Interest in Suspense'),
        # check_176_DaysPastDueVsReasonForDefault(core_data, 'Days Past Due (Month-End)', 'Reason for Default', dpd_threshold = 30),
        check_177_DefaultDateVsCureDate(core_data, 'Default Date', 'Cure Date', max_gap_months = 12),
        check_178_RecoveryRateVsWorkoutType(core_data, 'Recovery Rate', 'Workout / Resolution Type'),
        check_179_FinalDefaultFlagVsCureDate(core_data, 'Final Default Flag (12 months)', 'Cure Date', months_threshold = 12),
        check_180_RestructuredLoanConcentration(core_data, 'Restructuring / Forbearance Flag', 'Segment', threshold_percentage = 20)
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