import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def check_121_InternalCreditRatingMissing(df, credit_rating_column, facility_status_column):
    """
    Validation Check 121: Internal Credit Rating must not be null for active facilities.
    Ensures ratings are present for active credit exposures.
    """
    # Identify active facilities (assuming non-null Facility Status means active)
    active_facilities = df[facility_status_column].notna()

    # Identify cases where Internal Credit Rating is missing
    missing_credit_rating = df[credit_rating_column].isna()

    # Flagged cases
    failed_records = active_facilities & missing_credit_rating
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 121,
        "Variable checked": f"{credit_rating_column}, {facility_status_column}",
        "Description": "Missing check for Internal Credit Rating.",
        "Threshold / Condition": "Must not be null for active facilities.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_122_ValidCreditRating(df, credit_rating_column, valid_ratings):
    """
    Validation Check 122: Internal Credit Rating must be within the bank’s rating scale.
    Invalid ratings are flagged.
    """
    # Identify cases where the credit rating is not in the valid rating set
    invalid_ratings = ~df[credit_rating_column].isin(valid_ratings)

    failed_count = invalid_ratings.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 122,
        "Variable checked": credit_rating_column,
        "Description": "Must be within the bank’s rating scale (e.g., 1–10 or AAA–D).",
        "Threshold / Condition": "Invalid ratings flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_123_RatingEffectiveDateMissing(df, rating_effective_date_column, credit_rating_column):
    """
    Validation Check 123: Rating Effective Date must exist if a rating is provided.
    Ensures rating records are complete.
    """
    # Identify cases where a rating is provided but the effective date is missing
    rating_provided = df[credit_rating_column].notna()
    missing_effective_date = df[rating_effective_date_column].isna()

    # Flagged cases
    failed_records = rating_provided & missing_effective_date
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 123,
        "Variable checked": f"{rating_effective_date_column}, {credit_rating_column}",
        "Description": "Rating Effective Date must exist if a rating is provided.",
        "Threshold / Condition": "Missing rating dates flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_124_RatingEffectiveDateNotFuture(df, rating_effective_date_column, snapshot_date_column):
    """
    Validation Check 124: Rating Effective Date must be <= Snapshot Date.
    No future rating dates allowed.
    """
    # Convert date columns to datetime format
    df[rating_effective_date_column] = pd.to_datetime(df[rating_effective_date_column], errors='coerce')
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    # Identify cases where Rating Effective Date is in the future (after Snapshot Date)
    future_dates = df[rating_effective_date_column] > df[snapshot_date_column]

    failed_count = future_dates.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 124,
        "Variable checked": f"{rating_effective_date_column}, {snapshot_date_column}",
        "Description": "Rating Effective Date must be <= Snapshot Date.",
        "Threshold / Condition": "No future rating date allowed.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_125_RatingEffectiveVsRelationshipStart(df, rating_effective_date_column, relationship_start_column):
    """
    Validation Check 125: Rating Effective Date cannot predate the Relationship Start.
    Ensures chronological consistency.
    """
    # Convert date columns to datetime format
    df[rating_effective_date_column] = pd.to_datetime(df[rating_effective_date_column], errors='coerce')
    df[relationship_start_column] = pd.to_datetime(df[relationship_start_column], errors='coerce')

    # Identify cases where Rating Effective Date is before Relationship Start Date
    invalid_dates = df[rating_effective_date_column] < df[relationship_start_column]

    failed_count = invalid_dates.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 125,
        "Variable checked": f"{rating_effective_date_column}, {relationship_start_column}",
        "Description": "Rating Effective Date cannot predate the Relationship Start.",
        "Threshold / Condition": "Rating Effective >= Relationship Start.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_126_IFRS9StageMissing(df, stage_column, facility_status_column):
    """
    Validation Check 126: IFRS 9 Stage must not be null if the facility is active.
    Ensures proper classification under IFRS 9.
    """
    # Identify active facilities (assuming non-null Facility Status means active)
    active_facilities = df[facility_status_column].notna()

    # Identify cases where IFRS 9 Stage is missing
    missing_stage = df[stage_column].isna()

    # Flagged cases
    failed_records = active_facilities & missing_stage
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 126,
        "Variable checked": f"{stage_column}, {facility_status_column}",
        "Description": "IFRS 9 Stage must not be null if the facility is active.",
        "Threshold / Condition": "Must not be null if facility is active.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_127_IFRS9StageValidity(df, stage_column, valid_stages={1, 2, 3}):
    """
    Validation Check 127: IFRS 9 Stage must be in {1, 2, 3}.
    Flags invalid stage values.
    """
    # Identify cases where IFRS 9 Stage is not within the valid set
    invalid_stage = ~df[stage_column].isin(valid_stages)

    failed_count = invalid_stage.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 127,
        "Variable checked": stage_column,
        "Description": "IFRS 9 Stage must be in {1, 2, 3}.",
        "Threshold / Condition": "Others flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_128_IFRS9StageVsCreditRating(df, credit_rating_column, stage_column, worst_rating_threshold):
    """
    Validation Check 128: If Internal Credit Rating is at worst level, IFRS 9 Stage should be 3.
    Ensures cross-consistency between credit rating and stage classification.
    """
    # Identify cases where Credit Rating is at the worst level
    worst_rating = df[credit_rating_column] <= worst_rating_threshold  # Assuming lower numbers indicate worse ratings

    # Identify cases where IFRS 9 Stage is not 3
    incorrect_stage = df[stage_column] != 3

    # Flagged cases where worst credit rating does not align with Stage 3
    failed_records = worst_rating & incorrect_stage
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 128,
        "Variable checked": f"{credit_rating_column}, {stage_column}",
        "Description": "If rating is at worst level, IFRS 9 Stage should be 3.",
        "Threshold / Condition": "Cross-consistency check.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_129_CreditRatingAbruptChange(df, credit_rating_column, snapshot_date_column, max_notch_change):
    """
    Validation Check 129: Trend check for abrupt movements in credit rating.
    Flags if rating changes > X notches at once in one month.
    """
    # Convert date column to datetime format
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    # Sort by date to ensure correct change tracking
    df = df.sort_values(by=snapshot_date_column)

    # Calculate the change in credit rating month-over-month
    df['Credit Rating Change'] = df[credit_rating_column].astype(float).diff().abs()

    # Identify cases where the credit rating change exceeds the threshold
    excessive_change = df['Credit Rating Change'] > max_notch_change

    failed_count = excessive_change.sum()
    total_count = len(df) - 1  # Exclude the first row due to diff() calculation
    failed_percent = (failed_count / total_count) * 100 if total_count > 0 else 0

    result_df = pd.DataFrame([{
        "Check ID": 129,
        "Variable checked": credit_rating_column,
        "Description": "Trend check: abrupt movement from top to bottom rating in 1 month is suspicious.",
        "Threshold / Condition": f"Flags if rating changes > {max_notch_change} notches at once.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_130_IFRS9StageVsDaysPastDue(df, stage_column, days_past_due_column, overdue_threshold):
    """
    Validation Check 130: Cross-check IFRS 9 Stage with Days Past Due.
    If Days Past Due > 90 days, Stage likely 3.
    Flags mismatches.
    """
    # Identify cases where Days Past Due exceeds the threshold
    overdue_cases = df[days_past_due_column].astype(float) > overdue_threshold

    # Identify cases where IFRS 9 Stage is not 3 despite being overdue
    incorrect_stage = df[stage_column] != 3

    # Flagged cases where past due days indicate default, but stage is not 3
    failed_records = overdue_cases & incorrect_stage
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 130,
        "Variable checked": f"{stage_column}, {days_past_due_column}",
        "Description": "Cross-check IFRS 9 Stage with Days Past Due. If >90 days, Stage likely 3.",
        "Threshold / Condition": "Mismatches flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_131_DefaultFlagVsStage(df, stage_column, default_flag_column):
    """
    Validation Check 131: If Default Flag = 1, IFRS 9 Stage must be 3.
    Ensures defaulted accounts are classified correctly under IFRS 9.
    """
    # Identify cases where Default Flag is set to 1 (indicating default)
    defaulted_accounts = df[default_flag_column] == 1

    # Identify cases where IFRS 9 Stage is not 3 despite default
    incorrect_stage = df[stage_column] != 3

    # Flagged cases where default flag is set, but Stage is not 3
    failed_records = defaulted_accounts & incorrect_stage
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 131,
        "Variable checked": f"{stage_column}, {default_flag_column}",
        "Description": "If Default Flag=1, Stage must be 3.",
        "Threshold / Condition": "Hard rule.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_132_CreditRatingDistribution(df, credit_rating_column, max_percentage_threshold):
    """
    Validation Check 132: Distribution check for Internal Credit Ratings.
    Ensure not all ratings are concentrated in the middle or a single category.
    Flags if more than X% of ratings fall into one category.
    """
    # Calculate the percentage distribution of each credit rating
    rating_distribution = df[credit_rating_column].value_counts(normalize=True) * 100

    # Identify cases where any single rating exceeds the allowed threshold
    over_concentration = rating_distribution > max_percentage_threshold

    failed_count = over_concentration.sum()
    total_count = len(rating_distribution)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 132,
        "Variable checked": credit_rating_column,
        "Description": "Distribution check: ensure not all are in middle or one category.",
        "Threshold / Condition": f"More than {max_percentage_threshold}% in a single rating is suspicious.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_133_StaleCreditRating(df, rating_effective_date_column, snapshot_date_column, max_age_years):
    """
    Validation Check 133: If rating is older than X years, flagged as stale.
    Must have updated rating after a certain period.
    """
    # Convert date columns to datetime format
    df[rating_effective_date_column] = pd.to_datetime(df[rating_effective_date_column], errors='coerce')
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    # Calculate the age of the rating in days
    df['Rating Age (Days)'] = (df[snapshot_date_column] - df[rating_effective_date_column]).dt.days

    # Identify cases where the rating age exceeds the maximum allowed period (converted to days)
    stale_ratings = df['Rating Age (Days)'] > (max_age_years * 365)

    failed_count = stale_ratings.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 133,
        "Variable checked": f"{rating_effective_date_column}, {snapshot_date_column}",
        "Description": "If rating is older than X years, flagged as stale.",
        "Threshold / Condition": f"Must have updated rating after {max_age_years} years.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_134_CreditRatingVsBorrowerSegment(df, credit_rating_column, borrower_segment_column, high_performance_segments, worst_rating_threshold):
    """
    Validation Check 134: Large Corporate typically might not have the worst rating if performing.
    High-level plausibility check (business-specific).
    """
    # Identify cases where the Borrower Segment is a high-performance category
    high_performance_borrowers = df[borrower_segment_column].isin(high_performance_segments)

    # Identify cases where the Credit Rating is at the worst level despite being a high-performance borrower
    worst_rating_assigned = df[credit_rating_column] <= worst_rating_threshold  # Assuming lower numbers indicate worse ratings

    # Flagged cases
    failed_records = high_performance_borrowers & worst_rating_assigned
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 134,
        "Variable checked": f"{credit_rating_column}, {borrower_segment_column}",
        "Description": "Large Corporate typically might not have worst rating if performing.",
        "Threshold / Condition": "High-level plausibility check (business-specific).",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_135_IFRS9StageAbruptTransition(df, stage_column, snapshot_date_column, max_allowed_jump, time_window_days):
    """
    Validation Check 135: Analyze transitions from Stage 1 to Stage 3 in one step.
    Flags abrupt jumps exceeding a threshold within a short time window.
    """
    # Convert date column to datetime format
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    # Sort by date to ensure correct transition tracking
    df = df.sort_values(by=snapshot_date_column)

    # Calculate the stage transition difference
    df['Stage Change'] = df[stage_column].diff()

    # Identify cases where the stage change is greater than the allowed jump
    abrupt_transition = (df['Stage Change'].abs() >= max_allowed_jump)

    failed_count = abrupt_transition.sum()
    total_count = len(df) - 1  # Exclude the first row due to diff() calculation
    failed_percent = (failed_count / total_count) * 100 if total_count > 0 else 0

    result_df = pd.DataFrame([{
        "Check ID": 135,
        "Variable checked": stage_column,
        "Description": "Analyze transitions from Stage 1 to 3 in one step.",
        "Threshold / Condition": f"If jump from 1 to 3 in {time_window_days} days > {max_allowed_jump}, flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_136_ConsistentCreditRatingAcrossFacilities(df, borrower_id_column, credit_rating_column):
    """
    Validation Check 136: If borrower has multiple facilities, ensure consistent rating across them (or explanation).
    Flags large rating differences within the same borrower.
    """
    # Calculate the range of ratings for each borrower
    rating_range_per_borrower = df.groupby(borrower_id_column)[credit_rating_column].nunique()

    # Identify borrowers with more than one unique rating
    inconsistent_ratings = rating_range_per_borrower > 1

    # Get borrowers with inconsistent ratings
    inconsistent_borrowers = inconsistent_ratings[inconsistent_ratings].index

    # Count failed records where borrower's facilities have inconsistent ratings
    failed_count = df[df[borrower_id_column].isin(inconsistent_borrowers)].shape[0]
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 136,
        "Variable checked": f"{borrower_id_column}, {credit_rating_column}",
        "Description": "If borrower has multiple facilities, ensure consistent rating across them (or explanation).",
        "Threshold / Condition": "Large rating differences within same borrower flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_137_IFRS9StageCureValidation(df, stage_column, cure_date_column, snapshot_date_column):
    """
    Validation Check 137: If Cure Date is present, ensure IFRS 9 Stage eventually returns to 1 or 2.
    Ensures timely IFRS 9 stage update post-cure.
    """
    # Convert date columns to datetime format
    df[cure_date_column] = pd.to_datetime(df[cure_date_column], errors='coerce')
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    # Identify cases where Cure Date is present
    cured_accounts = df[cure_date_column].notna()

    # Identify cases where IFRS 9 Stage is still 3 after the Cure Date
    still_in_stage_3 = (df[stage_column] == 3) & (df[snapshot_date_column] > df[cure_date_column])

    # Flagged cases
    failed_records = cured_accounts & still_in_stage_3
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 137,
        "Variable checked": f"{stage_column}, {cure_date_column}, {snapshot_date_column}",
        "Description": "If Cure Date is present, ensure IFRS 9 Stage eventually returns to 1 or 2.",
        "Threshold / Condition": "Timely IFRS stage update post-cure.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_138_IFRS9StageVsRepaymentSchedule(df, stage_column, repayment_schedule_column, bullet_repayment_value):
    """
    Validation Check 138: If bullet repayment, watch IFRS 9 Stage transitions near maturity.
    Flags cases where bullet repayment is due but stage remains unchanged.
    """
    # Identify cases where the repayment schedule is "Bullet"
    bullet_repayment_cases = df[repayment_schedule_column] == bullet_repayment_value

    # Identify cases where IFRS 9 Stage should be adjusted (possibly Stage 2)
    incorrect_stage = df[stage_column] == 1  # Expected transition to Stage 2 if near maturity

    # Flagged cases where bullet repayment is near, but stage remains at 1
    failed_records = bullet_repayment_cases & incorrect_stage
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 138,
        "Variable checked": f"{stage_column}, {repayment_schedule_column}",
        "Description": "If bullet repayment, watch IFRS 9 Stage transitions near maturity.",
        "Threshold / Condition": "Possibly Stage 2 if bullet is near due date.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_139_CollateralCoverageVsCreditRating(df, credit_rating_column, collateral_coverage_column, high_coverage_threshold, expected_rating_threshold):
    """
    Validation Check 139: If collateral coverage is extremely high, credit rating might be better.
    Flags cases where collateral coverage is high but credit rating remains low.
    """
    # Identify cases where Collateral Coverage is extremely high
    high_collateral_coverage = df[collateral_coverage_column].astype(float) >= high_coverage_threshold

    # Identify cases where Credit Rating is still at a low level (bad rating)
    low_credit_rating = df[credit_rating_column].astype(float) <= expected_rating_threshold  # Assuming lower numbers indicate worse ratings

    # Flagged cases where collateral coverage is high, but rating remains poor
    failed_records = high_collateral_coverage & low_credit_rating
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 139,
        "Variable checked": f"{credit_rating_column}, {collateral_coverage_column}",
        "Description": "If collateral coverage is extremely high, credit rating might be better.",
        "Threshold / Condition": "Big mismatch flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_140_IFRS9RatingAge(df, rating_effective_date_column, snapshot_date_column, max_rating_age_years):
    """
    Validation Check 140: No rating older than 2 years for IFRS 9 compliance.
    Ensures timely re-rating.
    """
    # Convert date columns to datetime format
    df[rating_effective_date_column] = pd.to_datetime(df[rating_effective_date_column], errors='coerce')
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    # Calculate the age of the rating in days
    df['Rating Age (Days)'] = (df[snapshot_date_column] - df[rating_effective_date_column]).dt.days

    # Identify cases where the rating age exceeds the maximum allowed period (converted to days)
    outdated_ratings = df['Rating Age (Days)'] > (max_rating_age_years * 365)

    failed_count = outdated_ratings.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 140,
        "Variable checked": f"{rating_effective_date_column}, {snapshot_date_column}",
        "Description": "No rating older than 2 years for IFRS 9 compliance.",
        "Threshold / Condition": f"Must re-rate at least every {max_rating_age_years} years.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_141_BestRatingWithNoDefaults(df, credit_rating_column, late_payments_column, default_flag_column, best_rating_value):
    """
    Validation Check 141: If rating = 1 (best), verify no late payments or defaults.
    Flags contradictions if a top-rated borrower has late payments or defaults.
    """
    # Identify cases where the rating is at the best level
    best_rating = df[credit_rating_column] == best_rating_value

    # Identify cases where there are late payments or defaults
    late_payments_or_defaults = (df[late_payments_column].astype(float) > 0) | (df[default_flag_column] == 1)

    # Flagged cases where rating is best but late payments or defaults exist
    failed_records = best_rating & late_payments_or_defaults
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 141,
        "Variable checked": f"{credit_rating_column}, {late_payments_column}, {default_flag_column}",
        "Description": "If rating=1 (best), verify no late payments or default.",
        "Threshold / Condition": "Contradiction flagged if any default.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_142_DefaultDateVsIFRS9Stage(df, default_date_column, stage_column, snapshot_date_column):
    """
    Validation Check 142: If Default Date is given, IFRS 9 Stage must be 3 from that month onward.
    Flags mismatches where default is recorded, but Stage is not set correctly.
    """
    # Convert date columns to datetime format
    df[default_date_column] = pd.to_datetime(df[default_date_column], errors='coerce')
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    # Identify cases where a Default Date is provided
    default_occurred = df[default_date_column].notna()

    # Identify cases where IFRS 9 Stage is not 3 from that month onward
    incorrect_stage = (df[stage_column] != 3) & (df[snapshot_date_column] >= df[default_date_column])

    # Flagged cases where default date exists but Stage is incorrect
    failed_records = default_occurred & incorrect_stage
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 142,
        "Variable checked": f"{default_date_column}, {stage_column}, {snapshot_date_column}",
        "Description": "If default date is given, Stage=3 from that month onward.",
        "Threshold / Condition": "Stage mismatch flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_143_RestructuringFlagVsIFRS9Stage(df, restructuring_flag_column, stage_column):
    """
    Validation Check 143: If Restructuring Flag is set, IFRS 9 Stage is typically 2 or 3.
    Flags cases where restructuring occurs but Stage is still 1.
    """
    # Identify cases where Restructuring Flag is set
    restructured_accounts = df[restructuring_flag_column] == 1

    # Identify cases where IFRS 9 Stage is still 1 despite restructuring
    incorrect_stage = df[stage_column] == 1

    # Flagged cases where restructuring occurred but Stage remains 1
    failed_records = restructured_accounts & incorrect_stage
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 143,
        "Variable checked": f"{restructuring_flag_column}, {stage_column}",
        "Description": "Typically Stage 2 or 3 if restructured.",
        "Threshold / Condition": "Stage=1 with restructuring = suspicious.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_144_RatingDateVsFacilityStart(df, rating_effective_date_column, facility_start_date_column):
    """
    Validation Check 144: Ensure Rating Effective Date isn’t assigned before Facility Start Date.
    Flags cases where ratings exist before the facility was established.
    """
    # Convert date columns to datetime format
    df[rating_effective_date_column] = pd.to_datetime(df[rating_effective_date_column], errors='coerce')
    df[facility_start_date_column] = pd.to_datetime(df[facility_start_date_column], errors='coerce')

    # Identify cases where Rating Effective Date is earlier than Facility Start Date
    invalid_dates = df[rating_effective_date_column] < df[facility_start_date_column]

    failed_count = invalid_dates.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 144,
        "Variable checked": f"{rating_effective_date_column}, {facility_start_date_column}",
        "Description": "Ensure Rating isn’t assigned before Facility Start Date.",
        "Threshold / Condition": "Rating date >= facility start.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_145_CreditRatingRange(df, credit_rating_column, min_rating, max_rating):
    """
    Validation Check 145: If rating is numeric, check range is within allowed values.
    Flags out-of-range ratings.
    """
    # Identify cases where Credit Rating is outside the allowed range
    out_of_range = (df[credit_rating_column].astype(float) < min_rating) | (df[credit_rating_column].astype(float) > max_rating)

    failed_count = out_of_range.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 145,
        "Variable checked": credit_rating_column,
        "Description": "If rating is numeric, check range is 1–10 (example).",
        "Threshold / Condition": "Out-of-range flagged.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_146_IFRS9StageDistribution(df, stage_column, min_percentage_threshold):
    """
    Validation Check 146: Check IFRS 9 Stage distribution across the portfolio.
    Ensures at least some records exist in each stage.
    """
    # Calculate the percentage distribution of each IFRS 9 Stage
    stage_distribution = df[stage_column].value_counts(normalize=True) * 100

    # Identify cases where any stage has less than the allowed threshold
    underrepresented_stages = stage_distribution < min_percentage_threshold

    failed_count = underrepresented_stages.sum()
    total_count = len(stage_distribution)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 146,
        "Variable checked": stage_column,
        "Description": "Check IFRS 9 Stage distribution across the portfolio.",
        "Threshold / Condition": f"At least some records in each stage (if relevant).",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_147_RapidCreditRatingImprovement(df, credit_rating_column, snapshot_date_column, max_improvement_threshold, time_window_days):
    """
    Validation Check 147: If rating improved from worst to best in < 1 month, investigate.
    Flags suspicious credit rating improvements.
    """
    # Convert date column to datetime format
    df[snapshot_date_column] = pd.to_datetime(df[snapshot_date_column], errors='coerce')

    # Sort by date to ensure correct transition tracking
    df = df.sort_values(by=snapshot_date_column)

    # Calculate the rating change difference
    df['Rating Change'] = df[credit_rating_column].astype(float).diff()

    # Identify cases where the rating change exceeds the allowed threshold within a short time
    rapid_improvement = df['Rating Change'] >= max_improvement_threshold

    failed_count = rapid_improvement.sum()
    total_count = len(df) - 1  # Exclude the first row due to diff() calculation
    failed_percent = (failed_count / total_count) * 100 if total_count > 0 else 0

    result_df = pd.DataFrame([{
        "Check ID": 147,
        "Variable checked": credit_rating_column,
        "Description": "If rating improved from worst to best in < 1 month, investigate.",
        "Threshold / Condition": f"Possibly erroneous update if rating changes > {max_improvement_threshold} in {time_window_days} days.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_148_IFRS9StageVsRevolvingCredit(df, stage_column, repayment_schedule_column, days_past_due_column, overdue_threshold):
    """
    Validation Check 148: For revolving credit, watch short-term delinquencies’ effect on IFRS 9 Stage.
    If Days Past Due > 30, the stage should possibly be 2.
    """
    # Identify cases where the repayment schedule is "Revolving"
    revolving_credit = df[repayment_schedule_column] == "Revolving"

    # Identify cases where Days Past Due exceeds the threshold
    overdue_accounts = df[days_past_due_column].astype(float) > overdue_threshold

    # Identify cases where IFRS 9 Stage is not 2 despite overdue status
    incorrect_stage = df[stage_column] != 2

    # Flagged cases where revolving credit is overdue, but stage remains incorrect
    failed_records = revolving_credit & overdue_accounts & incorrect_stage
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 148,
        "Variable checked": f"{stage_column}, {repayment_schedule_column}, {days_past_due_column}",
        "Description": "For revolving credit, watch short-term delinquencies’ effect on stage.",
        "Threshold / Condition": f"If Days Past Due > {overdue_threshold}, possibly Stage = 2.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_149_IFRS9StageVsInterestSuspense(df, stage_column, interest_suspense_column):
    """
    Validation Check 149: If there’s an interest in suspense, IFRS 9 Stage must be 3.
    Ensures that loans with interest in suspense are classified correctly under IFRS 9.
    """
    # Identify cases where Interest in Suspense exists (assuming non-null means it's in suspense)
    interest_in_suspense = df[interest_suspense_column].notna()

    # Identify cases where IFRS 9 Stage is not 3 despite interest being in suspense
    incorrect_stage = df[stage_column] != 3

    # Flagged cases where interest is in suspense, but stage remains incorrect
    failed_records = interest_in_suspense & incorrect_stage
    failed_count = failed_records.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 149,
        "Variable checked": f"{stage_column}, {interest_suspense_column}",
        "Description": "If there’s an interest in suspense, IFRS 9 Stage must be 3.",
        "Threshold / Condition": "Hard rule if that’s the bank policy.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def check_150_CureDateVsRatingEffectiveDate(df, cure_date_column, rating_effective_date_column):
    """
    Validation Check 150: If Cure Date < Rating Effective Date, check if rating was updated properly post-cure.
    Ensures that the cure date is correctly reflected in the rating updates.
    """
    # Convert date columns to datetime format
    df[cure_date_column] = pd.to_datetime(df[cure_date_column], errors='coerce')
    df[rating_effective_date_column] = pd.to_datetime(df[rating_effective_date_column], errors='coerce')

    # Identify cases where Cure Date is before Rating Effective Date
    invalid_sequence = df[cure_date_column] < df[rating_effective_date_column]

    failed_count = invalid_sequence.sum()
    total_count = len(df)
    failed_percent = (failed_count / total_count) * 100

    result_df = pd.DataFrame([{
        "Check ID": 150,
        "Variable checked": f"{cure_date_column}, {rating_effective_date_column}",
        "Description": "If Cure Date < Rating Effective Date, check if rating was updated properly post-cure.",
        "Threshold / Condition": "Cure date should come before next rating effective date if improved.",
        "Failed Count": failed_count,
        "% of overall data": f"{failed_percent:.2f}%"
    }])

    return result_df

def section_e_checks(core_data):
    """Run all 30 checks for Section A and return a combined results table."""

    # Run each check individually (Ensure all check functions are defined)
    checks = [
        check_121_InternalCreditRatingMissing(core_data, 'Internal Credit Rating', 'Facility Status'),
        check_122_ValidCreditRating(core_data, 'Internal Credit Rating', valid_ratings = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"}),
        check_123_RatingEffectiveDateMissing(core_data, 'Rating Effective Date', 'Internal Credit Rating'),
        check_124_RatingEffectiveDateNotFuture(core_data, 'Rating Effective Date', 'Snapshot Date'),
        check_125_RatingEffectiveVsRelationshipStart(core_data, 'Rating Effective Date', 'Relationship Start Date'),
        check_126_IFRS9StageMissing(core_data, 'Stage (IFRS 9)', 'Facility Status'),
        check_127_IFRS9StageValidity(core_data, 'Stage (IFRS 9)'),
        check_128_IFRS9StageVsCreditRating(core_data, 'Internal Credit Rating', 'Stage (IFRS 9)', worst_rating_threshold = 1),
        check_129_CreditRatingAbruptChange(core_data, 'Internal Credit Rating', 'Snapshot Date', max_notch_change = 3),
        check_130_IFRS9StageVsDaysPastDue(core_data, 'Stage (IFRS 9)', 'Days Past Due (Month-End)', overdue_threshold = 90),
        check_131_DefaultFlagVsStage(core_data, 'Stage (IFRS 9)', 'Default Flag'),
        check_132_CreditRatingDistribution(core_data, 'Internal Credit Rating', max_percentage_threshold = 40),
        check_133_StaleCreditRating(core_data, 'Rating Effective Date', 'Snapshot Date', max_age_years = 2),
        check_134_CreditRatingVsBorrowerSegment(core_data, 'Internal Credit Rating', 'Borrower Segment', high_performance_segments = {"Large Corporate", "Multinational", "Government Entity"}, worst_rating_threshold = 1),
        # check_135_IFRS9StageAbruptTransition(core_data, 'Stage (IFRS 9)', 'Snapshot Date', max_allowed_jump = 2, time_window_days = 30),
        check_136_ConsistentCreditRatingAcrossFacilities(core_data, 'Borrower ID', 'Internal Credit Rating'),
        check_137_IFRS9StageCureValidation(core_data, 'Stage (IFRS 9)', 'Cure Date', 'Snapshot Date'),
        check_138_IFRS9StageVsRepaymentSchedule(core_data, 'Stage (IFRS 9)', 'Repayment Schedule', bullet_repayment_value = 'Bullet'),
        check_139_CollateralCoverageVsCreditRating(core_data, 'Internal Credit Rating', 'Collateral Coverage', high_coverage_threshold = 200, expected_rating_threshold = 5),
        check_140_IFRS9RatingAge(core_data, 'Rating Effective Date', 'Snapshot Date', max_rating_age_years = 2),
        check_141_BestRatingWithNoDefaults(core_data, 'Internal Credit Rating', 'Late Payments Count', 'Default Flag', best_rating_value = 1),
        check_142_DefaultDateVsIFRS9Stage(core_data, 'Default Date', 'Stage (IFRS 9)', 'Snapshot Date'),
        check_143_RestructuringFlagVsIFRS9Stage(core_data, 'Restructuring Flag', 'Stage (IFRS 9)'),
        check_144_RatingDateVsFacilityStart(core_data, 'Rating Effective Date', 'Facility Start Date'),
        check_145_CreditRatingRange(core_data, 'Internal Credit Rating', min_rating = 1, max_rating = 10),
        check_146_IFRS9StageDistribution(core_data, 'Stage (IFRS 9)', min_percentage_threshold = 5),
        check_147_RapidCreditRatingImprovement(core_data, 'Internal Credit Rating', 'Snapshot Date', max_improvement_threshold = 5, time_window_days = 30),
        check_148_IFRS9StageVsRevolvingCredit(core_data, 'Stage (IFRS 9)', 'Repayment Schedule', 'Days Past Due (Month-End)', overdue_threshold = 30),
        check_149_IFRS9StageVsInterestSuspense(core_data, 'Stage (IFRS 9)', 'Interest in Suspense'),
        check_150_CureDateVsRatingEffectiveDate(core_data, 'Cure Date', 'Rating Effective Date')
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