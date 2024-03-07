"""Feature engineers the state_DC dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def delete_columns(df):
    df = df.drop(columns=[
        'activity_year', 'lei', 'derived_msa-md', 'state_code', 'county_code', 'census_tract', 'tract_population',
        'tract_minority_population_percent', 'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage',
        'tract_owner_occupied_units', 'tract_one_to_four_family_homes', 'tract_median_age_of_housing_units',
        'applicant_credit_score_type', 'co-applicant_credit_score_type', 'applicant_ethnicity_observed', 
        'co-applicant_ethnicity_observed', 'applicant_race_observed', 'co-applicant_race_observed',
        'applicant_sex_observed', 'co-applicant_sex_observed', 'submission_of_application', 'hoepa_status',
        'aus-1', 'aus-2', 'aus-3', 'aus-4', 'aus-5', 'initially_payable_to_institution', 'applicant_race-3',
        'applicant_race-4', 'applicant_race-5', 'co-applicant_race-3', 'co-applicant_race-4','co-applicant_race-5', 
        'co-applicant_ethnicity-2', 'co-applicant_ethnicity-3', 'co-applicant_ethnicity-4', 'co-applicant_ethnicity-5',
        'applicant_ethnicity-2', 'applicant_ethnicity-3', 'applicant_ethnicity-4', 'applicant_ethnicity-5', 
        'denial_reason-1', 'denial_reason-2', 'denial_reason-3', 'denial_reason-4', 'multifamily_affordable_units',
        'manufactured_home_land_property_interest', 'manufactured_home_secured_property_type', 'purchaser_type',
        'derived_loan_product_type', 'derived_dwelling_category', 'interest_rate', 'rate_spread', 'origination_charges',
        'total_points_and_fees', 'total_loan_costs', 'discount_points', 'lender_credits', 'prepayment_penalty_term',
        'intro_rate_period'
        ])
    return df

def create_derived_age_above_62(df):
    conditions = [
        (df['applicant_age_above_62'] == 'Yes') & ((df['co-applicant_age_above_62'] == 'Yes') | (pd.isna(df['co-applicant_age_above_62']))),
        (df['applicant_age_above_62'] == 'No') & ((df['co-applicant_age_above_62'] == 'No') | (pd.isna(df['co-applicant_age_above_62']))),
        (df['applicant_age_above_62'] == 'Yes') & (df['co-applicant_age_above_62'] == 'No') | (df['applicant_age_above_62'] == 'No') & (df['co-applicant_age_above_62'] == 'Yes'),
        (pd.isna(df['applicant_age_above_62'])) & (df['co-applicant_age_above_62'] == 'Yes'),
        (pd.isna(df['applicant_age_above_62'])) & (df['co-applicant_age_above_62'] == 'No'),
        (pd.isna(df['applicant_age_above_62'])) & (pd.isna(df['co-applicant_age_above_62']))
    ]
    choices = ['Yes', 'No', 'Joint', 'Yes', 'No', pd.NA]
    df['derived_age_above_62'] = np.select(conditions, choices, default=0)
    return df

def create_derived_age_below_25(df):
    #df.replace({'co-applicant_age':{'9999': 'No co-applicant', '8888': 'NaN'}, 'applicant_age':{'8888': 'NaN'}}, inplace = True)
    df['derived_age_below_25'] = np.where((df['applicant_age'] == '<25') & ((df['co-applicant_age'] == '<25') | (df['co-applicant_age'] == '9999')), 'Yes', 'No')
    return df

def create_derived_race_revisited(df):
    conditions = [
        #(df['applicant_race-1'] == 'White') & (df['derived_ethnicity'] == 'Hispanic or Latino'),
        (df['applicant_race-1'] == 5.0) & (pd.isna(df['applicant_race-2'])) & (df['applicant_ethnicity-1'] == 2.0) & 
        ((df['co-applicant_race-1'] == 5.0) & (pd.isna(df['co-applicant_race-2'])) & (df['co-applicant_ethnicity-1'] == 2)
         | (df['co-applicant_race-1'] == 8.0) & (df['co-applicant_ethnicity-1'] == 5)),
        (df['applicant_race-1'].isin([5.0, 6.0])) & (pd.isna(df['applicant_race-2'])) & (df['applicant_ethnicity-1'].isin([1.0, 11.0, 12.0, 13.0, 14.0])) & 
        ((df['co-applicant_race-1'].isin([5.0, 6.0])) & (pd.isna(df['co-applicant_race-2'])) & (df['co-applicant_ethnicity-1'].isin([1.0, 11.0, 12.0, 13.0, 14.0]))
         | (df['co-applicant_race-1'] == 8.0) & (df['co-applicant_ethnicity-1'] == 5)),
        (df['applicant_race-1'] == 3.0) & ((df['co-applicant_race-1'] == 3.0) 
         | (df['co-applicant_race-1'] == 8.0) & (df['co-applicant_ethnicity-1'] == 5)),
        (df['applicant_race-1'].isin([2.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0])) & ((df['applicant_race-2'].isin([2.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 5.0])) | (pd.isna(df['applicant_race-2']))) & 
        ((df['co-applicant_race-1'].isin([2.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0])) & ((df['co-applicant_race-2'].isin([2.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 5.0])) | (pd.isna(df['co-applicant_race-2'])))
         | (df['co-applicant_race-1'] == 8.0) & (df['co-applicant_ethnicity-1'] == 5)),
        (df['applicant_race-1'] == 1.0) & ((pd.isna(df['applicant_race-2'])) | (df['applicant_race-2'] == 5.0)) & 
        ((df['co-applicant_race-1'] == 1.0) & ((pd.isna(df['co-applicant_race-2'])) | (df['applicant_race-2'] == 5.0))
         | (df['co-applicant_race-1'] == 8.0) & (df['co-applicant_ethnicity-1'] == 5)),
        (df['applicant_race-1'].isin([4.0, 41.0, 42.0, 43.0, 44.0])) & ((pd.isna(df['applicant_race-2'])) | (df['applicant_race-2'].isin([4.0, 41.0, 42.0, 43.0, 44.0, 5.0]))) & 
        ((df['co-applicant_race-1'].isin([4.0, 41.0, 42.0, 43.0, 44.0])) & ((pd.isna(df['co-applicant_race-2'])) | (df['co-applicant_race-2'].isin([4.0, 41.0, 42.0, 43.0, 44.0, 5.0])))
         | (df['co-applicant_race-1'] == 8.0) & (df['co-applicant_ethnicity-1'] == 5)),
        (df['applicant_ethnicity-1'].isin([2.0, 3.0, 4.0])) & (df['co-applicant_ethnicity-1'].isin([2.0, 3.0, 4.0, 5.0]))
        & (df['applicant_race-1'].isin([6.0, 7.0])) & (df['co-applicant_race-1'].isin([6.0, 7.0, 8.0])),
        (df['co-applicant_race-1'] == 6.0) & (df['co-applicant_ethnicity-1'].isin([2, 3])),
        (df['applicant_ethnicity-1'] == 3.0) & (df['co-applicant_ethnicity-1'] == 5) & (df['applicant_race-1'] == 5.0) & (df['co-applicant_race-1'] == 8.0) & (pd.isna(df['applicant_race-2']))   
        | (df['applicant_ethnicity-1'].isin([2.0, 3.0])) & (df['co-applicant_ethnicity-1'].isin([2, 3])) & (df['applicant_race-1'] == 5.0) & (df['co-applicant_race-1'] == 5.0) & (pd.isna(df['applicant_race-2'])) & (pd.isna(df['co-applicant_race-2']))
    ]
    choices = ['White', 'Hispanic or Latino', 'Black or African American', 'Asian', 'American Indian or Alaska Native', 
               'Native Hawaiian or Other Pacific Islander', 'Race Not Available', 'Possibly Mixed', 'Possibly White'
              ]
    df['derived_race_revisited'] = np.select(conditions, choices, default = 'Mixed')
    return df

def turn_action_taken_into_binary_target_variable(df):
    df = df.query('action_taken != 4 & action_taken != 5 & action_taken != 6')
    # Checks whether loan_purpose is '1 - Home purchase' - a condition for a preapproval request to be considered an application
    # Federal Financial Institutions Examination Council. A GUIDE TO HMDA Reporting. Getting It Right! 
    # url: https://www.ffiec.gov/hmda/pdf/2023Guide.pdf
    df = df.drop(df[(df['action_taken'] == 7) & (df['loan_purpose'] != 1)].index) 
    df = df.drop(df[(df['action_taken'] == 8) & (df['loan_purpose'] != 1)].index)
    df.reset_index(drop=True, inplace=True)
    # ------------------------------------------------------------------------------------------------
    df.loc[(df['action_taken'].isin([1, 2, 8])), 'action_taken'] = 1
    df.loc[(df['action_taken'].isin([3, 7])), 'action_taken'] = 0                              
    return df
    
# According to 2023 "Guide To HMDA Reporting. Getting It Right!" applications that fall under the categories filtered out
# below are applications submitted by legal entities as opposed to natural persons.

def delete_non_natural_person_entries(df):
    df = df.drop(df[((df['applicant_ethnicity-1'] == 4.0) | (df['co-applicant_ethnicity-1'] == 4))
                 & ((df['applicant_race-1'] == 7.0) | (df['co-applicant_race-1'] == 7.0))
                 & ((df['applicant_sex'] == 4) | (df['co-applicant_sex'] == 4)) 
                 & ((df['applicant_age'] == '8888') | (df['co-applicant_age'] == '8888'))
                 & pd.isna(df['income']) & (pd.isna(df['debt_to_income_ratio']) | (df['debt_to_income_ratio'] == 'Exempt'))].index)
    df.reset_index(drop=True, inplace=True)
    #df = df.query('applicant_ethnicity-1 != 4.0 & applicant_race-1 != 7.0 & applicant_sex != 4 & applicant_age != 8888')
    return df
     
def create_if_co_applicant(df):
    df['if_co-applicant'] = np.where(df['co-applicant_race-1'] == 8.0, 'No', 'Yes')
    return df

def replace_numeric_applicant_age(df):
    df.replace({'co-applicant_age':{'9999': 'No co-applicant', '8888': pd.NA}, 'applicant_age':{'8888': pd.NA}}, inplace = True)
    return df

def replace_numeric_applicant_sex(df):
    collection = ['applicant_sex', 'co-applicant_sex']
    for applicant_type in collection:
        conditions = [(df[applicant_type] == 1), (df[applicant_type] == 2), (df[applicant_type] == 3),
                      (df[applicant_type] == 4) | (pd.isna(df[applicant_type])), (df[applicant_type] == 6)
                     ]
        choices = ['Male', 'Female', 'Not Provided', pd.NA, 'Both']
        df[applicant_type] = np.select(conditions, choices, default = 'No co-applicant')
    return df
     
def replace_numeric_applicant_race(df):
    collection = ['applicant_race-1', 'applicant_race-2', 'co-applicant_race-1', 'co-applicant_race-2']
    for applicant_type in collection:
        conditions = [(df[applicant_type] == 1.0), (df[applicant_type].isin([2.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0])),
                      (df[applicant_type] == 3.0), (df[applicant_type].isin([4.0, 41.0, 42.0, 43.0, 44.0])), 
                      (df[applicant_type] == 5.0), (df[applicant_type] == 6.0), 
                      (df[applicant_type] == 7.0) | (pd.isna(df[applicant_type]))
                     ]
        choices = ['American Indian or Alaska Native', 'Asian', 'Black or African American',
                   'Native Hawaiian or Other Pacific Islander', 'White', 'Not Provided', pd.NA
                  ]
        df[applicant_type] = np.select(conditions, choices, default = 'No co-applicant')
    return df

def replace_numeric_applicant_ethnicity(df):
    collection = ['applicant_ethnicity-1', 'co-applicant_ethnicity-1']
    for applicant_type in collection:
        conditions = [(df[applicant_type].isin([1.0, 11.0, 12.0, 13.0, 14.0])), (df[applicant_type] == 2),
                      (df[applicant_type] == 3), (df[applicant_type] == 4) | (pd.isna(df[applicant_type]))
                     ]
        choices = ['Hispanic or Latino', 'Not Hispanic or Latino', 'Not Provided', pd.NA]
        df[applicant_type] = np.select(conditions, choices, default = 'No co-applicant')
    return df

def replace_numeric_occupancy_type(df):
    conditions = [(df['occupancy_type'] == 1), (df['occupancy_type'] == 2),
                  (df['occupancy_type'] == 3)
                 ]
    choices = ['Principal residence', 'Second residence', 'Investment property']
    df['occupancy_type'] = np.select(conditions, choices)
    return df

#def replace_numeric_construction_method(df):
#    conditions = [(df['construction_method'] == 1), (df['construction_method'] == 2)
#                 ]
#    choices = ['Site-built', 'Manufactured home']
#    df['construction_method'] = np.select(conditions, choices)
#    return df
    
def replace_exempt(df):
    collection = ['debt_to_income_ratio', 'property_value', 'loan_term', 'loan_to_value_ratio']
    for feature_type in collection:
        df.replace({feature_type: {'Exempt': pd.NA}}, inplace = True)
    return df

#def replace_numeric_hoepa_status(df):
#    conditions = [(df['hoepa_status'] == 1), (df['hoepa_status'] == 2), (df['hoepa_status'] == 3)
#                 ]
#    choices = ['High-cost mortgage', 'Not a high-cost mortgage', pd.NA]
#    df['hoepa_status'] = np.select(conditions, choices)
#    return df

def replace_numeric_3_conditions_with_exempt(df):
    feature_collection = ['other_nonamortizing_features', 'balloon_payment', 'interest_only_payment', 'negative_amortization',
                          'business_or_commercial_purpose', 'open-end_line_of_credit', 'reverse_mortgage'
                         ]
    substitute_collection = [
        ['Other non-fully amortizing features', 'No other non-fully amortizing features', pd.NA],
        ['Balloon payment', 'No balloon payment', pd.NA],
        ['Interest-only payments', 'No interest-only payments', pd.NA],
        ['Negative amortization', 'No negative amortization', pd.NA],
        ['Primarily for a business or commercial purpose', 'Not primarily for a business or commercial purpose', pd.NA],
        ['Open-end line of credit', 'Not an open-end line of credit', pd.NA],
        ['Reverse mortgage', 'Not a reverse mortgage', pd.NA]
    ]
    for feature_type, list in zip(feature_collection, substitute_collection):
        conditions = [(df[feature_type] == 1), (df[feature_type] == 2),
                      (df[feature_type] == 1111)
                     ]
        choices = list
        df[feature_type] = np.select(conditions, choices, default=0)
    return df

def replace_numeric_2_conditions(df):
    feature_collection = ['construction_method', 'lien_status', 'preapproval'
                         ]
    substitute_collection = [
        ['Site-built', 'Manufactured home'],
        ['Secured by a first lien', 'Secured by a subordinate lien'],
        ['Preapproval requested', 'Preapproval not requested']
    ]
    for feature_type, list in zip(feature_collection, substitute_collection):
        conditions = [(df[feature_type] == 1), (df[feature_type] == 2)
                     ]
        choices = list
        df[feature_type] = np.select(conditions, choices, default=0)
    return df

#def replace_numeric_3_conditions(df):
#    feature_collection = ['occupancy_type', 'hoepa_status'
#                         ]
#    substitute_collection = [
#        ['Principal residence', 'Second residence', 'Investment property'],
#        ['High-cost mortgage', 'Not a high-cost mortgage', pd.NA]     
#    ]
#    for feature_type, list in zip(feature_collection, substitute_collection):
#       conditions = [(df[feature_type] == 1), (df[feature_type] == 2),
#                      (df[feature_type] == 3)
#                     ]
#        choices = list
#        df[feature_type] = np.select(conditions, choices, default=0)
#    return df    

#def replace_numeric_other_nonamortizing_features(df):
#    conditions = [(df['other_nonamortizing_features'] == 1), (df['other_nonamortizing_features'] == 2),
#                  (df['other_nonamortizing_features'] == 1111)
#                 ]
#    choices = ['Other non-fully amortizing features', 'No other non-fully amortizing features', pd.NA]
#    df['other_nonamortizing_features'] = np.select(conditions, choices)
#    return df

#def replace_numeric_balloon_payment(df):
#    conditions = [(df['balloon_payment'] == 1), (df['balloon_payment'] == 2),
#                  (df['balloon_payment'] == 1111)
#                 ]
#    choices = ['Balloon payment', 'No balloon payment', pd.NA]
#    df['balloon_payment'] = np.select(conditions, choices)
#    return df

#def replace_numeric_interest_only_payment(df):
#    conditions = [(df['interest_only_payment'] == 1), (df['interest_only_payment'] == 2),
#                  (df['interest_only_payment'] == 1111)
#                 ]
#    choices = ['Interest-only payments', 'No interest-only payments', pd.NA]
#    df['interest_only_payment'] = np.select(conditions, choices)
#    return df

#def replace_numeric_negative_amortization(df):
#    conditions = [(df['negative_amortization'] == 1), (df['negative_amortization'] == 2),
#                  (df['negative_amortization'] == 1111)
#                 ]
#    choices = ['Negative amortization', 'No negative amortization', pd.NA]
#    df['negative_amortization'] = np.select(conditions, choices)
#    return df

#def replace_numeric_business_or_commercial_purpose(df):
#    conditions = [(df['business_or_commercial_purpose'] == 1), (df['business_or_commercial_purpose'] == 2),
#                  (df['business_or_commercial_purpose'] == 1111)
#                 ]
#    choices = ['Primarily for a business or commercial purpose', 'Not primarily for a business or commercial purpose', pd.NA]
#    df['business_or_commercial_purpose'] = np.select(conditions, choices)
#    return df

#def replace_numeric_open_end_line_of_credit(df):
#    conditions = [(df['open-end_line_of_credit'] == 1), (df['open-end_line_of_credit'] == 2),
#                  (df['open-end_line_of_credit'] == 1111)
#                 ]
#    choices = ['Open-end line of credit', 'Not an open-end line of credit', pd.NA]
#    df['open-end_line_of_credit'] = np.select(conditions, choices)
#    return df

#def replace_numeric_reverse_mortgage(df):
#    conditions = [(df['reverse_mortgage'] == 1), (df['reverse_mortgage'] == 2),
#                  (df['reverse_mortgage'] == 1111)
#                 ]
#    choices = ['Reverse mortgage', 'Not a reverse mortgage', pd.NA]
#    df['reverse_mortgage'] = np.select(conditions, choices)
#    return df

#def replace_numeric_lien_status(df):
#    conditions = [(df['lien_status'] == 1), (df['lien_status'] == 2)
#                 ]
#   choices = ['Secured by a first lien', 'Secured by a subordinate lien']
#    df['lien_status'] = np.select(conditions, choices)
#    return df

def replace_numeric_loan_purpose(df):
    conditions = [(df['loan_purpose'] == 1), (df['loan_purpose'] == 2), (df['loan_purpose'] == 31), (df['loan_purpose'] == 32), 
                 (df['loan_purpose'] == 4), (df['loan_purpose'] == 5)]
    choices = ['Home purchase', 'Home improvement', 'Refinancing', 'Cash-out refinancing', 'Other purpose', pd.NA]
    df['loan_purpose'] = np.select(conditions, choices, default=0)
    return df

def replace_numeric_loan_type(df):
    conditions = [(df['loan_type'] == 1), (df['loan_type'] == 2), (df['loan_type'] == 3), (df['loan_type'] == 4)]
    choices = ['Conventional (not insured or guaranteed)', 'Federal Housing Administration insured', 
               'Veterans Affairs guaranteed', 'USDA Rural Housing Service or Farm Service Agency guaranteed']
    df['loan_type'] = np.select(conditions, choices, default=0)
    return df
    
def move_target_column_to_the_front(df):
    col = df.pop("action_taken")
    df.insert(0, col.name, col)
    return df    

#def replace_numeric_preapproval(df):
#    conditions = [(df['preapproval'] == 1), (df['preapproval'] == 2)]
#    choices = ['Preapproval requested', 'Preapproval not requested']
#    df['preapproval'] = np.select(conditions, choices)
#    return df


    
#def replace_numerical_co_applicant_sex(df):
#    conditions = [(df['applicant_sex'] == 1), (df['applicant_sex'] == 2), (df['applicant_sex'] == 3), (df['applicant_sex'] == 6)
#    ]
#    choices = ['Male', 'Female', 'Not Provided', 'Both']
#    df['co-applicant_sex'] = np.select(conditions, choices, default = 'NaN')
#    return df

#def replace_numerical_applicant_race_1(df):
#    conditions = [(df['applicant_race'] == 1), (df['applicant_race'] == 2), (df['applicant_race'] == 3), 
#                  (df['applicant_race'] == 6)
#    ]
#    choices = ['American Indian or Alaska Native', 'Asian', 'Black or African American', 
#               'Native Hawaiian or Other Pacific Islander', 'White'
#              ]
#    df['applicant_race'] = np.select(conditions, choices, default = 'NaN')
#    return df

#---------------------NOTES FOR LATER------------------------------
#def code_to_description(df): 
#    return df

#First apply turn_action_taken_into_binary_target_variable, then the rest
#------------------------------------------------------------------
#def create_derived_race_revisited(df):
#    conditions = [
#        (df['derived_race'] == 'White') & (df['derived_ethnicity'] == 'Hispanic or Latino'),
#        ((df['applicant_race-1'] == 3.0) & ((df['co-applicant_race-1'] == 3.0) | (df['co-applicant_race-1'] == 6.0) | (df['co-applicant_race-1'] == 7.0) | (df['co-applicant_race-1'] == 8.0))) |
#        ((df['co-applicant_race-1'] == 3.0) & ((df['applicant_race-1'] == 3.0) | (df['applicant_race-1'] == 6.0) | (df['applicant_race-1'] == 7.0) | (df['applicant_race-1'] == 8.0))),
#        ((df['applicant_race-1'].isin([2, 21, 22, 23, 24, 25, 26, 27])) & ((df['co-applicant_race-1'].isin([2, 21, 22, 23, 24, 25, 26, 27])) | (df['co-applicant_race-1'] == 6.0) | (df['co-applicant_race-1'] == 7.0) | (df['co-applicant_race-1'] == 8.0))) |
#        ((df['co-applicant_race-1'].isin([2, 21, 22, 23, 24, 25, 26, 27])) & ((df['applicant_race-1'].isin([2, 21, 22, 23, 24, 25, 26, 27])) | (df['applicant_race-1'] == 6.0) | (df['applicant_race-1'] == 7.0) | (df['applicant_race-1'] == 8.0)))
#    ]
#    choices = ['Hispanic or Latino', 'Black or African American', 'Asian']
#    df['derived_race_revisited'] = np.select(conditions, choices, df['derived_race'])
#    return df

##df_after_initial_deletion = delete_columns(df_raw)
##df_new_feature_derived_age_above_62 = create_derived_age_above_62(df_after_initial_deletion)
##df_new_feature_derived_age_below_25 = create_derived_age_below_25(df_new_feature_derived_age_above_62)
##df_new_feature_derived_race_revisited = create_derived_race_revisited(df_new_feature_derived_age_below_25)
#df_new_feature_derived_race_revisited
#df_new_feature_derived_race_revisited.dropna(subset=['denial_reason-4'])
#df_new_feature_derived_race_revisited.loc[df_new_feature_derived_race_revisited['action_taken'] == 6].tail(60)
##df_binary_target_variable = turn_action_taken_into_binary_target_variable(df_new_feature_derived_race_revisited) 
#df_binary_target_variable.loc[df_binary_target_variable['action_taken'] == 7].count()
##df_new_feature_if_co_applicant = create_if_co_applicant(df_binary_target_variable)
#df_non_numerical_identifiers=replace_numerical_applicant_sex(df_new_feature_if_co_applicant)
#df_non_numerical_identifiers
##df_natural_persons = replace_numeric(delete_non_natural_person_entries(df_new_feature_if_co_applicant))
##df_natural_persons
#delete_non_natural_person_entries(turn_action_taken_into_binary_target_variable(

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()
    
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])
    
    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/state_DC.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)
    
    logger.debug("Reading downloaded data.")
    df = pd.read_csv(
        fn
    )
    os.unlink(fn)    

    df_preprocessed = move_target_column_to_the_front(
        replace_numeric_loan_type(
            replace_numeric_loan_purpose(
                replace_numeric_occupancy_type(
                    replace_numeric_2_conditions(
                        replace_numeric_3_conditions_with_exempt(
                            replace_exempt(
                                replace_numeric_applicant_ethnicity(
                                    replace_numeric_applicant_race(
                                        replace_numeric_applicant_sex(
                                            replace_numeric_applicant_age(
                                                create_if_co_applicant(
        create_derived_race_revisited(
            create_derived_age_below_25(
                create_derived_age_above_62(
                    delete_non_natural_person_entries(
                        turn_action_taken_into_binary_target_variable(
                            delete_columns(df))))))))))))))))))
    
    df_full_feature_set = df_preprocessed.drop(
        columns=[
            'if_co-applicant', 'derived_race_revisited', 'derived_age_below_25', 'derived_age_above_62'
            ]
        )
    
    df_minimal_set = df_preprocessed.drop(
        columns=[
            'derived_ethnicity', 'derived_race', 'applicant_ethnicity-1', 'co-applicant_ethnicity-1',
            'applicant_race-1', 'applicant_race-2', 'co-applicant_race-1', 'co-applicant_race-2',
            'applicant_sex', 'co-applicant_sex', 'applicant_age', 'co-applicant_age',
            'applicant_age_above_62', 'co-applicant_age_above_62'
            ]
        )
   
    logger.info("Splitting %d rows of data into train, and test datasets.", len(df_minimal_set.index))
    train, test = train_test_split(df_minimal_set, test_size=0.2)
    
    logger.info("Writing out datasets to %s.", base_dir)
    train.to_csv(f"{base_dir}/train/train.csv", index = False)
    train.to_csv(f"{base_dir}/train_no_header/train_no_header.csv", header = False, index = False)    
    test.to_csv(f"{base_dir}/test/test.csv", header = False, index = False)
    
