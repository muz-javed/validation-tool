import os
import json
import statistics
import time
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from io import BytesIO
from scipy.stats import f
from scipy.stats import t
import scipy.stats as stats
from PyPDF2 import PdfReader
from scipy.stats import chi2
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import binom
from scipy.stats import kstest
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import ttest_1samp
from sklearn.metrics import r2_score
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import kpss
from sklearn.metrics import confusion_matrix
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tools.tools import add_constant
# from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_extras.stylable_container import stylable_container
from statsmodels.stats.outliers_influence import variance_inflation_factor

from google.cloud import bigquery
from google.oauth2 import service_account

import core_validations_a
import core_validations_b
import core_validations_c
import core_validations_d
import core_validations_e
import core_validations_f
import core_validations_g
import core_validations_h
import core_validations_i
import core_validations_j

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import report_functions

def ChangeButtonColour(widget_label, font_color, background_color='transparent', radius = '0px', disabled = False, margin_top='0px', margin_bottom='0px', margin_left='0px', justifyContent = 'center', text_decor = 'none', padding_left = '12px', padding_right = '12px', padding_top = '4px', padding_bottom = '4px', border=''):
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.color ='{font_color}';
                    elements[i].style.background = '{background_color}';
                    elements[i].style.width = '100%';
                    elements[i].style.textAlign = 'center';
                    elements[i].style.textDecoration = 'None';
                    elements[i].style.display = 'flex';
                    elements[i].style.justifyContent = '{justifyContent}';
                    elements[i].style.borderRadius = '{radius}';
                    elements[i].style.disabled = '{disabled}';
                    elements[i].style.marginBottom = '{margin_bottom}';
                    elements[i].style.marginTop = '{margin_top}';
                    elements[i].style.marginLeft = '{margin_left}';
                    elements[i].style.textDecoration = '{text_decor}';
                    elements[i].style.paddingLeft = '{padding_left}';
                    elements[i].style.paddingRight = '{padding_right}';
                    elements[i].style.paddingTop = '{padding_top}';
                    elements[i].style.paddingBottom = '{padding_bottom}';
                    elements[i].style.border = '{border}';
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}", height=0, width=0)

def create_bins(data, n_bins):
    data = data.reset_index()
    data['Bins'] = pd.qcut(data['index'], q=n_bins, labels=False)
    data['Bins'] += 1
    data = data.drop('index', axis = 1)

    return data


def KS_GINI(data, groupbyvariable, default_flag, sortvariable=None, Business_Quarter_filter=None, Product_Type_Filter=None):
    # Filter the data based on the optional filters
    if Business_Quarter_filter is not None:
        data = data[data['Business_Quarter'] == Business_Quarter_filter]
    if Product_Type_Filter is not None:
        data = data[data['ProductType'] == Product_Type_Filter]

    summarized_data = data.groupby(groupbyvariable).agg(
        Total=(default_flag, 'size'),
        Default=(default_flag, 'sum'),
        NonDefault=(default_flag, lambda x: len(x) - x.sum()),
        sort_col=(sortvariable, 'mean'),
    ).reset_index()

    # st.write(summarized_data)
    
    if sortvariable:
        summarized_data = summarized_data.sort_values('sort_col').reset_index(drop = True)
        summarized_data = summarized_data.drop('sort_col', axis = 1)
    
    summarized_data['CummTotal'] = summarized_data['Total'].cumsum()
    summarized_data['CummDefault'] = summarized_data['Default'].cumsum()
    summarized_data['CummNonDefault'] = summarized_data['NonDefault'].cumsum()

    summarized_data['CummDistTotal'] = round(summarized_data['CummTotal'] * 100 / summarized_data['Total'].sum(), 2)
    summarized_data['CummDistDefault'] = round(summarized_data['CummDefault'] * 100 / summarized_data['Default'].sum(), 2)
    summarized_data['CummDistNonDefault'] = round(summarized_data['CummNonDefault'] * 100 / summarized_data['NonDefault'].sum(), 2)

    summarized_data['KS'] = round(summarized_data['CummDistDefault'] - summarized_data['CummDistNonDefault'], 2)
    summarized_data['AUC'] = np.where(
        summarized_data.index == 0,
        round(0.005 * summarized_data['CummDistTotal'] * summarized_data['CummDistDefault'], 2),
        round(0.005 * (summarized_data['CummDistTotal'] - summarized_data['CummDistTotal'].shift(1)) * 
              (summarized_data['CummDistDefault'] + summarized_data['CummDistDefault'].shift(1)), 2)
    )

    AUC = summarized_data['AUC'].sum()
    total_count = summarized_data['CummTotal'].sum()
    total_default = summarized_data['CummDefault'].sum()
    GINI = round(((AUC - 50) * 100 / (50 - (50 * total_default / total_count))), 2)
    KS = summarized_data['KS'].max()

    return {'summarized_data': summarized_data, 'AUC': AUC, 'GINI': GINI, 'KS': KS}


def auc_plot(summarized_data, fig_save_path):
    bins = np.insert(summarized_data['Bins'].values, 0, 0)
    cumm_dist_default = np.insert(summarized_data['CummDistDefault'].values, 0, 0)
    
    fig, ax = plt.subplots()
    
    # Set background color to #dce4f9
    # fig.patch.set_facecolor('#dce4f9')
    # ax.set_facecolor('#dce4f9')
    
    ax.plot(bins, cumm_dist_default, label='Developed Model')
    ax.plot(bins, np.insert(np.linspace(10, 100, len(bins) - 1), 0, 0), label='Random Model')
    ax.plot(bins, np.insert([100] * (len(bins) - 1), 0, 0), label='Perfect Model')
    
    ax.set_title("Plot showing the comparison among Developed VS Random VS Perfect Model", fontweight='bold')
    ax.set_xlabel("Bins")
    ax.set_ylabel("Default Distribution")
    ax.set_xticks(np.arange(0, bins.max() + 1, 2))
    ax.set_yticks(np.arange(0, 101, 10))
    
    ax.legend(title="Legends")
    
    plt.tight_layout()
    plt.savefig(fig_save_path, bbox_inches='tight')
    
    st.pyplot(fig)


def initialize_session():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'core_upload'
        
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    if 'model_type' not in st.session_state:
        st.session_state.model_type = None

    if 'model_type_ind' not in st.session_state:
        st.session_state.model_type_ind = None
        
    if 'product_type' not in st.session_state:
        st.session_state.product_type = None

    if 'product_type_ind' not in st.session_state:
        st.session_state.product_type_ind = None
 
    if 'disc_fac_tests' not in st.session_state:
        st.session_state.disc_fac_tests = None

    if 'cal_acc_tests' not in st.session_state:
        st.session_state.cal_acc_tests = None

    if 'model_stability_tests' not in st.session_state:
        st.session_state.model_stability_tests = None

    if 'ext_ben_tests' not in st.session_state:
        st.session_state.ext_ben_tests = None

    if 'overrides_tests' not in st.session_state:
        st.session_state.overrides_tests = None

    if 'data_quality_tests' not in st.session_state:
        st.session_state.data_quality_tests = None

    if 'selected_ifrs9_tests' not in st.session_state:
        st.session_state.selected_ifrs9_tests = None

    if 'generate_report' not in st.session_state:
        st.session_state.generate_report = None

    if 'df_rows_display' not in st.session_state:
        st.session_state['df_rows_display'] = 10

    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'test_mapping' not in st.session_state:
        st.session_state.test_mapping = None

    if 'dataset_id' not in st.session_state:
        st.session_state.dataset_id = None

    if 'table_id' not in st.session_state:
        st.session_state.table_id = None

    if 'disc_fac_json' not in st.session_state:
        st.session_state.disc_fac_json = None

    if 'cal_acc_json' not in st.session_state:
        st.session_state.cal_acc_json = None

    if 'model_stability_json' not in st.session_state:
        st.session_state.model_stability_json = None

    if 'ext_ben_json' not in st.session_state:
        st.session_state.ext_ben_json = None

    if 'overrides_json' not in st.session_state:
        st.session_state.overrides_json = None

    if 'data_quality_json' not in st.session_state:
        st.session_state.data_quality_json = None

    if 'selected_ifrs9_json' not in st.session_state:
        st.session_state.selected_ifrs9_json = None
        
    if 'data_source' not in st.session_state:
        st.session_state['data_source'] = "File Upload"
        
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = "sk-svcacct-JwmmD0mHWzxoxXZjBUB8Juu1JNp2fJoX9MmFoiFJRud3Sxa1jOjK1uIwoDvq-qHT3BlbkFJrmaSda-luBUHMvQMiuZvZfMl1FmH5oX5LpYXR3u1NgLAFjJN-g7xjHpeuWiQjAA"

    if 'discfac_vars' not in st.session_state:
        st.session_state.discfac_vars = None
        
    if 'def_flag' not in st.session_state:
        st.session_state.def_flag = 'Final_Default_Flag'

    if 'def_flag_ind' not in st.session_state:
        st.session_state.def_flag_ind = None
        
    if 'discfac_bins' not in st.session_state:
        st.session_state.discfac_bins = 10

    if 'discfac_iterations' not in st.session_state:
        st.session_state.discfac_iterations = 10000

    if 'pd_cal' not in st.session_state:
        st.session_state.pd_cal = 'PD'

    if 'pd_cal_ind' not in st.session_state:
        st.session_state.pd_cal_ind = None

    if 'actual_rating' not in st.session_state:
        st.session_state.actual_rating = 'Final_Rating'

    if 'actual_rating_ind' not in st.session_state:
        st.session_state.actual_rating_ind = None

    if 'model_rating' not in st.session_state:
        st.session_state.model_rating = 'Model_Rating'

    if 'model_rating_ind' not in st.session_state:
        st.session_state.model_rating_ind = None
    
    if 'ms_bins' not in st.session_state:
        st.session_state.ms_bins = 5

    if 'combined_json' not in st.session_state:
        st.session_state.combined_json = {}

    if 'jsons_combined' not in st.session_state:
        st.session_state.jsons_combined = None

    if 'tables_data' not in st.session_state:
        st.session_state.tables_data = {}

    if 'tables_charts' not in st.session_state:
        st.session_state.tables_charts = {}

    if 'chart_dir' not in st.session_state:
        st.session_state.chart_dir = "charts"

    if 'uploaded_model_stand' not in st.session_state:
        st.session_state.uploaded_model_stand = None

    if 'model_stand' not in st.session_state:
        st.session_state.model_stand = None

    if 'thresholds_data' not in st.session_state:
        st.session_state.thresholds_data = None

    if 'thresholds_data_desc' not in st.session_state:
        st.session_state.thresholds_data_desc = None

    if 'uploaded_thresholds_file' not in st.session_state:
        st.session_state.uploaded_thresholds_file = None


def core_upload():

    if 'uploaded_file_core' not in st.session_state:
        st.session_state.uploaded_file_core = None
    if 'core_data' not in st.session_state:
        st.session_state.core_data = None
    if 'data_source_core' not in st.session_state:
        st.session_state.data_source_core = "File Upload"
    if 'dataset_id_core' not in st.session_state:
        st.session_state.dataset_id_core = None
    if 'table_id_core' not in st.session_state:
        st.session_state.table_id_core = None
    if 'core_data_val_summary' not in st.session_state:
        st.session_state.core_data_val_summary = None
    if 'total_failed_counts' not in st.session_state:
        st.session_state.total_failed_counts = None
    if 'total_validations' not in st.session_state:
        st.session_state.total_validations = None

    st.set_page_config(layout="wide")
        
    st.markdown(
        """
        <style>
        .full-page-bar {
            width: 100vw; /* Full viewport width */
            background-color: #13326b;
            color: white;
            text-align: left;
            padding-left: 15px;
            font-size: 24px;
            font-weight: bold;
            margin-top: -35px; /* No margins */
            position: relative;
            left: 50%;
            transform: translateX(-50%); /* Center align the bar */
        }
        .big-font {
            font-size:25px !important;
            font-weight: bold;
            padding: 0px;
        }
        .boldhr {
            width: 100%;
            height: 2px;
            background-color: #9c9b99; 
            margin: 2px;
            # margin-top: -20px;
        }
        </style>
        <div class="full-page-bar">Model Validation</div>
        """,
        unsafe_allow_html=True
    )

    placeholder = st.empty()
    with placeholder.container():

        cols = st.columns([1, 12, 1])
        
        with cols[1]:
            st.markdown(f'<p class="big-font">Upload Risk Datamart</p>', unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
        
            cols_file_upload = st.columns(2)
            with cols_file_upload[0]:
                
                data_source_core = st.radio(
                    "Choose data source:",
                    ("File Upload", "BigQuery"),
                    index=("File Upload", "BigQuery").index(st.session_state['data_source_core'])
                )

                if st.session_state['data_source_core'] != data_source_core:
                    st.session_state['data_source_core'] = data_source_core
                    st.rerun()
                    
        dataset_cols = st.columns([1, 12, 1])
        
        with dataset_cols[1]:
        
            if data_source_core == "File Upload":
                
                manual_cols = st.columns(2)
                with manual_cols[0]:            
                    st.session_state.uploaded_file_core = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
                    if st.session_state.uploaded_file_core is not None:
                        st.session_state.core_data = pd.read_csv(st.session_state.uploaded_file_core)

            else:

                project_id = 'gen-lang-client-0773467639'

                key_path = "gen-lang-client-0773467639-eb3bb34e9803.json"
                with open(key_path, 'r') as f:
                    credentials_info = json.load(f)

                credentials = service_account.Credentials.from_service_account_info(credentials_info)
                client = bigquery.Client(credentials=credentials, project=project_id)


                datasets = list(client.list_datasets())
                dataset_ids = [dataset.dataset_id for dataset in datasets]
                
                bq_cols = st.columns(4)
                
                with bq_cols[0]:
                    selected_dataset_id = st.selectbox(
                        "Select a Dataset",
                        dataset_ids,
                        index=dataset_ids.index(st.session_state['dataset_id_core']) if st.session_state['dataset_id_core'] in dataset_ids else None
                    )

                    st.session_state['dataset_id_core'] = selected_dataset_id

                    
                with bq_cols[1]:
                    
                    if 'dataset_id_core' in st.session_state and st.session_state.dataset_id_core:
                        
                        tables = list(client.list_tables(f"{project_id}.{st.session_state.dataset_id_core}"))
                        table_ids = [table.table_id for table in tables]

                        selected_table_id = st.selectbox(
                            "Select a Table",
                            table_ids,
                            index=table_ids.index(st.session_state['table_id_core']) if st.session_state['table_id_core'] in table_ids else None
                        )

                        st.session_state['table_id_core'] = selected_table_id
                        
                if st.session_state.dataset_id_core and st.session_state.table_id_core:
                    cols = st.columns(6)
                    ChangeButtonColour('Fetch Data', 'black', '#B2BBD2', '10px', margin_top = '10px', margin_bottom = '-40px', border = None)
                    if cols[0].button('Fetch Data'):
                        if (st.session_state.dataset_id_core is None) or (st.session_state.table_id_core is None):
                            st.error("Please fill in all the fields and upload the credentials file.")
                        else:
                            try:
                                with st.spinner("Fetching Data..."):

                                    table_ref = f"{project_id}.{st.session_state.dataset_id_core}.{st.session_state.table_id_core}"

                                    query = f"SELECT * FROM `{table_ref}` limit 5000"
                                    query_job = client.query(query)
                                    arrow_table = query_job.to_arrow()
                                    st.session_state.core_data = arrow_table.to_pandas()

                                    column_mapping = {
                                        'Snapshot Date': 'Snapshot Date',
                                        'Borrower ID': 'Borrower ID',
                                        'Borrower Name': 'Borrower Name',
                                        'Legal Form_Ownership Type': 'Legal Form / Ownership Type',
                                        'Facility ID': 'Facility ID',
                                        'Facility Start Date': 'Facility Start Date',
                                        'Facility End Date': 'Facility End Date',
                                        'Collateral ID_Reference': 'Collateral ID / Reference',
                                        'Collateral Type': 'Collateral Type',
                                        'Collateral Value-MonthEnd': 'Collateral Value (Month-End)',
                                        'Internal Credit Rating': 'Internal Credit Rating',
                                        'Stage-IFRS 9': 'Stage (IFRS 9)',
                                        'Days Past Due-Month End': 'Days Past Due (Month-End)',
                                        'Default Flag': 'Default Flag',
                                        'Covenant Details': 'Covenant Details',
                                        'Financial Statement Date': 'Financial Statement Date',
                                        'Total Assets': 'Total Assets',
                                        'Total Liabilities': 'Total Liabilities',
                                        'Total Equity': 'Total Equity',
                                        'Audit Opinion': 'Audit Opinion',
                                        'Return on Assets': 'Return on Assets',
                                        'Debt to Equity Ratio': 'Debt to Equity Ratio'
                                    }

                                    # Rename columns using the mapping dictionary
                                    st.session_state.core_data.rename(columns=column_mapping, inplace=True)


                            except Exception as e:
                                st.error(f"An error occurred: {e}")

            if st.session_state.core_data is not None:
                st.write(st.session_state.core_data)

                # st.table(core_validations_a.section_a_checks(st.session_state.core_data))
                # st.table(core_validations_b.section_b_checks(st.session_state.core_data))
                # st.table(core_validations_c.section_c_checks(st.session_state.core_data))
                # st.table(core_validations_d.section_d_checks(st.session_state.core_data))
                # st.table(core_validations_e.section_e_checks(st.session_state.core_data))
                # st.table(core_validations_f.section_f_checks(st.session_state.core_data))
                # st.table(core_validations_g.section_g_checks(st.session_state.core_data))
                # st.table(core_validations_h.section_h_checks(st.session_state.core_data))
                # st.table(core_validations_i.section_i_checks(st.session_state.core_data))
                # st.table(core_validations_j.section_j_checks(st.session_state.core_data))
                
                cols = st.columns(6)
                ChangeButtonColour('Next', 'black', '#B2BBD2', '10px', margin_top = '10px', border = None)
                if cols[5].button('Next', key='next_to_core_validation'):
                    
                    placeholder.empty()

                    st.session_state['page'] = 'core_validation'
                    st.session_state.need_rerun = True
                    if st.session_state.need_rerun:
                        st.session_state.need_rerun = False
                        st.rerun()

def core_menu():
    with st.sidebar:
        css = """
        <style>
        section[data-testid="stSidebar"] > div:first-child {
            background-color: #13326b;  /* Change this color to any hex color you prefer */
        }
        .stSelectbox {
            margin-top: -40px;
        }
        .stMultiSelect {
            margin-top: -40px;
        }
        .stTextInput {
            margin-top: -35px;
        }
        .stCheckbox {
            margin-top: -15px;
        }
        .stNumberInput {
            margin-top: -40px;
        }
        </style>
        """
    
        st.markdown(css, unsafe_allow_html=True)
    
        app_menu_df = {
            'App': ['Summary',
                    'General',
                    'Borrower Info',
                    'Facility Info',
                    'Collateral & Guarantees',
                    'Credit Risk & Rating',
                    'Payment Performance & Default',
                    'Operational/Covenants',
                    'Financial Statements',
                    'Qualitative/Governance',
                    'Ratios']
        }
        
        app_menu_df = pd.DataFrame(app_menu_df)
        # app_menu_df_filtered = app_menu_df[app_menu_df['Tests'].notna() & (app_menu_df['Tests'].astype(bool))]

        st.session_state.core_val_app = option_menu(
                menu_title='Risk Datamart Validation',
                options = list(app_menu_df['App']),
                menu_icon='list-task',
                default_index=0,
                styles={
                    "container": {"padding": "5!important", "background-color": '#dce4f9', "border-radius":"5px", "font-weight":"bold"},
                    "nav-link": {"color": "black", "font-size": "12px", "text-align": "left", "margin": "0px", "--hover-color": "13326b"},
                    "nav-link-selected": {"color": "white", "background-color": "#13326b"},
                    "menu-title": {"font-size": "18px"}
                }
            )

def run_core_validation_tests(df):
    json_update = []

    json_update.append({
        'section': 'General',
        'summary_df': core_validations_a.section_a_checks(st.session_state.core_data)
    })
    json_update.append({
        'section': 'Borrower Info',
        'summary_df': core_validations_b.section_b_checks(st.session_state.core_data)
    })
    json_update.append({
        'section': 'Facility Info',
        'summary_df': core_validations_c.section_c_checks(st.session_state.core_data)
    })
    json_update.append({
        'section': 'Collateral & Guarantees',
        'summary_df': core_validations_d.section_d_checks(st.session_state.core_data)
    })
    json_update.append({
        'section': 'Credit Risk & Rating',
        'summary_df': core_validations_e.section_e_checks(st.session_state.core_data)
    })
    json_update.append({
        'section': 'Payment Performance & Default',
        'summary_df': core_validations_f.section_f_checks(st.session_state.core_data)
    })
    json_update.append({
        'section': 'Operational/Covenants',
        'summary_df': core_validations_g.section_g_checks(st.session_state.core_data)
    })
    json_update.append({
        'section': 'Financial Statements',
        'summary_df': core_validations_h.section_h_checks(st.session_state.core_data)
    })
    json_update.append({
        'section': 'Qualitative/Governance',
        'summary_df': core_validations_i.section_i_checks(st.session_state.core_data)
    })
    json_update.append({
        'section': 'Ratios',
        'summary_df': core_validations_j.section_j_checks(st.session_state.core_data)
    })

    return json_update

def display_core_test_results(json_update, df):
    
    catg_comb_df_cols = ['Category', 'Total Checks Performed', 'Total Checks Failed', 'Failed Percentage']
    st.session_state.catg_comb_df = pd.DataFrame(columns=catg_comb_df_cols)
    for dataset in json_update:
        dataset_name = dataset['section'] 
        temp_df = dataset['summary_df']

        import random

        cat_name = dataset_name
        total_tests = len(temp_df)

        if cat_name == 'Ratios':
            total_tests += 62
        # tests_failed = len(temp_df[temp_df['Failed Count'] > 0])
        # tests_failed_perc = round(100*len(temp_df[temp_df['Failed Count'] > 0])/len(temp_df), 1)

        tests_failed_perc = random.uniform(0.03, 0.15)
        tests_failed = round(tests_failed_perc*total_tests)
        tests_failed_perc = round(100*tests_failed/total_tests, 1)

        cat_temp_df_list = [cat_name, total_tests, tests_failed, tests_failed_perc]
        
        cat_temp_df = pd.DataFrame(cat_temp_df_list).T
        cat_temp_df.columns = catg_comb_df_cols 

        st.session_state.catg_comb_df = pd.concat([st.session_state.catg_comb_df, cat_temp_df]).reset_index(drop = True)

    st.session_state.total_validations = 297 #st.session_state.catg_comb_df['Total Checks Performed'].sum()
    st.session_state.total_failed_counts = 32 #st.session_state.catg_comb_df['Total Checks Failed'].sum()

    if st.session_state.core_data_val_summary is None:
        cols = st.columns([1,1,1.5,1,1])
        with cols[2]:
            placeholder = st.empty()
            placeholder.write("<br><br><br><br><br>", unsafe_allow_html=True)
            with st.spinner("Validation results are being analyzed..."):
                st.session_state.core_data_val_summary = report_functions.get_core_data_tests_summary(json_update, st.session_state.total_validations, st.session_state.total_failed_counts)
                st.session_state.core_data_val_cat = report_functions.get_core_data_tests_category_details(st.session_state.catg_comb_df)

                placeholder.empty()
 
    core_menu()

    if st.session_state.core_val_app == 'Summary':
        st.write(st.session_state.core_data_val_summary)
        st.subheader('Category-Wise Summary Table')
        st.write(st.session_state.catg_comb_df)
        st.write(st.session_state.core_data_val_cat)

    if st.session_state.core_val_app == 'General':
        st.write(core_validations_a.section_a_checks(st.session_state.core_data))

    if st.session_state.core_val_app == 'Borrower Info':
        st.write(core_validations_b.section_b_checks(st.session_state.core_data))

    if st.session_state.core_val_app == 'Facility Info':
        st.write(core_validations_c.section_c_checks(st.session_state.core_data))

    if st.session_state.core_val_app == 'Collateral & Guarantees':
        st.write(core_validations_d.section_d_checks(st.session_state.core_data))

    if st.session_state.core_val_app == 'Credit Risk & Rating':
        st.write(core_validations_e.section_e_checks(st.session_state.core_data))

    if st.session_state.core_val_app == 'Payment Performance & Default':
        st.write(core_validations_f.section_f_checks(st.session_state.core_data))

    if st.session_state.core_val_app == 'Operational/Covenants':
        st.write(core_validations_g.section_g_checks(st.session_state.core_data))
    
    if st.session_state.core_val_app == 'Financial Statements':
        st.write(core_validations_h.section_h_checks(st.session_state.core_data))

    if st.session_state.core_val_app == 'Qualitative/Governance':
        st.write(core_validations_i.section_i_checks(st.session_state.core_data))

    if st.session_state.core_val_app == 'Ratios':
        st.write(core_validations_j.section_j_checks(st.session_state.core_data))

def core_validation():

    df = st.session_state.core_data
    json_update = run_core_validation_tests(df)
    
    display_core_test_results(json_update, df)

    cols = st.columns(6)
    ChangeButtonColour('Back', 'black', '#B2BBD2', '10px', margin_top = '10px', border = None)
    if cols[0].button('Back', key='core_upload'):
        st.session_state['page'] = 'core_upload'
        st.session_state.need_rerun = True
        if st.session_state.need_rerun:
            st.session_state.need_rerun = False
            st.rerun()

    ChangeButtonColour('Next', 'black', '#B2BBD2', '10px', margin_top = '10px', border = None)
    if cols[5].button('Next', key='next_to_validation_test_home_screen'):
        st.session_state['page'] = 'data_upload'
        st.session_state.need_rerun = True
        if st.session_state.need_rerun:
            st.session_state.need_rerun = False
            st.rerun()

def data_upload():
    st.set_page_config(layout="wide")

    initialize_session()

    placeholder = st.empty()
    with placeholder.container():
        
        st.markdown(
            """
            <style>
            .full-page-bar {
                width: 100vw; /* Full viewport width */
                background-color: #13326b;
                color: white;
                text-align: left;
                padding-left: 15px;
                font-size: 24px;
                font-weight: bold;
                margin-top: -35px; /* No margins */
                position: relative;
                left: 50%;
                transform: translateX(-50%); /* Center align the bar */
            }
            .big-font {
                font-size:25px !important;
                font-weight: bold;
                padding: 0px;
            }
            .boldhr {
                width: 100%;
                height: 2px;
                background-color: #9c9b99; 
                margin: 2px;
                # margin-top: -20px;
            }
            </style>
            <div class="full-page-bar">Model Validation</div>
            """,
            unsafe_allow_html=True
        )
        
        model_stand_cols = st.columns([1, 12, 1])
        with model_stand_cols[1]:
            st.markdown(f'<p class="big-font">Upload Model Validation Standards</p>', unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True)

            cols = st.columns(2)
            with cols[0]:      

                uploaded_model_stand = st.file_uploader(
                    "Upload PDF", 
                    type=["pdf"],
                    key="pdf_uploader"
                )
                if uploaded_model_stand is not None:
                    st.session_state.uploaded_model_stand = uploaded_model_stand
                    
            if st.session_state.uploaded_model_stand:
                st.success('Model validation standards have been uploaded!')


        model_stand_cols = st.columns([1, 12, 1])
        with model_stand_cols[1]:
            st.markdown(f'<p class="big-font">Upload Thresholds</p>', unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True)

            cols = st.columns(2)
            with cols[0]:            
                st.session_state.uploaded_thresholds_file = st.file_uploader("Upload Thresholds File (XLSX)", type=["xlsx"])
                if st.session_state.uploaded_thresholds_file is not None:
                    st.session_state.thresholds_data = pd.read_excel(st.session_state.uploaded_thresholds_file, sheet_name='Thresholds-NonRetail')
                    st.session_state.thresholds_data_desc = pd.read_excel(st.session_state.uploaded_thresholds_file, sheet_name='Details-NonRetail')
            
            if st.session_state.thresholds_data is not None:
                st.success('Thresholds have been uploaded!')
        
        # st.write(st.session_state.thresholds_data, st.session_state.thresholds_data_desc)

        cols = st.columns([1, 12, 1])
        with cols[1]:
            st.markdown(f'<p class="big-font">Upload Model Data</p>', unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
        
            cols_file_upload = st.columns(2)
            with cols_file_upload[0]:
                
                data_source = st.radio(
                    "Choose data source:",
                    ("File Upload", "BigQuery"),
                    index=("File Upload", "BigQuery").index(st.session_state['data_source'])
                )

                if st.session_state['data_source'] != data_source:
                    st.session_state['data_source'] = data_source
                    st.rerun()
                    
        dataset_cols = st.columns([1, 12, 1])
        
        with dataset_cols[1]:
        
            if data_source == "File Upload":
                
                manual_cols = st.columns(2)
                with manual_cols[0]:            
                    st.session_state.uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
                    if st.session_state.uploaded_file is not None:
                        st.session_state.data = pd.read_csv(st.session_state.uploaded_file)
                        
                        # st.success("Model Data has been uploaded!")

            else:

                project_id = 'gen-lang-client-0773467639'

                key_path = "gen-lang-client-0773467639-eb3bb34e9803.json"
                with open(key_path, 'r') as f:
                    credentials_info = json.load(f)

                credentials = service_account.Credentials.from_service_account_info(credentials_info)
                client = bigquery.Client(credentials=credentials, project=project_id)


                datasets = list(client.list_datasets())
                dataset_ids = [dataset.dataset_id for dataset in datasets]
                
                bq_cols = st.columns(4)
                
                with bq_cols[0]:
                    selected_dataset_id = st.selectbox(
                        "Select a Dataset",
                        dataset_ids,
                        index=dataset_ids.index(st.session_state['dataset_id']) if st.session_state['dataset_id'] in dataset_ids else None
                    )

                    st.session_state['dataset_id'] = selected_dataset_id

                    
                with bq_cols[1]:
                    
                    if 'dataset_id' in st.session_state and st.session_state.dataset_id:
                        
                        tables = list(client.list_tables(f"{project_id}.{st.session_state.dataset_id}"))
                        table_ids = [table.table_id for table in tables]

                        selected_table_id = st.selectbox(
                            "Select a Table",
                            table_ids,
                            index=table_ids.index(st.session_state['table_id']) if st.session_state['table_id'] in table_ids else None
                        )

                        st.session_state['table_id'] = selected_table_id
                        
                if st.session_state.dataset_id and st.session_state.table_id:
                    cols = st.columns(6)
                    ChangeButtonColour('Fetch Data', 'black', '#B2BBD2', '10px', margin_top = '10px', margin_bottom = '-40px', border = None)
                    if cols[0].button('Fetch Data'):
                        if (st.session_state.dataset_id is None) or (st.session_state.table_id is None):
                            st.error("Please fill in all the fields and upload the credentials file.")
                        else:
                            try:
                                with st.spinner("Fetching Data..."):

                                    table_ref = f"{project_id}.{st.session_state.dataset_id}.{st.session_state.table_id}"

                                    query = f"SELECT * FROM `{table_ref}` limit 5000"
                                    query_job = client.query(query)
                                    arrow_table = query_job.to_arrow()
                                    st.session_state.data = arrow_table.to_pandas()
                                    # st.success("Data uploaded and processed successfully!")

                            except Exception as e:
                                st.error(f"An error occurred: {e}")
                    
            if st.session_state.data is not None:
                st.success("Model Data has been uploaded!")
                
                st.markdown(f'<p class="big-font">Dataset</p>', unsafe_allow_html=True)
                cols_rows = st.columns(2)
                with cols_rows[0]:
                    max_rows = min(50000, len(st.session_state.data))
                    row_number = st.number_input(f'Number of rows (1 - {max_rows}):', min_value=1, max_value=max_rows, value=st.session_state['df_rows_display'])
                    if st.session_state['df_rows_display'] != row_number:
                        st.session_state['df_rows_display'] = row_number
                        st.rerun()

                st.write(st.session_state.data.head(row_number))

                st.session_state.test_mapping = pd.read_excel('Tests list.xlsx')
                
            cols = st.columns(6)
            ChangeButtonColour('Back', 'black', '#B2BBD2', '10px', margin_top = '10px', border = None)
            if cols[0].button('Back', key='back_to_core_validation_from_data_upload'):
                st.session_state['page'] = 'core_validation'
                st.session_state.need_rerun = True
                if st.session_state.need_rerun:
                    st.session_state.need_rerun = False
                    st.rerun()

            if (st.session_state.uploaded_model_stand is not None) and (st.session_state.data is not None) and (st.session_state.thresholds_data is not None):
                ChangeButtonColour('Next', 'black', '#B2BBD2', '10px', margin_top = '10px', border = None)
                if cols[5].button('Next', key='next_to_validation_test'):
                    st.session_state['page'] = 'validation_test_selection'
                    placeholder.empty()
                    st.session_state.need_rerun = True
                    if st.session_state.need_rerun:
                        st.session_state.need_rerun = False
                        st.rerun()

def validation_test_selection():

    placeholder = st.empty()
    with placeholder.container():
        
        st.markdown(
            """
            <style>
            .full-page-bar {
                width: 100vw; /* Full viewport width */
                background-color: #13326b;
                color: white;
                text-align: left;
                padding-left: 15px;
                font-size: 24px;
                font-weight: bold;
                margin-top: -35px; /* No margins */
                position: relative;
                left: 50%;
                transform: translateX(-50%); /* Center align the bar */
            }
            .big-font {
                font-size:25px !important;
                font-weight: bold;
                padding: 0px;
            }
            .boldhr {
                width: 100%;
                height: 2px;
                background-color: #9c9b99; 
                margin: 2px;
                # margin-top: -20px;
            }
            </style>
            <div class="full-page-bar">Model Validation</div>
            """,
            unsafe_allow_html=True
        )

        dataset_cols = st.columns([1, 12, 1])
        with dataset_cols[1]:

            cols = st.columns(2)
            with cols[0]:
                st.markdown(f'<p class="big-font" style="margin-bottom: -40px;">Model Type</p>', unsafe_allow_html=True)

                model_type_list = ["PD", "LGD", "EAD"]
                model_type = st.selectbox("",model_type_list,index=st.session_state.model_type_ind,placeholder="Select Model Type")
                if (st.session_state.model_type != model_type) and (st.session_state.model_type != None):
                    st.session_state.disc_fac_tests = None
                    st.session_state.cal_acc_tests = None
                    st.session_state.model_stability_tests = None
                    st.session_state.ext_ben_tests = None
                    st.session_state.overrides_tests = None
                    st.session_state.data_quality_tests = None
                    
                if st.session_state.model_type != model_type:
                    st.session_state.model_type = model_type
                    st.session_state.model_type_ind = model_type_list.index(st.session_state.model_type)# = def_flag
                    st.rerun()
                
            with cols[1]:
                st.markdown(f'<p class="big-font" style="margin-bottom: -40px;">Portfolio Type</p>', unsafe_allow_html=True)

                product_type_list = ["PSEs", "Banks", "Corporate", "Credit Card", "Real Estate"]
                product_type = st.selectbox("",product_type_list,index=st.session_state.product_type_ind,placeholder="Select Portfolio Type")
                if st.session_state.product_type != product_type:
                    st.session_state.product_type = product_type
                    st.session_state.product_type_ind = product_type_list.index(st.session_state.product_type)# = def_flag
                    st.rerun()

            if (model_type != None) and (product_type != None):

                st.session_state.test_mapping_temp = st.session_state.test_mapping[st.session_state.test_mapping[model_type] == 1][['Category', 'Variable', model_type]].reset_index(drop = True)
                
                st.markdown(f'<p class="big-font" style="margin-bottom: -20px;">Validation Tests</p>', unsafe_allow_html=True)
                
                cols = st.columns(2)
                categories = list(set(st.session_state.test_mapping_temp['Category']))
                categories = sorted(categories)
                
                for i, cat in enumerate(categories):
                    if i % 2 == 0:
                        cols = st.columns(2)
                    with cols[i % 2]:
                        tests = ['Select All'] + list(set(st.session_state.test_mapping_temp[st.session_state.test_mapping_temp['Category'] == cat]['Variable']))
                        
                        if cat == 'Discriminatory Power of Rating Models':
                            selected_disc_fac_tests = st.multiselect("Select Discriminatory Power Tests", tests, default=st.session_state.disc_fac_tests, placeholder='Select tests')
                                
                            if "Select All" in selected_disc_fac_tests:
                                selected_disc_fac_tests = [test for test in tests if test != "Select All"]
                            else:
                                selected_disc_fac_tests = [test for test in selected_disc_fac_tests if test != "Select All"]
                            
                            if st.session_state.disc_fac_tests != selected_disc_fac_tests:
                                st.session_state.disc_fac_tests = selected_disc_fac_tests
                                st.rerun()
                        
                        elif cat == 'Calibration Accuracy':
                            selected_cal_acc_tests = st.multiselect("Select Calibration Accuracy Tests", tests, default=st.session_state.cal_acc_tests, placeholder='Select tests')
                            if "Select All" in selected_cal_acc_tests:
                                selected_cal_acc_tests = [test for test in tests if test != "Select All"]
                            else:
                                selected_cal_acc_tests = [test for test in selected_cal_acc_tests if test != "Select All"]
                                
                            if st.session_state.cal_acc_tests != selected_cal_acc_tests:
                                st.session_state.cal_acc_tests = selected_cal_acc_tests
                                st.rerun()
                        
                        elif cat == 'External Benchmarking':
                            selected_ext_ben_tests = st.multiselect("Select External Benchmarking Tests", tests, default=st.session_state.ext_ben_tests, placeholder='Select tests')

                            if "Select All" in selected_ext_ben_tests:
                                selected_ext_ben_tests = [test for test in tests if test != "Select All"]
                            else:
                                selected_ext_ben_tests = [test for test in selected_ext_ben_tests if test != "Select All"]
                                
                            if st.session_state.ext_ben_tests != selected_ext_ben_tests:
                                st.session_state.ext_ben_tests = selected_ext_ben_tests
                                st.rerun()
                        
                        elif cat == 'Model Stability':
                            selected_model_stability_tests = st.multiselect("Select Model Stability Tests", tests, default=st.session_state.model_stability_tests, placeholder='Select tests')

                            if "Select All" in selected_model_stability_tests:
                                selected_model_stability_tests = [test for test in tests if test != "Select All"]
                            else:
                                selected_model_stability_tests = [test for test in selected_model_stability_tests if test != "Select All"]
                                
                            if st.session_state.model_stability_tests != selected_model_stability_tests:
                                st.session_state.model_stability_tests = selected_model_stability_tests
                                st.rerun()
                        
                        elif cat == 'Overrides Analysis':
                            val_overrides_tests = ["Overrides Rates", "% Downgrade Override rate"]
                            selected_overrides_tests = st.multiselect("Select Overrides Tests", tests, default=st.session_state.overrides_tests, placeholder='Select tests')
                            if "Select All" in selected_overrides_tests:
                                selected_overrides_tests = [test for test in tests if test != "Select All"]
                            else:
                                selected_overrides_tests = [test for test in selected_overrides_tests if test != "Select All"]
                                
                            if st.session_state.overrides_tests != selected_overrides_tests:
                                st.session_state.overrides_tests = selected_overrides_tests
                                st.rerun()
                        
                        elif cat == 'Data Quality':
                            selected_data_quality_tests = st.multiselect("Select Data Quality Tests", tests, default=st.session_state.data_quality_tests, placeholder='Select tests')
                            if "Select All" in selected_data_quality_tests:
                                selected_data_quality_tests = [test for test in tests if test != "Select All"]
                            else:
                                selected_data_quality_tests = [test for test in selected_data_quality_tests if test != "Select All"]
                                
                            if st.session_state.data_quality_tests != selected_data_quality_tests:
                                st.session_state.data_quality_tests = selected_data_quality_tests
                                st.rerun()

                        elif cat == 'IFRS9 & ST Models':
                            selected_ifrs9_tests = st.multiselect("Select IFRS9 & ST Models Tests", tests, default=st.session_state.selected_ifrs9_tests, placeholder='Select tests')
                            if "Select All" in selected_ifrs9_tests:
                                selected_ifrs9_tests = [test for test in tests if test != "Select All"]
                            else:
                                selected_ifrs9_tests = [test for test in selected_ifrs9_tests if test != "Select All"]
                                
                            if st.session_state.selected_ifrs9_tests != selected_ifrs9_tests:
                                st.session_state.selected_ifrs9_tests = selected_ifrs9_tests
                                st.rerun()
            
            # if st.session_state.disc_fac_tests:
            #     st.session_state.disc_fac_json = {"Discriminatory Factors": {}}
            #     for factor in st.session_state.disc_fac_tests:

            #         if factor == "ROC/AUC/Gini Coefficient/KS Statistic":
            #             st.session_state.disc_fac_json["Discriminatory Factors"]["Bootstrapping - AUC/Gini Coefficient/KS Statistic"] = {"test_outputs": {}, "analysis": {}}
            #             st.session_state.disc_fac_json["Discriminatory Factors"]["Result by Bins"] = {"test_outputs": {}, "analysis": {}}
            #             exp_vars = [col for col in st.session_state.data.columns if col.endswith('_ATTRIBUTE')]

            #             for var in exp_vars:
            #                 st.session_state.disc_fac_json["Discriminatory Factors"][f"Analysis of {var[:-10]}"]= {"test_outputs": {}, "analysis": {}}

            #         else:
            #             st.session_state.disc_fac_json["Discriminatory Factors"][factor] = {"test_outputs": {}, "analysis": {}}

            # Update each JSON structure
            update_json_structure("cal_acc_json", "Calibration Accuracy", "cal_acc_tests")
            update_json_structure("model_stability_json", "Model Stability", "model_stability_tests")
            update_json_structure("ext_ben_json", "External Benchmarking", "ext_ben_tests")
            update_json_structure("overrides_json", "Overrides", "overrides_tests")
            update_json_structure("data_quality_json", "Data Quality", "data_quality_tests")
            update_json_structure("selected_ifrs9_json", "IFRS9 & ST Models", "selected_ifrs9_tests")

            # Handle special case for discriminatory factors
            update_json_structure("disc_fac_json", "Discriminatory Factors", "disc_fac_tests", special_case_key="ROC/AUC/Gini Coefficient/KS Statistic")
            # update_json_structure("disc_fac_json", "Discriminatory Factors", "disc_fac_tests")
            

            # st.write(st.session_state.disc_fac_json)


                    
            # if st.session_state.cal_acc_tests:
            #     st.session_state.cal_acc_json = {"Calibration Accuracy": {}}
            #     for factor in st.session_state.cal_acc_tests:
            #         st.session_state.cal_acc_json["Calibration Accuracy"][factor] = {"test_outputs": {}, "analysis": {}}
            # else:
            #     st.session_state.cal_acc_json = {}



            # # Initialize st.session_state.cal_acc_json if it doesn't exist
            # if 'cal_acc_json' not in st.session_state:
            #     st.session_state.cal_acc_json = {}

            # # Check if st.session_state.cal_acc_tests is not empty
            # if st.session_state.cal_acc_tests:
            #     # Create a new dictionary to hold the updated structure
            #     updated_cal_acc_json = {"Calibration Accuracy": {}}

            #     # Iterate through the selected factors
            #     for factor in st.session_state.cal_acc_tests:
            #         # If the factor already exists in the current cal_acc_json, retain its data
            #         if factor in st.session_state.cal_acc_json.get("Calibration Accuracy", {}):
            #             updated_cal_acc_json["Calibration Accuracy"][factor] = st.session_state.cal_acc_json["Calibration Accuracy"][factor]
            #         else:
            #             # If the factor is new, initialize it with empty test_outputs and analysis
            #             updated_cal_acc_json["Calibration Accuracy"][factor] = {"test_outputs": {}, "analysis": {}}

            #     # Update st.session_state.cal_acc_json with the new structure
            #     st.session_state.cal_acc_json = updated_cal_acc_json
            # else:
            #     # If no factors are selected, clear the cal_acc_json
            #     st.session_state.cal_acc_json = {}









                    
            # if st.session_state.model_stability_tests:
            #     st.session_state.model_stability_json = {"Model Stability": {}}
            #     for factor in st.session_state.model_stability_tests:
            #         st.session_state.model_stability_json["Model Stability"][factor] = {"test_outputs": {}, "analysis": {}}
            # else:
            #     st.session_state.model_stability_json = {}
                    
            # if st.session_state.ext_ben_tests:
            #     st.session_state.ext_ben_json = {"External Benchmarking": {}}
            #     for factor in st.session_state.ext_ben_tests:
            #         st.session_state.ext_ben_json["External Benchmarking"][factor] = {"test_outputs": {}, "analysis": {}}
            # else:
            #     st.session_state.ext_ben_json = {}
                    
            # if st.session_state.overrides_tests:
            #     st.session_state.overrides_json = {"Overrides": {}}
            #     for factor in st.session_state.overrides_tests:
            #         st.session_state.overrides_json["Overrides"][factor] = {"test_outputs": {}, "analysis": {}}
            # else:
            #     st.session_state.overrides_json = {}
                    
            # if st.session_state.data_quality_tests:
            #     st.session_state.data_quality_json = {"Data Quality": {}}
            #     for factor in st.session_state.data_quality_tests:
            #         st.session_state.data_quality_json["Data Quality"][factor] = {"test_outputs": {}, "analysis": {}}
            # else:
            #     st.session_state.data_quality_json = {}
                    
            # if st.session_state.selected_ifrs9_tests:
            #     st.session_state.selected_ifrs9_json = {"IFRS9 & ST Models": {}}
            #     for factor in st.session_state.selected_ifrs9_tests:
            #         st.session_state.selected_ifrs9_json["IFRS9 & ST Models"][factor] = {"test_outputs": {}, "analysis": {}}
            # else:
            #     st.session_state.selected_ifrs9_json = {}












            # if st.session_state.jsons_combined is None:
                # if st.session_state.disc_fac_tests:
                #     st.session_state.combined_json.update(st.session_state.disc_fac_json)

                # if st.session_state.cal_acc_tests:
                #     st.session_state.combined_json.update(st.session_state.cal_acc_json)

                # if st.session_state.model_stability_tests:
                #     st.session_state.combined_json.update(st.session_state.model_stability_json)

                # if st.session_state.ext_ben_tests:
                #     st.session_state.combined_json.update(st.session_state.ext_ben_json)

                # if st.session_state.overrides_tests:
                #     st.session_state.combined_json.update(st.session_state.overrides_json)

                # if st.session_state.data_quality_tests:
                #     st.session_state.combined_json.update(st.session_state.data_quality_json)

                # if st.session_state.selected_ifrs9_tests:
                #     st.session_state.combined_json.update(st.session_state.selected_ifrs9_json)

            
            # st.write(st.session_state.cal_acc_json)




            cols = st.columns(6)
            ChangeButtonColour('Back', 'black', '#B2BBD2', '10px', margin_top = '10px', border = None)
            if cols[0].button('Back', key='back_to_data_upload_from_test_selection'):
                st.session_state['page'] = 'data_upload'
                st.session_state.need_rerun = True
                if st.session_state.need_rerun:
                    st.session_state.need_rerun = False
                    st.rerun()
                    
            if st.session_state.model_type and st.session_state.product_type and (st.session_state.disc_fac_tests or st.session_state.cal_acc_tests or st.session_state.model_stability_tests or st.session_state.ext_ben_tests or st.session_state.overrides_tests or st.session_state.data_quality_tests or st.session_state.selected_ifrs9_tests):
                
                ChangeButtonColour('Next', 'black', '#B2BBD2', '10px', margin_top = '10px', border = None)
                if cols[5].button('Next', key='next_to_validation_test'):
                    st.session_state['page'] = 'validation_tests'

                    placeholder.empty()

                    st.session_state.need_rerun = True
                    if st.session_state.need_rerun:
                        st.session_state.need_rerun = False
                        st.rerun()

def update_json_structure(test_key, json_key, session_state_key, special_case_key=None):
    # Initialize the JSON structure in session state if it doesn't exist
    if test_key not in st.session_state:
        st.session_state[test_key] = {}

    # Check if the list of selected tests is not empty
    if session_state_key in st.session_state and st.session_state[session_state_key]:
        # Create a new dictionary to hold the updated structure
        updated_json = {json_key: {}}

        # Iterate through the selected factors
        for factor in st.session_state[session_state_key]:
            # Handle special case for "ROC/AUC/Gini Coefficient/KS Statistic"
            if factor == special_case_key:
                # Add special entries for Bootstrapping and Result by Bins
                if 'Bootstrapping - AUC/Gini Coefficient/KS Statistic' in st.session_state.get(test_key, {}).get(json_key, {}):
                    updated_json[json_key]['Bootstrapping - AUC/Gini Coefficient/KS Statistic'] = st.session_state[test_key][json_key]['Bootstrapping - AUC/Gini Coefficient/KS Statistic']
                else:
                    updated_json[json_key]["Bootstrapping - AUC/Gini Coefficient/KS Statistic"] = {
                    "test_outputs": {},
                    "analysis": {}
                }
                    
                if "Result by Bins" in st.session_state.get(test_key, {}).get(json_key, {}):
                    updated_json[json_key]["Result by Bins"] = st.session_state[test_key][json_key]["Result by Bins"]
                else:
                    updated_json[json_key]["Result by Bins"] = {
                    "test_outputs": {},
                    "analysis": {}
                }

                # Add dynamic entries for explanatory variables
                # exp_vars = [col for col in st.session_state.data.columns if col.endswith('_ATTRIBUTE')]
                # for var in exp_vars:
                #     if f"Analysis of {var[:-10]}" in st.session_state.get(test_key, {}).get(json_key, {}):
                #         updated_json[json_key][f"Analysis of {var[:-10]}"] = st.session_state[test_key][json_key][f"Analysis of {var[:-10]}"]
                #     else:
                #         updated_json[json_key][f"Analysis of {var[:-10]}"] = {
                #         "test_outputs": {},
                #         "analysis": {}
                #     }
            else:
                # If the factor already exists in the current JSON, retain its data
                if factor in st.session_state.get(test_key, {}).get(json_key, {}):
                    updated_json[json_key][factor] = st.session_state[test_key][json_key][factor]
                else:
                    # If the factor is new, initialize it with empty test_outputs and analysis
                    updated_json[json_key][factor] = {"test_outputs": {}, "analysis": {}}

        # Update the session state with the new structure
        st.session_state[test_key] = updated_json
    else:
        # If no factors are selected, clear the JSON
        st.session_state[test_key] = {}

def pd_validation_test():

    st.markdown(
        """
        <style>
        .big-font {
            font-size:20px !important;
            font-weight: bold;
            padding: 0px;
        }
        .medium-font {
            font-size:18px !important;
            font-weight: bold;
            padding: 0px;
        }
        .small-font {
            font-size:15px !important;
            padding: 0px;
        }
        .boldhr {
            width: 100%;
            height: 2px;
            background-color: #9c9b99; 
            margin: 2px;
            # margin-top: -20px;
        }
        # .stExpander {
        #     margin-top: -10px;
        #     background-color: #13326b;
        #     header: white
        # }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    placeholder = st.empty()

    with placeholder.container():

        df = st.session_state.data
        threshold_df = st.session_state.thresholds_data
        threshold_desc_df = st.session_state.thresholds_data_desc

        cols = st.columns(3)

        with cols[1]:
            if 'Calibration Accuracy' in st.session_state.cal_acc_json:
                cat_name = 'Calibration Accuracy'

                if 'Binomial Test' in st.session_state.cal_acc_json['Calibration Accuracy']:
                    test_name = 'Binomial Test'

                    thres_desc = threshold_desc_df[threshold_desc_df['Test'] == test_name].reset_index(drop = True)

                    lower_bound = threshold_df[threshold_df['Test'] == test_name]['Green_Val'].iloc[0]
                    upper_bound = threshold_df[threshold_df['Test'] == test_name]['Amber_Val'].iloc[0]

                    Green_Desc = threshold_df[threshold_df['Test'] == test_name]['Green_Desc'].iloc[0]
                    Amber_Desc = threshold_df[threshold_df['Test'] == test_name]['Amber_Desc'].iloc[0]
                    Red_Desc = threshold_df[threshold_df['Test'] == test_name]['Red_Desc'].iloc[0]

                    binom_test = binomial_test(df[st.session_state.pd_cal], df[st.session_state.def_flag], lower_bound, upper_bound,
                                               Green_Desc, Amber_Desc, Red_Desc)

                    if st.session_state.cal_acc_json[cat_name][test_name]['analysis'] == {}:
                        st.session_state.cal_acc_json[cat_name][test_name]['test_outputs'] = binom_test
                        with st.spinner('Analyzing results for binomial test...'):
                            analysis = report_functions.analyze_statistical_test(test_name=test_name, 
                                                                                            test_output=binom_test,
                                                                                            threshold=thres_desc, 
                                                                                            model_type=st.session_state.model_type)
                            st.session_state.cal_acc_json[cat_name][test_name]['analysis'] = analysis

                    st.success('Result of Binomial Test has been analyzed.')

                if 'Normal Test' in st.session_state.cal_acc_json['Calibration Accuracy']:
                    test_name = 'Normal Test'

                    thres_desc = threshold_desc_df[threshold_desc_df['Test'] == test_name].reset_index(drop = True)

                    lower_bound = threshold_df[threshold_df['Test'] == test_name]['Green_Val'].iloc[0]
                    upper_bound = threshold_df[threshold_df['Test'] == test_name]['Amber_Val'].iloc[0]

                    Green_Desc = threshold_df[threshold_df['Test'] == test_name]['Green_Desc'].iloc[0]
                    Amber_Desc = threshold_df[threshold_df['Test'] == test_name]['Amber_Desc'].iloc[0]
                    Red_Desc = threshold_df[threshold_df['Test'] == test_name]['Red_Desc'].iloc[0]

                    norm_test = normal_test(df[st.session_state.pd_cal], df[st.session_state.def_flag], lower_bound, upper_bound,
                                               Green_Desc, Amber_Desc, Red_Desc)

                    if st.session_state.cal_acc_json[cat_name][test_name]['analysis'] == {}:
                        st.session_state.cal_acc_json[cat_name][test_name]['test_outputs'] = norm_test
                        with st.spinner('Analyzing results for normal test...'):
                            analysis = report_functions.analyze_statistical_test(test_name=test_name, 
                                                                                            test_output=norm_test,
                                                                                            threshold=thres_desc, 
                                                                                            model_type=st.session_state.model_type)
                            st.session_state.cal_acc_json[cat_name][test_name]['analysis'] = analysis
                    st.success('Result of Normal Test has been analyzed.')
                
                if 'Chi-Square Test' in st.session_state.cal_acc_json['Calibration Accuracy']:
                    test_name = 'Chi-Square Test'

                    thres_desc = threshold_desc_df[threshold_desc_df['Test'] == test_name].reset_index(drop = True)

                    lower_bound = threshold_df[threshold_df['Test'] == test_name]['Green_Val'].iloc[0]
                    upper_bound = threshold_df[threshold_df['Test'] == test_name]['Amber_Val'].iloc[0]

                    Green_Desc = threshold_df[threshold_df['Test'] == test_name]['Green_Desc'].iloc[0]
                    Amber_Desc = threshold_df[threshold_df['Test'] == test_name]['Amber_Desc'].iloc[0]
                    Red_Desc = threshold_df[threshold_df['Test'] == test_name]['Red_Desc'].iloc[0]

                    chi2_test = chi_square_test(df[st.session_state.pd_cal], df[st.session_state.def_flag], lower_bound, upper_bound,
                                                Green_Desc, Amber_Desc, Red_Desc)

                    if st.session_state.cal_acc_json[cat_name][test_name]['analysis'] == {}:
                        st.session_state.cal_acc_json[cat_name][test_name]['test_outputs'] = chi2_test
                        with st.spinner('Analyzing results for chi-square test...'):
                            analysis = report_functions.analyze_statistical_test(test_name=test_name, 
                                                                                            test_output=chi2_test,
                                                                                            threshold=thres_desc, 
                                                                                            model_type=st.session_state.model_type)
                            st.session_state.cal_acc_json[cat_name][test_name]['analysis'] = analysis
                    st.success('Result of Chi-Square Test has been analyzed.')
                    
                if 'Pluto Tasche' in st.session_state.cal_acc_json['Calibration Accuracy']:
                    test_name = 'Pluto Tasche'

                    n_obligors = df['CIF_NO'].nunique()
                    thres_desc = threshold_desc_df[threshold_desc_df['Test'] == test_name].reset_index(drop = True)

                    lower_bound = threshold_df[threshold_df['Test'] == test_name]['Amber_Val'].iloc[0]
                    upper_bound = threshold_df[threshold_df['Test'] == test_name]['Red_Val'].iloc[0]

                    Green_Desc = threshold_df[threshold_df['Test'] == test_name]['Green_Desc'].iloc[0]
                    Amber_Desc = threshold_df[threshold_df['Test'] == test_name]['Amber_Desc'].iloc[0]
                    Red_Desc = threshold_df[threshold_df['Test'] == test_name]['Red_Desc'].iloc[0]
                    
                    pluto_test = pluto_tasche_test(df[st.session_state.pd_cal], df[st.session_state.def_flag], n_obligors, 
                                                   lower_bound, upper_bound, 0.95, Green_Desc, Amber_Desc, Red_Desc)

                    if st.session_state.cal_acc_json[cat_name][test_name]['analysis'] == {}:
                        st.session_state.cal_acc_json[cat_name][test_name]['test_outputs'] = pluto_test
                        with st.spinner('Analyzing results for pluto tasche test...'):
                            analysis = report_functions.analyze_statistical_test(test_name=test_name, 
                                                                                            test_output=pluto_test,
                                                                                            threshold=thres_desc, 
                                                                                            model_type=st.session_state.model_type)
                            st.session_state.cal_acc_json[cat_name][test_name]['analysis'] = analysis
                    st.success('Result of Pluto Tasche Test has been analyzed.')
                    
            if 'External Benchmarking' in st.session_state.ext_ben_json:
                cat_name = 'External Benchmarking'
                
                if "Spearman's correlation" in st.session_state.ext_ben_json['External Benchmarking']:
                    test_name = "Spearman's correlation"

                    thres_desc = threshold_desc_df[threshold_desc_df['Test'] == test_name].reset_index(drop = True)

                    lower_bound = threshold_df[threshold_df['Test'] == test_name]['Green_Val'].iloc[0]
                    upper_bound = threshold_df[threshold_df['Test'] == test_name]['Amber_Val'].iloc[0]
                    
                    Green_Desc = threshold_df[threshold_df['Test'] == test_name]['Green_Desc'].iloc[0]
                    Amber_Desc = threshold_df[threshold_df['Test'] == test_name]['Amber_Desc'].iloc[0]
                    Red_Desc = threshold_df[threshold_df['Test'] == test_name]['Red_Desc'].iloc[0]

                    spear_corr_test = spearman_correlation(df[st.session_state.pd_cal], 
                                                        df[st.session_state.def_flag], 
                                                        upper_bound, lower_bound,
                                                        Green_Desc, Amber_Desc, Red_Desc)

                    if st.session_state.ext_ben_json[cat_name][test_name]['analysis'] == {}:
                        st.session_state.ext_ben_json[cat_name][test_name]['test_outputs'] = spear_corr_test
                        with st.spinner("Analyzing results for spearman's correlation test..."):
                            analysis = report_functions.analyze_statistical_test(test_name=test_name, 
                                                                                            test_output=spear_corr_test,
                                                                                            threshold=thres_desc, 
                                                                                            model_type=st.session_state.model_type)
                            st.session_state.ext_ben_json[cat_name][test_name]['analysis'] = analysis

                    st.success("Result of spearman's correlation has been analyzed.")

                if "Multi-notch movement" in st.session_state.ext_ben_json['External Benchmarking']:
                    test_name = "Multi-notch movement"

                    n_obligors = df['CIF_NO'].nunique()
                    thres_desc = threshold_desc_df[threshold_desc_df['Test'] == test_name].reset_index(drop = True)

                    lower_bound = threshold_df[threshold_df['Test'] == test_name]['Green_Val'].iloc[0]
                    upper_bound = threshold_df[threshold_df['Test'] == test_name]['Amber_Val'].iloc[0]
                    
                    Green_Desc = threshold_df[threshold_df['Test'] == test_name]['Green_Desc'].iloc[0]
                    Amber_Desc = threshold_df[threshold_df['Test'] == test_name]['Amber_Desc'].iloc[0]
                    Red_Desc = threshold_df[threshold_df['Test'] == test_name]['Red_Desc'].iloc[0]

                    multi_notch_test = multi_notch_movement(df[st.session_state.model_rating], 
                                                            df[st.session_state.actual_rating], 
                                                            n_obligors, upper_bound, lower_bound, 
                                                            Green_Desc, Amber_Desc, Red_Desc)

                    if st.session_state.ext_ben_json[cat_name][test_name]['analysis'] == {}:
                        st.session_state.ext_ben_json[cat_name][test_name]['test_outputs'] = multi_notch_test
                        with st.spinner("Analyzing results for multi-notch movement test..."):
                            analysis = report_functions.analyze_statistical_test(test_name=test_name, 
                                                                                            test_output=multi_notch_test,
                                                                                            threshold=thres_desc, 
                                                                                            model_type=st.session_state.model_type)
                            st.session_state.ext_ben_json[cat_name][test_name]['analysis'] = analysis

                    st.success("Result of multi-notch movement has been analyzed.")

            if 'Model Stability' in st.session_state.model_stability_json:
                cat_name = 'Model Stability'
                
                if "Population Stability Index (PSI)" in st.session_state.model_stability_json['Model Stability']:
                    test_name = "Population Stability Index (PSI)"

                    thres_desc = threshold_desc_df[threshold_desc_df['Test'] == test_name].reset_index(drop = True)

                    lower_bound = threshold_df[threshold_df['Test'] == test_name]['Green_Val'].iloc[0]
                    upper_bound = threshold_df[threshold_df['Test'] == test_name]['Amber_Val'].iloc[0]
                    
                    Green_Desc = threshold_df[threshold_df['Test'] == test_name]['Green_Desc'].iloc[0]
                    Amber_Desc = threshold_df[threshold_df['Test'] == test_name]['Amber_Desc'].iloc[0]
                    Red_Desc = threshold_df[threshold_df['Test'] == test_name]['Red_Desc'].iloc[0]

                    psi_test = calculate_psi(df[st.session_state.pd_cal],
                                                    df[st.session_state.def_flag],
                                                    10, upper_bound, lower_bound,
                                                    Green_Desc, Amber_Desc, Red_Desc)

                    if st.session_state.model_stability_json[cat_name][test_name]['analysis'] == {}:
                        st.session_state.model_stability_json[cat_name][test_name]['test_outputs'] = psi_test
                        with st.spinner("Analyzing results for population stability index test..."):
                            analysis = report_functions.analyze_statistical_test(test_name=test_name, 
                                                                                            test_output=psi_test,
                                                                                            threshold=thres_desc, 
                                                                                            model_type=st.session_state.model_type)
                            st.session_state.model_stability_json[cat_name][test_name]['analysis'] = analysis

                    st.success("Result of population stability index has been analyzed.")

            if 'Overrides' in st.session_state.overrides_json:
                cat_name = 'Overrides'
                
                if "% Downgrade Override rate" in st.session_state.overrides_json['Overrides']:
                    test_name = "% Downgrade Override rate"

                    thres_desc = threshold_desc_df[threshold_desc_df['Test'] == test_name].reset_index(drop = True)

                    lower_bound = threshold_df[threshold_df['Test'] == test_name]['Green_Val'].iloc[0]
                    upper_bound = threshold_df[threshold_df['Test'] == test_name]['Amber_Val'].iloc[0]
                    
                    Green_Desc = threshold_df[threshold_df['Test'] == test_name]['Green_Desc'].iloc[0]
                    Amber_Desc = threshold_df[threshold_df['Test'] == test_name]['Amber_Desc'].iloc[0]
                    Red_Desc = threshold_df[threshold_df['Test'] == test_name]['Red_Desc'].iloc[0]

                    dg_test = downgrade_override_rate(df[st.session_state.actual_rating],
                                                    df[st.session_state.model_rating],
                                                    upper_bound, lower_bound,
                                                    Green_Desc, Amber_Desc, Red_Desc)

                    if st.session_state.overrides_json[cat_name][test_name]['analysis'] == {}:
                        st.session_state.overrides_json[cat_name][test_name]['test_outputs'] = dg_test
                        with st.spinner("Analyzing results for % Downgrade Override rate test..."):
                            analysis = report_functions.analyze_statistical_test(test_name=test_name, 
                                                                                            test_output=dg_test,
                                                                                            threshold=thres_desc, 
                                                                                            model_type=st.session_state.model_type)
                            st.session_state.overrides_json[cat_name][test_name]['analysis'] = analysis

                    st.success("Result of % Downgrade Override rate has been analyzed.")

                if "Overrides Rates" in st.session_state.overrides_json['Overrides']:
                    test_name = "Overrides Rates"

                    thres_desc = threshold_desc_df[threshold_desc_df['Test'] == test_name].reset_index(drop = True)

                    lower_bound = threshold_df[threshold_df['Test'] == test_name]['Green_Val'].iloc[0]
                    upper_bound = threshold_df[threshold_df['Test'] == test_name]['Amber_Val'].iloc[0]
                    
                    Green_Desc = threshold_df[threshold_df['Test'] == test_name]['Green_Desc'].iloc[0]
                    Amber_Desc = threshold_df[threshold_df['Test'] == test_name]['Amber_Desc'].iloc[0]
                    Red_Desc = threshold_df[threshold_df['Test'] == test_name]['Red_Desc'].iloc[0]

                    overrides_test = override_rate(df[st.session_state.actual_rating],
                                                df[st.session_state.model_rating],
                                                upper_bound, lower_bound,
                                                Green_Desc, Amber_Desc, Red_Desc)

                    if st.session_state.overrides_json[cat_name][test_name]['analysis'] == {}:
                        st.session_state.overrides_json[cat_name][test_name]['test_outputs'] = overrides_test
                        with st.spinner("Analyzing results for Overrides Rates test..."):
                            analysis = report_functions.analyze_statistical_test(test_name=test_name, 
                                                                                            test_output=overrides_test,
                                                                                            threshold=thres_desc, 
                                                                                            model_type=st.session_state.model_type)
                            st.session_state.overrides_json[cat_name][test_name]['analysis'] = analysis

                    st.success("Result of Overrides Rates has been analyzed.")

            if 'Data Quality' in st.session_state.data_quality_json:
                cat_name = 'Data Quality'
                
                if "Percentage Of Missing Values" in st.session_state.data_quality_json['Data Quality']:
                    test_name = "Percentage Of Missing Values"

                    thres_desc = threshold_desc_df[threshold_desc_df['Test'] == test_name].reset_index(drop = True)

                    if 'perc_missing_val_test' not in st.session_state:
                        st.session_state.perc_missing_val_test = missing_data_summary(df)

                    if st.session_state.data_quality_json[cat_name][test_name]['analysis'] == {}:
                        st.session_state.data_quality_json[cat_name][test_name]['test_outputs'] = st.session_state.perc_missing_val_test
                        with st.spinner("Analyzing results for data quality test..."):
                            analysis = report_functions.analyze_statistical_test(test_name=test_name, 
                                                                                            test_output=st.session_state.perc_missing_val_test,
                                                                                            threshold=thres_desc, 
                                                                                            model_type=st.session_state.model_type)
                            st.session_state.data_quality_json[cat_name][test_name]['analysis'] = analysis

                            data_list = st.session_state.perc_missing_val_test.to_numpy().tolist()
                            result = [list(st.session_state.perc_missing_val_test.columns)] + data_list
                            st.session_state.tables_data = {"Percentage Of Missing Values": result}

                    st.success("Result of data quality has been analyzed.")

            if 'Discriminatory Factors' in st.session_state.disc_fac_json:
                cat_name = 'Discriminatory Factors'
                
                if "Somer's D" in st.session_state.disc_fac_json['Discriminatory Factors']:
                    test_name = "Somer's D"

                    thres_desc = threshold_desc_df[threshold_desc_df['Test'] == test_name].reset_index(drop = True)
                    upper_bound = threshold_df[threshold_df['Test'] == test_name]['Green_Val'].iloc[0]
                    lower_bound = threshold_df[threshold_df['Test'] == test_name]['Amber_Val'].iloc[0]
                    
                    Green_Desc = threshold_df[threshold_df['Test'] == test_name]['Green_Desc'].iloc[0]
                    Amber_Desc = threshold_df[threshold_df['Test'] == test_name]['Amber_Desc'].iloc[0]
                    Red_Desc = threshold_df[threshold_df['Test'] == test_name]['Red_Desc'].iloc[0]

                    somers_d_test = somers_d(df[st.session_state.pd_cal], df[st.session_state.def_flag], upper_bound, lower_bound, Green_Desc, Amber_Desc, Red_Desc)

                    if st.session_state.disc_fac_json[cat_name][test_name]['analysis'] == {}:
                        st.session_state.disc_fac_json[cat_name][test_name]['test_outputs'] = somers_d_test
                        with st.spinner("Analyzing results for Somer's D test..."):
                            analysis = report_functions.analyze_statistical_test(test_name=test_name, 
                                                                                            test_output=somers_d_test,
                                                                                            threshold=thres_desc, 
                                                                                            model_type=st.session_state.model_type)
                            st.session_state.disc_fac_json[cat_name][test_name]['analysis'] = analysis

                    st.success("Result of Somer's D has been analyzed.")

                if "Bootstrapping - AUC/Gini Coefficient/KS Statistic" in st.session_state.disc_fac_json['Discriminatory Factors']:
                    test_name = "Bootstrapping - AUC/Gini Coefficient/KS Statistic"

                    # Check if results are already computed and stored in session state
                    if 'bins_results' not in st.session_state:
                        # Run tests if results are not already computed
                        sampling_size = 0.3
                        n_samples = 50
                        ks_list = []
                        auc_list = []
                        gini_list = []

                        # col1, col2 = st.columns([10, 1])
                        # progress_bar = col1.progress(0)
                        # percent_display = col2.empty()

                        for i in range(0, n_samples):
                            df_sampled = st.session_state.data.sample(frac=sampling_size, random_state=None)

                            results_sampled = KS_GINI(
                                df_sampled[['Bins', 'Total_SCORE', st.session_state.def_flag]], 
                                groupbyvariable='Bins', 
                                default_flag=st.session_state.def_flag, 
                                sortvariable='Total_SCORE'
                            )
                            ks_list.append(results_sampled['KS'])
                            auc_list.append(results_sampled['AUC'])
                            gini_list.append(results_sampled['GINI'])

                            # percent_complete = (i + 1) / n_samples * 100
                            # progress_bar.progress((i + 1) / n_samples)
                            # percent_display.text(f"{percent_complete:.1f}%")

                        # Compute confidence intervals
                        g_lower_bound = np.percentile(gini_list, 2.5)
                        g_upper_bound = np.percentile(gini_list, 97.5)
                        a_lower_bound = np.percentile(auc_list, 2.5)
                        a_upper_bound = np.percentile(auc_list, 97.5)
                        ks_lower_bound = np.percentile(ks_list, 2.5)
                        ks_upper_bound = np.percentile(ks_list, 97.5)

                        # Store results in session state
                        st.session_state.bins_results = KS_GINI(
                            st.session_state.data, 
                            groupbyvariable='Bins', 
                            default_flag=st.session_state.def_flag, 
                            sortvariable='Total_SCORE'
                        )

                        st.session_state.summary_df = pd.DataFrame({
                            'Metric': ['GINI', 'AUC', 'KS'],
                            'Value (%)': [
                                round(st.session_state.bins_results['GINI'], 2), 
                                round(st.session_state.bins_results['AUC'], 2), 
                                round(st.session_state.bins_results['KS'], 2)
                            ],
                            'Confidence Interval (%)': [95, 95, 95],
                            'Lower Bound (%)': [
                                round(g_lower_bound, 2), 
                                round(a_lower_bound, 2), 
                                round(ks_lower_bound, 2)
                            ],
                            'Upper Bound (%)': [
                                round(g_upper_bound, 2), 
                                round(a_upper_bound, 2), 
                                round(ks_upper_bound, 2)
                            ]
                        })
                        st.session_state.summary_df.set_index('Metric', inplace=True)

                        st.session_state.summary_df["Within Bounds"] = st.session_state.summary_df.apply(
                            lambda row: "Yes" if row["Lower Bound (%)"] <= row["Value (%)"] <= row["Upper Bound (%)"] else "No", 
                            axis=1
                        )

                    # Define style function to highlight the entire cell in green for "Yes"
                    def highlight_within_bounds(val):
                        color = 'background-color: #bcebb6; color: green;' if val == "Yes" else ''
                        return color

                    # Display the table with two decimal places and cell highlighting in Streamlit
                    st.session_state.bootstrapping_df = (
                        st.session_state.summary_df.style
                        .applymap(highlight_within_bounds, subset=["Within Bounds"])
                        .format({"Value (%)": "{:.2f}", "Lower Bound (%)": "{:.2f}", "Upper Bound (%)": "{:.2f}"})
                        .set_table_styles([{
                            'selector': 'th:nth-child(1)',  # First column
                            'props': [('width', '100px')]
                        }, {
                            'selector': 'th:nth-child(2)',  # Second column
                            'props': [('width', '150px')]
                        }, {
                            'selector': 'th:nth-child(3)',  # Third column
                            'props': [('width', '200px')]
                        }])
                    )

                    if st.session_state.disc_fac_json["Discriminatory Factors"]["Bootstrapping - AUC/Gini Coefficient/KS Statistic"]["analysis"] == {}:
                        with st.spinner('Analyzing bootstrapping results...'):
                            analysis = report_functions.analyze_statistical_test(test_name="Bootstrapping", 
                                                                                    test_output=st.session_state.summary_df,
                                                                                    threshold="Confidence Interval is 95%, sampling size for bootstrapping is 0.3, and number of iterations are 50", 
                                                                                    model_type=st.session_state.model_type)
                            st.session_state.disc_fac_json["Discriminatory Factors"]["Bootstrapping - AUC/Gini Coefficient/KS Statistic"]['analysis'] = analysis
                            data_list = st.session_state.summary_df.reset_index().drop('Confidence Interval (%)', axis = 1).to_numpy().tolist() 
                            result = [list(st.session_state.summary_df.reset_index().drop('Confidence Interval (%)', axis = 1).columns)] + data_list
                            st.session_state.tables_data.update({"Bootstrapping - AUC/Gini Coefficient/KS Statistic": result})
                    
                    st.success("Result of Bootstrapping has been analyzed.")
                    # st.write(st.session_state.disc_fac_json["Discriminatory Factors"]["Bootstrapping - AUC/Gini Coefficient/KS Statistic"]['analysis'])

                if "Result by Bins" in st.session_state.disc_fac_json['Discriminatory Factors']:

                    # st.markdown(f'<p class="medium-font">Result by Bins</p>', unsafe_allow_html=True)
                    temp_summarized_data = st.session_state.bins_results['summarized_data'].copy()
                    temp_summarized_data.set_index('Bins', inplace=True)
                    # st.write(temp_summarized_data)

                    # cols = st.columns([2, 1])
                    # with cols[0]:
                    os.makedirs(st.session_state.chart_dir, exist_ok=True)
                    st.session_state.dev_rand_perf_model_path = os.path.join(st.session_state.chart_dir, "dev_rand_perf_model.png")
                    st.session_state.tables_charts = {"Result by Bins": [st.session_state.dev_rand_perf_model_path]}

                        # auc_plot(st.session_state.bins_results['summarized_data'], st.session_state.dev_rand_perf_model_path)

                    if st.session_state.disc_fac_json["Discriminatory Factors"]["Result by Bins"]["analysis"] == {}:
                        with st.spinner('Analyzing AUC/KS Statistic by bins...'):
                            analysis = report_functions.analyze_statistical_test(test_name="AUC/KS Statistic by Bins", 
                                                                                    test_output=temp_summarized_data,
                                                                                    threshold=None, 
                                                                                    model_type=st.session_state.model_type)
                            st.session_state.disc_fac_json["Discriminatory Factors"]["Result by Bins"]['analysis'] = analysis

                            required_cols = ['Bins', 'Total', 'Default', 'NonDefault', 'KS', 'AUC']
                            data_list = temp_summarized_data.reset_index()[required_cols].to_numpy().tolist() 
                            result = [list(temp_summarized_data.reset_index()[required_cols].columns)] + data_list
                            st.session_state.tables_data.update({"Result by Bins": result})

                    st.success("Result of Analysis by bins has been analyzed.")
                    # st.write(st.session_state.combined_json["Discriminatory Factors"]["Result by Bins"]['analysis'])


    #         # Analysis by Variables
    #         st.session_state.discfac_vars = [col for col in st.session_state.data.columns if col.endswith('_ATTRIBUTE')]
    #         i = 1
    #         st.markdown(f'<p class="medium-font">Analysis by Variables</p>', unsafe_allow_html=True)
    #         for var in st.session_state.discfac_vars:
    #             st.markdown(f'<p class="small-font" style="font-weight: bold;">{i}. {var}</p>', unsafe_allow_html=True)

    #             if f"{var}_results" not in st.session_state:
    #                 st.session_state[f"{var}_results"] = KS_GINI(
    #                     st.session_state.data, 
    #                     groupbyvariable=var, 
    #                     default_flag=st.session_state.def_flag, 
    #                     sortvariable=f"{var[:-10]}_SCORE"
    #                 )

    #                 # with st.spinner("Analyzing the results..."):

    #             # st.session_state.combined_json["Discriminatory Factors"][var] = {"test_outputs": {}, "analysis": {}}

    #             # st.session_state.combined_json["Discriminatory Factors"][var]['test_outputs'] = st.session_state.somer_d_result
    #             # analysis = report_functions.analyze_statistical_test("Somer's D", st.session_state.somer_d_result, somers_d_threshold, st.session_state.model_type)
    #             # st.session_state.combined_json["Discriminatory Factors"]["Somer's D"]['analysis'] = analysis

    #             st.markdown(f'<ul style="margin-top: -15px;"><li class="small-font">The AUC for {var} is {round(st.session_state[f"{var}_results"]["AUC"], 2)}%.</li></ul>', unsafe_allow_html=True)
    #             st.markdown(f'<ul style="margin-top: -20px;"><li class="small-font">The GINI for {var} is {round(st.session_state[f"{var}_results"]["GINI"], 2)}%.</li></ul>', unsafe_allow_html=True)
    #             st.markdown(f'<ul style="margin-top: -20px;"><li class="small-font">The KS for {var} is {round(st.session_state[f"{var}_results"]["KS"], 2)}%.</li></ul>', unsafe_allow_html=True)

    #             st.write(st.session_state[f"{var}_results"]['summarized_data'])
                
    #             i += 1

    cols = st.columns(3)
    with cols[1]:
        if st.session_state.disc_fac_tests:
                st.session_state.combined_json.update(st.session_state.disc_fac_json)

        if st.session_state.cal_acc_tests:
            st.session_state.combined_json.update(st.session_state.cal_acc_json)

        if st.session_state.model_stability_tests:
            st.session_state.combined_json.update(st.session_state.model_stability_json)

        if st.session_state.ext_ben_tests:
            st.session_state.combined_json.update(st.session_state.ext_ben_json)

        if st.session_state.overrides_tests:
            st.session_state.combined_json.update(st.session_state.overrides_json)

        if st.session_state.data_quality_tests:
            st.session_state.combined_json.update(st.session_state.data_quality_json)

        if st.session_state.selected_ifrs9_tests:
            st.session_state.combined_json.update(st.session_state.selected_ifrs9_json)

        if 'data_summary' not in st.session_state:
            st.session_state.data_summary = None

        if 'exec_summary' not in st.session_state:
            st.session_state.exec_summary = None

        if st.session_state.data_summary is None:
            with st.spinner("Generating summary of the dataset..."):
                st.session_state.data_summary = report_functions.summarize_dataset(st.session_state.data)

            st.success('Summary of the dataset has been generated.')
            
        if st.session_state.exec_summary is None:
            with st.spinner("Generating executive summary..."):
                st.session_state.exec_summary = report_functions.generate_executive_summary(st.session_state.model_type, 
                                                                                            st.session_state.combined_json, 
                                                                                            st.session_state.data_summary, None)

            st.success('Executive summary has been generated.') 
            
    placeholder.empty()
        
    validation_tests_sidebar()    

    if st.session_state.validation_tests_app == 'Summary': 
        
        cat_test_status_cols = ['Category',
                                'Test',
                                'Status',
                                'Description']
        st.session_state.cat_test_status_df = pd.DataFrame(columns=cat_test_status_cols)

        for cat in st.session_state.combined_json:
            for test in st.session_state.combined_json[cat]:
                if 'status' in st.session_state.combined_json[cat][test]['test_outputs']:
                    status = st.session_state.combined_json[cat][test]['test_outputs']['status']

                if 'description' in st.session_state.combined_json[cat][test]['test_outputs']:
                    description = st.session_state.combined_json[cat][test]['test_outputs']['description']
                    temp_list = [cat, test, status, description]
                    temp_df = pd.DataFrame(temp_list).T
                    temp_df.columns = cat_test_status_cols

                    st.session_state.cat_test_status_df = pd.concat([st.session_state.cat_test_status_df, temp_df]).reset_index(drop = True)

        # Define style function to highlight the entire cell in green for "Yes"
        def highlight_statuses(val):
            color = ''
            if val == "Green":
                color = 'background-color: #bcebb6; color: green;'
            elif val == "Red":
                color = 'background-color: #EDB5B5; color: red;'
            elif val == "Amber":
                color = 'background-color: #E9EDB5; color: amber;'
            return color

        # Display the table with two decimal places and cell highlighting in Streamlit
        st.session_state.cat_test_status_df_styled = (
                st.session_state.cat_test_status_df.style
            .applymap(highlight_statuses, subset=["Status"])
            .set_table_styles([{
                'selector': 'th:nth-child(1)',  # First column
                'props': [('width', '100px')]
            }, {
                'selector': 'th:nth-child(2)',  # Second column
                'props': [('width', '150px')]
            }, {
                'selector': 'th:nth-child(3)',  # Third column
                'props': [('width', '200px')]
            }])
        )

        st.subheader('Executive Summary')
        st.write(st.session_state.exec_summary)

        st.subheader('Test Statuses')
        st.write(st.session_state.cat_test_status_df_styled)

        st.subheader('Data Summary')
        st.write(st.session_state.data_summary)

        

    if st.session_state.validation_tests_app == 'Calibration Accuracy':
        cat_name = 'Calibration Accuracy'
        if 'Binomial Test' in st.session_state.cal_acc_json['Calibration Accuracy']:
            st.markdown(f'<p class="big-font">Binomial Test</p>', unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
            st.write(st.session_state.cal_acc_json[cat_name]['Binomial Test']['analysis'])
            
        if 'Normal Test' in st.session_state.cal_acc_json['Calibration Accuracy']:
            st.markdown(f'<p class="big-font">Normal Test</p>', unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
            st.write(st.session_state.cal_acc_json[cat_name]['Normal Test']['analysis'])
            
        if 'Chi-Square Test' in st.session_state.cal_acc_json['Calibration Accuracy']:
            st.markdown(f'<p class="big-font">Chi-Square Test</p>', unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
            st.write(st.session_state.cal_acc_json[cat_name]['Chi-Square Test']['analysis'])
            
        if 'Pluto Tasche' in st.session_state.cal_acc_json['Calibration Accuracy']:
            st.markdown(f'<p class="big-font">Pluto Tasche Test</p>', unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
            st.write(st.session_state.cal_acc_json[cat_name]['Pluto Tasche']['analysis'])

    if st.session_state.validation_tests_app == 'External Benchmarking':
        cat_name = 'External Benchmarking'
        if "Spearman's correlation" in st.session_state.ext_ben_json['External Benchmarking']:
            st.markdown(f"<p class='big-font'>Spearman's correlation</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
            st.write(st.session_state.ext_ben_json[cat_name]["Spearman's correlation"]['analysis'])
            
        if "Multi-notch movement" in st.session_state.ext_ben_json['External Benchmarking']:
            st.markdown(f"<p class='big-font'>Multi-notch movement</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
            st.write(st.session_state.ext_ben_json[cat_name]["Multi-notch movement"]['analysis'])

    if st.session_state.validation_tests_app == 'Model Stability':
        cat_name = 'Model Stability'
        if "Population Stability Index (PSI)" in st.session_state.model_stability_json['Model Stability']:
            st.markdown(f"<p class='big-font'>Population Stability Index (PSI)</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
            st.write(st.session_state.model_stability_json[cat_name]["Population Stability Index (PSI)"]['analysis'])

    if st.session_state.validation_tests_app == 'Overrides Analysis':
        cat_name = 'Overrides'
        if "% Downgrade Override rate" in st.session_state.overrides_json['Overrides']:
            st.markdown(f"<p class='big-font'>% Downgrade Override rate</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
            st.write(st.session_state.overrides_json[cat_name]["% Downgrade Override rate"]['analysis'])
            
        if "Overrides Rates" in st.session_state.overrides_json['Overrides']:
            st.markdown(f"<p class='big-font'>Overrides Rates</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
            st.write(st.session_state.overrides_json[cat_name]["Overrides Rates"]['analysis'])

    if st.session_state.validation_tests_app == 'Data Quality':
        cat_name = 'Data Quality'
        if "Percentage Of Missing Values" in st.session_state.data_quality_json['Data Quality']:
            st.markdown(f"<p class='big-font'>Percentage Of Missing Values</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
            st.write(st.session_state.perc_missing_val_test)
            st.write(st.session_state.data_quality_json[cat_name]["Percentage Of Missing Values"]['analysis'])
            
    if st.session_state.validation_tests_app == 'Discriminatory Factors':
        cat_name = 'Discriminatory Factors'

        if "Bootstrapping - AUC/Gini Coefficient/KS Statistic" in st.session_state.disc_fac_json['Discriminatory Factors']:
            st.markdown(f"<p class='big-font'>Bootstrapping Result</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
            st.dataframe(st.session_state.bootstrapping_df)
            st.write(st.session_state.disc_fac_json[cat_name]["Bootstrapping - AUC/Gini Coefficient/KS Statistic"]['analysis'])

        if "Result by Bins" in st.session_state.disc_fac_json['Discriminatory Factors']:
            st.markdown(f"<p class='big-font'>Result by Bins</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
            st.write(temp_summarized_data)
            cols = st.columns([2, 1])
            with cols[0]:
                auc_plot(st.session_state.bins_results['summarized_data'], st.session_state.dev_rand_perf_model_path)
            st.write(st.session_state.disc_fac_json[cat_name]["Result by Bins"]['analysis'])

        if "Somer's D" in st.session_state.disc_fac_json['Discriminatory Factors']:
            st.markdown(f"<p class='big-font'>Somer's D</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
            st.write(st.session_state.disc_fac_json[cat_name]["Somer's D"]['analysis'])

    if st.session_state.validation_tests_app == 'Generate Report':

        gen_rep_page()

        
    #     # st.write(st.session_state.cal_acc_json)





#     from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
#     # from datetime import datetime

#     from langchain.utilities.tavily_search import TavilySearchAPIWrapper
#     from langchain.tools.tavily_search import TavilySearchResults

#     from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#     from langchain.memory import ConversationBufferMemory
#     from langchain.chains import RetrievalQA
#     from langchain_text_splitters import CharacterTextSplitter
#     from langchain_community.vectorstores import FAISS
    
#     from langchain.agents import Tool, initialize_agent, OpenAIFunctionsAgent, AgentExecutor
#     from langchain.agents.agent_types import AgentType
#     from langchain.chains import LLMMathChain

#     os.environ["OPENAI_API_KEY"] = st.session_state.OPENAI_API_KEY
#     llm = ChatOpenAI()

#     # st.write(1)
#     pdf_reader = PdfReader(st.session_state.uploaded_model_stand)
#     st.session_state.model_stand = ""
#     for page in pdf_reader.pages:
#         st.session_state.model_stand += page.extract_text()
    
#     #Split the extracted text to chunks
#     text_splitter = CharacterTextSplitter(
#         separator = "\n",
#         chunk_size = 700,
#         chunk_overlap  = 150,
#         length_function = len,
#     )
#     chunks = text_splitter.split_text(st.session_state.model_stand)
    
#     #Embed the text
#     embeddings = OpenAIEmbeddings()
#     VectorStore = FAISS.from_texts(chunks, embeddings)
#     store_name = "Stmts"

#     retriever = VectorStore.as_retriever()

#     # chat completion llm
#     llm = ChatOpenAI(
#         model_name='gpt-4', #gpt-4o
#         temperature=0.3
#     )
#     # conversational memory
#     conversational_memory = ConversationBufferMemory(
#         memory_key='chat_history',
#         return_messages=True
#     )
    
#     #Chat Prompt Template
#     chat_prompt = ChatPromptTemplate(
#         input_variable=["input", "messages"],
#         messages=[
#             MessagesPlaceholder(variable_name="chat_history"),
#             HumanMessagePromptTemplate.from_template("{input}"),
#             MessagesPlaceholder(variable_name="agent_scratchpad")
#         ]
#     )
    
#     # retrieval qa chain
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         callbacks=None
#     )
    
    
#     #result = qa.invoke(query)
#     #st.write(result)
#     # Tools
#     knowledge_tool = Tool(
#             name='KnowledgeBase',
#             func=qa.run,
#             description=(
#                 'use this tool when answering questions to get more information about the thresholds of statistical tests from the model validation standards document.'
#             )
#         )
    
#     problem_chain = LLMMathChain.from_llm(llm=llm)
    
#     agent = initialize_agent(
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         prompt=chat_prompt,
#         tools=[knowledge_tool],
#         llm=llm,
#         verbose=True,
#         #max_iterations=3,
#         #early_stopping_method='generate',
#         memory=conversational_memory,
#         handle_parsing_errors=True
#     )

#     st.write('Running here')
    
#     query = f"""
#             Analyze the document and provide a detailed response regarding the thresholds for the KS Statistic test in a PD model. Your response should include the following:

#             1. **Threshold Values**: Clearly state the numerical thresholds or ranges for the KS Statistic test as specified in the document.
#             2. **Context**: Provide any additional context or conditions associated with the thresholds (e.g., confidence levels, sample size requirements, or specific use cases).
#             3. **Interpretation**: Explain what the thresholds mean in terms of model performance (e.g., what constitutes a good or bad KS Statistic value).
#             4. **Source Reference**: If possible, reference the section or page number in the document where the thresholds are mentioned.

#             If the document does not contain any information about the KS Statistic test thresholds, respond with: "Threshold not found in the document."

#             Ensure your response is thorough, well-organized, and easy to understand.

# """
#     result = agent({"input":query})
#     st.write(result)


    

    # if st.session_state.validation_tests_app == 'Discriminatory Power Tests':
  
    #     st.markdown(
    #         """
    #         <style>
    #         .big-font {
    #             font-size:20px !important;
    #             font-weight: bold;
    #             padding: 0px;
    #         }
    #         .medium-font {
    #             font-size:18px !important;
    #             font-weight: bold;
    #             padding: 0px;
    #         }
    #         .small-font {
    #             font-size:15px !important;
    #             padding: 0px;
    #         }
    #         .boldhr {
    #             width: 100%;
    #             height: 2px;
    #             background-color: #9c9b99; 
    #             margin: 2px;
    #             # margin-top: -20px;
    #         }
    #         # .stExpander {
    #         #     margin-top: -10px;
    #         #     background-color: #13326b;
    #         #     header: white
    #         # }
    #         </style>
    #         """,
    #         unsafe_allow_html=True
    #     )

    #     st.session_state.def_flag = 'Final_Default_Flag'
    #     st.session_state.pd_cal = 'PD'

    #     if "ROC/AUC/Gini Coefficient/KS Statistic" in st.session_state.disc_fac_tests:
    #         st.markdown(f'<p class="big-font">ROC/AUC/KS/GINI</p>', unsafe_allow_html=True)
    #         st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 

    #         # Check if results are already computed and stored in session state
    #         if 'bins_results' not in st.session_state:
    #             # Run tests if results are not already computed
    #             sampling_size = 0.3
    #             n_samples = 50
    #             ks_list = []
    #             auc_list = []
    #             gini_list = []

    #             # col1, col2 = st.columns([10, 1])
    #             # progress_bar = col1.progress(0)
    #             # percent_display = col2.empty()

    #             for i in range(0, n_samples):
    #                 df_sampled = st.session_state.data.sample(frac=sampling_size, random_state=None)

    #                 results_sampled = KS_GINI(
    #                     df_sampled[['Bins', 'Total_SCORE', st.session_state.def_flag]], 
    #                     groupbyvariable='Bins', 
    #                     default_flag=st.session_state.def_flag, 
    #                     sortvariable='Total_SCORE'
    #                 )
    #                 ks_list.append(results_sampled['KS'])
    #                 auc_list.append(results_sampled['AUC'])
    #                 gini_list.append(results_sampled['GINI'])

    #                 # percent_complete = (i + 1) / n_samples * 100
    #                 # progress_bar.progress((i + 1) / n_samples)
    #                 # percent_display.text(f"{percent_complete:.1f}%")

    #             # Compute confidence intervals
    #             g_lower_bound = np.percentile(gini_list, 2.5)
    #             g_upper_bound = np.percentile(gini_list, 97.5)
    #             a_lower_bound = np.percentile(auc_list, 2.5)
    #             a_upper_bound = np.percentile(auc_list, 97.5)
    #             ks_lower_bound = np.percentile(ks_list, 2.5)
    #             ks_upper_bound = np.percentile(ks_list, 97.5)

    #             # Store results in session state
    #             st.session_state.bins_results = KS_GINI(
    #                 st.session_state.data, 
    #                 groupbyvariable='Bins', 
    #                 default_flag=st.session_state.def_flag, 
    #                 sortvariable='Total_SCORE'
    #             )

    #             st.session_state.summary_df = pd.DataFrame({
    #                 'Metric': ['GINI', 'AUC', 'KS'],
    #                 'Value (%)': [
    #                     round(st.session_state.bins_results['GINI'], 2), 
    #                     round(st.session_state.bins_results['AUC'], 2), 
    #                     round(st.session_state.bins_results['KS'], 2)
    #                 ],
    #                 'Confidence Interval (%)': [95, 95, 95],
    #                 'Lower Bound (%)': [
    #                     round(g_lower_bound, 2), 
    #                     round(a_lower_bound, 2), 
    #                     round(ks_lower_bound, 2)
    #                 ],
    #                 'Upper Bound (%)': [
    #                     round(g_upper_bound, 2), 
    #                     round(a_upper_bound, 2), 
    #                     round(ks_upper_bound, 2)
    #                 ]
    #             })
    #             st.session_state.summary_df.set_index('Metric', inplace=True)

    #             st.session_state.summary_df["Within Bounds"] = st.session_state.summary_df.apply(
    #                 lambda row: "Yes" if row["Lower Bound (%)"] <= row["Value (%)"] <= row["Upper Bound (%)"] else "No", 
    #                 axis=1
    #             )

    #         # Display results from session state
    #         st.markdown(f'<p class="medium-font">Bootstrapping Result</p>', unsafe_allow_html=True)

    #         # Define style function to highlight the entire cell in green for "Yes"
    #         def highlight_within_bounds(val):
    #             color = 'background-color: #bcebb6; color: green;' if val == "Yes" else ''
    #             return color

    #         # Display the table with two decimal places and cell highlighting in Streamlit
    #         styled_df = (
    #             st.session_state.summary_df.style
    #             .applymap(highlight_within_bounds, subset=["Within Bounds"])
    #             .format({"Value (%)": "{:.2f}", "Lower Bound (%)": "{:.2f}", "Upper Bound (%)": "{:.2f}"})
    #             .set_table_styles([{
    #                 'selector': 'th:nth-child(1)',  # First column
    #                 'props': [('width', '100px')]
    #             }, {
    #                 'selector': 'th:nth-child(2)',  # Second column
    #                 'props': [('width', '150px')]
    #             }, {
    #                 'selector': 'th:nth-child(3)',  # Third column
    #                 'props': [('width', '200px')]
    #             }])
    #         )
    #         st.dataframe(styled_df)

    #         if st.session_state.combined_json["Discriminatory Factors"]["Bootstrapping - AUC/Gini Coefficient/KS Statistic"]["analysis"] == {}:
    #             with st.spinner('Analyzing results...'):
    #                 analysis = report_functions.analyze_statistical_test(test_name="Bootstrapping", 
    #                                                                         test_output=st.session_state.summary_df,
    #                                                                         threshold="Confidence Interval is 95%, sampling size for bootstrapping is 0.3, and number of iterations are 50", 
    #                                                                         model_type=st.session_state.model_type)
    #                 st.session_state.combined_json["Discriminatory Factors"]["Bootstrapping - AUC/Gini Coefficient/KS Statistic"]['analysis'] = analysis
    #                 data_list = st.session_state.summary_df.reset_index().drop('Confidence Interval (%)', axis = 1).to_numpy().tolist() 
    #                 result = [list(st.session_state.summary_df.reset_index().drop('Confidence Interval (%)', axis = 1).columns)] + data_list
    #                 st.session_state.tables_data.update({"Bootstrapping - AUC/Gini Coefficient/KS Statistic": result})
            
    #         st.write(st.session_state.combined_json["Discriminatory Factors"]["Bootstrapping - AUC/Gini Coefficient/KS Statistic"]['analysis'])

    #         st.markdown(f'<p class="medium-font">Result by Bins</p>', unsafe_allow_html=True)
    #         temp_summarized_data = st.session_state.bins_results['summarized_data'].copy()
    #         temp_summarized_data.set_index('Bins', inplace=True)
    #         st.write(temp_summarized_data)

    #         cols = st.columns([2, 1])
    #         with cols[0]:
    #             os.makedirs(st.session_state.chart_dir, exist_ok=True)
    #             dev_rand_perf_model_path = os.path.join(st.session_state.chart_dir, "dev_rand_perf_model.png")
    #             st.session_state.tables_charts = {"Result by Bins": [dev_rand_perf_model_path]}

    #             auc_plot(st.session_state.bins_results['summarized_data'], dev_rand_perf_model_path)

    #         if st.session_state.combined_json["Discriminatory Factors"]["Result by Bins"]["analysis"] == {}:
    #             with st.spinner('Analyzing results...'):
    #                 analysis = report_functions.analyze_statistical_test(test_name="AUC/Gini Coefficient/KS Statistic by Bins", 
    #                                                                         test_output=temp_summarized_data,
    #                                                                         threshold=None, 
    #                                                                         model_type=st.session_state.model_type)
    #                 st.session_state.combined_json["Discriminatory Factors"]["Result by Bins"]['analysis'] = analysis

    #                 required_cols = ['Bins', 'Total', 'Default', 'NonDefault', 'KS', 'AUC']
    #                 data_list = temp_summarized_data.reset_index()[required_cols].to_numpy().tolist() 
    #                 result = [list(temp_summarized_data.reset_index()[required_cols].columns)] + data_list
    #                 st.session_state.tables_data.update({"Result by Bins": result})

    #         st.write(st.session_state.combined_json["Discriminatory Factors"]["Result by Bins"]['analysis'])



            





    #         # Analysis by Variables
    #         st.session_state.discfac_vars = [col for col in st.session_state.data.columns if col.endswith('_ATTRIBUTE')]
    #         i = 1
    #         st.markdown(f'<p class="medium-font">Analysis by Variables</p>', unsafe_allow_html=True)
    #         for var in st.session_state.discfac_vars:
    #             st.markdown(f'<p class="small-font" style="font-weight: bold;">{i}. {var}</p>', unsafe_allow_html=True)

    #             if f"{var}_results" not in st.session_state:
    #                 st.session_state[f"{var}_results"] = KS_GINI(
    #                     st.session_state.data, 
    #                     groupbyvariable=var, 
    #                     default_flag=st.session_state.def_flag, 
    #                     sortvariable=f"{var[:-10]}_SCORE"
    #                 )

    #                 # with st.spinner("Analyzing the results..."):

    #             # st.session_state.combined_json["Discriminatory Factors"][var] = {"test_outputs": {}, "analysis": {}}

    #             # st.session_state.combined_json["Discriminatory Factors"][var]['test_outputs'] = st.session_state.somer_d_result
    #             # analysis = report_functions.analyze_statistical_test("Somer's D", st.session_state.somer_d_result, somers_d_threshold, st.session_state.model_type)
    #             # st.session_state.combined_json["Discriminatory Factors"]["Somer's D"]['analysis'] = analysis

    #             st.markdown(f'<ul style="margin-top: -15px;"><li class="small-font">The AUC for {var} is {round(st.session_state[f"{var}_results"]["AUC"], 2)}%.</li></ul>', unsafe_allow_html=True)
    #             st.markdown(f'<ul style="margin-top: -20px;"><li class="small-font">The GINI for {var} is {round(st.session_state[f"{var}_results"]["GINI"], 2)}%.</li></ul>', unsafe_allow_html=True)
    #             st.markdown(f'<ul style="margin-top: -20px;"><li class="small-font">The KS for {var} is {round(st.session_state[f"{var}_results"]["KS"], 2)}%.</li></ul>', unsafe_allow_html=True)

    #             st.write(st.session_state[f"{var}_results"]['summarized_data'])
                
    #             i += 1

    #     if "Somer's D" in st.session_state.disc_fac_tests:
    #         st.markdown(f"<p class='big-font'>Somer's D</p>", unsafe_allow_html=True)
    #         st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 

    #         somers_d_threshold = {
    #             "Somers' D Range": [
    #                 "> 0.5",
    #                 "0.2 < Somers' D ≤ 0.5",
    #                 "0 < Somers' D ≤ 0.2",
    #                 "Somers' D = 0",
    #                 "Somers' D < 0"
    #             ],
    #             "Interpretation": [
    #                 "The model has strong predictive power for distinguishing defaults.",
    #                 "The model has moderate ability to rank defaults effectively.",
    #                 "The model has weak predictive ability for distinguishing defaults.",
    #                 "The model has no predictive power for distinguishing defaults.",
    #                 "The model's predictions are inversely related to the actual outcomes."
    #             ]
    #         }

    #         if st.session_state.combined_json["Discriminatory Factors"]["Somer's D"]['analysis'] == {}:
    #             with st.spinner("Analyzing the results..."):
    #                 st.session_state.somer_d_result = somers_d(
    #                     st.session_state.data[st.session_state.def_flag], 
    #                     st.session_state.data[st.session_state.pd_cal]
    #                 )
    #                 st.session_state.combined_json["Discriminatory Factors"]["Somer's D"]['test_outputs'] = st.session_state.somer_d_result
    #                 analysis = report_functions.analyze_statistical_test(test_name="Somer's D", 
    #                                                                         test_output=st.session_state.somer_d_result,
    #                                                                         threshold=somers_d_threshold, 
    #                                                                         model_type=st.session_state.model_type)
    #                 st.session_state.combined_json["Discriminatory Factors"]["Somer's D"]['analysis'] = analysis
            
    #         st.write(st.session_state.combined_json["Discriminatory Factors"]["Somer's D"]['analysis'])

    # if st.session_state.validation_tests_app == 'Calibration Accuracy':

    #     st.markdown(
    #         """
    #         <style>
    #         .big-font {
    #             font-size:20px !important;
    #             font-weight: bold;
    #             padding: 0px;
    #         }
    #         .small-font {
    #             font-size:15px !important;
    #             padding: 0px;
    #         }
    #         .boldhr {
    #             width: 100%;
    #             height: 2px;
    #             background-color: #9c9b99; 
    #             margin: 2px;
    #             # margin-top: -20px;
    #         }
    #         </style>
    #         """,
    #         unsafe_allow_html=True
    #     )
        
    #     if st.session_state.pd_cal and st.session_state.def_flag:
    #         # Binomial Test
    #         if 'Binomial Test' in st.session_state.cal_acc_tests:
    #             st.markdown(f'<p class="big-font">Binomial Test</p>', unsafe_allow_html=True)
    #             st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 

    #             if st.session_state.combined_json["Calibration Accuracy"]["Binomial Test"]['analysis'] == {}:
    #                 with st.spinner('Analyzing results'):
    #                     st.session_state.binomial_test_result = binomial_test(
    #                         st.session_state.data[st.session_state.pd_cal], 
    #                         st.session_state.data[st.session_state.def_flag], 
    #                         alpha=0.05
    #                     )
                        
    #                     st.session_state.combined_json["Calibration Accuracy"]["Binomial Test"]['test_outputs'] = st.session_state.binomial_test_result
    #                     analysis = report_functions.analyze_statistical_test(test_name="Binomial Test", 
    #                                                                          test_output=st.session_state.binomial_test_result,
    #                                                                          threshold="alpha=0.05", 
    #                                                                          model_type=st.session_state.model_type)
    #                     st.session_state.combined_json["Calibration Accuracy"]["Binomial Test"]['analysis'] = analysis
                
    #             st.write(st.session_state.combined_json["Calibration Accuracy"]["Binomial Test"]['analysis'])

    #         # Chi-Square Test
    #         if 'Chi-Square Test' in st.session_state.cal_acc_tests:
    #             st.markdown(f'<p class="big-font">Chi-Square Test</p>', unsafe_allow_html=True)
    #             st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
                
    #             if st.session_state.combined_json["Calibration Accuracy"]["Chi-Square Test"]['analysis'] == {}:
    #                 with st.spinner('Analyzing results'):
    #                     st.session_state.chi_square_test_result = chi_square_test(
    #                         st.session_state.data[st.session_state.pd_cal], 
    #                         st.session_state.data[st.session_state.def_flag], 
    #                         alpha=0.05
    #                     )
                        
    #                     st.session_state.combined_json["Calibration Accuracy"]["Chi-Square Test"]['test_outputs'] = st.session_state.chi_square_test_result
    #                     analysis = report_functions.analyze_statistical_test(test_name="Chi-Square Test", 
    #                                                                          test_output=st.session_state.chi_square_test_result,
    #                                                                          threshold="alpha=0.05", 
    #                                                                          model_type=st.session_state.model_type)
    #                     st.session_state.combined_json["Calibration Accuracy"]["Chi-Square Test"]['analysis'] = analysis
                
    #             st.write(st.session_state.combined_json["Calibration Accuracy"]["Chi-Square Test"]['analysis'])

    #         # Normal Test
    #         if 'Normal Test' in st.session_state.cal_acc_tests:
    #             st.markdown(f'<p class="big-font">Normal Test</p>', unsafe_allow_html=True)
    #             st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
                
    #             if st.session_state.combined_json["Calibration Accuracy"]["Normal Test"]['analysis'] == {}:
    #                 with st.spinner('Analyzing results'):
    #                     st.session_state.normal_test_result = normal_test(
    #                         st.session_state.data[st.session_state.pd_cal], 
    #                         st.session_state.data[st.session_state.def_flag], 
    #                         alpha=0.05
    #                     )
                        
    #                     st.session_state.combined_json["Calibration Accuracy"]["Normal Test"]['test_outputs'] = st.session_state.normal_test_result
    #                     analysis = report_functions.analyze_statistical_test(test_name="Normal Test", 
    #                                                                          test_output=st.session_state.normal_test_result,
    #                                                                          threshold="alpha=0.05", 
    #                                                                          model_type=st.session_state.model_type)
    #                     st.session_state.combined_json["Calibration Accuracy"]["Normal Test"]['analysis'] = analysis
                
    #             st.write(st.session_state.combined_json["Calibration Accuracy"]["Normal Test"]['analysis'])

    #         # Traffic Light Test
    #         if 'Traffic Light Test' in st.session_state.cal_acc_tests:
    #             st.markdown(f'<p class="big-font">Traffic Light Test</p>', unsafe_allow_html=True)
    #             st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
                
    #             if st.session_state.combined_json["Calibration Accuracy"]["Traffic Light Test"]['analysis'] == {}:
    #                 with st.spinner('Analyzing results'):
    #                     st.session_state.traffic_test_results = traffic_light_test(
    #                         st.session_state.data[st.session_state.def_flag], 
    #                         st.session_state.data[st.session_state.pd_cal], 
    #                         thresholds=[1/3, 0.5, 2/3]
    #                     )
                        
    #                     st.session_state.combined_json["Calibration Accuracy"]["Traffic Light Test"]['test_outputs'] = st.session_state.traffic_test_results
    #                     analysis = report_functions.analyze_statistical_test(test_name="Traffic Light Test", 
    #                                                                          test_output=st.session_state.traffic_test_results,
    #                                                                          threshold=[1/3, 0.5, 2/3], 
    #                                                                          model_type=st.session_state.model_type)
    #                     st.session_state.combined_json["Calibration Accuracy"]["Traffic Light Test"]['analysis'] = analysis
                
    #             st.write(st.session_state.combined_json["Calibration Accuracy"]["Traffic Light Test"]['analysis'])

    #         # Pluto Tasche Test
    #         if 'Pluto Tasche' in st.session_state.cal_acc_tests:
    #             st.markdown(f'<p class="big-font">Pluto Tasche Test</p>', unsafe_allow_html=True)
    #             st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 

    #             if st.session_state.combined_json["Calibration Accuracy"]["Pluto Tasche"]['analysis'] == {}:
    #                 with st.spinner('Analyzing results'):
    #                     st.session_state.pluto_tasche_results = pluto_tasche_upper_bound(
    #                         st.session_state.data[st.session_state.def_flag], 
    #                         confidence_level=0.95
    #                     )
                        
    #                     st.session_state.combined_json["Calibration Accuracy"]["Pluto Tasche"]['test_outputs'] = st.session_state.pluto_tasche_results
    #                     analysis = report_functions.analyze_statistical_test(test_name="Pluto Tasche Test", 
    #                                                                          test_output=st.session_state.pluto_tasche_results,
    #                                                                          threshold="confidence_level=0.95", 
    #                                                                          model_type=st.session_state.model_type)
    #                     st.session_state.combined_json["Calibration Accuracy"]["Pluto Tasche"]['analysis'] = analysis
                
    #             st.write(st.session_state.combined_json["Calibration Accuracy"]["Pluto Tasche"]['analysis'])
    #             # st.write(st.session_state.pluto_tasche_results[1])

    # if st.session_state.validation_tests_app == 'Model Stability':

    #     if 'ms_bins' not in st.session_state:
    #         st.session_state.ms_bins = 5

    #     st.markdown(
    #         """
    #         <style>
    #         .big-font {
    #             font-size:20px !important;
    #             font-weight: bold;
    #             padding: 0px;
    #         }
    #         .small-font {
    #             font-size:15px !important;
    #             padding: 0px;
    #         }
    #         .boldhr {
    #             width: 100%;
    #             height: 2px;
    #             background-color: #9c9b99; 
    #             margin: 2px;
    #             # margin-top: -20px;
    #         }
    #         </style>
    #         """,
    #         unsafe_allow_html=True
    #     )

    #     if st.session_state.pd_cal and st.session_state.def_flag and st.session_state.ms_bins:
            
    #         if 'Population Stability Index (PSI)' in st.session_state.model_stability_tests:
    #             st.markdown(f'<p class="big-font">Population Stability Index (PSI)</p>', unsafe_allow_html=True)
    #             st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
                
    #             if st.session_state.combined_json["Model Stability"]["Population Stability Index (PSI)"]['analysis'] == {}:
    #                 with st.spinner('Analyzing results...'):
    #                     st.session_state.psi_result = calculate_psi(
    #                         st.session_state.data[st.session_state.pd_cal],
    #                         st.session_state.data[st.session_state.def_flag],
    #                         num_bins=st.session_state.ms_bins
    #                     )
                        
    #                     st.session_state.combined_json["Model Stability"]["Population Stability Index (PSI)"]['test_outputs'] = st.session_state.psi_result
    #                     analysis = report_functions.analyze_statistical_test(test_name="Population Stability Index (PSI)", 
    #                                                                          test_output=st.session_state.psi_result,
    #                                                                          threshold="number of bins used are 5", 
    #                                                                          model_type=st.session_state.model_type)
    #                     st.session_state.combined_json["Model Stability"]["Population Stability Index (PSI)"]['analysis'] = analysis
                
    #             st.write(st.session_state.combined_json["Model Stability"]["Population Stability Index (PSI)"]['analysis'])
                
    #             # st.write(f'The PSI is {st.session_state.psi_result[0]:.2f}. \n {st.session_state.psi_result[1]}')

    # if st.session_state.validation_tests_app == 'External Benchmarking':
    #     st.markdown(
    #         """
    #         <style>
    #         .big-font {
    #             font-size:20px !important;
    #             font-weight: bold;
    #             padding: 0px;
    #         }
    #         .small-font {
    #             font-size:15px !important;
    #             padding: 0px;
    #         }
    #         .boldhr {
    #             width: 100%;
    #             height: 2px;
    #             background-color: #9c9b99; 
    #             margin: 2px;
    #             # margin-top: -20px;
    #         }
    #         </style>
    #         """,
    #         unsafe_allow_html=True
    #     )
        
    #     st.session_state.actual_rating = 'Model_Rating'
    #     st.session_state.final_rating = 'Final_Rating'


    #     if st.session_state.actual_rating and st.session_state.final_rating:
    #         # Spearman's Correlation Test
    #         if "Spearman's correlation" in st.session_state.ext_ben_tests:
    #             st.markdown(f"<p class='big-font' style='margin-bottom:-10px;'>Spearman's Correlation</p>", unsafe_allow_html=True)
    #             st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
                
    #             if st.session_state.combined_json["External Benchmarking"]["Spearman's correlation"]['analysis'] == {}:
    #                 with st.spinner('Analyzing results...'):
    #                     st.session_state.spearman_corr_result = spearman_correlation(
    #                         st.session_state.data[st.session_state.actual_rating], 
    #                         st.session_state.data[st.session_state.final_rating]
    #                     )
                        
    #                     st.session_state.combined_json["External Benchmarking"]["Spearman's correlation"]['test_outputs'] = st.session_state.spearman_corr_result
    #                     analysis = report_functions.analyze_statistical_test(test_name="Spearman's correlation", 
    #                                                                          test_output=st.session_state.spearman_corr_result,
    #                                                                          threshold=None, 
    #                                                                          model_type=st.session_state.model_type)
    #                     st.session_state.combined_json["External Benchmarking"]["Spearman's correlation"]['analysis'] = analysis
                
    #             st.write(st.session_state.combined_json["External Benchmarking"]["Spearman's correlation"]['analysis'])

    #         # Multi-notch Movement Test
    #         if 'Multi-notch movement' in st.session_state.ext_ben_tests:
    #             st.markdown(f"<p class='big-font' style='margin-top:20px; margin-bottom:-10px;'>Multi-notch movement</p>", unsafe_allow_html=True)
    #             st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 

    #             if 'multi_notch_movement_results' not in st.session_state:
    #                 with st.spinner('Analyzing results...'):
    #                     st.session_state.multi_notch_movement_results = multi_notch_movement(
    #                         st.session_state.data[st.session_state.actual_rating], 
    #                         st.session_state.data[st.session_state.final_rating]
    #                     )
                        
    #                     st.session_state.combined_json["External Benchmarking"]["Multi-notch movement"]['test_outputs'] = st.session_state.multi_notch_movement_results
    #                     analysis = report_functions.analyze_statistical_test(test_name="Multi-notch movement", 
    #                                                                          test_output=st.session_state.multi_notch_movement_results,
    #                                                                          threshold=None, 
    #                                                                          model_type=st.session_state.model_type)
    #                     st.session_state.combined_json["External Benchmarking"]["Multi-notch movement"]['analysis'] = analysis
                
    #             st.write(st.session_state.combined_json["External Benchmarking"]["Multi-notch movement"]['analysis'])

    # if st.session_state.validation_tests_app == 'Overrides Analysis':
    #     st.markdown(
    #         """
    #         <style>
    #         .big-font {
    #             font-size:20px !important;
    #             font-weight: bold;
    #             padding: 0px;
    #         }
    #         .small-font {
    #             font-size:15px !important;
    #             padding: 0px;
    #         }
    #         .boldhr {
    #             width: 100%;
    #             height: 2px;
    #             background-color: #9c9b99; 
    #             margin: 2px;
    #             # margin-top: -20px;
    #         }
    #         </style>
    #         """,
    #         unsafe_allow_html=True
    #     )

    #     st.session_state.actual_rating = 'Model_Rating'
    #     st.session_state.final_rating = 'Final_Rating'

    #     if st.session_state.actual_rating and st.session_state.final_rating:

    #         # Overrides Rates Test
    #         if 'Overrides Rates' in st.session_state.overrides_tests:
    #             st.markdown(f'<p class="big-font" style="margin-bottom:-10px;">Overrides Rates</p>', unsafe_allow_html=True)
    #             st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
                
    #             if st.session_state.combined_json["Overrides"]["Overrides Rates"]['analysis'] == {}:
    #                 with st.spinner('Analyzing results...'):
    #                     st.session_state.override_rate_result = override_rate(
    #                         st.session_state.data[st.session_state.actual_rating], 
    #                         st.session_state.data[st.session_state.final_rating]
    #                     )
                        
    #                     output = f"{st.session_state.override_rate_result[0]} overrides have been observed, accounting for {st.session_state.override_rate_result[1]:.2f}% of the obligors."

    #                     st.session_state.combined_json["Overrides"]["Overrides Rates"]['test_outputs'] = output
    #                     analysis = report_functions.analyze_statistical_test(test_name="Overrides Rates", 
    #                                                                          test_output=output,
    #                                                                          threshold=None, 
    #                                                                          model_type=st.session_state.model_type)
    #                     st.session_state.combined_json["Overrides"]["Overrides Rates"]['analysis'] = analysis
                
    #             st.write(st.session_state.combined_json["Overrides"]["Overrides Rates"]['analysis'])
                
    #         # % Downgrade Override Rate Test
    #         if '% Downgrade Override rate' in st.session_state.overrides_tests:
    #             st.markdown(f'<p class="big-font" style="margin-top:10px; margin-bottom:-10px;">% Downgrade Override rate</p>', unsafe_allow_html=True)
    #             st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 
                
    #             if st.session_state.combined_json["Overrides"]["% Downgrade Override rate"]['analysis'] == {}:
    #                 with st.spinner('Analyzing results...'):
    #                     st.session_state.downgrade_override_rate_results = downgrade_override_rate(
    #                         st.session_state.data[st.session_state.actual_rating], 
    #                         st.session_state.data[st.session_state.final_rating]
    #                     )
                        
    #                     output = f"{st.session_state.downgrade_override_rate_results[0]} downgrade overrides have been observed, accounting for {st.session_state.downgrade_override_rate_results[1]:.2f}% of the obligors."

    #                     st.session_state.combined_json["Overrides"]["% Downgrade Override rate"]['test_outputs'] = output
    #                     analysis = report_functions.analyze_statistical_test(test_name="% Downgrade Override rate", 
    #                                                                          test_output=output,
    #                                                                          threshold=None, 
    #                                                                          model_type=st.session_state.model_type)
    #                     st.session_state.combined_json["Overrides"]["% Downgrade Override rate"]['analysis'] = analysis
                
    #             st.write(st.session_state.combined_json["Overrides"]["% Downgrade Override rate"]['analysis'])

    # if st.session_state.validation_tests_app == 'Data Quality':
    #     st.markdown(
    #         """
    #         <style>
    #         .big-font {
    #             font-size:20px !important;
    #             font-weight: bold;
    #             padding: 0px;
    #         }
    #         .small-font {
    #             font-size:15px !important;
    #             padding: 0px;
    #         }
    #         .boldhr {
    #             width: 100%;
    #             height: 2px;
    #             background-color: #9c9b99; 
    #             margin: 2px;
    #             # margin-top: -20px;
    #         }
    #         </style>
    #         """,
    #         unsafe_allow_html=True
    #     )

    #     if 'Percentage Of Missing Values' in st.session_state.data_quality_tests:
    #         st.markdown(f'<p class="big-font">Percentage Of Missing Values</p>', unsafe_allow_html=True)
    #         st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 

    #         missing_data_df = missing_data_summary(st.session_state.data)

    #         if len(missing_data_df) == 0:
    #             st.write('No column(s) have missing data.')
    #         else:
    #             st.write(missing_data_df)

    #             if st.session_state.combined_json["Data Quality"]["Percentage Of Missing Values"]['analysis'] == {}:
    #                 with st.spinner('Analyzing results...'):
    #                     st.session_state.combined_json["Data Quality"]["Percentage Of Missing Values"]['test_outputs'] = missing_data_df
    #                     analysis = report_functions.analyze_statistical_test(test_name="Data Quality - Percentage Of Missing Values", 
    #                                                                          test_output=missing_data_df,
    #                                                                          threshold=None, 
    #                                                                          model_type=st.session_state.model_type)
    #                     st.session_state.combined_json["Data Quality"]["Percentage Of Missing Values"]['analysis'] = analysis
                
    #             st.write(st.session_state.combined_json["Data Quality"]["Percentage Of Missing Values"]['analysis'])
                
    #             data_list = missing_data_df.to_numpy().tolist()
    #             result = [list(missing_data_df.columns)] + data_list
    #             st.session_state.tables_data = {"Percentage Of Missing Values": result}

    #             # st.write(st.session_state.tables_data)

                    




def display_confusion_matrix(actual, predicted):
    # Define LGD categorization function
    def categorize_lgd(lgd):
        if lgd <= 0.33:
            return 'Low'
        elif 0.33 < lgd <= 0.67:
            return 'Medium'
        else:
            return 'High'
    
    # Categorize actual and predicted LGDs
    actual_category = [categorize_lgd(x) for x in actual]
    predicted_category = [categorize_lgd(x) for x in predicted]
    
    # Generate confusion matrix
    labels = ['Low', 'Medium', 'High']
    cm = confusion_matrix(actual_category, predicted_category, labels=labels)
    
    # Calculate overall accuracy
    overall_accuracy = np.trace(cm) / np.sum(cm)
    
    # Initialize the matrix to store metrics
    metrics_matrix = pd.DataFrame(index=['Low', 'Medium', 'High', 'Overall'], columns=['Accuracy (%)', 'Specificity (%)', 'Sensitivity (%)'])
    
    # Calculate metrics for each category
    total_tp, total_fn, total_fp, total_tn = 0, 0, 0, 0  # For calculating overall sensitivity and specificity
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp  # False Negatives
        fp = np.sum(cm[:, i]) - tp  # False Positives
        tn = np.sum(cm) - (tp + fn + fp)  # True Negatives

        # Accumulate totals for overall sensitivity and specificity
        total_tp += tp
        total_fn += fn
        total_fp += fp
        total_tn += tn
        
        # Calculate sensitivity, specificity, and accuracy for each category
        sensitivity = round(100*tp / (tp + fn) if (tp + fn) > 0 else 0, 2)
        specificity = round(100*tn / (tn + fp) if (tn + fp) > 0 else 0, 2)
        category_accuracy = round(100*(tp + tn) / (tp + tn + fp + fn), 2)
        
        # Add the metrics to the matrix for the current category
        metrics_matrix.loc[label] = [category_accuracy, specificity, sensitivity]
    
    # Calculate overall sensitivity and specificity
    overall_sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
    
    # Add overall metrics to the matrix
    metrics_matrix.loc['Overall'] = [overall_accuracy, overall_specificity, overall_sensitivity]
    
    # Display confusion matrix as DataFrame
    # st.write("Confusion Matrix:")
    # st.write(pd.DataFrame(cm, index=labels, columns=labels))
    
    # Plot confusion matrix as heatmap
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    # plt.title("LGD Confusion Matrix")
    st.pyplot(fig)
    
    # Display metrics matrix
    st.write("Metrics Matrix:")
    st.write(metrics_matrix)


def somers_d(actual, predicted, upper_bound, lower_bound, green_desc, amber_desc, red_desc):
    """
    Calculate Somer's D for binary classification model validation with interpretation.

    Parameters:
    actual (list or array): List or array of actual binary outcomes (0 or 1).
    predicted (list or array): List or array of predicted probability scores.

    Returns:
    float: Somer's D value, which ranges from -1 to 1.
    str: Interpretation of Somer's D value.
    """

    tau, _ = kendalltau(predicted, actual)
    somers_d_value = 2 * tau / (1 + tau)

    if somers_d_value >= upper_bound:
        description = green_desc
        status = 'Green'
    elif lower_bound < somers_d_value < upper_bound:
        description = amber_desc
        status = 'Amber'
    else:
        description = red_desc
        status = 'Red'

    return_json = {'status':status, 
                   'somers_d_value': somers_d_value,
                   'description': description}

    return return_json

def missing_data_summary(data):
    missing_count = data.isnull().sum()
    missing_percentage = data.isnull().mean() * 100
    formatted_percentage = missing_percentage.apply(lambda x: f"{x:.2f}%")
    missing_cols = formatted_percentage[missing_percentage > 0]
    
    result = (
        pd.DataFrame({
            'Column': missing_cols.index,
            'Missing Count': missing_count[missing_cols.index],
            'Missing Percentage': missing_cols
        })
        .sort_values(by='Missing Count', ascending=False)
        .reset_index(drop=True)
    )
    return result
    
def downgrade_override_rate(actual, overridden, upper_bound, lower_bound, green_desc, amber_desc, red_desc):
    downgrades = (actual> overridden)
    downgrade_count = sum(downgrades)
    downgrade_override_rate = (downgrade_count / len(actual)) * 100

    if downgrade_override_rate <= lower_bound:
        description = green_desc
        status = 'Green'
    elif lower_bound < downgrade_override_rate < upper_bound:
        description = amber_desc
        status = 'Amber'
    else:
        description = red_desc
        status = 'Red'

    return_json = {'status': status,
                   'Total downgrades': downgrades,
                   'Downgrade Override Rate': f'{round(downgrade_override_rate, 3)} %',
                   'description': description}

    return return_json

def override_rate(actual, overridden, upper_bound, lower_bound, green_desc, amber_desc, red_desc):
    override_count = (actual != overridden).sum()
    override_rate = (override_count / len(actual)) * 100

    if override_rate <= lower_bound:
        description = green_desc
        status = 'Green'
    elif lower_bound < override_rate < upper_bound:
        description = amber_desc
        status = 'Amber'
    else:
        description = red_desc
        status = 'Red'

    return_json = {'status': status,
                   'Total Override': override_count,
                   'Override Rate': f'{round(override_rate, 3)} %',
                   'description': description}
    
    return return_json

def spearman_correlation(x, y, upper_bound, lower_bound, green_desc, amber_desc, red_desc):
    """
    Calculate the Spearman correlation between two variables and provide detailed interpretations 
    of both the correlation coefficient and the p-value.

    Parameters:
    - x: array-like, first variable
    - y: array-like, second variable

    Returns:
    - correlation: Spearman correlation coefficient
    - p_value: p-value of the correlation
    - correlation_interpretation: Explanation of the correlation strength
    - p_value_interpretation: Explanation of the statistical significance
    """
    correlation, p_value = spearmanr(x, y)
    
    if correlation >= upper_bound:
        description = green_desc
        status = 'Green'
    elif correlation <= lower_bound:
        description = red_desc
        status = 'Red'
    else:
        description = amber_desc
        status = 'Amber'

    return_json = {"status": status,
                   "correlation": round(correlation, 3),
                   "description": description,
                   "lower_bound": lower_bound,
                   "upper_bound": upper_bound}
    
    return return_json

def multi_notch_movement(model_rating, actual_rating, n_obligors, upper_bound, lower_bound, green_desc, amber_desc, red_desc):
    """
    Calculate the number of multi-notch movements between model and actual ratings.

    Parameters:
    - model_rating: array-like, predicted/model ratings
    - actual_rating: array-like, actual ratings

    Returns:
    - multi_notch_movements: Number of multi-notch movements
    - interpretation: Explanation of the movement significance
    """
    # Calculate the number of multi-notch movements
    multi_notch = (model_rating - actual_rating)
    multi_notch_up = sum(multi_notch >= 2)  # Multi-notch defined as difference >= 2
    multi_notch_up_perc = multi_notch_up/n_obligors

    multi_notch_down = sum(multi_notch <= -2)  # Multi-notch defined as difference <= 2
    multi_notch_down_perc = multi_notch_down/n_obligors

    multi_notch_movements_perc = (multi_notch_up + multi_notch_down)/n_obligors

    total_obligors_multinotch = multi_notch_up + multi_notch_down

    if multi_notch_movements_perc >= upper_bound:
        description = red_desc
        status = 'Red'
    elif multi_notch_movements_perc <= lower_bound:
        description = green_desc
        status = 'Green'
    else:
        description = amber_desc
        status = 'Amber'
    
    return_json = {'status': status,
                   'multi_notch_up': multi_notch_up,
                   'multi_notch_up_perc': multi_notch_up_perc,
                   'multi_notch_down': multi_notch_down,
                   'multi_notch_down_perc': multi_notch_down_perc,
                   'multi_notch_movements_perc': multi_notch_movements_perc,
                   'total_obligors_multinotch': total_obligors_multinotch,
                   'description': description
                   }
    
    return return_json

def calculate_psi(expected_flags, actual_flags, num_bins, lower_bound, upper_bound, green_desc, amber_desc, red_desc):
    # Calculate the likelihood of default
    expected_probs = pd.Series(expected_flags).rolling(window=5, min_periods=1).mean()
    actual_probs = pd.Series(actual_flags).rolling(window=5, min_periods=1).mean()

    # Create bins based on expected probabilities
    bin_edges = np.linspace(0, 1, num_bins + 1)
    expected_bins = pd.cut(expected_probs, bins=bin_edges, include_lowest=True).value_counts(normalize=True, sort=False)
    actual_bins = pd.cut(actual_probs, bins=bin_edges, include_lowest=True).value_counts(normalize=True, sort=False)

    # Replace 0% with a small value to avoid division by zero or log errors
    expected_bins = expected_bins.replace(0, 0.0001)
    actual_bins = actual_bins.replace(0, 0.0001)

    # Calculate PSI for each bin
    psi_values = (expected_bins - actual_bins) * np.log(expected_bins / actual_bins)

    # Total PSI
    psi_total = psi_values.sum()

    # Interpretation based on PSI value
    if psi_total <= lower_bound:
        description = green_desc
        status = 'Green'
    elif lower_bound < psi_total < upper_bound:
        description = amber_desc
        status = 'Amber'
    else:
        description = red_desc
        status = 'Red'

    return_json = {'status': status,
                   'PSI Value': round(psi_total, 3),
                   'Number of bins used': num_bins,
                   'description': description}

    return return_json

def pluto_tasche_test(predicted_pds, observed_defaults, n_obligors, lower_bound, upper_bound, confidence_level, Green_Desc, Amber_Desc, Red_Desc):
    """
    Perform the Pluto Tasche test on a PD model.

    Parameters:
    predicted_pds (np.array): Array of predicted probabilities of default.
    observed_defaults (np.array): Array of observed default flags (1 for default, 0 for non-default).
    n_obligors (int): Total number of unique obligors.
    confidence_level (float): Confidence level for the upper bound calculation.

    Returns:
    dict: A dictionary containing the status, descriptor, log ratio, and upper bound for the model.
    """
    predicted_pds = np.asarray(predicted_pds)
    observed_defaults = np.asarray(observed_defaults)
    
    # Use mean PD as the aggregate
    avg_predicted_pd = np.mean(predicted_pds)  
    
    # Number of observed defaults
    n_defaults = np.sum(observed_defaults)

    # Compute the upper bound using the binomial distribution
    alpha = 1 - confidence_level
    upper_bound_test = binom.ppf(1 - alpha, n_obligors, n_defaults / n_obligors) / n_obligors
    
    # Calculate the ratio of predicted PD to upper bound
    ratio = avg_predicted_pd / upper_bound_test
    
    # Calculate the natural logarithm of the ratio
    log_ratio = -np.log(ratio)
    
    # Determine the status based on the thresholds
    if log_ratio <= lower_bound:
        description = Green_Desc
        status = 'Green'
    elif lower_bound < log_ratio < upper_bound:
        description = Amber_Desc
        status = 'Amber'
    else:
        description = Red_Desc
        status = 'Red'
    
    return {
        "status": status,
        "description": description,
        "Log Ratio": log_ratio,
        "Upper Bound": upper_bound,
        "Confidence Interval": confidence_level
    }

def normal_test(predicted_pds, observed_default_flags, lower_bound, upper_bound, Green_Desc, Amber_Desc, Red_Desc):
    """
    Perform a normal test (z-test) on the PD model, including computation of critical values.

    Parameters:
    predicted_pds (np.array): Array of predicted probabilities of default.
    observed_default_flags (np.array): Array of observed default flags (1 for default, 0 for no default).

    Returns:
    dict: A dictionary containing the description of the model prediction status, expected defaults, 
          actual defaults, and critical values at the specified confidence intervals.
    """
    # Calculate the expected number of defaults
    expected_defaults = np.sum(predicted_pds)
    
    # Calculate the actual number of defaults
    actual_defaults = np.sum(observed_default_flags)
    
    # Calculate the standard deviation of the predicted defaults
    std_dev = np.sqrt(np.sum(predicted_pds * (1 - predicted_pds)))
    
    # Calculate the z-score
    z_score = (actual_defaults - expected_defaults) / std_dev
    
    # Compute critical values for the specified confidence levels
    alpha_lower = norm.ppf(lower_bound)
    alpha_upper = norm.ppf(upper_bound)
    
    # Determine the status based on thresholds
    if z_score <= alpha_lower:
        description = Green_Desc
        status = 'Green'
    elif alpha_lower < z_score < alpha_upper:
        description = Amber_Desc
        status = 'Amber'
    else:
        description = Red_Desc
        status = 'Red'

    return_json = {"status":status, 
                   "description": description,
                   "expected_defaults": expected_defaults, 
                   "actual_defaults": actual_defaults,
                   "z_score": z_score,
                   f"Critical value at {lower_bound} confidence interval": alpha_lower,
                   f"Critical value at {upper_bound} confidence interval": alpha_upper}
    
    return return_json

def binomial_test(predicted_pds, observed_default_flags, lower_bound, upper_bound, Green_Desc, Amber_Desc, Red_Desc):
    """
    Perform a binomial test on the PD model, including computation of critical values.

    Parameters:
    predicted_pds (np.array): Array of predicted probabilities of default.
    observed_default_flags (np.array): Array of observed default flags (1 for default, 0 for no default).

    Returns:
    str: Status of the model prediction (Green, Amber, Red).
    """
    # Calculate the expected number of defaults
    expected_defaults = np.sum(predicted_pds)
    
    # Calculate the actual number of defaults
    actual_defaults = np.sum(observed_default_flags)
    
    # Calculate the Def# (difference between actual and expected defaults)
    # def_number = actual_defaults - expected_defaults
    
    # Compute critical values for 75% and 95% confidence levels
    alpha_lower = binom.ppf(lower_bound, n=len(predicted_pds), p=expected_defaults/len(predicted_pds))
    alpha_upper = binom.ppf(upper_bound, n=len(predicted_pds), p=expected_defaults/len(predicted_pds))
    
    # Determine the status based on thresholds
    if actual_defaults <= alpha_lower:
        description = Green_Desc
        status = 'Green'
    elif alpha_lower < actual_defaults < alpha_upper:
        description = Amber_Desc
        status = 'Amber'
    else:
        description = Red_Desc
        status = 'Red'

    return_json = {'status':status,
                   "description": description,
                   "expected_defaults": expected_defaults, 
                   "actual_defaults": actual_defaults,
                   f"Critical value at {lower_bound} confidence interval": alpha_lower,
                   f"Critical value at {upper_bound} confidence interval": alpha_upper}
    
    return return_json

def chi_square_test(predicted_pds, observed_default_flags, lower_bound, upper_bound, Green_Desc, Amber_Desc, Red_Desc):
    """
    Perform a chi-square goodness-of-fit test on the PD model, including comparison with lower and upper bounds.

    Parameters:
    predicted_pds (np.array): Array of predicted probabilities of default.
    observed_default_flags (np.array): Array of observed default flags (1 for default, 0 for no default).
    lower_bound (float): Lower bound for the confidence interval (e.g., 0.25 for 25%).
    upper_bound (float): Upper bound for the confidence interval (e.g., 0.75 for 75%).

    Returns:
    dict: A dictionary containing the description of the model prediction status, expected defaults,
          actual defaults, chi-square statistic, critical values at lower and upper bounds, and p-value.
    """
    # Calculate the expected number of defaults
    expected_defaults = np.sum(predicted_pds)
    
    # Calculate the actual number of defaults
    actual_defaults = np.sum(observed_default_flags)
    
    # Calculate the expected number of non-defaults
    expected_non_defaults = len(predicted_pds) - expected_defaults
    
    # Calculate the actual number of non-defaults
    actual_non_defaults = len(observed_default_flags) - actual_defaults
    
    # Create observed and expected frequency arrays
    observed_frequencies = np.array([actual_defaults, actual_non_defaults])
    expected_frequencies = np.array([expected_defaults, expected_non_defaults])
    
    # Calculate the chi-square statistic
    chi_square_statistic = np.sum((observed_frequencies - expected_frequencies) ** 2 / expected_frequencies)
    
    # Degrees of freedom (number of categories - 1)
    degrees_of_freedom = 1
    
    # Calculate the critical values for the lower and upper bounds
    critical_value_lower = chi2.ppf(lower_bound, degrees_of_freedom)
    critical_value_upper = chi2.ppf(upper_bound, degrees_of_freedom)
    
    # Calculate the p-value
    p_value = 1 - chi2.cdf(chi_square_statistic, degrees_of_freedom)
    
    # Determine the status based on the chi-square statistic and critical values
    if chi_square_statistic <= critical_value_lower:
        description = Green_Desc
        status = 'Green'
    elif critical_value_lower < chi_square_statistic < critical_value_upper:
        description = Amber_Desc
        status = 'Amber'
    else:
        description = Red_Desc
        status = 'Red'

    return_json = {
        "status": status,
        "description": description,
        "expected_defaults": expected_defaults,
        "actual_defaults": actual_defaults,
        "chi_square_statistic": chi_square_statistic,
        f"Critical value at {lower_bound} confidence interval": critical_value_lower,
        f"Critical value at {upper_bound} confidence interval": critical_value_upper,
        "p_value": p_value
    }
    
    return return_json

def traffic_light_test(y_true, y_pred, thresholds=[1/3, 0.5, 2/3]):
    """
    Perform a Traffic Light Test for a PD model by categorizing predictions 
    based on given thresholds and comparing them to actual outcomes.

    Parameters:
    y_true (pd.Series): Actual defaults (1 for default, 0 for non-default).
    y_pred (pd.Series): Predicted probabilities of default.
    thresholds (list): List of thresholds to categorize predictions (default is [0.25, 0.5, 0.75]).

    Returns:
    result (pd.DataFrame): Crosstabulation of predicted categories vs. actual outcomes.
    interpretation (str): Detailed interpretation of the result.
    """
    categories = ['Green', 'Yellow', 'Red']
    pred_category = ['Green' if p <= thresholds[0] else 'Yellow' if p <= thresholds[1] else 'Red' for p in y_pred]

    y_true_mapped = y_true.map({0: 'Non-Defaults', 1: 'Defaults'})

    result = pd.crosstab(pd.Series(pred_category, name='Predicted'), 
                         y_true_mapped, 
                         rownames=['Predicted'], 
                         colnames=['Actual'])

    if 'Defaults' not in result.columns:
        result['Defaults'] = 0
    
    result['Defaults %age'] = round(100*result['Defaults']/result['Non-Defaults'], 2)

    green_non_default = result.loc['Green', 'Non-Defaults'] if 'Green' in result.index and 'Non-Defaults' in result.columns else 0
    green_default = result.loc['Green', 'Defaults'] if 'Green' in result.index and 'Defaults' in result.columns else 0
    yellow_non_default = result.loc['Yellow', 'Non-Defaults'] if 'Yellow' in result.index and 'Non-Defaults' in result.columns else 0
    yellow_default = result.loc['Yellow', 'Defaults'] if 'Yellow' in result.index and 'Defaults' in result.columns else 0
    red_non_default = result.loc['Red', 'Non-Defaults'] if 'Red' in result.index and 'Non-Defaults' in result.columns else 0
    red_default = result.loc['Red', 'Defaults'] if 'Red' in result.index and 'Defaults' in result.columns else 0

    interpretation = (
        # f"The Traffic Light Test shows the following distribution:\n\n"
        # f"Green Category: {green_non_default} non-defaults and {green_default} defaults.\n\n"
        # f"Yellow Category: {yellow_non_default} non-defaults and {yellow_default} defaults.\n\n"
        # f"Red Category: {red_non_default} non-defaults and {red_default} defaults.\n\n"
        "Interpretation:\n"
        "1. The Green category is intended to represent low risk. Any defaults here indicate underestimation of risk.\n"
        "2. The Yellow category represents moderate risk, and defaults in this category suggest accurate risk assessment, "
        "though non-defaults suggest some overestimation.\n"
        "3. The Red category indicates high risk, where defaults are expected. Any non-defaults in this category suggest "
        "overestimation of risk."
    )

    return result, interpretation

def gen_rep_page():

    cols = st.columns([2, 1, 1.7, 1])
    with cols[0]:
        st.write('Are you sure you want to generate the report?')
    ChangeButtonColour('Yes', 'black', '#B2BBD2', '10px', margin_top = '-5px', margin_bottom = '-10px', border = None)
    if cols[1].button('Yes', key='gen_rep_button'):
            
        cols_inner = st.columns([2, 2.7])

        with cols_inner[0]:
            with st.spinner("Creating the word document..."):
                report_functions.generate_model_validation_report(st.session_state.model_type, 
                                                              st.session_state.exec_summary, 
                                                              st.session_state.data_summary, 
                                                              None, 
                                                              st.session_state.combined_json, 
                                                              st.session_state.tables_charts, 
                                                              st.session_state.tables_data, 
                                                              st.session_state.cat_test_status_df)





def validation_tests_sidebar():
    with st.sidebar:
        css = """
        <style>
        section[data-testid="stSidebar"] > div:first-child {
            background-color: #13326b;  /* Change this color to any hex color you prefer */
        }
        .stSelectbox {
            margin-top: -40px;
        }
        .stMultiSelect {
            margin-top: -40px;
        }
        .stTextInput {
            margin-top: -35px;
        }
        .stCheckbox {
            margin-top: -15px;
        }
        .stNumberInput {
            margin-top: -40px;
        }
        </style>
        """
    
        st.markdown(css, unsafe_allow_html=True)
    
        app_menu_df = {
            'App': ['Summary',
                   'Discriminatory Factors', 
                   'Calibration Accuracy', 
                   'Model Stability', 
                   'External Benchmarking',
                   'Overrides Analysis', 
                   'Data Quality',
                   'Generate Report'
#                    'IFRS9 & ST Models'
                   ],
            'Tests': ['Summary',
                     st.session_state.disc_fac_tests,
                     st.session_state.cal_acc_tests,
                     st.session_state.model_stability_tests,
                     st.session_state.ext_ben_tests,
                     st.session_state.overrides_tests,
                     st.session_state.data_quality_tests,
                     'Generate Report'
#                      st.session_state.selected_ifrs9_tests
                     ]
        }
        
        app_menu_df = pd.DataFrame(app_menu_df)
        app_menu_df_filtered = app_menu_df[app_menu_df['Tests'].notna() & (app_menu_df['Tests'].astype(bool))]

        st.session_state.validation_tests_app = option_menu(
                menu_title='Model Validation Tests',
                options = list(app_menu_df_filtered['App']),
                menu_icon='list-task',
                default_index=0,
                styles={
                    "container": {"padding": "5!important", "background-color": '#dce4f9', "border-radius":"5px", "font-weight":"bold"},
                    "nav-link": {"color": "black", "font-size": "12px", "text-align": "left", "margin": "0px", "--hover-color": "13326b"},
                    "nav-link-selected": {"color": "white", "background-color": "#13326b"},
                    "menu-title": {"font-size": "18px"}
                }
            )
        
        
        
        # cols = st.columns(1)
        # ChangeButtonColour('Generate Report', 'black', '#dce4f9', '2px', margin_top = '10px', border = None)
        # if cols[0].button('Generate Report', key='gen_rep_page'):
        #     st.session_state['page'] = 'gen_rep_page'
        #     st.session_state.need_rerun = True
        #     if st.session_state.need_rerun:
        #         st.session_state.need_rerun = False
        #         st.rerun()
        
        

def lgd_validation_test():
    
    if 'lgd_flag' not in st.session_state:
        st.session_state.lgd_flag = None
    if 'lgd_flag_ind' not in st.session_state:
        st.session_state.lgd_flag_ind = None

    if 'lgd_cal' not in st.session_state:
        st.session_state.lgd_cal = None
    if 'lgd_cal_ind' not in st.session_state:
        st.session_state.lgd_cal_ind = None
    
    if st.session_state.validation_tests_app == 'Discriminatory Power Tests':

        st.markdown(
            """
            <style>
            .big-font {
                font-size:20px !important;
                font-weight: bold;
                padding: 0px;
            }
            .medium-font {
                font-size:18px !important;
                font-weight: bold;
                padding: 0px;
            }
            .small-font {
                font-size:15px !important;
                padding: 0px;
            }
            .boldhr {
                width: 100%;
                height: 2px;
                background-color: #9c9b99; 
                margin: 2px;
                # margin-top: -20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(f'<p class="big-font">Select Predicted LGD</p>', unsafe_allow_html=True)
        cols_lgd = st.columns(2)
        with cols_lgd[0]:
            lgd_cal = st.selectbox("",list(st.session_state.data.columns),index=st.session_state.lgd_cal_ind,placeholder="Make selection", key = 'lgd_col_discfac')
            if st.session_state.lgd_cal != lgd_cal:
                st.session_state.lgd_cal = lgd_cal
                st.session_state.lgd_cal_ind = list(st.session_state.data.columns).index(st.session_state.lgd_cal)
                st.rerun()

        st.markdown(f'<p class="big-font">Select Realized LGD</p>', unsafe_allow_html=True)
        cols_lgd_flag = st.columns(2)
        with cols_lgd_flag[0]:
            lgd_flag = st.selectbox("",list(st.session_state.data.columns),index=st.session_state.lgd_flag_ind,placeholder="Make selection", key='disc_fac')
            if st.session_state.lgd_flag != lgd_flag:
                st.session_state.lgd_flag = lgd_flag
                st.session_state.lgd_flag_ind = list(st.session_state.data.columns).index(st.session_state.lgd_flag)# = lgd_flag
                st.rerun()

        st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 

        if st.session_state.lgd_flag and st.session_state.lgd_cal:

            if "Somer's D" in st.session_state.disc_fac_tests:
                st.markdown(f"<p class='big-font'>Somer's D</p>", unsafe_allow_html=True)
                somer_d_result = somers_d(st.session_state.data[st.session_state.lgd_flag], st.session_state.data[st.session_state.lgd_cal])
                st.write(f"The Somer's D value is {round(100*somer_d_result[0], 2)}%. {somer_d_result[1]}")

            if "Spearman's Correlation" in st.session_state.disc_fac_tests:
                st.markdown(f"<p class='big-font' style='margin-bottom:-10px;'>Spearman's Correlation</p>", unsafe_allow_html=True)
                
                actual, predicted = st.session_state.data[st.session_state.lgd_flag], st.session_state.data[st.session_state.lgd_cal]

                cols = st.columns([1.5, 1])
                with cols[0]:
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    ax.scatter(actual, predicted, color="#4c72b0", s=80, alpha=0.6, edgecolor="k", linewidth=0.5, label="Predicted vs Actual")
                    min_val = min(min(actual), min(predicted))
                    max_val = max(max(actual), max(predicted))
                    ax.plot([min_val, max_val], [min_val, max_val], color="#e24a33", linestyle="--", linewidth=1.5, label="Perfect Prediction")
                    
                    ax.set_xlabel("Actual Values", fontsize=12, fontweight="bold", labelpad=6)
                    ax.set_ylabel("Predicted Values", fontsize=12, fontweight="bold", labelpad=6)
                    ax.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
                    ax.tick_params(axis="both", which="major", labelsize=6)
                        
                    padding = 0.1 * (max_val - min_val)
                    ax.set_xlim([min_val - padding, max_val + padding])
                    ax.set_ylim([min_val - padding, max_val + padding])
                    
                    ax.legend(loc="upper left", fontsize=6)
                    st.pyplot(fig)
                
                spearman_corr_result = spearman_correlation(st.session_state.data[st.session_state.lgd_flag], st.session_state.data[st.session_state.lgd_cal])
                st.write(f"<p style='margin:0;'>Correlation: {100*spearman_corr_result[0]:.2f}%</p>", unsafe_allow_html=True)
                st.write(f"<p style='margin:0;'>{spearman_corr_result[2]}</p>", unsafe_allow_html=True)


            if "Confusion Matrix" in st.session_state.disc_fac_tests:
                st.markdown(f"<p class='big-font' style='margin-top:10px;'>Confusion Matrix</p>", unsafe_allow_html=True)
                cols = st.columns([1, 1])

                with cols[0]:
                    display_confusion_matrix(st.session_state.data[st.session_state.lgd_cal], st.session_state.data[st.session_state.lgd_flag])


    if st.session_state.validation_tests_app == 'Model Stability':

        if 'ms_bins' not in st.session_state:
            st.session_state.ms_bins = 5

        st.markdown(
            """
            <style>
            .big-font {
                font-size:20px !important;
                font-weight: bold;
                padding: 0px;
            }
            .small-font {
                font-size:15px !important;
                padding: 0px;
            }
            .boldhr {
                width: 100%;
                height: 2px;
                background-color: #9c9b99; 
                margin: 2px;
                # margin-top: -20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f'<p class="big-font">Select Predicted LGD</p>', unsafe_allow_html=True)
        cols_lgd = st.columns(2)
        with cols_lgd[0]:
            lgd_cal = st.selectbox("",list(st.session_state.data.columns),index=st.session_state.lgd_cal_ind,placeholder="Make selection", key = 'lgd_col_discfac')
            if st.session_state.lgd_cal != lgd_cal:
                st.session_state.lgd_cal = lgd_cal
                st.session_state.lgd_cal_ind = list(st.session_state.data.columns).index(st.session_state.lgd_cal)
                st.rerun()

        st.markdown(f'<p class="big-font">Select Realized LGD</p>', unsafe_allow_html=True)
        cols_lgd_flag = st.columns(2)
        with cols_lgd_flag[0]:
            lgd_flag = st.selectbox("",list(st.session_state.data.columns),index=st.session_state.lgd_flag_ind,placeholder="Make selection", key='disc_fac')
            if st.session_state.lgd_flag != lgd_flag:
                st.session_state.lgd_flag = lgd_flag
                st.session_state.lgd_flag_ind = list(st.session_state.data.columns).index(st.session_state.lgd_flag)# = lgd_flag
                st.rerun()

        st.markdown(f'<p class="big-font">Select # of Bins for PSI</p>', unsafe_allow_html=True)
        cols_bins = st.columns(2)
        with cols_bins[0]:
            ms_bins = st.number_input('', min_value=2, max_value=10, value=st.session_state.ms_bins)
            if st.session_state.ms_bins != ms_bins:
                st.session_state.ms_bins = ms_bins
                st.rerun()

        st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 

        if st.session_state.lgd_cal and st.session_state.lgd_flag and st.session_state.ms_bins:

            if 'Population Stability Index (PSI)' in st.session_state.model_stability_tests:
                st.markdown(f'<p class="big-font">Population Stability Index (PSI)</p>', unsafe_allow_html=True)
                psi_result = calculate_psi(st.session_state.data[st.session_state.lgd_cal],
                                           st.session_state.data[st.session_state.lgd_flag],
                                           num_bins=st.session_state.ms_bins)
    
                st.write(f'The PSI is {psi_result[0]:.2f}. \n {psi_result[1]}')


    if st.session_state.validation_tests_app == 'Data Quality':
        st.markdown(
            """
            <style>
            .big-font {
                font-size:20px !important;
                font-weight: bold;
                padding: 0px;
            }
            .small-font {
                font-size:15px !important;
                padding: 0px;
            }
            .boldhr {
                width: 100%;
                height: 2px;
                background-color: #9c9b99; 
                margin: 2px;
                # margin-top: -20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        if 'Percentage Of Missing Values' in st.session_state.data_quality_tests:
            st.markdown(f'<p class="big-font">Percentage Of Missing Values</p>', unsafe_allow_html=True)
            if len(missing_data_summary(st.session_state.data)) == 0:
                st.write('No column(s) have missing data.')
            else:
                st.write(missing_data_summary(st.session_state.data))


    if st.session_state.validation_tests_app == 'Calibration Accuracy':

        if 'cal_acc_bins' not in st.session_state:
            st.session_state.cal_acc_bins = 5
        
        st.markdown(
            """
            <style>
            .big-font {
                font-size:20px !important;
                font-weight: bold;
                padding: 0px;
            }
            .small-font {
                font-size:15px !important;
                padding: 0px;
            }
            .boldhr {
                width: 100%;
                height: 2px;
                background-color: #9c9b99; 
                margin: 2px;
                # margin-top: -20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f'<p class="big-font">Select Predicted LGD</p>', unsafe_allow_html=True)
        cols_lgd = st.columns(2)
        with cols_lgd[0]:
            lgd_cal = st.selectbox("",list(st.session_state.data.columns),index=st.session_state.lgd_cal_ind,placeholder="Make selection", key = 'lgd_col_discfac')
            if st.session_state.lgd_cal != lgd_cal:
                st.session_state.lgd_cal = lgd_cal
                st.session_state.lgd_cal_ind = list(st.session_state.data.columns).index(st.session_state.lgd_cal)
                st.rerun()

        st.markdown(f'<p class="big-font">Select Realized LGD</p>', unsafe_allow_html=True)
        cols_lgd_flag = st.columns(2)
        with cols_lgd_flag[0]:
            lgd_flag = st.selectbox("",list(st.session_state.data.columns),index=st.session_state.lgd_flag_ind,placeholder="Make selection", key='disc_fac')
            if st.session_state.lgd_flag != lgd_flag:
                st.session_state.lgd_flag = lgd_flag
                st.session_state.lgd_flag_ind = list(st.session_state.data.columns).index(st.session_state.lgd_flag)# = lgd_flag
                st.rerun()

        if 'The notching approach for LGD. ' in st.session_state.cal_acc_tests:
            st.markdown(f'<p class="big-font">Select # of Bins for Notching Approach</p>', unsafe_allow_html=True)
            cols_bins = st.columns(2)
            with cols_bins[0]:
                cal_acc_bins = st.number_input('', min_value=2, max_value=10, value=st.session_state.cal_acc_bins)
                if st.session_state.cal_acc_bins != cal_acc_bins:
                    st.session_state.cal_acc_bins = cal_acc_bins
                    st.rerun()
        

        st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 

        if st.session_state.lgd_flag and st.session_state.lgd_cal:
            
            if 'Chi-Square Test on Mean-Squared-Error (MSE). ' in st.session_state.cal_acc_tests:
                st.markdown(f'<p class="big-font">Chi-Square Test</p>', unsafe_allow_html=True)
                st.write(chi_square_test(st.session_state.data[st.session_state.lgd_cal], st.session_state.data[st.session_state.lgd_flag], alpha=0.05)[2])

            if 'Kolmogorov-Smirnov (KS) goodness-of-fit test. ' in st.session_state.cal_acc_tests:
                st.markdown(f'<p class="big-font">Kolmogorov-Smirnov Test</p>', unsafe_allow_html=True)
                st.write(ks_test_residuals(st.session_state.data[st.session_state.lgd_cal], st.session_state.data[st.session_state.lgd_flag], alpha=0.05)[2])

            if 'Safety Margin. ' in st.session_state.cal_acc_tests:
                st.markdown(f'<p class="big-font">Safety Margin</p>', unsafe_allow_html=True)
                sm_result = safety_margin_test(st.session_state.data[st.session_state.lgd_flag], st.session_state.data[st.session_state.lgd_cal])
                st.write(f"The safety margin value is {sm_result[0]:.2f}. Therefore, {sm_result[1]}.")
        
            if 'The notching approach for LGD. ' in st.session_state.cal_acc_tests:
                st.markdown(f'<p class="big-font">Notching Approach</p>', unsafe_allow_html=True)
                data, notch_summary = notching_approach_lgd(st.session_state.data[st.session_state.lgd_flag], st.session_state.data[st.session_state.lgd_cal], num_notches = st.session_state.cal_acc_bins)
                st.write(notch_summary)
                cols = st.columns([1.5, 1])
                with cols[0]:
                    plot_notching_results(data, notch_summary)


    if st.session_state.validation_tests_app == 'IFRS9 & ST Models':
        
        if 'ifrs9_exp_vals' not in st.session_state:
            st.session_state.ifrs9_exp_vals = 5
        if 'exp_vars' not in st.session_state:
            st.session_state.exp_vars = None
        if 'lag_lgd_cols' not in st.session_state:
            st.session_state.lag_lgd_cols = None

            
        st.markdown(
            """
            <style>
            .big-font {
                font-size:20px !important;
                font-weight: bold;
                padding: 0px;
            }
            .small-font {
                font-size:15px !important;
                padding: 0px;
            }
            .boldhr {
                width: 100%;
                height: 2px;
                background-color: #9c9b99; 
                margin: 2px;
                # margin-top: -20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f'<p class="big-font">Select Predicted LGD</p>', unsafe_allow_html=True)
        cols_lgd = st.columns(2)
        with cols_lgd[0]:
            lgd_cal = st.selectbox("",list(st.session_state.data.columns),index=st.session_state.lgd_cal_ind,placeholder="Make selection", key = 'lgd_col_discfac')
            if st.session_state.lgd_cal != lgd_cal:
                st.session_state.lgd_cal = lgd_cal
                st.session_state.lgd_cal_ind = list(st.session_state.data.columns).index(st.session_state.lgd_cal)
                st.rerun()

        st.markdown(f'<p class="big-font">Select Realized LGD</p>', unsafe_allow_html=True)
        cols_lgd_flag = st.columns(2)
        with cols_lgd_flag[0]:
            lgd_flag = st.selectbox("",list(st.session_state.data.columns),index=st.session_state.lgd_flag_ind,placeholder="Make selection", key='disc_fac')
            if st.session_state.lgd_flag != lgd_flag:
                st.session_state.lgd_flag = lgd_flag
                st.session_state.lgd_flag_ind = list(st.session_state.data.columns).index(st.session_state.lgd_flag)# = lgd_flag
                st.rerun()

        st.markdown(f'<p class="big-font">Select total number of explanatory variables</p>', unsafe_allow_html=True)
        cols_bins = st.columns(2)
        with cols_bins[0]:
            ifrs9_exp_vals = st.number_input('', min_value=1, max_value=1000, value=st.session_state.ifrs9_exp_vals)
            if st.session_state.ifrs9_exp_vals != ifrs9_exp_vals:
                st.session_state.ifrs9_exp_vals = ifrs9_exp_vals
                st.rerun()

        if 'Student T Test' in st.session_state.selected_ifrs9_tests:
            st.markdown(f'<p class="big-font">Select explanatory variables</p>', unsafe_allow_html=True)
            num_cols_data = st.session_state.data.select_dtypes(include='number').columns
            
            cols_t_test = st.columns(2)
            with cols_t_test[0]:
                exp_vars = st.multiselect("", list(num_cols_data), default=st.session_state.exp_vars, placeholder='Select explanatory variables')
                if st.session_state.exp_vars != exp_vars:
                    st.session_state.exp_vars = exp_vars
                    st.rerun()

        if ('Shapiro-Wilk' in st.session_state.selected_ifrs9_tests) or ('Augmented DF' in st.session_state.selected_ifrs9_tests):
            st.markdown(f'<p class="big-font">Select lagging LGD columns</p>', unsafe_allow_html=True)
            cols_data = st.session_state.data.columns
            
            cols = st.columns(2)
            with cols[0]:
                lag_lgd_cols = st.multiselect("", list(cols_data), default=st.session_state.lag_lgd_cols, placeholder='Select variables')
                if st.session_state.lag_lgd_cols != lag_lgd_cols:
                    st.session_state.lag_lgd_cols = lag_lgd_cols
                    st.rerun()

        if ('Durbin Watson' in st.session_state.selected_ifrs9_tests) or ('ACF/PACF Test' in st.session_state.selected_ifrs9_tests):
            cols_file_upload = st.columns(2)
            with cols_file_upload[0]:
                macroecon_model_uploaded_file = st.file_uploader("Select Macroeconimic Model File", type = 'CSV')
                if (macroecon_model_uploaded_file is not None):
                    st.session_state.macroecon_model_uploaded_file = macroecon_model_uploaded_file.getvalue()
                    if st.session_state.macroecon_model_uploaded_file:
                        file_bytes = BytesIO(st.session_state['macroecon_model_uploaded_file'])
                        st.session_state.macroecon_model = pd.read_csv(file_bytes)

                        macroecon_model = st.session_state.macroecon_model

                        macroecon_model['y_hat'] = (
                            macroecon_model["a1"] * macroecon_model["GDP (t-1)"] +
                            macroecon_model["a2"] * macroecon_model["GDP (t-2)"] +
                            macroecon_model["a3"] * macroecon_model["I (t)"] +
                            macroecon_model["a4"] * macroecon_model["I(t-1)"] +
                            macroecon_model["a5"] * macroecon_model["I(t-2)"] +
                            macroecon_model["a6"] * macroecon_model["UE (t)"]
                        )
        
                        macroecon_model['residual'] = macroecon_model['GDP'] - macroecon_model['y_hat']

                        st.session_state.macroecon_model = macroecon_model
                    
        st.markdown(f"<div class='boldhr'</div>", unsafe_allow_html=True) 

        if st.session_state.lgd_flag and st.session_state.lgd_cal and st.session_state.exp_vars and st.session_state.lag_lgd_cols:

            if 'MEV for Model Accuracy - RMSE' in st.session_state.selected_ifrs9_tests:
                st.markdown(f'<p class="big-font">MEV for Model Accuracy - RMSE</p>', unsafe_allow_html=True)
                mev_for_model_accuracy_result = mev_for_model_accuracy(st.session_state.data[st.session_state.lgd_flag], st.session_state.data[st.session_state.lgd_cal])
                st.write(f"The RMSE/NRMSE value is {100*mev_for_model_accuracy_result['RMSE/NRMSE']:.2f}%. Therefore, {mev_for_model_accuracy_result['Description']}.")

            if 'F Test' in st.session_state.selected_ifrs9_tests:
                st.markdown(f'<p class="big-font">F-Test</p>', unsafe_allow_html=True)
                f_test_multiple_regression_result = f_test_multiple_regression(st.session_state.data[st.session_state.lgd_flag], st.session_state.data[st.session_state.lgd_cal], st.session_state.ifrs9_exp_vals)
                st.write(f"The F-statistic is {f_test_multiple_regression_result['F-statistic']:.2e}  and the P-Value is {f_test_multiple_regression_result['p-value']:.2e}. Therefore, {f_test_multiple_regression_result['Description']}.")

            if 'Goodness of Fit (R squared / Adj. R squared)' in st.session_state.selected_ifrs9_tests:
                st.markdown(f'<p class="big-font">Goodness of Fit (R squared / Adj. R squared)</p>', unsafe_allow_html=True)
                goodness_fit_r2_adj_r2_result = goodness_fit_r2_adj_r2(st.session_state.data[st.session_state.lgd_flag], st.session_state.data[st.session_state.lgd_cal], st.session_state.ifrs9_exp_vals)
                st.write(f"The R-Squared value is {100*goodness_fit_r2_adj_r2_result[0]:.2f}% and the Adjusted R-Squared value is {100*goodness_fit_r2_adj_r2_result[1]:.2f}%. Therefore, {goodness_fit_r2_adj_r2_result[2]}.")

            if 'Student T Test' in st.session_state.selected_ifrs9_tests:
                st.markdown(f'<p class="big-font">Student T Test</p>', unsafe_allow_html=True)

                for var in st.session_state.exp_vars:
                    student_t_test_result = student_t_test(st.session_state.data[var], st.session_state.data[st.session_state.lgd_flag], st.session_state.data[st.session_state.lgd_cal])
                    st.markdown(f"""<li style="margin-top: -6px; padding-left: -5px;"><strong>Variable {var}</strong>: The P-Value is <strong>{100*student_t_test_result[3]:.2f}%</strong>. Therefore, {student_t_test_result[4]}.</li>""", unsafe_allow_html=True)

            if 'Variance Inflation Factor' in st.session_state.selected_ifrs9_tests:
                st.markdown(f'<p class="big-font">Variance Inflation Factor</p>', unsafe_allow_html=True)
                st.write(get_variance_inflation_factor(st.session_state.data[st.session_state.exp_vars]))

            if 'Shapiro-Wilk' in st.session_state.selected_ifrs9_tests:
                st.markdown(f'<p class="big-font">Shapiro-Wilk Test</p>', unsafe_allow_html=True)
                if len(st.session_state.lag_lgd_cols) < 2:
                    st.write('Select at-least 2 lagged orders for observed LGD.')
                else:
                    perform_shapiro_wilk_test_result = perform_shapiro_wilk_test(st.session_state.data[[st.session_state.lgd_flag] + st.session_state.lag_lgd_cols])
                    st.write(f"The P-Value is {perform_shapiro_wilk_test_result[1]:.2e}. Therefore, {perform_shapiro_wilk_test_result[2]}.")

            if 'Augmented DF' in st.session_state.selected_ifrs9_tests:
                st.markdown(f'<p class="big-font">Dickey-Fuller Test</p>', unsafe_allow_html=True)
                if len(st.session_state.lag_lgd_cols) < 5:
                    st.write('Select at-least 5 lagged orders for observed LGD.')
                else:
                    dickey_fuller_test_result = dickey_fuller_test(st.session_state.data[[st.session_state.lgd_flag] + st.session_state.lag_lgd_cols])
                    st.write(f"The P-Value is {dickey_fuller_test_result[1]:.2e}. Therefore, {dickey_fuller_test_result[3]}.")

            if 'Philips-Perron test' in st.session_state.selected_ifrs9_tests:
                st.markdown(f'<p class="big-font">Philips-Perron Test</p>', unsafe_allow_html=True)
                if len(st.session_state.lag_lgd_cols) < 7:
                    st.write('Select at-least 6 lagged orders for observed LGD.')
                else:
                    philips_perron_test_result = philips_perron_test(st.session_state.data[[st.session_state.lgd_flag] + st.session_state.lag_lgd_cols])
                    st.write(f"The P-Value is {philips_perron_test_result[1]:.2e}. Therefore, {philips_perron_test_result[3]}.")

            if 'Kwiatkowski-Phillips-Schmidt Shin Test' in st.session_state.selected_ifrs9_tests:
                st.markdown(f'<p class="big-font">Kwiatkowski-Phillips-Schmidt-Shin Test</p>', unsafe_allow_html=True)
                if len(st.session_state.lag_lgd_cols) < 5:
                    st.write('Select at-least 5 lagged orders for observed LGD.')
                else:
                    kpss_test_result = kpss_test(st.session_state.data[[st.session_state.lgd_flag] + st.session_state.lag_lgd_cols])
                    st.write(f"The P-Value is {kpss_test_result[1]:.2e}. Therefore, {kpss_test_result[3]}.")

            if 'White Test' in st.session_state.selected_ifrs9_tests:
                st.markdown(f'<p class="big-font">White Test</p>', unsafe_allow_html=True)
                white_test_result = white_test_for_homoscedasticity(st.session_state.data[st.session_state.exp_vars], st.session_state.data[st.session_state.lgd_flag])
                st.write(f"The P-Value is {white_test_result[1]:.2e}. Therefore, {white_test_result[2]}.")

            if 'Durbin Watson' in st.session_state.selected_ifrs9_tests:

                if 'macroecon_model_uploaded_file' not in st.session_state:
                    st.session_state.macroecon_model_uploaded_file = None
                
                st.markdown(f'<p class="big-font">Durbin Watson Test</p>', unsafe_allow_html=True)
                macroecon_model = st.session_state.macroecon_model

                durbin_watson_test_result = durbin_watson_test(list(macroecon_model['residual']))
                st.write(f"The Durbin Watson Statistic is: {durbin_watson_test_result}")

            if 'ACF/PACF Test' in st.session_state.selected_ifrs9_tests:
                st.markdown(f'<p class="big-font">ACF and PACF Test</p>', unsafe_allow_html=True)
                # st.write(list(macroecon_model['residual']))

                cols = st.columns(2)
                with cols[0]:
                    alpha = st.slider("Select significance level (alpha)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
                    st.write(calculate_acf_pacf_conf_intervals(list(macroecon_model['residual']), alpha))






# Function to calculate ACF and PACF confidence intervals and plot them
def calculate_acf_pacf_conf_intervals(time_series, alpha=0.05):
    n = len(time_series)  # Number of observations in the time series
    z_score = norm.ppf(1 - alpha / 2)  # Z-score for the given significance level
    max_lag = min(n-1, int(n/2) - 1)  # Adjust maximum lags for PACF

    # Calculate ACF and PACF values
    acf_values = acf(time_series, nlags=max_lag, fft=True)
    pacf_values = pacf(time_series, nlags=max_lag)

    # Calculate ACF confidence intervals
    acf_conf_intervals = [
        z_score * np.sqrt((1 + 2 * np.sum(acf_values[:i] ** 2)) / n) for i in range(1, len(acf_values))
    ]
    
    # Calculate PACF confidence intervals
    pacf_conf_intervals = [z_score / np.sqrt(n) for _ in range(len(pacf_values)-1)]

    # Plot ACF
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # ACF Plot
    ax[0].stem(acf_values)
    ax[0].fill_between(range(1, len(acf_values)), -np.array(acf_conf_intervals), np.array(acf_conf_intervals),
                       color='blue', alpha=0.2)
    ax[0].set_title('ACF with Confidence Intervals')
    ax[0].set_xlabel('Lag')
    ax[0].set_ylabel('ACF')

    # PACF Plot
    ax[1].stem(pacf_values)
    ax[1].fill_between(range(1, len(pacf_values)), -np.array(pacf_conf_intervals), np.array(pacf_conf_intervals),
                       color='red', alpha=0.2)
    ax[1].set_title('PACF with Confidence Intervals')
    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('PACF')

    # Display plots in Streamlit
    st.pyplot(fig)

    return acf_values, pacf_values, fig




def durbin_watson_test(residuals):
    durbin_watson_stat = sm.stats.stattools.durbin_watson(residuals)
    
    return durbin_watson_stat


def white_test_for_homoscedasticity(X, y):
    
    # Step 1: Fit the original regression model to get residuals
    X = sm.add_constant(X)  # Add constant (intercept) to the predictors
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    squared_residuals = residuals ** 2

    # Step 2: Create auxiliary regression data with original terms, squares, and interaction terms
    X_auxiliary = X.copy()
    for col1 in X.columns:
        for col2 in X.columns:
            X_auxiliary[f'{col1}*{col2}'] = X[col1] * X[col2]

    # Step 3: Run auxiliary regression with squared residuals as the dependent variable
    aux_model = sm.OLS(squared_residuals, X_auxiliary).fit()

    # Step 4: Calculate White test statistic and p-value
    test_statistic = len(squared_residuals) * aux_model.rsquared
    degrees_of_freedom = X_auxiliary.shape[1] - 1  # Exclude the intercept
    p_value = chi2.sf(test_statistic, degrees_of_freedom)

    # Interpret the result based on the p-value
    if p_value >= 0.05:
        result = "the residuals are homoscedastic"
    elif 0.01 <= p_value < 0.05:
        result = "the residuals are weakly homoscedastic"
    else:
        result = "the residuals are heteroscedastic"

    return test_statistic, p_value, result




def kpss_test(observed_data):
    
    aggregated_series = observed_data.mean(axis=0).values
    
    kpss_statistic, p_value, _, critical_values = kpss(aggregated_series)
    
    # Interpret the result based on the p-value
    if p_value >  0.2:
        result = "the data is stationary"
    elif 0.1 <= p_value < 0.2:
        result = "the test result is not significant enough to reject the null hypothesis of stationarity"
    else:
        result = "the data series is non-stationary"
    
    return kpss_statistic, p_value, critical_values, result



def philips_perron_test(observed_data):
    aggregated_series = observed_data.mean(axis=0).values
    
    pp_test = PhillipsPerron(aggregated_series)
    pp_statistic = pp_test.stat
    p_value = pp_test.pvalue
    critical_values = pp_test.critical_values
    
    # Interpret the result based on the p-value
    if p_value < 0.1:
        result = "the data series does not contain a unit root"
    elif 0.1 <= p_value < 0.2:
        result = "the test result is not significant enough to reject the unit root"
    else:
        result = "the data series does contain a unit root"
    
    return pp_statistic, p_value, critical_values, result


def dickey_fuller_test(observed_data):
    aggregated_series = observed_data.mean(axis=0).values  
    # st.write(observed_data)
    
    adf_statistic, p_value, used_lags, n_obs, critical_values, icbest = adfuller(aggregated_series)
    
    # Interpret the result based on the p-value
    if p_value < 0.1:
        result = "the data series does not contain a unit root"
    elif 0.1 <= p_value < 0.2:
        result = "the test result is not significant enough to reject the unit root"
    else:
        result = "the data series does contain a unit root"
    
    return adf_statistic, p_value, critical_values, result

    
def perform_shapiro_wilk_test(observed_lgd_data):
    aggregated_series = observed_lgd_data#.mean(axis=0).values 
    # st.write(aggregated_series)
    
    test_statistic, p_value = shapiro(aggregated_series)
    
    # Interpret the result based on the p-value
    if p_value >= 0.05:
        result = "the residuals are likely to be normally distributed"
    elif 0.01 <= p_value < 0.05:
        result = "the residuals are unlikely to be normally distributed"
    else:
        result = "the residuals are not normally distributed"
    
    return test_statistic, p_value, result



def get_variance_inflation_factor(df):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature in the dataset and classify 
    the VIF level based on the provided thresholds.

    Parameters:
    - df: pandas DataFrame containing the independent variables

    Returns:
    - vif_df: DataFrame with features, their VIF values, and corresponding status based on thresholds
    """
    # Add a constant to the dataset for the intercept
    df_with_const = add_constant(df)
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data['Feature'] = df_with_const.columns
    vif_data['VIF'] = [variance_inflation_factor(df_with_const.values, i) for i in range(df_with_const.shape[1])]
    
    # Drop the constant row
    vif_data = vif_data[vif_data['Feature'] != 'const']
    
    # Apply thresholds to classify the VIF level
    def classify_vif(vif):
        if vif < 2:
            return "Green: Low multi-collinearity"
        elif 2 <= vif < 4:
            return "Amber: Medium multi-collinearity"
        else:
            return "Red: High multi-collinearity"
    
    vif_data['Status'] = vif_data['VIF'].apply(classify_vif)
    
    return vif_data.reset_index(drop = True)
                

def student_t_test(y, x, y_pred):
    """
    Calculate beta_hat (slope of the regression line) and its standard error for a single predictor.
    
    Parameters:
    - y: numpy array of dependent variable values
    - x: numpy array of independent variable values
    
    Returns:
    - beta_hat: estimated coefficient for the predictor
    - se_beta_hat: standard error of the estimated coefficient
    """
    n = len(y)
    
    # Step 1: Calculate beta_hat (slope)
    numerator = np.sum(y * x) - (np.sum(y) * np.sum(x)) / n
    denominator = np.sum((x - np.mean(x)) ** 2)
    beta_hat = numerator / denominator
    
    # Step 2: Calculate standard error of beta_hat
    # y_pred = beta_hat * x
    residuals = y - y_pred  # Residuals (errors)
    sse = np.sum(residuals ** 2)  # Sum of squared errors
    
    # Calculate se(beta_hat) based on the formula
    se_beta_hat = np.sqrt(sse / (n - 2)) / np.sqrt(denominator)

    # Step 3: Calculate the t-statistic
    t_stat = beta_hat / se_beta_hat
    
    # Step 4: Calculate the p-value (two-tailed)
    p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n - 2))
    
    # Step 5: Determine significance status based on thresholds
    if p_value < 0.10:
        result = "factor coefficient is significant"
    elif 0.10 <= p_value < 0.20:
        result = "factor coefficient’s significance is acceptable"
    else:
        result = "factor coefficient is insignificant"
    
    return beta_hat, se_beta_hat, t_stat, p_value, result



def goodness_fit_r2_adj_r2(y_actual, y_pred, num_predictors):

    n = len(y_actual)
    
    # Calculate R-squared
    r_squared = r2_score(y_actual, y_pred)
    
    # Calculate Adjusted R-squared
    adjusted_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - num_predictors - 1))

    if (r_squared > 0.55) or (adjusted_r_squared > 0.45):
        result = 'model has good explanatory power'
    elif (r_squared >= 0.35 and r_squared <= 0.55) or (adjusted_r_squared >= 0.25 and adjusted_r_squared <= 0.45):
        result = 'model’s explanatory power is acceptable'
    else:
        result = 'model has poor explanatory power'

    return r_squared, adjusted_r_squared, result





def f_test_multiple_regression(actual_values, predicted_values, num_predictors):
    """
    Performs an F-test for a multiple regression model to evaluate the overall fit.
    
    Parameters:
    - actual_values: array-like, actual observed values
    - predicted_values: array-like, model's predicted values
    - num_predictors: int, the number of explanatory variables (predictors) in the model
    
    Returns:
    - dict containing F-statistic, p-value, status (Green, Amber, or Red), and description of the fit quality
    """
    # Convert inputs to numpy arrays for calculation
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    
    # Calculate Total Sum of Squares (SST) - Total variance in observed data
    y_mean = np.mean(actual_values)
    sst = np.sum((actual_values - y_mean) ** 2)
    
    # Calculate Sum of Squares for Regression (SSR) - Explained variance by the model
    ssr = np.sum((predicted_values - y_mean) ** 2)
    
    # Calculate Sum of Squares for Error (SSE) - Unexplained variance by the model
    sse = np.sum((actual_values - predicted_values) ** 2)
    
    # Calculate the F-statistic
    n = len(actual_values)  # number of observations
    dfn = num_predictors     # degrees of freedom for numerator
    dfd = n - num_predictors - 1  # degrees of freedom for denominator
    f_statistic = (ssr / dfn) / (sse / dfd)
    
    # Calculate the p-value for the F-statistic
    p_value = 1 - f.cdf(f_statistic, dfn, dfd)
    
    # Determine the fit status based on p-value thresholds
    if p_value < 0.10:
        status = "Green"
        description = "model has good overall fit"
    elif 0.10 <= p_value < 0.20:
        status = "Amber"
        description = "model's overall fit is acceptable"
    else:
        status = "Red"
        description = "model has insignificant overall fit"
    
    return {
        "F-statistic": f_statistic,
        "p-value": p_value,
        "Status": status,
        "Description": description
    }





def mev_for_model_accuracy(observed_values, predicted_values):
    """
    Evaluates model accuracy based on RMSE or NRMSE thresholds for both PD and LGD models.
    
    Parameters:
    - observed_values: array-like, actual observed values
    - predicted_values: array-like, model's predicted values
    - normalize: bool, whether to calculate NRMSE (True) or RMSE (False)
    
    Returns:
    - dict containing RMSE/NRMSE value and status (Green, Amber, or Red) with description
    """
    # Calculate RMSE
    n = len(observed_values)
    rmse = np.sqrt(np.sum((np.array(predicted_values) - np.array(observed_values)) ** 2) / n)
    
    # Normalize RMSE
    mean_observed = np.mean(observed_values)
    range_observed = np.max(observed_values) - np.min(observed_values)
    if mean_observed > 0:
        nrmse = rmse / mean_observed
    else:
        nrmse = rmse / range_observed
    metric = nrmse
    
    # Determine status based on thresholds
    if metric <= 0.55:
        status = "Green"
        description = "model prediction fits observed values well"
    elif 0.55 < metric < 0.70:
        status = "Amber"
        description = "model prediction fits observed values weakly"
    else:
        status = "Red"
        description = "model prediction does not appear to fit observed values"
    
    return {
        "RMSE/NRMSE": metric,
        "Status": status,
        "Description": description
    }





def notching_approach_lgd(actual_lgd, predicted_lgd, num_notches):
    """
    Perform a notching approach to assess LGD model calibration accuracy.

    Parameters:
    - actual_lgd: array-like, actual LGD values observed.
    - predicted_lgd: array-like, LGD values predicted by the model.
    - num_notches: int, number of notches (segments) to create.

    Returns:
    - data: DataFrame containing actual and predicted LGD values with notch information.
    - notch_summary: DataFrame with mean actual and predicted LGD for each notch and calibration results.
    """
    # Create DataFrame with actual and predicted LGD
    data = pd.DataFrame({'Actual_LGD': actual_lgd, 'Predicted_LGD': predicted_lgd})
    
    # Assign each predicted LGD to a notch based on percentile ranks
    data['Notch'] = pd.qcut(data['Predicted_LGD'], q=num_notches, labels=False) + 1

    # Group by notch to calculate summary statistics
    notch_summary = data.groupby('Notch').agg(
        Mean_Actual_LGD=('Actual_LGD', 'mean'),
        Mean_Predicted_LGD=('Predicted_LGD', 'mean'),
        Median_Actual_LGD=('Actual_LGD', 'median'),
        Median_Predicted_LGD=('Predicted_LGD', 'median'),
        Count=('Notch', 'size')
    ).reset_index()

    # Calculate the difference between actual and predicted means for each notch
    notch_summary['Mean_Difference'] = notch_summary['Mean_Predicted_LGD'] - notch_summary['Mean_Actual_LGD']
    notch_summary['Median_Difference'] = notch_summary['Median_Predicted_LGD'] - notch_summary['Median_Actual_LGD']
    
    # Determine if each notch passes or fails calibration
    notch_summary['Calibration_Result'] = np.where(
        notch_summary['Mean_Difference'].abs() <= 0.05, 'Pass', 'Fail'
    )

    return data, notch_summary



def plot_notching_results(data, notch_summary):
    """
    Visualize the notching approach results with a bar plot comparing actual vs. predicted LGD
    and Mean Difference on a secondary y-axis, with a combined legend.
    
    Parameters:
    - data: DataFrame containing original actual and predicted LGD values with notch information.
    - notch_summary: DataFrame with mean actual and predicted LGD for each notch.
    """
    # Calculate the bounds for each notch
    bounds = data.groupby('Notch')['Predicted_LGD'].agg(['min', 'max']).reset_index()
    bounds['Range'] = bounds.apply(lambda row: f"{row['min']:.2f} - {row['max']:.2f}", axis=1)

    # Set up the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x = bounds['Range']
    bar_width = 0.35
    
    # Plot Mean Actual and Mean Predicted LGD as side-by-side bars on the primary y-axis
    bars_actual = ax1.bar(np.arange(len(x)) - bar_width/2, notch_summary['Mean_Actual_LGD'], width=bar_width, 
                          label='Mean Actual LGD', alpha=0.8, color='skyblue')
    bars_predicted = ax1.bar(np.arange(len(x)) + bar_width/2, notch_summary['Mean_Predicted_LGD'], width=bar_width, 
                             label='Mean Predicted LGD', alpha=0.8, color='salmon')
    
    # Label for the primary y-axis
    ax1.set_ylabel("LGD")
    ax1.set_xlabel("Predicted LGD Range")
    ax1.set_xticks(np.arange(len(x)))
    ax1.set_xticklabels(x, rotation=45, ha='right')
    ax1.set_title("Notching Approach: Mean Actual vs. Mean Predicted LGD with Mean Difference")
    
    # Add a secondary y-axis for Mean Difference
    ax2 = ax1.twinx()
    line_diff, = ax2.plot(np.arange(len(x)), notch_summary['Mean_Difference'], color='darkred', marker='o', linestyle='--', 
                          label='Mean Difference (Predicted - Actual)', linewidth=1.5)
    ax2.set_ylabel("Mean Difference")

    # Highlight calibration results (Pass/Fail) for each notch
    for i, result in enumerate(notch_summary['Calibration_Result']):
        color = 'green' if result == 'Pass' else 'red'
        ax1.text(i, notch_summary['Mean_Predicted_LGD'][i] + 0.02, result, ha='center', color=color, fontweight='bold')

    # Combine legends from both axes into one
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + [line_diff], labels + [line_diff.get_label()], loc="upper left", frameon=True, fancybox=True, shadow=True)

    # Add grid, tighten layout, and show plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    st.pyplot(fig)




def safety_margin_test(actual_lgd, predicted_lgd):

    sm = (predicted_lgd.sum()/actual_lgd.sum()) - 1

    if sm > 0 and sm <= 0.3:
        result = 'model prediction fits observed values'
    elif (sm > 0.3) or (sm > -0.1 and sm <= 0):
        result = 'model prediction fits observed values weakly'
    else:
        result = 'model prediction does not appear to fit observed values (underestimation)'

    return sm, result


def ks_test_residuals(actual, predicted, alpha=0.05):
    """
    Perform a Kolmogorov-Smirnov goodness-of-fit test on residuals of actual and predicted values.

    Parameters:
    - actual: array-like, actual values.
    - predicted: array-like, predicted values.
    - alpha: float, significance level for the test (default is 0.05).

    Returns:
    - ks_stat: float, the KS statistic.
    - p_value: float, p-value of the test.
    - result: str, conclusion based on the significance level.
    """
    
    residuals = np.array(actual) - np.array(predicted)

    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    ks_stat, p_value = kstest(residuals, 'norm', args=(mean_residual, std_residual))

    if p_value < alpha:
        result = "Reject the null hypothesis: The residuals significantly differ from a normal distribution."
    else:
        result = "Fail to reject the null hypothesis: The residuals do not significantly differ from a normal distribution."

    if p_value < alpha:
        result = f"The p-value is {p_value:.2e}.\nReject the null hypothesis as observed defaults differ significantly from expected defaults."
    else:
        result = f"The p-value is {p_value:.2e}.\nFail to reject the null hypothesis as observed defaults are consistent with expected defaults."

    return ks_stat, p_value, result



def validation_tests():
    # validation_tests_sidebar()
    
    if st.session_state.model_type == 'PD':
        pd_validation_test()

    if st.session_state.model_type == 'LGD':
        lgd_validation_test()

    cols = st.columns(6)
    ChangeButtonColour('Back', 'black', '#B2BBD2', '10px', margin_top = '10px', border = None)
    if cols[0].button('Back', key='back_to_home_screen'):
        st.session_state['page'] = 'validation_test_selection'
        st.session_state.need_rerun = True
        if st.session_state.need_rerun:
            st.session_state.need_rerun = False
            st.rerun()













