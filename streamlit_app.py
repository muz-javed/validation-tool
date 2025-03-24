import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.stylable_container import stylable_container
# from st_aggrid import AgGrid, GridOptionsBuilder

from functions import *
from report_functions import *

initialize_session()
        
if st.session_state['page'] == 'core_upload':
    core_upload()

if st.session_state['page'] == 'core_validation':
    core_validation()

if st.session_state['page'] == 'data_upload':
    data_upload()
    
if st.session_state['page'] == 'validation_test_selection':
    validation_test_selection()

if st.session_state['page'] == 'validation_tests':
    validation_tests()

if st.session_state['page'] == 'gen_rep_page':
    gen_rep_page()

# import pandas as pd

# data = {

#     "Snapshot Date": ["2023-01-31", None],

#     "Borrower ID": [101, 101],

#     "Borrower Name": ["ABC Corp", "12345"],  # suspicious

#     "Legal Form / Ownership Type": ["Public Limited", "Unknown"],

#     "Facility ID": [1001, None],

#     "Facility Start Date": ["2022-01-01", "2022-06-01"],

#     "Facility End Date": ["2022-12-31", "2021-12-31"],  # end < start for 2nd

#     "Collateral ID / Reference": ["C123", None],

#     "Collateral Type": ["Real Estate", "InvalidType"],

#     "Collateral Value (Month-End)": [100000, -100],

#     "Internal Credit Rating": [10, 11],  # 11 out of range

#     "Stage (IFRS 9)": [1, 5],           # 5 invalid

#     "Days Past Due (Month-End)": [0, -10],

#     "Default Flag": [0, 2],  # 2 invalid

#     "Covenant Details": [None, "Debt/EBITDA<3"],

#     "Financial Statement Date": ["2022-12-31", "2023-02-15"],

#     "Total Assets": [1_000_000, 2_000_000],

#     "Total Liabilities": [600_000, 2_100_000],

#     "Total Equity": [400_000, -100_000],

#     "Audit Opinion": ["Unqualified", "Suspicious"],

#     "Return on Assets": [0.05, 2.0],  # 200%

#     "Debt to Equity Ratio": [None, 6.0]

# }

# df = pd.DataFrame(data)

# df.to_csv('Core-Temp.csv')