# ## case 1:
# from ast import Num
# from asyncore import close_all
# from curses import COLS
# from datetime import datetime
# from nis import cat
# from matplotlib.pyplot import text

# from numpy import empty, expand_dims


# case 1: input cols are empty
# required inputs
#     pri, sec, cat cols, num cols, datetime cols
# output
#     pri, sec, cat cols, num cols, datetime cols
# logic
#     check if each l is empty

# case 2: changing mapping fn to dict format
# required inputs:
#     c.mapping
# output:
#     mapping_dict
# logic:
#     nested for loops

# case 3: mapping is empty
# required inputs:
#     c.mapping
# outputs:
#     mapping_dict
# logic:
#     if empty:
#         try to convert each col into diff datatypes => classify col into 1 of 4 types:
#             datetime => pd.to_datetime() or df.astype("datetime64[ns]")
#             cat => df.astype("int64")
#             Num => df.astype("float64")
#             text => df.astype()

from pyparsing import col
import jazz_context
import jazz_context_show
import jazz_context_unshow
import jazz_antiformat
import briefology_ps_to_task_search as briefology_graph
import opera_util_common
import os
import datetime
import pandas as pd

def get_ctx(file):

    C = jazz_context.context
    antiformat = jazz_antiformat.main
    show = jazz_context_show.show
    unshow = jazz_context_unshow.unshow
    c = opera_util_common.parse_ps_to_ctx(file)
    return c

def convert(a):
    return list(dict(a).keys())

def appending_to_cols(l,name_of_l, c):
    if not l:
        if (name_of_l == "pri_cols") or (name_of_l == "sec_cols") or (name_of_l == "cat_cols_to_expand"):
            l = convert(c.mapping.categorical)
        elif (name_of_l == "num_cols"):
            l = convert(c.mapping.numerical)
        elif (name_of_l == "datetime_cols"):
            l = convert(c.mapping.datetime)
        return l
    else:
        return []

# converting the config file format into the dict format
            
def create_dest_folder(df_file, pri_cols, sec_cols, cat_cols_to_expand, num_cols, datetime_cols):
    current_dir = os.getcwd()
    current_datetime = datetime.datetime.now().strftime("%d.%m.%Y_%H%M")
    dest_path = current_dir+'\\'+current_datetime + "_Visualisation_" + df_file
    univariate_path = dest_path + "\\Univariate_Plots"
    bivariate_path = dest_path + "\\Bivariate_Plots"
    if not os.path.exists(dest_path):
        os.makedirs(univariate_path)
        os.mkdir(bivariate_path)
    readme_l = ["File Naming Conventions\nUnivariate Plots:\n[Name of Visual] of [x axis] & [legend]\n\nBivariate Plots:\n[Name of Visual] of [x axis] & [y axis] & [legend]\n\nConfig File Inputs\nPri Cols: ", str(pri_cols), "\nSec Cols: ", str(sec_cols), "\nCat Cols to expand: ", str(cat_cols_to_expand), "\nNumerical Cols:", str(num_cols) + "\nDatetime Cols: " + str(datetime_cols)]
    with open(dest_path + "/README.txt", "w") as f:
        f.writelines(readme_l)
    return dest_path


## Case 3: if mapping dict is empty
def create_mapping_dict(mapping, c, df):
    # if mapping empty:
    # try to convert the df into a certain dtype => whichever can be converted will be sorted into that particular dtype
    # qns is: how to check whether the mapping dict is empty?
    print(df.info())
    for k, v in dict(c.mapping).items():
        keys_l = list(v.keys())
        if keys_l:
            for i in keys_l:
                mapping[i] = k
        else:
            # empty col
            # datetime first
            datetime_col_l = []
            
            for col_name, col_val in df.iteritems():
                print("col name is: " + str(col_name))
                try:
                    df[[col_name]].apply(pd.to_datetime)
    print(df.info())

    
        # print("v: " + str(v))
        # #print(c.mapping.k)
        # print(keys_l)
