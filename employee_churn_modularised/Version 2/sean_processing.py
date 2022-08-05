import itertools
from matplotlib.pyplot import axis
import numpy as np
import opera_util_common
import os
import datetime
import pandas as pd
import argparse
import pathlib
from jazz_antiformat import main


# ----------------------------------------------------
# df parsing functions
# ----------------------------------------------------

def get_ctx(file):
    ctx = opera_util_common.parse_ps_to_ctx(file)
    return ctx

def convert(a):
    """
    Returns elements of [a] in txt config file
    """
    return list(dict(a).keys())

def arg_parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", nargs="?", default="sean_config_qqaltar", type=str, help="name of config file")
    args, unknown_args = parser.parse_known_args(argv)
    ctx = get_ctx(args.c)
    compute_table = convert(ctx.compute_table)[0]
    threshold_nan_pct = convert(ctx.threshold_nan_pct)[0]
    threshold_x_tick_labels_cat = convert(ctx.threshold_x_tick_labels_cat)[0]
    threshold_x_tick_labels_num = convert(ctx.threshold_x_tick_labels_num)[0]
    df_file_path = str(convert(ctx.df_file_path)[0]).replace("/", "\\")
    pri_cols = convert(ctx.pri_cols)
    sec_cols = convert(ctx.sec_cols)
    cat_cols_to_expand = convert(ctx.cat_cols_to_expand)
    num_cols = convert(ctx.num_cols)
    datetime_cols = convert(ctx.datetime_cols)
    text_cols = convert(ctx.text_col)
    return ctx, compute_table, threshold_nan_pct, threshold_x_tick_labels_cat, threshold_x_tick_labels_num, df_file_path, pri_cols, sec_cols, cat_cols_to_expand, num_cols, datetime_cols, text_cols

# ----------------------------------------------------
# mapping dict functions
# ----------------------------------------------------

def check_if_datetime_1(s: pd.Series) -> bool:
    try:
        s = s.astype("datetime64[ns]")
        return True
    except:
        return False

def check_if_datetime_2(s: pd.Series) -> bool:
    pattern = '{}/{}/{}'
    s = s[1]
    l = s.split("/", 3)[-1]
    if main(pattern, s) and (len(l[-1]) > 11):
        return True
    else:
        return False

def check_if_float(s: pd.Series) -> bool:
    try:
        s = s.astype("float")
        return True
    except:
        return False

def check_if_int(s: pd.Series) -> bool:
    try:
        s = s.astype("int")
        return True
    except:
        return False

def check_if_str(s: pd.Series) -> bool:
    try:
        s = s.astype("str")
        return True
    except:
        return False

def check_if_cat(s: pd.Series) -> bool:
    n_unique_vals = s.nunique()
    sqrt_val = np.sqrt(s.shape[0])
    if n_unique_vals < sqrt_val:
        return True
    else:
        return False


def create_mapping_dict(c, df):
    """
    Generation of mapping dict => from input mapping AND/OR df
    """
    d = {}
    for k, v in dict(c.mapping).items():
        keys_l = list(v.keys())
        if keys_l:
            d.update({i: k for i in keys_l})
    for col in df.columns:
        if col not in d:
            if check_if_datetime_1(df[col]):
                d[col] = 'datetime'
                continue

            elif check_if_datetime_2(df[col]):
                d[col] = 'datetime'
                continue
            elif check_if_float(df[col]):
                if check_if_cat(df[col]):
                    d[col] = 'categorical'
                else:
                    d[col] = 'numerical'
                continue
            elif check_if_int(df[col]):
                if check_if_cat(df[col]):
                    d[col] = 'categorical'
                else:
                    d[col] = 'numerical'
                continue
            elif check_if_cat(df[col]): 
                d[col] = 'categorical'
                continue
            else:
                d[col] = 'text'
                continue
    return d

# ----------------------------------------------------
# misc functions
# ----------------------------------------------------

def df_drop(df, threshold: float = 0.2):
    """
    Drop cols with :
        1) >= 20% NaN values
        2) 1 unique value
    """
    df = df.dropna(axis=1, thresh = threshold*len(df))
    for col in df.columns:
        if df[col].nunique() == 1:
            df = df.drop(col, axis=1)
    df = df.astype(str)
    return df

def df_replace(df):
    """
    Remove symbols in col headers + values
    """
    symbols_c = ["?", "%", "#", "$", "^", "<", ">"]
    symbols_v = ["%", "#", "$", "^", "<", ">"]
    for i in symbols_c:
        df.columns = df.columns.str.replace(i,"")

    for j in symbols_v:
        df = df.replace(j, "", regex=True)
    df.replace("-", "/", regex=True)
    return df

def dtype_conversion(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Based on mapping dict => converts df cols to their respective dtype

    len() > 11 => Incoming datetimeformat e.g " 2022 05:10"

    """
    for k, v in mapping.items():
        if v == "numerical":
            df[k] = df[k].astype("float64")
            continue
        elif v == "datetime":
            df[k] = pd.to_datetime(df[k])
            df[k] = df[k].dt.strftime("%d/%m/%Y")
            continue
    return df

def create_dest_folder(df_file):
    current_dir = os.getcwd()
    current_datetime = datetime.datetime.now().strftime("%d.%m.%Y_%H%M")
    dest_path = current_dir+'\\'+current_datetime + "_Visualisation_" + df_file
    univariate_path = dest_path + "\\Univariate_Plots"
    bivariate_path = dest_path + "\\Bivariate_Plots"
    wordcloud_path = dest_path + "\\WordCloud_Plots"
    path_l = [univariate_path, bivariate_path, wordcloud_path]
    for i in path_l:
        pathlib.Path(i).mkdir(parents=True, exist_ok=True)
    return dest_path

def create_readme(dest_path, pri_cols, sec_cols, cat_cols_to_expand, num_cols, datetime_cols, text_cols):
    readme_l = ["File Naming Conventions\nUnivariate Plots:\n[Name of Visual] of [x axis] & [legend]\n\nBivariate Plots:\n[Name of Visual] of [x axis] & [y axis] & [legend]\n\nConfig File Inputs\n\nPri Cols: ", str(pri_cols), "\n\nSec Cols: ", str(sec_cols), "\n\nCat Cols to expand: ", str(cat_cols_to_expand), "\n\nNumerical Cols: ", str(num_cols) + "\n\nDatetime Cols: " + str(datetime_cols), "\n\nText Cols: " +str(text_cols)]
    with open(dest_path + "/README.txt", "w") as f:
        f.writelines(readme_l)

def appending_to_cols(l,name_of_l, mapping):
    """
    Returns list of df cols, classified based on col dtype
    """
    if not l:
        if (name_of_l == "pri_cols") or (name_of_l == "sec_cols") or (name_of_l == "cat_cols_to_expand") and (mapping.get("categorical") != None):
            l = mapping["categorical"]
        elif (name_of_l == "num_cols") and (mapping.get("numerical") != None):
            l = mapping["numerical"]
        elif (name_of_l == "datetime_cols") and (mapping.get("datetime") != None):
            l = mapping["datetime"]
        elif (name_of_l == "text_cols") and (mapping.get("text") != None):
            l = mapping["text"]
        return l
    else:
        return []



def set_x_tick_limit(viz_df: pd.DataFrame, l: list, datatype: str, mapping: dict, threshold: int = 8) -> list:
    """
    Returns minimised list of df cols
    """
    l1 = l[:]
    for i in l:
        if (mapping[i] == datatype) and (viz_df[i].nunique() > threshold):
            l1.remove(i)
            continue
    return l1

def rotate_axis(fig) -> bool:
    """
    Dynamic rotation of x axis labels
    """
    num_labels = len(fig.get_xticklabels())
    if num_labels > 9:
        return True
    else:
        return False

