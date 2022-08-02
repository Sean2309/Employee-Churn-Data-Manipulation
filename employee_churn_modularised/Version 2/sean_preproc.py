from matplotlib.pyplot import axis
import numpy as np
import opera_util_common
import os
import datetime
import pandas as pd

def get_ctx(file):
    c = opera_util_common.parse_ps_to_ctx(file)
    return c

def convert(a):
    return list(dict(a).keys())

def df_replace(df):
    df.columns = df.columns.str.replace("?","")
    df.columns = df.columns.str.replace("%","")
    df.columns = df.columns.str.replace("#","")
    df.columns = df.columns.str.replace("$","")
    df.columns = df.columns.str.replace("%","")
    df.columns = df.columns.str.replace("^","")
    df.columns = df.columns.str.replace("<","")
    df.columns = df.columns.str.replace(">","")

    df = df.replace("/", " ", regex=True)
    df = df.replace(" ", "_", regex=True)
    df = df.replace("<", "", regex=True)
    df = df.replace(">", "", regex=True)
    df = df.replace("%", "", regex=True)
    #df = df.replace("?", "", regex=True)
    df = df.dropna(axis=1, thresh= 0.2 * len(df))
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df = df.drop(col, axis=1)
    df = df.astype(str)
    return df

def appending_to_cols_from_dict(l,name_of_l, mapping):
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

def create_dest_folder(df_file, pri_cols, sec_cols, cat_cols_to_expand, num_cols, datetime_cols, text_cols):
    current_dir = os.getcwd()
    current_datetime = datetime.datetime.now().strftime("%d.%m.%Y_%H%M")
    dest_path = current_dir+'\\'+current_datetime + "_Visualisation_" + df_file
    univariate_path = dest_path + "\\Univariate_Plots"
    bivariate_path = dest_path + "\\Bivariate_Plots"
    wordcloud_path = dest_path + "\\WordCloud_Plots"
    if not os.path.exists(dest_path):
        os.makedirs(univariate_path)
        os.mkdir(bivariate_path)
        os.mkdir(wordcloud_path)
    readme_l = ["File Naming Conventions\nUnivariate Plots:\n[Name of Visual] of [x axis] & [legend]\n\nBivariate Plots:\n[Name of Visual] of [x axis] & [y axis] & [legend]\n\nConfig File Inputs\nPri Cols: ", str(pri_cols), "\nSec Cols: ", str(sec_cols), "\nCat Cols to expand: ", str(cat_cols_to_expand), "\nNumerical Cols: ", str(num_cols) + "\nDatetime Cols: " + str(datetime_cols), "\nText Cols: " +str(text_cols)]
    with open(dest_path + "/README.txt", "w") as f:
        f.writelines(readme_l)
    return dest_path


def check_if_datetime(s: pd.Series) -> bool:
    try:
        s = s.astype("datetime64[ns]")
        return True
    except:
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
    if n_unique_vals < np.sqrt(s.shape[0]):
        return True
    else:
        return False

def create_mapping_dict(c, df):
    d = {}
    for k, v in dict(c.mapping).items():
        keys_l = list(v.keys())
        if keys_l:
            for i in keys_l:
                d[i] = k

        else:
            for col in df.columns:
                if check_if_float(df[col]):
                    if check_if_cat(df[col]):
                        d[col] = 'categorical'
                        df[col] = df[col].astype("str")
                    else:
                        d[col] = 'numerical'
                        df[col] = df[col].astype("float")
                elif check_if_datetime(df[col]):
                    d[col] = 'datetime'
                    df[col] = df[col].astype("datetime64[ns]")
                    continue
                elif check_if_int(df[col]):
                    if check_if_cat(df[col]):
                        d[col] = 'categorical'
                        df[col] = df[col].astype("str")
                    else:
                        d[col] = 'numerical'
                        df[col] = df[col].astype("float")
                    continue
                elif check_if_cat(df[col]): 
                    d[col] = 'categorical'
                    df[col] = df[col].astype("str")
                    continue
                else:
                    d[col] = 'text'
                    df[col] = df[col].astype("str")
                    continue
    return d

def dropping_unnecessary_plots_list(viz_df: pd.DataFrame, l: list, datatype: str, mapping: dict) -> list:
    print(viz_df)
    for i in l:
        print(viz_df[i].values)
        if (mapping[i] == datatype) and (viz_df[i].values.nunique() > 8):
            l.remove(i)
    return l

def dropping_unncessary_plots_label(viz_df: pd.DataFrame, label: str, datatype: str, mapping: dict) -> str:
    if (mapping[label] == datatype) and (viz_df[label].nunique() > 8):
        return False
    else:
        return True

# def appending_to_cols(l,name_of_l, c):
#     if not l:
#         if (name_of_l == "pri_cols") or (name_of_l == "sec_cols") or (name_of_l == "cat_cols_to_expand"):
#             l = convert(c.mapping.categorical)
#         elif (name_of_l == "num_cols"):
#             l = convert(c.mapping.numerical)
#         elif (name_of_l == "datetime_cols"):
#             l = convert(c.mapping.datetime)
#         elif (name_of_l == "text_cols"):
#             l = convert(c.mapping.text)
#         return l
#     else:
#         return []