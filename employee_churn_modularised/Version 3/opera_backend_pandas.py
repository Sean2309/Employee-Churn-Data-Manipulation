import pandas as pd
import numpy as np
from jazz_antiformat import main

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

def check_if_cat(s: pd.Series) -> bool:
    n_unique_vals = s.nunique()
    sqrt_val = np.sqrt(s.shape[0])
    if n_unique_vals < sqrt_val:
        return True
    else:
        return False

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
    # df = df.astype(str)
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

def expand_categorical_cols(input_df_calc: pd.DataFrame, mapping: dict , pri_cols: list, sec_cols: list, cat_cols_to_expand: list, text_cols: list, cat_cols_no_expand: list = []):
    """
    Returns n cols from n unique vals in each categorical col input
    """
    expanded_cat_cols = []
    try:
        group_by_col = list(set(pri_cols + sec_cols))
        combined_col = group_by_col + cat_cols_to_expand + cat_cols_no_expand + text_cols
    except Exception as e:
        print(e)
    new_df = pd.DataFrame()
    temp_df = input_df_calc.loc[:, combined_col]
    temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]
    cat_cols_to_expand = set_x_tick_limit(input_df_calc, cat_cols_to_expand, "categorical", mapping)
    for val in cat_cols_to_expand:
        eval_col = pd.get_dummies(temp_df[val]).add_prefix(f"{val}_")
        new_df = pd.concat([new_df, eval_col], axis=1)
    temp_df = pd.concat([temp_df, new_df], axis=1, join="inner")

    for i in new_df.columns:
        expanded_cat_cols.append(i)
        mapping[i] = "categorical"

    temp_df = temp_df.drop_duplicates(group_by_col)    
    return temp_df, mapping, expanded_cat_cols
    