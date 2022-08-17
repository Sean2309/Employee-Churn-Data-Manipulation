import pandas as pd
import numpy as np
import opera_util_common

def set_x_tick_limit(
    viz_df: pd.DataFrame, 
    l: list, 
    datatype: str, 
    mapping: dict, 
    threshold: int = 8
) -> list:
    """
    Returns minimised list of df cols
    """
    
    l1 = l[:]
    l2 = []
    for i in l:
        if (mapping[i] == datatype) and (viz_df[i].nunique() > threshold):
            l1.remove(i)
            l2.append(i)
            continue
    return l1, l2

def split_l(
    l: list,
    n: int
):  
    return [l[i::n] for i in range(n)]

def gen_var(
    n: int
):  
    return [i+1 for i in range(0, n)]

def split_cols_exceeding_thresh(
    df: pd.DataFrame,
    thresh: int,
    label_l: list
) -> dict:
    """
    For the cols that exceed the categorical threshold 
        => This function will break down the unique vals of the col into n groups
            where n = n_unique values of the col / categorical threshold
    Returns nested dict
        keys == vars with arbitrary names
        values == list of cols within the threshold 
    """
    d = {}
    for i in label_l:
        d[i] = {}
        n_cuts = int(np.floor((df[i].nunique()) / thresh))
        l = df[i].unique()
        col_names_l = split_l(l, n_cuts)
        var_name_l = gen_var(n_cuts)
        for j in range(len(var_name_l)):
            d[i][var_name_l[j]] = col_names_l[j]
    return d

def check_if_datetime_1(s: pd.Series) -> bool:
    try:
        s = s.astype("datetime64[ns]")
        return True
    except:
        return False

def check_if_datetime_2(s: pd.Series) -> bool:
    pattern = '{}/{}/{}'
    i = s.first_valid_index()
    s = s[i]
    l = s.split("/", 3)[-1]
    if opera_util_common.jazz_antiformat.main(pattern, s) and (len(l[-1]) < 11):
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

# TODO Explore other methods to determining cat dtype
def check_if_cat(s: pd.Series) -> bool:
    n_unique_vals = s.nunique()
    sqrt_val = np.sqrt(s.shape[0])
    if n_unique_vals < sqrt_val:
        return True
    else:
        return False

def df_read(
    df_file_path: str,
    sheet_name: str = ""
):
    """
    DF Parsing according to the file type: 
        xlsx
        csv
    """
    if sheet_name:
        df = pd.read_excel(df_file_path, sheet_name=sheet_name)
    else:
        try:
            df = pd.read_csv(df_file_path, encoding="utf8", infer_datetime_format=True)
        except:
            df = pd.read_csv(df_file_path, encoding="cp1252", infer_datetime_format=True)
    return df

def df_drop(
    df: pd.DataFrame, 
    threshold: float = 0.2
) -> pd.DataFrame: 
    """
    Drop cols with :
        1) >= 20% NaN values
        2) 1 unique value
    """
    df = df.dropna(axis=1, thresh = threshold*len(df))
    for col in df.columns:
        if df[col].nunique() == 1:
            df = df.drop(col, axis=1)
    return df

def df_clean_colnames_colvals(df: pd.DataFrame) -> pd.DataFrame:
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

def dtype_conversion(
    df: pd.DataFrame, 
    mapping: dict
) -> pd.DataFrame:
    """
    Based on mapping dict => converts df cols to their respective dtype

    len() > 11 => Incoming datetimeformat e.g " 2022 05:10"

    .dt accessor must operate on the datetime col => returns an Index of formatted strs => 2 conversions to datetime needed

    """
    for k, v in mapping.items():
        if v == "numerical":
            df[k] = df[k].astype("float64")
            continue
        elif v == "datetime":
            df[k] = df[k].astype("datetime64")
            if df[k].nunique() > 500:
                df[k] = df[k].dt.strftime("%Y")
            elif df[k].nunique() > 200:
                df[k] = df[k].dt.strftime("%m/%Y")
            else:
                df[k] = df[k].dt.strftime("%d/%m/%Y")
            df[k] = pd.to_datetime(df[k])
            continue
    return df