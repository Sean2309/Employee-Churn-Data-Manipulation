import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import time
import copy
import datetime
import itertools
import pathlib
import opera_util_common

from warnings import simplefilter
from opera_backend_wordcloud import wordcloud_plot
from opera_backend_seaborn import plot_all_relevant
from opera_backend_pandas import df_read, split_cols_by_nunique, split_cols_exceeding_thresh, check_if_datetime_1, check_if_datetime_2, check_if_float, check_if_int, check_if_cat, df_drop, df_clean_colnames_colvals, dtype_conversion


UNIVARIATE = "Univariate_Plots"
BIVARIATE = "Bivariate_Plots"
WORDCLOUD = "WordCloud_Plots"


# ----------------------------------------------------
# mapping dict functions
# ----------------------------------------------------
def _create_mapping_dict(df: pd.DataFrame, mapping: dict) -> dict:
    """
    Generation of mapping dict => from input mapping AND/OR df
    """
    col_to_type = {}
    df = df.astype("str")
    for k, v in mapping.items():
        for col in v.keys():
            col_to_type[col] = k
    for col in df.columns:
        if col not in col_to_type:
            if check_if_datetime_1(df[col]) and check_if_datetime_2(df[col]):
                col_to_type[col] = 'datetime'
                continue
            elif check_if_float(df[col]):
                if check_if_cat(df[col]):
                    col_to_type[col] = 'categorical'
                else:
                    col_to_type[col] = 'numerical'
                continue
            elif check_if_int(df[col]):
                if check_if_cat(df[col]):
                    col_to_type[col] = 'categorical'
                else:
                    col_to_type[col] = 'numerical'
                continue
            elif check_if_cat(df[col]): 
                col_to_type[col] = 'categorical'
                continue
            else:
                col_to_type[col] = 'text'
                continue
    return col_to_type

# ----------------------------------------------------
# misc functions
# ----------------------------------------------------

def get_value_from_ctx(a) -> list:
    """
    Returns elements of [a] in txt config file
    """
    return list(dict(a).keys())

def custom_settings(
    markersize: float = 2,
    linewidth: float = 0.5,
    labelsize: float = 6.5
):
    mpl.rcdefaults()
    mpl.rcParams["lines.markersize"] = markersize
    mpl.rcParams["lines.linewidth"] = linewidth
    mpl.rcParams["xtick.labelsize"] = labelsize
    ax = plt.gca()
    ax.set_xlim((0, 55))
    simplefilter(action="ignore")

def create_dest_folder(
    df_file: str,
    sheet_name: str = ""
) -> str:
    current_dir = os.getcwd()
    current_datetime = datetime.datetime.now().strftime("%d.%m.%Y_%H%M")
    dest_path = current_dir+'\\'+current_datetime + "_Visualisation_" + df_file + sheet_name
    UNIVARIATE_PATH = os.path.join(dest_path, UNIVARIATE)
    BIVARIATE_PATH = os.path.join(dest_path, BIVARIATE)
    WORDCLOUD_PATH = os.path.join(dest_path, WORDCLOUD)
    path_l = [UNIVARIATE_PATH, BIVARIATE_PATH, WORDCLOUD_PATH]
    for i in path_l:
        pathlib.Path(i).mkdir(parents=True, exist_ok=True)
    return dest_path

def create_readme(
    pri_cols: list,
    sec_cols: list,
    num_cols: list,
    datetime_cols: list,
    text_cols: list,
    dest_path: str,
    encoding: str
):
    readme_l = ["File Naming Conventions\nUnivariate Plots:\n[Name of Visual] of [x axis] & [legend] [iteration*]\n\nBivariate Plots:\n[Name of Visual] of [x axis] & [y axis] & [legend] [iteration*]\n\nConfig File Inputs\n\nPri Cols: ", str(pri_cols), "\n\nSec Cols: ", str(sec_cols), "\n\nNumerical Cols: ", str(num_cols) + "\n\nDatetime Cols: " + str(datetime_cols), "\n\nText Cols: " +str(text_cols)]
    with open(os.path.join(dest_path,"README.txt"), "w", encoding=encoding) as f:
        f.writelines(readme_l)

def get_additional_cols_mappings(l: list, name_of_l: str, mapping: dict):
    """
    Returns list of df cols, classified based on col dtype
    """
    if not l:
        if (name_of_l == "pri_cols") or (name_of_l == "sec_cols") and (mapping.get("categorical") != None):
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


# ----------------------------------------------------
# viz function
# ----------------------------------------------------

def viz(
    viz_df: pd.DataFrame,
    mapping: dict,
    threshold_x_tick_labels_cat: int,
    x_label_l: list,
    hue_label_l: list,
    y_label_l: list,
    text_cols_l: list,
    dest_path: str,
    palette: str
):  
    """
    Iterative function that forms different combinations of the lists of cols for:
        Univariate analysis
        Bivariate analysis
    """
    UNIVARIATE_PATH = os.path.join(dest_path, UNIVARIATE)
    BIVARIATE_PATH = os.path.join(dest_path, BIVARIATE)
    WORDCLOUD_PATH = os.path.join(dest_path, WORDCLOUD)
    x_valid, x_exceed_thresh = split_cols_by_nunique(viz_df, x_label_l, "categorical", mapping, threshold_x_tick_labels_cat)
    hue_label_l, _ = split_cols_by_nunique(viz_df, hue_label_l, "categorical", mapping, threshold_x_tick_labels_cat)
    univariate_label_l = list(pd.Series(x_valid + hue_label_l + y_label_l + x_exceed_thresh).drop_duplicates())
    dict_split_cols = split_cols_exceeding_thresh(df=viz_df, thresh=threshold_x_tick_labels_cat, label_l=x_exceed_thresh)
    for j in text_cols_l:
        wordcloud_plot(viz_df, WORDCLOUD_PATH, j)

    for i in hue_label_l:
        hue_label = i
        hue_label = ''.join(hue_label)
        ## Univariate analysis
        for x_label in univariate_label_l:
            dtype = ""
            if x_label in x_exceed_thresh:
                for i in range(len(dict_split_cols[x_label].keys())): 
                    dtype = ""
                    viz_df_a = viz_df.loc[viz_df[x_label].isin(list(dict_split_cols[x_label][i]))]
                    plot_all_relevant(
                        viz_df=viz_df_a, 
                        mapping=mapping, 
                        x_label=x_label, 
                        hue_label=hue_label, 
                        dest_path=UNIVARIATE_PATH, 
                        dtype=dtype, 
                        palette=palette,
                        iteration=i
                    )
            else:
                plot_all_relevant(
                            viz_df=viz_df, 
                            mapping=mapping, 
                            x_label=x_label, 
                            hue_label=hue_label, 
                            dest_path=UNIVARIATE_PATH, 
                            dtype=dtype, 
                            palette=palette
                        )
        print(f"\nUnivariate Analysis completed for {i}\n\nBivariate Analysis starting")

        # Bivariate analysis
        for (x_label, y_label) in itertools.product(x_valid, y_label_l):
            dtype = ""
            if x_label in x_exceed_thresh:
                for i in range(len(dict_split_cols[x_label].keys())): 
                    dtype = ""
                    viz_df_a = viz_df.loc[viz_df[x_label].isin(list(dict_split_cols[x_label][i]))]
                    plot_all_relevant(
                        viz_df=viz_df_a, 
                        mapping=mapping, 
                        x_label=x_label, 
                        hue_label=hue_label, 
                        dest_path=BIVARIATE_PATH, 
                        dtype=dtype, 
                        palette=palette,
                        y_label=y_label,
                        iteration=i
                    )
            plot_all_relevant(
                        viz_df=viz_df, 
                        mapping=mapping, 
                        x_label=x_label, 
                        hue_label=hue_label, 
                        dest_path=BIVARIATE_PATH, 
                        dtype=dtype, 
                        palette=palette,
                        y_label=y_label
                    )
        print(f"\nBivariate Analysis completed for {i}")

# ----------------------------------------------------
# init functions
# ----------------------------------------------------
def gen_viz_workflow(
    df: pd.DataFrame,
    threshold_nan_pct: float,
    threshold_x_tick_labels_cat: int,
    pri_cols: list,
    sec_cols: list,
    num_cols: list,
    datetime_cols: list,
    text_cols: list,
    df_file: str, 
    palette: str,
    mapping: dict,
    encoding: str,
    sheet_name: str = ""
):
    """
    DF Cleaning 
    Mapping Dict generation
    Label Lists generation 
    """
    type_to_ls_cols = {}
    df = df_drop(df, threshold_nan_pct)
    df = df_clean_colnames_colvals(df)
    mapping = _create_mapping_dict(df, mapping) 
    df = dtype_conversion(df, mapping)
    l = [pri_cols, sec_cols, num_cols, datetime_cols, text_cols]
    names_l = ["pri_cols", "sec_cols", "num_cols", "datetime_cols", "text_cols"]
    for k,v in mapping.items():
        type_to_ls_cols[v] = type_to_ls_cols.get(v, []) + [k]
    for ls, names in zip(l, names_l):
        additional_cols = get_additional_cols_mappings(ls, names, type_to_ls_cols)
        ls.extend(additional_cols)
    pri_cols += datetime_cols
    print("Mapping:\n"+str(type_to_ls_cols))
    """
    Creation of dest folder for viz
    """
    dest_path = create_dest_folder(
        df_file=df_file,
        sheet_name=sheet_name
    )
    create_readme(
        pri_cols=pri_cols, 
        sec_cols=sec_cols, 
        num_cols=num_cols, 
        datetime_cols=datetime_cols, 
        text_cols=text_cols, 
        dest_path=dest_path,
        encoding=encoding
    )

    """
    Main fn workflow:
        Expanding cat cols 
        Generation of plots
    """
    viz(
        viz_df=df,
        mapping=mapping,
        threshold_x_tick_labels_cat=threshold_x_tick_labels_cat, 
        x_label_l=pri_cols, 
        hue_label_l=sec_cols, 
        y_label_l=num_cols, 
        text_cols_l=text_cols, 
        dest_path=dest_path, 
        palette=palette)

def gen_viz(context_engine: C):
    start = time.time()

    """
    Parsing inputs
    """
    c = context_engine.k
    df_file_path = c.df.replace("/", os.sep)
    config = c.config
    ctx = opera_util_common.parse_ps_to_ctx(config)
    threshold_nan_pct = get_value_from_ctx(ctx.threshold_nan_pct)[0]
    threshold_x_tick_labels_cat = get_value_from_ctx(ctx.threshold_x_tick_labels_cat)[0]
    markersize = get_value_from_ctx(ctx.markersize)[0]
    linewidth = get_value_from_ctx(ctx.linewidth)[0]
    labelsize = get_value_from_ctx(ctx.labelsize)[0]
    palette = get_value_from_ctx(ctx.palette)[0]

    sheet_names = get_value_from_ctx(ctx.sheet_name)
    pri_cols = get_value_from_ctx(ctx.pri_cols)
    sec_cols = get_value_from_ctx(ctx.sec_cols)
    num_cols = get_value_from_ctx(ctx.num_cols)
    datetime_cols = get_value_from_ctx(ctx.datetime_cols)
    text_cols = get_value_from_ctx(ctx.text_cols)
    mapping = ctx.mapping
    custom_settings(markersize=markersize, linewidth=linewidth, labelsize=labelsize)
    full_file_name, ext = os.path.splitext(df_file_path)
    df_file = os.path.basename(full_file_name)
    encoding = sys.getdefaultencoding()
    """
    Read df based on file type => Main workflow fn starts
    """
    if ext == ".xlsx":
        if not sheet_names:
            xl = pd.ExcelFile(df_file_path)
            sheet_names = xl.sheet_names
        for sheet in sheet_names:
            #TODO Change this brute force copying if possible
            print("Current Sheet: "+str(sheet) + "\n")
            mapping_init = copy.copy(mapping)
            pri_cols_init = copy.copy(pri_cols)
            sec_cols_init = copy.copy(sec_cols)
            num_cols_init = copy.copy(num_cols)
            datetime_cols_init = copy.copy(datetime_cols)
            text_cols_init = copy.copy(text_cols)
            df, encoding = df_read(
                df_file_path=df_file_path,
                encoding=encoding,
                sheet_name=sheet,
            )
            gen_viz_workflow(
                df=df,
                threshold_nan_pct=threshold_nan_pct, 
                threshold_x_tick_labels_cat=threshold_x_tick_labels_cat,
                pri_cols=pri_cols_init,
                sec_cols=sec_cols_init,
                num_cols=num_cols_init,
                datetime_cols=datetime_cols_init,
                text_cols=text_cols_init,
                df_file=df_file,
                palette=palette,
                mapping=mapping_init,
                encoding=encoding,
                sheet_name=sheet
            )
    else:
        df, encoding = df_read(
        df_file_path=df_file_path,
        encoding=encoding
        )
        gen_viz_workflow(
                df=df,
                threshold_nan_pct=threshold_nan_pct, 
                threshold_x_tick_labels_cat=threshold_x_tick_labels_cat,
                pri_cols=pri_cols,
                sec_cols=sec_cols,
                num_cols=num_cols,
                datetime_cols=datetime_cols,
                text_cols=text_cols,
                df_file=df_file,
                palette=palette,
                mapping=mapping,
                encoding=encoding
            )

    end = time.time()
    print("Time Taken: " + str(end-start))
    return True
#TODO factorise looping across sheets function
def infer_col_types(context_engine: C):
    simplefilter(action="ignore")
    c = context_engine.k
    df_file_path = c.df.replace("/", os.sep)
    mapping = opera_util_common.parse_ps_to_ctx(c.config).mapping
    mapping = {k:list(v.keys()) for k, v in mapping}
    try:
        df = pd.read_csv(df_file_path, encoding="utf8")
    except:
        df = pd.read_csv(df_file_path, encoding="cp1252")
    inferred_mappings = _create_mapping_dict(df=df, mapping=mapping)
    return inferred_mappings