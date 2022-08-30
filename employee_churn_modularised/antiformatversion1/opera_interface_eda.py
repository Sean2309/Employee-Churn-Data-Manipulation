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
from opera_backend_pandas import read_df, split_cols_by_nunique, split_cols_exceeding_thresh, check_if_datetime_1, check_if_datetime_2, check_if_float, check_if_int, check_if_cat, df_drop, df_clean_colnames_colvals, dtype_conversion

UNIVARIATE = "Univariate_Plots"
BIVARIATE = "Bivariate_Plots"
WORDCLOUD = "WordCloud_Plots"

# ----------------------------------------------------
# mapping dict fn
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
# misc fns
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
    df_file_name: str,
    sheet_name: str = ""
) -> str:
    current_dir = os.getcwd()
    current_datetime = datetime.datetime.now().strftime("%d.%m.%Y_%H%M")
    dest_path = current_dir+'\\'+current_datetime + "_Visualisation_" + df_file_name + sheet_name
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

def convert_type_to_ls_cols(
    mapping: dict
) -> dict:
    type_to_ls_cols = {}
    for k,v in mapping.items():
        type_to_ls_cols[v] = type_to_ls_cols.get(v, []) + [k]
    return type_to_ls_cols

# ----------------------------------------------------
# viz fn
# ----------------------------------------------------
def viz(
    viz_df: pd.DataFrame,
    mapping: dict,
    max_number_of_x_ticklabels_per_graphplot: int,
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
    x_valid, x_exceed_thresh = split_cols_by_nunique(viz_df, x_label_l, "categorical", mapping, max_number_of_x_ticklabels_per_graphplot)
    hue_label_l, _ = split_cols_by_nunique(viz_df, hue_label_l, "categorical", mapping, max_number_of_x_ticklabels_per_graphplot)
    univariate_label_l = list(pd.Series(x_valid + hue_label_l + y_label_l + x_exceed_thresh).drop_duplicates())
    dict_split_cols = split_cols_exceeding_thresh(df=viz_df, thresh=max_number_of_x_ticklabels_per_graphplot, label_l=x_exceed_thresh)
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
# gen viz workflow fn
# ----------------------------------------------------
def gen_viz_workflow(
    df: pd.DataFrame,
    df_file_name: str, 
    max_NaN_percent_in_dfcol: float,
    max_number_of_x_ticklabels_per_graphplot: int,
    pri_cols: list,
    sec_cols: list,
    num_cols: list,
    datetime_cols: list,
    text_cols: list,
    palette: str,
    mapping: dict,
    encoding: str,
    sheet_name: str = ""
):
    """
    (A) Preprocessing of Data + Generation of mapping dict
    (B) Init of plot variables
    (C) Creation of dest folder
    (D) Viz fn
    """
    type_to_ls_cols = {}

    # (A1) Preprocessing DF
    df = df_drop(df, max_NaN_percent_in_dfcol)
    df = df_clean_colnames_colvals(df)
    # (A2) Generation of mapping dict
    mapping = _create_mapping_dict(df, mapping) 
    # (A1) Preprocessing DF
    df = dtype_conversion(df, mapping)

    # (B)
    l = [pri_cols, sec_cols, num_cols, datetime_cols, text_cols]
    names_l = ["pri_cols", "sec_cols", "num_cols", "datetime_cols", "text_cols"]
    type_to_ls_cols = convert_type_to_ls_cols(mapping=mapping)
    for ls, names in zip(l, names_l):
        additional_cols = get_additional_cols_mappings(ls, names, type_to_ls_cols)
        ls.extend(additional_cols)
    pri_cols += datetime_cols
    print(f"Mapping:\n{type_to_ls_cols}")
    
    # (C)
    dest_path = create_dest_folder(
        df_file_name=df_file_name,
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

    # (D)
    viz(
        viz_df=df,
        mapping=mapping,
        max_number_of_x_ticklabels_per_graphplot=max_number_of_x_ticklabels_per_graphplot, 
        x_label_l=pri_cols, 
        hue_label_l=sec_cols, 
        y_label_l=num_cols, 
        text_cols_l=text_cols, 
        dest_path=dest_path, 
        palette=palette)

# ----------------------------------------------------
# interate sheets fn
# ----------------------------------------------------
def iterate_through_sheets(
    engine: str,
    ext: str,
    df_path_input: str,
    mapping: dict,
    encoding: str = "",
    max_NaN_percent_in_dfcol: float = None,
    max_number_of_x_ticklabels_per_graphplot: int = None,
    pri_cols: list = None,
    sec_cols: list = None,
    num_cols: list = None,
    datetime_cols: list = None,
    text_cols: list = None,
    df_file_name: str = None, 
    palette: str = None,
    sheet_names: list = []
):  
    """
    If xlsx => Iterates through sheets => Pass into respective engine's workflow
    If csv => Pass into respective engine's workflow
    """
    if ext == ".xlsx":
        if not sheet_names:
            xl = pd.ExcelFile(df_path_input)
            sheet_names = xl.sheet_names
        for sheet in sheet_names:
            #TODO Change this brute force copying if possible
            mapping_init = copy.copy(mapping)
            df, encoding = read_df(
                df_path_input=df_path_input,
                encoding=encoding,
                sheet_name=sheet,
            )
            if engine == "gen_viz":
                print(f"\nCurrent Sheet: {sheet}")
                pri_cols_init = copy.copy(pri_cols)
                sec_cols_init = copy.copy(sec_cols)
                num_cols_init = copy.copy(num_cols)
                datetime_cols_init = copy.copy(datetime_cols)
                text_cols_init = copy.copy(text_cols)
                gen_viz_workflow(
                    df=df,
                    df_file_name=df_file_name,
                    max_NaN_percent_in_dfcol=max_NaN_percent_in_dfcol, 
                    max_number_of_x_ticklabels_per_graphplot=max_number_of_x_ticklabels_per_graphplot,
                    pri_cols=pri_cols_init,
                    sec_cols=sec_cols_init,
                    num_cols=num_cols_init,
                    datetime_cols=datetime_cols_init,
                    text_cols=text_cols_init,
                    palette=palette,
                    mapping=mapping_init,
                    encoding=encoding,
                    sheet_name=sheet
                )
            elif engine == "infer_col_types":
                inferred_mappings = _create_mapping_dict(
                    df=df,
                    mapping=mapping_init,
                )
                type_to_ls_cols = convert_type_to_ls_cols(mapping=inferred_mappings)
                print(f"Inferred Col Types of Sheet {sheet}:\n" + str(type_to_ls_cols) + "\n\n\n")
    else:
        df, encoding = read_df(
            df_path_input=df_path_input,
            encoding=encoding
        )
        if engine == "gen_viz":
            gen_viz_workflow(
                    df=df,
                    df_file_name=df_file_name,
                    max_NaN_percent_in_dfcol=max_NaN_percent_in_dfcol, 
                    max_number_of_x_ticklabels_per_graphplot=max_number_of_x_ticklabels_per_graphplot,
                    pri_cols=pri_cols,
                    sec_cols=sec_cols,
                    num_cols=num_cols,
                    datetime_cols=datetime_cols,
                    text_cols=text_cols,
                    palette=palette,
                    mapping=mapping,
                    encoding=encoding
                )
        elif engine == "infer_col_types":
            inferred_mappings = _create_mapping_dict(
                df=df,
                mapping=mapping,
            )
            type_to_ls_cols = convert_type_to_ls_cols(mapping=inferred_mappings)
            print("Inferred Col Types:\n" + str(type_to_ls_cols))

# ----------------------------------------------------
# init fns
# ----------------------------------------------------
def gen_viz(context_engine: C):
    """
    Engine that automatically generates viz plots with 3 functions:
        - This gen_viz init fn
        - iterate_through_sheets fn
        - gen_viz_workflow fn =>    Data Processing 
                                    Inferring col data types
                                    Gen plots 

    This init function contains 3 main steps:
        (A) Parsing + Gen of inputs
        (B) Customisation of plotting settings
        (C) Pass into an iterate_through_sheets fn
    """
    start = time.time()
    engine = "gen_viz"

    # (A1) Parse Config File
    c = context_engine.k
    config = c.config
    ctx = opera_util_common.parse_ps_to_ctx(config)

    # (A2) Parse + Gen file path inputs
    df_path_input = c.df.replace("/", os.sep)
    stripped_df_path_input, ext = os.path.splitext(df_path_input)
    df_file_name = os.path.basename(stripped_df_path_input)
    encoding = sys.getdefaultencoding()

    # (A3) Parse Viz inputs
    max_NaN_percent_in_dfcol = get_value_from_ctx(ctx.max_NaN_percent_in_dfcol)[0]
    max_number_of_x_ticklabels_per_graphplot = get_value_from_ctx(ctx.max_number_of_x_ticklabels_per_graphplot)[0]
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
    mapping = dict(ctx.mapping)
    
    # (B)
    custom_settings(markersize=markersize, linewidth=linewidth, labelsize=labelsize)

    # (C)
    iterate_through_sheets(
        engine=engine,
        ext=ext,
        df_path_input=df_path_input,
        mapping=mapping,
        encoding=encoding,
        max_NaN_percent_in_dfcol=max_NaN_percent_in_dfcol,
        max_number_of_x_ticklabels_per_graphplot=max_number_of_x_ticklabels_per_graphplot,
        pri_cols=pri_cols,
        sec_cols=sec_cols,
        num_cols=num_cols,
        datetime_cols=datetime_cols,
        text_cols=text_cols,
        df_file_name=df_file_name, 
        palette=palette,
        sheet_names=sheet_names
    )  
    end = time.time()
    print("\nTime Taken: " + str(end-start))
    return True

def infer_col_types(context_engine: C):
    """
    Engine to automatically infer col types
    Contains 2 main steps:
        (A) Parsing Input
        (B) Passing into Intermediate fn
    """
    
    simplefilter(action="ignore")

    # (A)
    engine = "infer_col_types"
    encoding = sys.getdefaultencoding()

    c = context_engine.k
    df_path_input = c.df.replace("/", os.sep)
    mapping = dict(opera_util_common.parse_ps_to_ctx(c.config).mapping)
    sheet_names = get_value_from_ctx(opera_util_common.parse_ps_to_ctx(c.config).sheet_name)
    stripped_df_path_input, ext = os.path.splitext(df_path_input)
    # (B)
    iterate_through_sheets(
        engine=engine,
        ext=ext,
        df_path_input=df_path_input,
        mapping=mapping,
        encoding=encoding,
        sheet_names=sheet_names
    )
    return True