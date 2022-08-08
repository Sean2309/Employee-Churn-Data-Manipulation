from asyncio import Future
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import time
import datetime
import itertools
import pathlib
import opera_util_common

from warnings import simplefilter
# from wordcloud import STOPWORDS  # TODO: opera_backend_wordcloud
from opera_backend_wordcloud import wordcloud_plot
from opera_backend_seaborn import hist_plot, kde_plot, ecdf_plot, box_plot, bar_plot, violin_plot, scatter_plot, line_plot
from opera_backend_pandas import set_x_tick_limit, check_if_datetime_1, check_if_datetime_2, check_if_float, check_if_int, check_if_cat, df_drop, df_replace, dtype_conversion, expand_categorical_cols

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

# ----------------------------------------------------
# mapping dict functions
# ----------------------------------------------------
def _create_mapping_dict(df: pd.DataFrame, mapping: dict): # taking 
    """
    Generation of mapping dict => from input mapping AND/OR df
    """
    df = df.astype("str")
    d = {}
    for k, v in mapping.items():
        keys_l = list(v.keys())
        if keys_l:
            d.update({i: k for i in keys_l})
    for col in df.columns:
        if col not in d:
            if check_if_datetime_1(df[col]):
                if check_if_datetime_2(df[col]):
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

def custom_settings():
    mpl.rcdefaults()
    mpl.rcParams["lines.markersize"] = 2
    mpl.rcParams["lines.linewidth"] = 0.5
    mpl.rcParams["xtick.labelsize"] = 6.5
    ax = plt.gca()
    ax.set_xlim((0, 55))
    simplefilter(action="ignore")
    simplefilter(action="ignore", category=UserWarning)


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



# ----------------------------------------------------
# viz function
# ----------------------------------------------------

def viz(viz_df: pd.DataFrame, mapping: dict, threshold_x_tick_labels_cat: int, threshold_x_tick_labels_num: int, x_label_list: list, hue_label_l: list, y_label_list_cat: list, y_label_list_num: list, text_cols: list, dest_path: str):
    univariate_path = dest_path + "\\Univariate_Plots"
    bivariate_path = dest_path + "\\Bivariate_Plots"
    wordcloud_path = dest_path + "\\WordCloud_Plots"

    x_label_list = set_x_tick_limit(viz_df, x_label_list, "categorical", mapping, threshold_x_tick_labels_cat)
    x_label_list = set_x_tick_limit(viz_df, x_label_list, "datetime", mapping, threshold_x_tick_labels_num)
    hue_label_l = set_x_tick_limit(viz_df, hue_label_l, "categorical", mapping, threshold_x_tick_labels_cat)
    univariate_label_l = list(pd.Series(x_label_list + hue_label_l + y_label_list_num).drop_duplicates())
    y_label_list = y_label_list_cat + y_label_list_num
    
    for j in text_cols:
        wordcloud_plot(viz_df, wordcloud_path, j)

    for i in hue_label_l:
        hue_label = i
        hue_label = ''.join(hue_label)
        ## Univariate analysis
        for x_label in univariate_label_l:
            dtype = ""
            print("\nx label iteration: " + str(x_label) + "    hue label: " + str(hue_label))
            if (x_label != hue_label) and (mapping[x_label] == "numerical"):
                dtype = "num"
                hist_plot(viz_df, x_label, hue_label, univariate_path, dtype)
                kde_plot(viz_df, x_label, hue_label, univariate_path, dtype)
                ecdf_plot(viz_df, x_label, hue_label, univariate_path, dtype)
            elif (x_label != hue_label) and (mapping[x_label] == "categorical"):
                hist_plot(viz_df, x_label, hue_label, univariate_path, dtype)
        print(f"\nUnivariate Analysis completed for {i}\n\nBivariate Analysis starting")

        ## Bivariate analysis
        for (x_label, y_label) in itertools.product(x_label_list, y_label_list):
            dtype = ""
            print("\nx label iteration: " + str(x_label) + "    y label iteration: " + str(y_label) + "    hue label: " + str(hue_label))

            if (x_label != hue_label) and (mapping[x_label] == "categorical") and (mapping[y_label] == "numerical"):
                box_plot(viz_df, x_label, hue_label, bivariate_path, dtype, y_label)
                bar_plot(viz_df, x_label, hue_label, bivariate_path, dtype, y_label)
                violin_plot(viz_df, x_label, hue_label, bivariate_path, dtype, y_label)

            elif (x_label != hue_label) and (mapping[x_label] == "numerical") and (mapping[y_label] == "numerical"):
                dtype = "num"
                hist_plot(viz_df, x_label, hue_label, bivariate_path, dtype, y_label)
                scatter_plot(viz_df, x_label, hue_label, bivariate_path, dtype, y_label)
                line_plot(viz_df, x_label, hue_label, bivariate_path, dtype, y_label)

            elif (x_label != hue_label) and (mapping[x_label] == "datetime") and (mapping[y_label] == "numerical"):
                line_plot(viz_df, x_label, hue_label, bivariate_path, dtype, y_label)

        print(f"\nBivariate Analysis completed for {i}")

# ----------------------------------------------------
# init functions
# ----------------------------------------------------


def gen_viz(context_engine: C):
    start = time.time()
    custom_settings()
    c = context_engine.k
    config = c.config
    
    ctx = get_ctx(config) # returns a list of dictionaries
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
    mapping = dict(ctx.mapping)
    inv_mapping = {}
    df_file = df_file_path.rsplit(sep="\\")[-1].split(sep=".")[0]
    l = [pri_cols, sec_cols, cat_cols_to_expand, num_cols, datetime_cols, text_cols]
    names_l = ["pri_cols", "sec_cols", "cat_cols_to_expand", "num_cols", "datetime_cols", "text_cols"]


    """
    Reading of df 
    +
    Generation of init vars vals
    """
    try:
        df = pd.read_csv(df_file_path, encoding="utf8", infer_datetime_format=True)
    except:
        df = pd.read_csv(df_file_path, encoding="cp1252", infer_datetime_format=True)
    df = df_drop(df, threshold_nan_pct)
    mapping = _create_mapping_dict(df, mapping)
    df = dtype_conversion(df, mapping)
    df = df_replace(df)
    for k,v in mapping.items():
        inv_mapping[v] = inv_mapping.get(v, []) + [k]
    for ls, names in zip(l, names_l):
        a = appending_to_cols(ls, names, inv_mapping)
        ls.extend(a)
    pri_cols += datetime_cols
    print("Mapping:\n"+str(inv_mapping))


    """
    Creation of dest folder for viz
    """
    dest_path = create_dest_folder(df_file)
    create_readme(dest_path, pri_cols, sec_cols, cat_cols_to_expand, num_cols, datetime_cols, text_cols)
    if compute_table == True:
        temp_df, mapping, expanded_cat_cols = expand_categorical_cols(df, mapping, pri_cols, sec_cols, cat_cols_to_expand, text_cols, num_cols)
        viz(temp_df, mapping, threshold_x_tick_labels_cat, threshold_x_tick_labels_num, pri_cols, sec_cols, expanded_cat_cols, num_cols, text_cols, dest_path)
        end = time.time()
        print("Time Taken: " + str(end-start))
    else:
        viz(df, mapping, threshold_x_tick_labels_cat, threshold_x_tick_labels_num, pri_cols, sec_cols, cat_cols_to_expand, num_cols, text_cols, dest_path)
        end = time.time()
        print("Time Taken: " + str(end-start))
    return True


def infer_col_types(context_engine: C):
    simplefilter(action="ignore")
    c = context_engine.k
    mapping = dict(get_ctx(c.mapping))
    try:
        df = pd.read_csv(c.df, encoding="utf8")
    except:
        df = pd.read_csv(c.df, encoding="cp1252")
    inferred_mappings = _create_mapping_dict(df=df, mapping=mapping)
    return inferred_mappings
