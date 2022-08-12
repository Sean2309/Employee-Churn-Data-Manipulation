import argparse
import configparser
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import time
import datetime
import itertools
import pathlib



from warnings import simplefilter
# from opera_backend_wordcloud import wordcloud_plot
# from opera_backend_seaborn import plot
from eda_backend_pandas import set_x_tick_limit, check_if_datetime_1, check_if_float, check_if_int, check_if_cat, df_drop, df_replace, dtype_conversion, expand_categorical_cols

# ----------------------------------------------------
# df parsing functions
# ----------------------------------------------------

def convert(a):
    """
    Returns elements of [a] in txt config file
    """
    return list(dict(a).keys())

# ----------------------------------------------------
# mapping dict functions
# ----------------------------------------------------
def _create_mapping_dict(df: pd.DataFrame, mapping: dict):
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

def create_readme(**kwargs):
    pri_cols=kwargs.get("pri_cols")
    sec_cols=kwargs.get("sec_cols")
    cat_cols_to_expand=kwargs.get("cat_cols_to_expand")
    num_cols=kwargs.get("num_cols")
    datetime_cols=kwargs.get("datetime_cols")
    text_cols=kwargs.get("text_cols")
    dest_path=kwargs.get("dest_path")
    readme_l = ["File Naming Conventions\nUnivariate Plots:\n[Name of Visual] of [x axis] & [legend]\n\nBivariate Plots:\n[Name of Visual] of [x axis] & [y axis] & [legend]\n\nConfig File Inputs\n\nPri Cols: ", str(pri_cols), "\n\nSec Cols: ", str(sec_cols), "\n\nCat Cols to expand: ", str(cat_cols_to_expand), "\n\nNumerical Cols: ", str(num_cols) + "\n\nDatetime Cols: " + str(datetime_cols), "\n\nText Cols: " +str(text_cols)]
    with open(dest_path + "/README.txt", "w") as f:
        f.writelines(readme_l)

def appending_to_cols(l: list,name_of_l: str, mapping: dict):
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
def viz(**kwargs):
    viz_df = kwargs.get("viz_df")
    mapping = kwargs.get("mapping")
    threshold_x_tick_labels_cat = kwargs.get("threshold_x_tick_labels_cat")
    threshold_x_tick_labels_num = kwargs.get("threshold_x_tick_labels_num")
    x_label_list = kwargs.get("x_label_list")
    hue_label_l = kwargs.get("hue_label_l")
    y_label_list_cat = kwargs.get("y_label_list_cat")
    y_label_list_num = kwargs.get("y_label_list_num")
    text_cols = kwargs.get("text_cols")
    dest_path = kwargs.get("dest_path")
    palette = kwargs.get("palette")

    univariate_path = dest_path + "\\Univariate_Plots"
    bivariate_path = dest_path + "\\Bivariate_Plots"
    wordcloud_path = dest_path + "\\WordCloud_Plots"

    x_label_list = set_x_tick_limit(viz_df, x_label_list, "categorical", mapping, threshold_x_tick_labels_cat)
    x_label_list = set_x_tick_limit(viz_df, x_label_list, "datetime", mapping, threshold_x_tick_labels_num)
    hue_label_l = set_x_tick_limit(viz_df, hue_label_l, "categorical", mapping, threshold_x_tick_labels_cat)
    univariate_label_l = list(pd.Series(x_label_list + hue_label_l + y_label_list_num).drop_duplicates())
    y_label_list = y_label_list_cat + y_label_list_num
    
    # for j in text_cols:
    #     wordcloud_plot(viz_df, wordcloud_path, j)

    # for i in hue_label_l:
    #     hue_label = i
    #     hue_label = ''.join(hue_label)
    #     ## Univariate analysis
    #     for x_label in univariate_label_l:
    #         dtype = ""
    #         print("\nx label iteration: " + str(x_label) + "    hue label: " + str(hue_label))
    #         if (x_label != hue_label) and (mapping[x_label] == "numerical"):
    #             dtype = "num"
    #             plot(plot_type="hist", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=univariate_path, dtype=dtype, palette=palette)
    #             plot(plot_type="kde", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=univariate_path, dtype=dtype, palette=palette)
    #             plot(plot_type="ecdf", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=univariate_path, dtype=dtype, palette=palette)
    #         elif (x_label != hue_label) and (mapping[x_label] == "categorical"):
    #             plot(plot_type="hist", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=univariate_path, dtype=dtype, palette=palette)
    #     print(f"\nUnivariate Analysis completed for {i}\n\nBivariate Analysis starting")

    #     ## Bivariate analysis
    #     for (x_label, y_label) in itertools.product(x_label_list, y_label_list):
    #         dtype = ""
    #         print("\nx label iteration: " + str(x_label) + "    y label iteration: " + str(y_label) + "    hue label: " + str(hue_label))

    #         if (x_label != hue_label) and (mapping[x_label] == "categorical") and (mapping[y_label] == "numerical"):
    #             plot(plot_type="box", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=bivariate_path, dtype=dtype, y_label=y_label, palette=palette)
    #             plot(plot_type="bar", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=bivariate_path, dtype=dtype, y_label=y_label, palette=palette)
    #             plot(plot_type="violin", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=bivariate_path, dtype=dtype, y_label=y_label, palette=palette)

    #         elif (x_label != hue_label) and (mapping[x_label] == "numerical") and (mapping[y_label] == "numerical"):
    #             dtype = "num"
    #             plot(plot_type="hist", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=bivariate_path, dtype=dtype, y_label=y_label, palette=palette)
    #             plot(plot_type="scatter", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=bivariate_path, dtype=dtype, y_label=y_label, palette=palette)
    #             plot(plot_type="line", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=bivariate_path, dtype=dtype, y_label=y_label, palette=palette)

    #         elif (x_label != hue_label) and (mapping[x_label] == "datetime") and (mapping[y_label] == "numerical"):
    #             plot(plot_type="line", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=bivariate_path, dtype=dtype, y_label=y_label, palette=palette)

    #     print(f"\nBivariate Analysis completed for {i}")

# ----------------------------------------------------
# init functions
# ----------------------------------------------------
def parse_config(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--c", nargs="?", default="sean_config_housing.ini", type=str, help="name of config file")
    args, unknown_args = arg_parser.parse_known_args(argv)
    config = configparser.ConfigParser()
    config.read(args.c)
    return config

def gen_viz(argv):
    start = time.time()
    custom_settings()
    config = parse_config(argv=argv)
    compute_table = config["input"]["compute_table"]
    threshold_nan_pct = config["input"]["threshold_nan_pct"]
    threshold_x_tick_labels_cat = config["input"]["threshold_x_tick_labels_cat"]
    threshold_x_tick_labels_num = config["input"]["threshold_x_tick_labels_num"]
    palette = config["input"]["palette"]
    pri_cols = config["input"]["pri_cols"]
    sec_cols = config["input"]["sec_cols"]
    cat_cols_to_expand= config["input"]["cat_cols_to_expand"]
    num_cols = config["input"]["num_cols"]
    datetime_cols = config["input"]["datetime_cols"]
    text_cols = config["input"]["text_cols"]
    df_file_path = config["input"]["df_file_path"]
    df_file = df_file_path.rsplit(sep="\\")[-1].split(sep=".")[0]
    d = {}
    inv_mapping = {}
    for k, v in config.items("mapping"):
        inv_mapping[k] = v
    
    df_file = df_file_path.rsplit(sep="\\")[-1].split(sep=".")[0]
    l = [pri_cols, sec_cols, cat_cols_to_expand, num_cols, datetime_cols, text_cols]
    names_l = ["pri_cols", "sec_cols", "cat_cols_to_expand", "num_cols", "datetime_cols", "text_cols"]

    """
    Reading of df 
    +
    Generating mapping dict + plotting variables
    """
    for k,v in inv_mapping.items():
        d[v] = d.get(v, []) + [k]
    print(d)

    # try:
    #     df = pd.read_csv(df_file_path, encoding="utf8", infer_datetime_format=True)
    # except:
    #     df = pd.read_csv(df_file_path, encoding="cp1252", infer_datetime_format=True)

    # df = df_drop(df, threshold_nan_pct)
    # mapping = _create_mapping_dict(df, mapping)
    # df = dtype_conversion(df, mapping)
    # df = df_replace(df)

    for k,v in inv_mapping.items():
        d[v] = d.get(v, []) + [k]
    print(d)
    # for ls, names in zip(l, names_l):
    #     a = appending_to_cols(ls, names, inv_mapping)
    #     ls.extend(a)
    # pri_cols += datetime_cols
    # print("Mapping:\n"+str(inv_mapping))

    # """
    # Creation of dest folder for viz
    # """
    # dest_path = create_dest_folder(df_file)
    # create_readme(
    #     pri_cols=pri_cols, 
    #     sec_cols=sec_cols, 
    #     cat_cols_to_expand=cat_cols_to_expand, 
    #     num_cols=num_cols, 
    #     datetime_cols=datetime_cols, 
    #     text_cols=text_cols, 
    #     dest_path=dest_path)

    # if compute_table == True:
    #     temp_df, mapping, expanded_cat_cols = expand_categorical_cols(
    #         df=df, 
    #         mapping=mapping, 
    #         pri_cols=pri_cols, 
    #         sec_cols=sec_cols, 
    #         cat_cols_to_expand=cat_cols_to_expand, 
    #         text_cols=text_cols, 
    #         cat_cols_no_expand=num_cols)
    #     viz(
    #         viz_df=temp_df,
    #         mapping=mapping,
    #         threshold_x_tick_labels_cat=threshold_x_tick_labels_cat, threshold_x_tick_labels_num=threshold_x_tick_labels_num, x_label_list=pri_cols, 
    #         hue_label_l=sec_cols, 
    #         y_label_list_cat=expanded_cat_cols, 
    #         y_label_list_num=num_cols, 
    #         text_cols=text_cols, 
    #         dest_path=dest_path, 
    #         palette=palette)
    #     end = time.time()
    #     print("Time Taken: " + str(end-start))
    # else:
    #     viz(
    #         viz_df=df, 
    #         mapping=mapping, threshold_x_tick_labels_cat=threshold_x_tick_labels_cat, threshold_x_tick_labels_num=threshold_x_tick_labels_num, x_label_list=pri_cols, 
    #         hue_label_l=sec_cols, 
    #         y_label_list_cat=cat_cols_to_expand, 
    #         y_label_list_num=num_cols, 
    #         text_cols=text_cols, 
    #         dest_path=dest_path, 
    #         palette=palette)
    #     end = time.time()
    #     print("Time Taken: " + str(end-start))
    # return True


# def infer_col_types(context_engine: C):
#     simplefilter(action="ignore")
#     c = context_engine.k
#     mapping = dict(get_ctx(c.mapping))
#     try:
#         df = pd.read_csv(c.df, encoding="utf8")
#     except:
#         df = pd.read_csv(c.df, encoding="cp1252")
#     inferred_mappings = _create_mapping_dict(df=df, mapping=mapping)
#     return inferred_mappings

if __name__ == "__main__":
    start = time.time()
    argv = sys.argv
    argv = argv[1:]
    gen_viz(argv=argv)