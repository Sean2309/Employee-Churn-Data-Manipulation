import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time
import copy
import datetime
import itertools
import pathlib
import opera_util_common

from warnings import simplefilter
from opera_backend_wordcloud import wordcloud_plot
from opera_backend_seaborn import plot
from opera_backend_pandas import df_read, set_x_tick_limit, split_cols_exceeding_thresh, check_if_datetime_1, check_if_datetime_2, check_if_float, check_if_int, check_if_cat, df_drop, df_clean_colnames_colvals, dtype_conversion

# ----------------------------------------------------
# mapping dict functions
# ----------------------------------------------------
def _create_mapping_dict(df: pd.DataFrame, mapping: dict) -> dict:
    """
    Generation of mapping dict => from input mapping AND/OR df
    """
    d = {}
    df = df.astype("str")
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
def get_ctx(file):
    ctx = opera_util_common.parse_ps_to_ctx(file)
    return ctx

def convert(a) -> list:
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
    univariate_path = dest_path + "\\Univariate_Plots"
    bivariate_path = dest_path + "\\Bivariate_Plots"
    wordcloud_path = dest_path + "\\WordCloud_Plots"
    path_l = [univariate_path, bivariate_path, wordcloud_path]
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
):
    readme_l = ["File Naming Conventions\nUnivariate Plots:\n[Name of Visual] of [x axis] & [legend]\n\nBivariate Plots:\n[Name of Visual] of [x axis] & [y axis] & [legend]\n\nConfig File Inputs\n\nPri Cols: ", str(pri_cols), "\n\nSec Cols: ", str(sec_cols), "\n\nNumerical Cols: ", str(num_cols) + "\n\nDatetime Cols: " + str(datetime_cols), "\n\nText Cols: " +str(text_cols)]
    with open(dest_path + "/README.txt", "w") as f:
        f.writelines(readme_l)

def appending_to_cols(l: list,name_of_l: str, mapping: dict):
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
class plotting_logic:
    def __init__(self, univariate, bivariate) -> None:
        self.univariate = univariate
        self.bivariate = bivariate

    def univariate_logic(self):
        ...
    ...

def viz(
    viz_df: pd.DataFrame,
    mapping: dict,
    threshold_x_tick_labels_cat: int,
    threshold_x_tick_labels_num: int,
    x_label_l: list,
    hue_label_l: list,
    y_label_l: list,
    text_cols_l: list,
    dest_path: str,
    palette: str
):  
    viz_df_og = copy.deepcopy(viz_df)
    univariate_path = dest_path + "\\Univariate_Plots"
    bivariate_path = dest_path + "\\Bivariate_Plots"
    wordcloud_path = dest_path + "\\WordCloud_Plots"
    x_label_l, x_label_split = set_x_tick_limit(viz_df, x_label_l, "categorical", mapping, threshold_x_tick_labels_cat)
    hue_label_l, l = set_x_tick_limit(viz_df, hue_label_l, "categorical", mapping, threshold_x_tick_labels_cat)
    # univariate_label_l = list(pd.Series(x_label_l + hue_label_l + y_label_l_num).drop_duplicates())
    univariate_label_l = list(pd.Series(x_label_l + hue_label_l + y_label_l + x_label_split).drop_duplicates())

    print((x_label_split[0]))
    """
    WIP

    viz_df_og = copy.deepcopy(viz_df)
    # univariate_label_l = list(pd.Series(x_label_l + hue_label_l + y_label_l_num + x_label_split).drop_duplicates())
    """
    d = split_cols_exceeding_thresh(df=viz_df, thresh=threshold_x_tick_labels_cat, label_l=x_label_split)

    # for j in text_cols_l:
    #     wordcloud_plot(viz_df, wordcloud_path, j)
    for i in hue_label_l:
        counter = 1
        hue_label = i
        hue_label = ''.join(hue_label)

        ## Univariate analysis
        for x_label in univariate_label_l:
            d1 = {}
            dtype = ""
            if x_label in x_label_split:
                for i in range(len(d[x_label].keys())):
                    viz_df = viz_df.loc[viz_df[x_label].isin(list(d[x_label][counter]))]
                    counter += 1
            print("\nx label iteration: " + str(x_label) + "    hue label: " + str(hue_label))
            if (x_label != hue_label) and (mapping[x_label] == "numerical"):
                dtype = "num"
                plot(plot_type="hist", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=univariate_path, dtype=dtype, palette=palette)
                plot(plot_type="kde", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=univariate_path, dtype=dtype, palette=palette)
                plot(plot_type="ecdf", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=univariate_path, dtype=dtype, palette=palette)
            elif (x_label != hue_label) and (mapping[x_label] == "categorical"):
                plot(plot_type="hist", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=univariate_path, dtype=dtype, palette=palette)
            viz_df = copy.deepcopy(viz_df_og)
        print(f"\nUnivariate Analysis completed for {i}\n\nBivariate Analysis starting")

    #     ## Bivariate analysis
    #     for (x_label, y_label) in itertools.product(x_label_l, y_label_l):
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
def gen_viz_workflow(
    df: pd.DataFrame,
    threshold_nan_pct: float,
    threshold_x_tick_labels_cat: int,
    threshold_x_tick_labels_num: int,
    pri_cols: list,
    sec_cols: list,
    num_cols: list,
    datetime_cols: list,
    text_cols: list,
    df_file: str, 
    palette: str,
    mapping: dict,
    sheet_name: str = ""
):
    """
    DF Cleaning 
    Mapping dict generation
    """
    inv_mapping = {}
    df = df_drop(df, threshold_nan_pct)
    df = df_clean_colnames_colvals(df)
    mapping = _create_mapping_dict(df, mapping)
    df = dtype_conversion(df, mapping)

    l = [pri_cols, sec_cols, num_cols, datetime_cols, text_cols]
    names_l = ["pri_cols", "sec_cols", "num_cols", "datetime_cols", "text_cols"]
    for k,v in mapping.items():
        inv_mapping[v] = inv_mapping.get(v, []) + [k]
    for ls, names in zip(l, names_l):
        a = appending_to_cols(ls, names, inv_mapping)
        ls.extend(a)
    pri_cols += datetime_cols
    # print("Mapping:\n"+str(inv_mapping))
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
        dest_path=dest_path)

    """
    Main fn workflow:
        Expanding cat cols 
        Generation of plots
    """
    viz(
        viz_df=df,
        mapping=mapping,
        threshold_x_tick_labels_cat=threshold_x_tick_labels_cat, 
        threshold_x_tick_labels_num=threshold_x_tick_labels_num, 
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
    df_file_path = c.df.replace("/", "\\")
    config = c.config
    ctx = get_ctx(config)
    threshold_nan_pct = convert(ctx.compulsory_inputs.threshold_nan_pct)[0]
    threshold_x_tick_labels_cat = convert(ctx.compulsory_inputs.threshold_x_tick_labels_cat)[0]
    threshold_x_tick_labels_num = convert(ctx.compulsory_inputs.threshold_x_tick_labels_num)[0]
    markersize = convert(ctx.compulsory_inputs.markersize)[0]
    linewidth = convert(ctx.compulsory_inputs.linewidth)[0]
    labelsize = convert(ctx.compulsory_inputs.labelsize)[0]
    palette = convert(ctx.compulsory_inputs.palette)[0]

    sheet_name = convert(ctx.situational_inputs.sheet_name)

    pri_cols = convert(ctx.optional_inputs.pri_cols)
    sec_cols = convert(ctx.optional_inputs.sec_cols)
    num_cols = convert(ctx.optional_inputs.num_cols)
    datetime_cols = convert(ctx.optional_inputs.datetime_cols)
    text_cols = convert(ctx.optional_inputs.text_cols)
    mapping = dict(ctx.optional_inputs.mapping)

    custom_settings(markersize=markersize, linewidth=linewidth, labelsize=labelsize)
    df_file = df_file_path.rsplit(sep="\\")[-1].split(sep=".")[0]
    ext = df_file_path.rsplit(".", 1)[-1]
    """
    Read df based on file type => Main workflow fn starts
    """
    if ext == "xlsx":
            sheet_name = str(sheet_name[0])
            df = df_read(
                df_file_path=df_file_path,
                sheet_name=sheet_name
            )
            gen_viz_workflow(
                df=df,
                threshold_nan_pct=threshold_nan_pct, 
                threshold_x_tick_labels_cat=threshold_x_tick_labels_cat,
                threshold_x_tick_labels_num=threshold_x_tick_labels_num,
                pri_cols=pri_cols,
                sec_cols=sec_cols,
                num_cols=num_cols,
                datetime_cols=datetime_cols,
                text_cols=text_cols,
                df_file=df_file,
                palette=palette,
                mapping=mapping,
                sheet_name=sheet_name
            )
    else:
        df = df_read(
        df_file_path=df_file_path
        )
        gen_viz_workflow(
                df=df,
                threshold_nan_pct=threshold_nan_pct, 
                threshold_x_tick_labels_cat=threshold_x_tick_labels_cat,
                threshold_x_tick_labels_num=threshold_x_tick_labels_num,
                pri_cols=pri_cols,
                sec_cols=sec_cols,
                num_cols=num_cols,
                datetime_cols=datetime_cols,
                text_cols=text_cols,
                df_file=df_file,
                palette=palette,
                mapping=mapping
            )

    end = time.time()
    print("Time Taken: " + str(end-start))
    return True

def infer_col_types(context_engine: C):
    simplefilter(action="ignore")
    c = context_engine.k
    df_file_path = c.df.replace("/", "\\")
    mapping = dict(get_ctx(c.config).optional_inputs.mapping)
    try:
        df = pd.read_csv(df_file_path, encoding="utf8")
    except:
        df = pd.read_csv(df_file_path, encoding="cp1252")
    inferred_mappings = _create_mapping_dict(df=df, mapping=mapping)
    return inferred_mappings