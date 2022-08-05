import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import time
from warnings import simplefilter

from sean_pandas import expand_categorical_cols
from sean_viz import viz
from sean_processing import arg_parse, create_readme, df_replace, dtype_conversion, create_mapping_dict, create_dest_folder, appending_to_cols, df_drop

# Customised Settings
start = time.time()
mpl.rcdefaults()
mpl.rcParams["lines.markersize"] = 2
mpl.rcParams["lines.linewidth"] = 0.5
mpl.rcParams["xtick.labelsize"] = 6.5
ax = plt.gca()
ax.set_xlim((0, 55))
simplefilter(action="ignore")

# ----------------------------------------------------
# init functions
# ----------------------------------------------------

def initialisation(calc_needed: bool, input_df: pd.DataFrame, df_mapping: dict, threshold_x_tick_labels_cat: int, threshold_x_tick_labels_num: int, pri_cols: list = [], sec_cols: list = [], cat_cols_to_expand: list = [], cat_cols_no_expand: list = [], text_cols: list = [],  dest_path: str = ""):

    if calc_needed == True:
        temp_df, mapping, expanded_cat_cols, cat_cols_no_expand = expand_categorical_cols(input_df, df_mapping, pri_cols, sec_cols, cat_cols_to_expand, text_cols, cat_cols_no_expand)
        viz(temp_df, mapping, threshold_x_tick_labels_cat, threshold_x_tick_labels_num, pri_cols, sec_cols, expanded_cat_cols, cat_cols_no_expand, text_cols, dest_path)
        end = time.time()
        print("Time Taken: " + str(end-start))
    else:
        viz(input_df, df_mapping, threshold_x_tick_labels_cat, threshold_x_tick_labels_num, pri_cols, sec_cols, cat_cols_to_expand, cat_cols_no_expand, text_cols, dest_path)
        end = time.time()
        print("Time Taken: " + str(end-start))

def main(argv):
    ctx, compute_table, threshold_nan_pct, threshold_x_tick_labels_cat, threshold_x_tick_labels_num, df_file_path, pri_cols, sec_cols, cat_cols_to_expand, num_cols, datetime_cols, text_cols = arg_parse(argv)
    mapping = {}
    inv_mapping = {}
    df_file = df_file_path.rsplit(sep="\\")[-1].split(sep=".")[0]
    l = [pri_cols, sec_cols, cat_cols_to_expand, num_cols, datetime_cols, text_cols]
    names_l = ["pri_cols", "sec_cols", "cat_cols_to_expand", "num_cols", "datetime_cols", "text_cols"]

    try:
        df = pd.read_csv(df_file_path, encoding="utf8")
    except:
        df = pd.read_csv(df_file_path, encoding="cp1252")
    df = df_drop(df, threshold_nan_pct)
    mapping = create_mapping_dict(ctx, df)
    df = dtype_conversion(df, mapping)
    df = df_replace(df)
    
    for k,v in mapping.items():
        inv_mapping[v] = inv_mapping.get(v, []) + [k]
    for ls, names in zip(l, names_l):
        a = appending_to_cols(ls, names, inv_mapping)
        ls.extend(a)
    pri_cols += datetime_cols
    print(inv_mapping)
    dest_path = create_dest_folder(df_file)
    create_readme(dest_path, pri_cols, sec_cols, cat_cols_to_expand, num_cols, datetime_cols, text_cols)
    initialisation(compute_table, df, mapping, threshold_x_tick_labels_cat,threshold_x_tick_labels_num, pri_cols, sec_cols, cat_cols_to_expand, num_cols, text_cols, dest_path)

if __name__ == '__main__':
    argv = sys.argv
    argv = argv[1:]
    main(argv)