import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import argparse
from collections import defaultdict
from warnings import simplefilter

from pyparsing import Regex
from sean_pandas import expand_categorical_cols
from sean_viz1 import viz
from sean_preproc import df_replace, get_ctx, convert, create_mapping_dict, create_dest_folder, appending_to_cols_from_dict

# Customised Settings
mpl.rcdefaults()
mpl.rcParams["lines.markersize"] = 2
mpl.rcParams["lines.linewidth"] = 0.5
ax = plt.gca()
ax.set_xlim((0, 55))
simplefilter(action="ignore")

# START OF INITIALISATION FUNCTION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def initialisation(calc_needed: bool, input_df: pd.DataFrame, df_mapping: dict, pri_cols: list = [], sec_cols: list = [], cat_cols_to_expand: list = [], cat_cols_no_expand: list = [], text_cols: list = [],  dest_path: str = ""):

    if calc_needed == True:
        temp_df, mapping, pri_cols, sec_cols, val_out, text_cols = expand_categorical_cols(input_df, df_mapping, pri_cols, sec_cols, cat_cols_to_expand, text_cols, cat_cols_no_expand)
        
        # viz(temp_df, mapping, pri_cols, sec_cols, val_out, text_cols, dest_path)
    else:
        cat_cols_to_expand += cat_cols_no_expand
        viz(input_df, df_mapping, pri_cols, sec_cols, cat_cols_to_expand, text_cols, dest_path)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# END OF INITIALISATION FUNCTION

def main(argv):

    # User Input
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs="?", default="sean_config_employeeattr", type=str, help="name of config file")

    # Init Vars
    args, unknown_args = parser.parse_known_args(argv)
    c = get_ctx(args.config_file)
    compute_table = convert(c.compute_table)[0]
    df_file_path = str(convert(c.df_file_path)[0]).replace("/", "\\")
    pri_cols = convert(c.pri_cols)
    sec_cols = convert(c.sec_cols)
    cat_cols_to_expand = convert(c.cat_cols_to_expand)
    num_cols = convert(c.num_cols)
    datetime_cols = convert(c.datetime_cols)
    text_cols = convert(c.text_cols)
    mapping = {}
    inv_mapping = {}
    # i dont have a way to change if the sep is "/" as of now
    df_file = df_file_path.rsplit(sep="\\")[-1].split(sep=".")[0]
    l = [pri_cols, sec_cols, cat_cols_to_expand, num_cols, datetime_cols, text_cols]
    names_l = ["pri_cols", "sec_cols", "cat_cols_to_expand", "num_cols", "datetime_cols", "text_cols"]

    # Pre processing
    try:
        df = pd.read_csv(df_file_path, encoding="utf8")
    except:
        df = pd.read_csv(df_file_path, encoding="cp1252")
    df = df_replace(df)
    mapping = create_mapping_dict(c, df)
    for k,v in mapping.items():
        inv_mapping[v] = inv_mapping.get(v, []) + [k]
    for ls, names in zip(l, names_l):
        a = appending_to_cols_from_dict(ls, names, inv_mapping)
        ls.extend(a)
    pri_cols += datetime_cols

    # Calling init fn
    dest_path = create_dest_folder(df_file, pri_cols, sec_cols, cat_cols_to_expand, num_cols, datetime_cols, text_cols)
    initialisation(compute_table, df, mapping, pri_cols, sec_cols, cat_cols_to_expand, num_cols, text_cols, dest_path)

if __name__ == '__main__':
    argv = sys.argv
    argv = argv[1:]
    main(argv)