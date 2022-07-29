import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
import datetime
import argparse
from warnings import simplefilter
from sean_test_read import get_ctx
from sean_test_pandas import expand_categorical_cols
from sean_test_viz import viz

# Customised Settings
mpl.rcdefaults()
mpl.rcParams["lines.markersize"] = 2
mpl.rcParams["lines.linewidth"] = 0.5
ax = plt.gca()
ax.set_xlim((0, 55))
simplefilter(action="ignore")

# START OF INITIALISATION FUNCTION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def initialisation(calc_needed: bool, input_df: pd.DataFrame, df_mapping: dict, pri_cols: list = [], sec_cols: list = [], cat_cols_to_expand: list = [], cat_cols_no_expand: list = [], dest_path: str = ""):

    if calc_needed == True:
        temp_df, mapping, pri_cols, sec_cols, val_out = expand_categorical_cols(input_df, df_mapping, pri_cols, sec_cols, cat_cols_to_expand, cat_cols_no_expand)

        viz(temp_df, mapping,  pri_cols, sec_cols, val_out, dest_path)
    else:
        cat_cols_to_expand += cat_cols_no_expand
        viz(input_df, df_mapping, pri_cols, sec_cols, cat_cols_to_expand)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# END OF INITIALISATION FUNCTION

def convert(a):
    return list(dict(a).keys())

def appending_to_cols(l,name_of_l, c):
    if not l:
        if (name_of_l == "pri_cols") or (name_of_l == "sec_cols") or (name_of_l == "cat_cols_to_expand"):
            l = convert(c.mapping.categorical)
        elif (name_of_l == "cat_cols_no_expand"):
            l = convert(c.mapping.numerical)
            l.extend(convert(c.mapping.datetime))
        return l
    else:
        return []

def main(argv):

    # User Input
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs="?", const="sean_test_config",
                        default="sean_test_config", type=str, help="name of config file")
    args, unknown_args = parser.parse_known_args(argv)
    c = get_ctx()
    compute_table = convert(c.compute_table)[0]
    df_file_path = convert(c.df_file_path)[0]
    pri_cols = convert(c.pri_cols)
    sec_cols = convert(c.sec_cols)
    cat_cols_to_expand = convert(c.cat_cols_to_expand)
    cat_cols_no_expand = convert(c.cat_cols_no_expand)
    mapping = {}
    
    l = [pri_cols, sec_cols, cat_cols_to_expand, cat_cols_no_expand]
    names_l = ["pri_cols", "sec_cols", "cat_cols_to_expand", "cat_cols_no_expand"]
    for ls, names in zip(l, names_l):
        a = appending_to_cols(ls, names, c)
        ls.extend(a)

    for k, v in dict(c.mapping).items():
        for i in list(v.keys()):
            mapping[i] = k

    # Importing DF
    df = pd.read_csv(df_file_path)
    df = df.replace("/", "_", regex=True).replace(" ", "_", regex=True)

    # Creating Dest Folder
    
    current_dir = os.getcwd()
    current_datetime = datetime.datetime.now().strftime("%d.%m.%Y_%H%M")
    dest_path = current_dir+'\\'+current_datetime + "_Visualisation"
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    readme_l = ["File Naming Conventions", "\n\n", "[Name of Visual] of [x axis] & [y axis] & [legend]"]
    with open(dest_path + "/(1)README.txt", "w") as f:
        f.writelines(readme_l)

    initialisation(compute_table, df, mapping, pri_cols, sec_cols, cat_cols_to_expand, cat_cols_no_expand, dest_path)

if __name__ == '__main__':
    argv = sys.argv 
    argv = argv[1:]
    main(argv)
