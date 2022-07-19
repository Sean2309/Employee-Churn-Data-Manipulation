import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys
import os
import datetime
import json
import argparse
import itertools
from tqdm import tqdm
from warnings import simplefilter

# Customised Settings
mpl.rcdefaults()
mpl.rcParams["lines.markersize"] = 2
mpl.rcParams["lines.linewidth"] = 0.5
ax = plt.gca()
ax.set_xlim((0, 55))
simplefilter(action="ignore")

# START OF PLOTTING FUNCTION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def save_viz(file_name):
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90, horizontalalignment="right")
    plt.savefig(os.path.join(dest_path, file_name),
                dpi=300, bbox_inches='tight')

def str_replacing(input_str: str):
    str1 = input_str.replace(" ", "_").replace("/", "_")
    return str1


def hist_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    hist_fig = sns.displot(data=viz_df, x=x_label, y=y_label,
                           kind="hist", hue=hue_label, palette="bright", height=4)
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"HistPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()


def kde_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    kde_fig = sns.displot(data=viz_df, x=x_label, y=y_label,
                          kind="kde", hue=hue_label, palette="bright", height=4)
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"KDEPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()


def bar_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    bar_fig = sns.catplot(data=viz_df, x=x_label, y=y_label,
                          kind="bar", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"BarPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()


def box_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    box_fig = sns.boxplot(data=viz_df, x=x_label, y=y_label,
                          hue=hue_label, palette="bright")
    box_fig = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"BoxPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()


def swarm_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    swarm_fig = sns.swarmplot(data=viz_df, x=x_label,
                              y=y_label, hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"SwarmPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()


def scatter_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    scatter_fig = sns.relplot(
        data=viz_df, x=x_label, y=y_label, kind="scatter", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"ScatterPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()


def line_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    line_fig = sns.relplot(data=viz_df, x=x_label, y=y_label,
                           kind="line", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"LinePlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()


def lm_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    lm_fig = sns.lmplot(data=viz_df, x=x_label, y=y_label,
                        hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"LMPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()


def ecdf_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str):
    ecdf_fig = sns.displot(data=viz_df, x=x_label,
                           kind="ecdf", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    file_name = f"ECDFPlot_of_{x_label}.png"
    save_viz(file_name)
    plt.close()


def viz(viz_df: pd.DataFrame, x_label_list: list, hue_label_l: list, y_label_list: list):
    for i in hue_label_l:
        hue_label = i
        hue_label = ''.join(hue_label)

        for (x_label, y_label) in tqdm(itertools.product(x_label_list, y_label_list)):

            if (mapping[x_label] == "categorical" or mapping[x_label] == "datetime") and (mapping[y_label] == "numerical") and (x_label != hue_label):
                bar_plot(viz_df, x_label, hue_label, y_label)
                box_plot(viz_df, x_label, hue_label, y_label)
                hist_plot(viz_df, x_label, hue_label, y_label)
                scatter_plot(viz_df, x_label, hue_label, y_label)
                swarm_plot(viz_df, x_label, hue_label, y_label)

            if (mapping[x_label] == "numerical" or mapping[x_label] == "datetime") and (mapping[y_label] == "numerical") and (x_label != hue_label):
                kde_plot(viz_df, x_label, hue_label, y_label)
                line_plot(viz_df, x_label, hue_label, y_label)

            if (mapping[x_label] == "numerical" or mapping[x_label] == "datetime") and (mapping[y_label] == "numerical" or mapping[y_label] == "datetime") and (x_label != hue_label):
                lm_plot(viz_df, x_label, hue_label, y_label)

            if (mapping[x_label] == "categorical" or mapping[x_label] == "numerical" or mapping[x_label] == "datetime") and (x_label != hue_label):
                ecdf_plot(viz_df, x_label, hue_label)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# END OF PLOTTING FUNCTION

# START OF EVAL COLS FUNCTION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def eval_cols(input_df_calc: pd.DataFrame, pri_cols_calc: list, sec_col_calc: list, val_col_calc: list, val_col_nocalc: list = []):
    val_out = []
    new_col_name_list = []
    try:
        group_by_col = list(set(pri_cols_calc + sec_col_calc))
        combined_col = group_by_col + val_col_calc + val_col_nocalc
    except Exception as e:
        print(e)
    new_df = pd.DataFrame()
    temp_df = input_df_calc.loc[:, combined_col]
    temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]

    for val in tqdm(val_col_calc):
        unique_vals = list(pd.Series(temp_df[val].unique()).astype("str"))
        new_col_name_list.extend(unique_vals)
        eval_col = pd.get_dummies(temp_df[val]).add_prefix(f"{val}_")
        new_df = pd.concat([new_df, eval_col], axis=1)
    new_df = new_df.add_prefix("Count_of_")
    temp_df = pd.concat([temp_df, new_df], axis=1, join="inner")

    for i in new_df.columns:
        val_out.append(i)
        mapping[i] = "numerical"

    temp_df = temp_df.drop_duplicates(group_by_col)
    val_out += val_col_nocalc
    viz(temp_df, pri_cols_calc, sec_col_calc, val_out)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# END OF EVAL COLS FUNCTION

# START OF INITIALISATION FUNCTION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def initialisation(calc_needed: bool, input_df: pd.DataFrame, df_mapping: dict, pri_cols: list = [], sec_cols: list = [], val_cols_calc: list = [], val_cols_nocalc: list = []):
    input_df = input_df.replace(
        "/", "_", regex=True).replace(" ", "_", regex=True)

    if (pri_cols == []) and (sec_cols == []) and (val_cols_calc == [] and (val_cols_nocalc == [])):
        for keys, values in df_mapping.items():
            try:
                if values == "categorical":
                    pri_cols.append(keys)
                    sec_cols.append(keys)
                    val_cols_calc.append(keys)

                elif values == "numerical" or values == "datetime":
                    val_cols_nocalc.append(keys)

            except Exception as e:
                print(e)

    if calc_needed == True:
        eval_cols(input_df, pri_cols, sec_cols, val_cols_calc, val_cols_nocalc)
    else:
        val_cols_calc += val_cols_nocalc
        viz(input_df, pri_cols, sec_cols, val_cols_calc)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# END OF INITIALISATION FUNCTION

def main(argv):

    # User Input
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs="?", const="config.json",
                        default="config.json", type=str, help="name of config file")
    args, unknown_args = parser.parse_known_args(argv)
    with open(args.config_file) as config_file:
        data = json.load(config_file)
        pri_cols = data["pri_cols"]
        sec_col = data["sec_col"]
        cols_with_eval_fn = data["list_of_val_cols_calc"]
        cols_no_eval_fn = data["list_of_val_cols_nocalc"]
        df_file_path = data["df_file_path"]
        mapping = data["mapping"]
        compute_table = bool(data["bool"])

    # Importing DF
    df = pd.read_csv(df_file_path)

    # Creating Dest Folder
    current_dir = os.getcwd()
    current_datetime = datetime.datetime.now().strftime("%d.%m.%Y_%H%M")
    dest_path = current_dir+'\\'+current_datetime + "_Visualisation"
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    readme_l = ["File Naming Conventions", "\n\n",
                "[Name of Visual] of [x axis] & [y axis] & [legend]"]
    with open(dest_path + "/(1)README.txt", "w") as f:
        f.writelines(readme_l)

    initialisation(compute_table, df, mapping, pri_cols, sec_col, cols_with_eval_fn, cols_no_eval_fn)

if __name__ == '__main__':
    argv = sys.argv 
    argv = argv[1:]
    main(argv)
