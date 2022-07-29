import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from tqdm import tqdm

def save_viz(file_name, dest_path: str):
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90, horizontalalignment="right")
    plt.savefig(os.path.join(dest_path, file_name),
                dpi=300, bbox_inches='tight')
    plt.close()

def str_replacing(input_str: str):
    str1 = input_str.replace(" ", "_").replace("/", "_")
    return str1

def hist_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str, dest_path: str):
    hist_fig = sns.displot(data=viz_df, x=x_label, y=y_label,
                           kind="hist", hue=hue_label, palette="bright", height=4)
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"HistPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def kde_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str, dest_path: str):
    kde_fig = sns.displot(data=viz_df, x=x_label, y=y_label,
                          kind="kde", hue=hue_label, palette="bright", height=4)
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"KDEPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def bar_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str, dest_path: str):
    bar_fig = sns.catplot(data=viz_df, x=x_label, y=y_label,
                          kind="bar", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"BarPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def box_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str = []):
    if y_label:
        box_fig = sns.boxplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
        x_label = str_replacing(x_label)
        y_label = str_replacing(y_label)
        hue_label = str_replacing(hue_label)
        file_name = f"BoxPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    else:
        box_fig = sns.boxplot(data=viz_df, x=x_label, hue=hue_label, palette="bright")
        x_label = str_replacing(x_label)
        y_label = str_replacing(y_label)
        hue_label = str_replacing(hue_label)
        file_name = f"BoxPlot_of_{x_label}_&_{hue_label}.png"
    box_fig = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    save_viz(file_name, dest_path)


def swarm_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str, dest_path: str):
    swarm_fig = sns.swarmplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    swarm_fig = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"SwarmPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def scatter_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str, dest_path: str):
    scatter_fig = sns.relplot(data=viz_df, x=x_label, y=y_label, kind="scatter", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"ScatterPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def line_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str, dest_path: str):
    line_fig = sns.relplot(data=viz_df, x=x_label, y=y_label,
                           kind="line", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"LinePlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def lm_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str, dest_path: str):
    lm_fig = sns.lmplot(data=viz_df, x=x_label, y=y_label,
                        hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"LMPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def ecdf_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str):
    ecdf_fig = sns.displot(data=viz_df, x=x_label,
                           kind="ecdf", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    file_name = f"ECDFPlot_of_{x_label}.png"
    save_viz(file_name, dest_path)


def viz(viz_df: pd.DataFrame, mapping: dict, x_label_list: list, hue_label_l: list, y_label_list: list, dest_path: str):
    for i in tqdm(hue_label_l):
        hue_label = i
        hue_label = ''.join(hue_label)

        for (x_label, y_label) in tqdm(itertools.product(x_label_list, y_label_list)):

            if (mapping[x_label] in ["numerical", "datetime"]) and (x_label != hue_label):
                box_plot(viz_df, x_label, hue_label, dest_path)

            if (mapping[x_label] in ["categorical", "datetime"]) and (mapping[y_label] == "numerical") and (x_label != hue_label):
                bar_plot(viz_df, x_label, hue_label, y_label, dest_path)
                box_plot(viz_df, x_label, hue_label, dest_path, y_label)
                hist_plot(viz_df, x_label, hue_label, y_label, dest_path)
                scatter_plot(viz_df, x_label, hue_label, y_label, dest_path)
                swarm_plot(viz_df, x_label, hue_label, y_label, dest_path)

            if (mapping[x_label] == "numerical" or mapping[x_label] == "datetime") and (mapping[y_label] == "numerical") and (x_label != hue_label):
                kde_plot(viz_df, x_label, hue_label, y_label, dest_path)   
                line_plot(viz_df, x_label, hue_label, y_label, dest_path)

            if (mapping[x_label] == "numerical" or mapping[x_label] == "datetime") and (mapping[y_label] == "numerical" or mapping[y_label] == "datetime") and (x_label != hue_label):
                lm_plot(viz_df, x_label, hue_label, y_label, dest_path)

            if (mapping[x_label] == "categorical" or mapping[x_label] == "numerical" or mapping[x_label] == "datetime") and (x_label != hue_label):
                ecdf_plot(viz_df, x_label, hue_label, dest_path)