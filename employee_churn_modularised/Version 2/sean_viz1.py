from cProfile import label
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from tqdm import tqdm

def save_viz(file_name, dest_path: str):
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45, horizontalalignment="right")
    
    plt.savefig(os.path.join(dest_path, file_name),
                dpi=300, bbox_inches='tight')
    plt.close()

def str_replacing(input_str: str):
    str1 = input_str.replace(" ", "_").replace("/", "_")
    return str1

def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)
    
def pie_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str = []):
    data = viz_df.groupby()
    palette_colour = sns.color_palette("bright")
    pie_fig = plt.pie()

def hist_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str = []):
    if y_label:
        hist_fig = sns.histplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
        sns.move_legend(hist_fig, "upper left", bbox_to_anchor=(1.04, 1))
        x_label = str_replacing(x_label)
        y_label = str_replacing(y_label)
        hue_label = str_replacing(hue_label)
        file_name = f"HistPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    else:
        hist_fig = sns.histplot(data=viz_df, x=x_label, hue=hue_label, palette="bright")
        sns.move_legend(hist_fig, "upper left", bbox_to_anchor=(1.04, 1))
        # plt.legend(title=hue_label, bbox_to_anchor=(1.04, 1), loc="upper left")
        print("Number of x axis tick labels: "+str(len(hist_fig.get_xticklabels())))
        x_label = str_replacing(x_label)
        hue_label = str_replacing(hue_label)
        file_name = f"HistPlot_of_{x_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def kde_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str = []):
    kde_fig = sns.displot(data=viz_df, x=x_label, kind="kde", hue=hue_label, palette="bright", height=4)
    sns.move_legend(kde_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    hue_label = str_replacing(hue_label)
    file_name = f"KDEPlot_of_{x_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def bar_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str):
    print("bar")
    print(x_label, y_label, hue_label)
    bar_fig = sns.barplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(bar_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"BarPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def box_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str):
    print("box")
    box_fig = sns.boxplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(box_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"BoxPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def violin_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str):
    violin_fig = sns.violinplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(violin_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"ViolinPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def scatter_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str):
    scatter_fig = sns.scatterplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(scatter_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"ScatterPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def line_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str):
    print("line")
    line_fig = sns.relplot(data=viz_df, x=x_label, y=y_label, kind="line", hue=hue_label, palette="bright")
    sns.move_legend(line_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"LinePlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def lm_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str):
    print("lm")
    lm_fig = sns.lmplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(lm_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"LMPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def ecdf_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str):
    print("ecdf")
    ecdf_fig = sns.displot(data=viz_df, x=x_label, kind="ecdf", hue=hue_label, palette="bright")
    sns.move_legend(ecdf_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    hue_label = str_replacing(hue_label)
    file_name = f"ECDFPlot_of_{x_label}_&_{hue_label}.png"
    save_viz(file_name, dest_path)


def viz(viz_df: pd.DataFrame, mapping: dict, x_label_list: list, hue_label_l: list, y_label_list: list, dest_path: str):
    univariate_label_l = list(pd.Series(x_label_list + y_label_list + hue_label_l).drop_duplicates())
    univariate_path = dest_path + "\\Univariate_Plots"
    bivariate_path = dest_path + "\\Bivariate_Plots"
    for i in hue_label_l:
        hue_label = i
        hue_label = ''.join(hue_label)

        ## Univariate analysis
        for x_label in tqdm(univariate_label_l):

            if (x_label != hue_label) and (mapping[x_label] == "numerical"):
                hist_plot(viz_df, x_label, hue_label, univariate_path)
                print("NUMMMM")
                kde_plot(viz_df, x_label, hue_label, univariate_path)
                ecdf_plot(viz_df, x_label, hue_label, univariate_path)
            elif (x_label != hue_label) and (mapping[x_label] == "categorical"):
                print("CATTT")
                hist_plot(viz_df, x_label, hue_label, univariate_path)

        # Bivariate analysis
        for (x_label, y_label) in tqdm(itertools.product(x_label_list, y_label_list)):
        #for (x_label, y_label) in itertools.product(x_label_list, y_label_list):
            if (x_label != hue_label) and (mapping[x_label] == "categorical") and (mapping[y_label] == "numerical"):
                print("\ncategorical")
                box_plot(viz_df, x_label, hue_label, bivariate_path, y_label)
                bar_plot(viz_df, x_label, hue_label, bivariate_path, y_label)
                violin_plot(viz_df, x_label, hue_label, bivariate_path, y_label)

            elif (x_label != hue_label) and (mapping[x_label] == "numerical") and (mapping[y_label] == "numerical"):
                print("numerical")
                hist_plot(viz_df, x_label, hue_label, bivariate_path, y_label)
                scatter_plot(viz_df, x_label, hue_label, bivariate_path, y_label)
                line_plot(viz_df, x_label, hue_label, bivariate_path, y_label)
                lm_plot(viz_df, x_label, hue_label, bivariate_path, y_label)

            elif (x_label != hue_label) and (mapping[x_label] == "datetime") and (mapping[y_label] == "numerical"):
                print("datetime")
                line_plot(viz_df, x_label, hue_label, bivariate_path, y_label)
                lm_plot(viz_df, x_label, hue_label, bivariate_path, y_label)