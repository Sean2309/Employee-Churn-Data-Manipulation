import os
from turtle import width
from typing import Mapping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import copy
from sean_preproc import dropping_unncessary_plots_label, dropping_unnecessary_plots_list
# from wordcloud import WordCloud
# from wordcloud import ImageColorGenerator
# from wordcloud import STOPWORDS
# from tqdm import tqdm

def save_sns_viz(file_name, dest_path: str):
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

# def wordcloud_plot(viz_df: pd.DataFrame, dest_path: str, sw: set, text_label: str = []):
#     text = " ".join(i for i in viz_df[text_label])
#     wordcloud = WordCloud(stopwords=sw, background_color="white", width=1600, height=800).generate(text)
#     text_label = str_replacing(text_label)
#     file_name = f"WordCloud_of_{text_label}.png"
#     plt.figure( figsize=(19.1,10), facecolor="k")
#     plt.imshow(wordcloud)
#     plt.axis("off")
#     plt.savefig(os.path.join(dest_path, file_name), facecolor="k", bbox_inches="tight")
#     plt.close()

def pie_plot(viz_df: pd.DataFrame, x_label: str, y_label: str, dest_path: str, mapping: dict):
    # if dropping_unncessary_plots_label(viz_df, y_label, "numerical", mapping):
    print(x_label, y_label)
    pie_fig = viz_df.groupby([x_label]).sum().plot(kind="pie", y=y_label, autopct="%1.0f%%", title=f"{y_label} by {x_label}")
    #sns.move_legend(pie_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    file_name = f"PiePlot_of_{x_label}_&_{y_label}.png"
    plt.savefig(os.path.join(dest_path, file_name), bbox_inches="tight")
    plt.close()

def hist_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str = []):
    print("Histogram")
    if y_label:
        hist_fig = sns.histplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright", bins=range(50))
        sns.move_legend(hist_fig, "upper left", bbox_to_anchor=(1.04, 1))
        x_label = str_replacing(x_label)
        y_label = str_replacing(y_label)
        hue_label = str_replacing(hue_label)
        file_name = f"HistPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    else:
        hist_fig = sns.histplot(data=viz_df, x=x_label, hue=hue_label, palette="bright", bins=range(50))
        sns.move_legend(hist_fig, "upper left", bbox_to_anchor=(1.04, 1))
        #print("Number of x axis tick labels: "+str(len(hist_fig.get_xticklabels())))
        x_label = str_replacing(x_label)
        hue_label = str_replacing(hue_label)
        file_name = f"HistPlot_of_{x_label}_&_{hue_label}.png"
    save_sns_viz(file_name, dest_path)


def kde_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str = []):
    print("KDE")
    kde_fig = sns.displot(data=viz_df, x=x_label, kind="kde", hue=hue_label, palette="bright", height=4)
    sns.move_legend(kde_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    hue_label = str_replacing(hue_label)
    file_name = f"KDEPlot_of_{x_label}_&_{hue_label}.png"
    save_sns_viz(file_name, dest_path)


def bar_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str):
    print("Bar")
    bar_fig = sns.barplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(bar_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"BarPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_sns_viz(file_name, dest_path)


def box_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str):
    print("Box")
    box_fig = sns.boxplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(box_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"BoxPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_sns_viz(file_name, dest_path)


def violin_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str):
    print("Violin")
    violin_fig = sns.violinplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(violin_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"ViolinPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_sns_viz(file_name, dest_path)


def scatter_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str):
    print("Scatter")
    scatter_fig = sns.scatterplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(scatter_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"ScatterPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_sns_viz(file_name, dest_path)


def line_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str):
    print("Line")
    line_fig = sns.relplot(data=viz_df, x=x_label, y=y_label, kind="line", hue=hue_label, palette="bright")
    sns.move_legend(line_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"LinePlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_sns_viz(file_name, dest_path)


def lm_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, y_label: str):
    print("LM")
    lm_fig = sns.lmplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(lm_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"LMPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_sns_viz(file_name, dest_path)


def ecdf_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str):
    print("ECDF")
    ecdf_fig = sns.displot(data=viz_df, x=x_label, kind="ecdf", hue=hue_label, palette="bright")
    sns.move_legend(ecdf_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    hue_label = str_replacing(hue_label)
    file_name = f"ECDFPlot_of_{x_label}_&_{hue_label}.png"
    save_sns_viz(file_name, dest_path)


def viz(viz_df: pd.DataFrame, mapping: dict, x_label_list: list, hue_label_l: list, y_label_list: list, text_cols: list, dest_path: str):
    univariate_label_l = list(pd.Series(x_label_list + y_label_list + hue_label_l).drop_duplicates())
    print(viz_df)
    pie_y_list = copy.deepcopy(y_label_list)
    univariate_path = dest_path + "\\Univariate_Plots"
    bivariate_path = dest_path + "\\Bivariate_Plots"
    wordcloud_path = dest_path + "\\WordCloud_Plots"
    pie_x_list = copy.deepcopy(x_label_list)
    
    x_label_list = dropping_unnecessary_plots_list(viz_df, x_label_list, "categorical", mapping)
    #print(pie_x_list)
    pie_x_list = dropping_unnecessary_plots_list(viz_df, pie_x_list, "numerical", mapping)

    # sw = set(STOPWORDS)
    # for j in text_cols:
    #     wordcloud_plot(viz_df, wordcloud_path, sw, j)

    # for (x_label, y_label) in itertools.product(pie_x_list, y_label_list):
    #     if (mapping[x_label] == "categorical") and (mapping[y_label] == "numerical"):
    #         pie_plot(viz_df, x_label, y_label, bivariate_path, mapping)
    # for i in hue_label_l:
    #     hue_label = i
    #     hue_label = ''.join(hue_label)
        ## Univariate analysis
        # for x_label in univariate_label_l:
        #     #print("\nx label iteration: " + str(x_label) + "    hue label: " + str(hue_label))
        #     if (x_label != hue_label) and (mapping[x_label] == "numerical"):
        #         hist_plot(viz_df, x_label, hue_label, univariate_path)
        #         kde_plot(viz_df, x_label, hue_label, univariate_path)
        #         ecdf_plot(viz_df, x_label, hue_label, univariate_path)
        #     elif (x_label != hue_label) and (mapping[x_label] == "categorical"):
        #         hist_plot(viz_df, x_label, hue_label, univariate_path)
        
        # print("\nUnivariate Analysis completed\n\nBivariate Analysis starting")
        ## Bivariate analysis
        # for (x_label, y_label) in itertools.product(x_label_list, y_label_list):
            #print("\nx label iteration: " + str(x_label) + "    y label iteration: " + str(y_label) + "    hue label: " + str(hue_label))

            # if (x_label != hue_label) and (mapping[x_label] == "categorical") and (mapping[y_label] == "numerical"):
            #     pie_plot(viz_df, x_label, y_label, bivariate_path, mapping)
                # box_plot(viz_df, x_label, hue_label, bivariate_path, y_label)
                # bar_plot(viz_df, x_label, hue_label, bivariate_path, y_label)
                # violin_plot(viz_df, x_label, hue_label, bivariate_path, y_label)

        #     elif (x_label != hue_label) and (mapping[x_label] == "numerical") and (mapping[y_label] == "numerical"):
        #         hist_plot(viz_df, x_label, hue_label, bivariate_path, y_label)
        #         scatter_plot(viz_df, x_label, hue_label, bivariate_path, y_label)
        #         line_plot(viz_df, x_label, hue_label, bivariate_path, y_label)
        #         lm_plot(viz_df, x_label, hue_label, bivariate_path, y_label)

        #     elif (x_label != hue_label) and (mapping[x_label] == "datetime") and (mapping[y_label] == "numerical"):
        #         line_plot(viz_df, x_label, hue_label, bivariate_path, y_label)
        #         lm_plot(viz_df, x_label, hue_label, bivariate_path, y_label)
        # print("\nBivariate Analysis completed")