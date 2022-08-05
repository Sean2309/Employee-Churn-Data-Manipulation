import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sean_processing import set_x_tick_limit, rotate_axis
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
from tqdm import tqdm

# ----------------------------------------------------
# misc functions
# ----------------------------------------------------

def save_sns_viz(s: pd.Series, file_name: str, dest_path: str, fig, dtype: str):
    locs, labels = plt.xticks()
    if rotate_axis(fig):
        plt.setp(labels, rotation=90, horizontalalignment="right")
    else:
        plt.setp(labels, horizontalalignment="right")
    if dtype == "num":
        min = np.min(s)
        max = np.max(s)
        dev = np.std(s)
        fig = fig.set(xlim=(min-dev, max+dev))
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

# ----------------------------------------------------
# plotting functions
# ----------------------------------------------------

def wordcloud_plot(viz_df: pd.DataFrame, dest_path: str, sw: set, text_label: str = []):
    text = " ".join(i for i in viz_df[text_label])
    wordcloud = WordCloud(stopwords=sw, background_color="white", width=1600, height=800).generate(text)
    text_label = str_replacing(text_label)
    file_name = f"WordCloud_of_{text_label}.png"
    plt.figure( figsize=(19.1,10), facecolor="k")
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(os.path.join(dest_path, file_name), facecolor="k", bbox_inches="tight")
    plt.close()

def hist_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, dtype: str, y_label: str = []):
    print("Histogram")
    s = viz_df[x_label]
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
        x_label = str_replacing(x_label)
        hue_label = str_replacing(hue_label)
        file_name = f"HistPlot_of_{x_label}_&_{hue_label}.png"
    save_sns_viz(s, file_name, dest_path, hist_fig, dtype)

def kde_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, dtype: str):
    print("KDE")
    s = viz_df[x_label]
    kde_fig = sns.kdeplot(data=viz_df, x=x_label, hue=hue_label, palette="bright")
    sns.move_legend(kde_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    hue_label = str_replacing(hue_label)
    file_name = f"KDEPlot_of_{x_label}_&_{hue_label}.png"
    save_sns_viz(s, file_name, dest_path, kde_fig, dtype)

def bar_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, dtype: str, y_label: str):
    print("Bar")
    s = viz_df[x_label]
    bar_fig = sns.barplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(bar_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"BarPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_sns_viz(s, file_name, dest_path, bar_fig, dtype)

def box_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, dtype: str, y_label: str):
    print("Box")
    s = viz_df[x_label]
    box_fig = sns.boxplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(box_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"BoxPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_sns_viz(s, file_name, dest_path, box_fig, dtype)

def violin_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, dtype: str, y_label: str):
    print("Violin")
    s = viz_df[x_label]
    violin_fig = sns.violinplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(violin_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"ViolinPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_sns_viz(s, file_name, dest_path, violin_fig, dtype)

def scatter_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, dtype: str, y_label: str):
    print("Scatter")
    s = viz_df[x_label]
    scatter_fig = sns.scatterplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(scatter_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"ScatterPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_sns_viz(s, file_name, dest_path, scatter_fig, dtype)

def line_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, dtype: str, y_label: str):
    print("Line")
    s = viz_df[x_label]
    line_fig = sns.lineplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    sns.move_legend(line_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"LinePlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_sns_viz(s, file_name, dest_path, line_fig, dtype)

def ecdf_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, dest_path: str, dtype: str = ""):
    print("ECDF")
    s = viz_df[x_label]
    ecdf_fig = sns.ecdfplot(data=viz_df, x=x_label, hue=hue_label, palette="bright")
    sns.move_legend(ecdf_fig, "upper left", bbox_to_anchor=(1.04, 1))
    x_label = str_replacing(x_label)
    hue_label = str_replacing(hue_label)
    file_name = f"ECDFPlot_of_{x_label}_&_{hue_label}.png"
    save_sns_viz(s, file_name, dest_path, ecdf_fig, dtype)

# ----------------------------------------------------
# main viz function
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

    sw = set(STOPWORDS)
    for j in text_cols:
        wordcloud_plot(viz_df, wordcloud_path, sw, j)

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