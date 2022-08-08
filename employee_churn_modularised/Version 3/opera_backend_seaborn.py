import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import opera_util_common
# ----------------------------------------------------
# misc functions
# ----------------------------------------------------


PLOTS_TO_FUNC = C(
    hist=sns.histplot,
    kde=sns.kdeplot,
    bar=sns.barplot,
    box=sns.boxplot,
    violin=sns.violinplot,
    scatter=sns.scatterplot,
    line=sns.lineplot,
    ecdf=sns.ecdfplot
)

def plot(plot_type: str, viz_df: pd.DataFrame,
         x_label: str, hue_label: str,
         dest_path: str, dtype: str,
         y_label: str=None, palette: str="bright"):
    fn = PLOTS_TO_FUNC[plot_type]
    fig = fn(data=viz_df, x=x_label, hue=hue_label, palette=palette)
    ...


def rotate_axis(fig) -> bool:
    """
    Dynamic rotation of x axis labels
    """
    num_labels = len(fig.get_xticklabels())
    if num_labels > 9:
        return True
    else:
        return False

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