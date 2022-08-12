import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def rotate_axis(fig) -> bool:
    """
    Dynamic rotation of x axis labels
    """
    num_labels = len(fig.get_xticklabels())
    if num_labels > 9:
        return True
    else:
        return False

def save_sns_viz(**kwargs):
    s = kwargs.get("s")
    file_name = kwargs.get("file_name")
    dest_path = kwargs.get("dest_path")
    fig = kwargs.get("fig")
    dtype = kwargs.get("dtype")
    
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
def plot(**kwargs):
    plot_type = kwargs.get("plot_type")
    viz_df = kwargs.get("viz_df")
    x_label = kwargs.get("x_label")
    hue_label = kwargs.get("hue_label")
    dest_path = kwargs.get("dest_path")
    dtype = kwargs.get("dtype")
    y_label = kwargs.get("y_label", None)
    palette = kwargs.get("palette", "bright")

    fn = PLOTS_TO_FUNC[plot_type]
    s = viz_df[x_label]
    if y_label is None:
        fig = fn(data=viz_df, x=x_label, hue=hue_label, palette=palette)
        print(plot_type)
        x_label = str_replacing(x_label)
        hue_label = str_replacing(hue_label)
        file_name = f"{plot_type}plot_of_{x_label}_&_{hue_label}.png"
    else:
        fig = fn(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette=palette)
        print(plot_type)
        x_label = str_replacing(x_label)
        y_label = str_replacing(y_label)
        hue_label = str_replacing(hue_label)
        file_name = f"{plot_type}plot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    
    sns.move_legend(fig, "upper left", bbox_to_anchor=(1.04, 1))
    save_sns_viz(s=s, file_name=file_name, dest_path=dest_path, fig=fig, dtype=dtype)