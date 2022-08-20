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

def rotate_axis(
    fig,
    plot_type: str
) -> bool:
    """
    Dynamic rotation of x axis labels
    """
    num_labels = len(fig.get_xticklabels())
    if plot_type == "num":
        if num_labels > 15:
            return True
        else:
            return False
    elif num_labels > 6:
        return True
    else:
        return False

def save_sns_viz(
    s: pd.Series,
    file_name: str,
    dest_path: str,
    fig,
    dtype: str,
    plot_type: str
):
    locs, labels = plt.xticks()
    if rotate_axis(fig, plot_type):
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

def str_replacing(input_str: str) -> str:
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

def plot(
    plot_type: str,
    viz_df: pd.DataFrame,
    x_label: str,
    hue_label: str,
    dest_path: str,
    dtype: str,
    iteration: int,
    y_label: str = None,
    palette: str = "bright",
):
    fn = PLOTS_TO_FUNC[plot_type]
    s = viz_df[x_label]
    if y_label is None:
        fig = fn(data=viz_df, x=x_label, hue=hue_label, palette=palette)
        print(plot_type)
        x_label = str_replacing(x_label)
        hue_label = str_replacing(hue_label)
        if iteration is None:
            file_name = f"{plot_type}plot_of_{x_label}_&_{hue_label}.png"
        else:
            file_name = f"{plot_type}plot_of_{x_label}_&_{hue_label}_{iteration}.png"
    else:
        fig = fn(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette=palette)
        print(plot_type)
        x_label = str_replacing(x_label)
        y_label = str_replacing(y_label)
        hue_label = str_replacing(hue_label)
        if iteration is None:
            file_name = f"{plot_type}plot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
        else:
            file_name = f"{plot_type}plot_of_{x_label}_&_{y_label}_&_{hue_label}_{iteration}.png"
    
    sns.move_legend(fig, "upper left", bbox_to_anchor=(1.04, 1))
    save_sns_viz(s=s, file_name=file_name, dest_path=dest_path, fig=fig, dtype=dtype, plot_type=plot_type)

def plot_all_relevant(
    viz_df: pd.DataFrame,
    mapping: dict, 
    x_label: str, 
    hue_label: str, 
    dest_path: str, 
    dtype: str, 
    palette: str,
    y_label: str = None, 
    iteration: int = None
):  
    """
    Generalised function for each iteration of viz fn
    """
    if y_label is None:
        print("\nx label iteration: " + str(x_label) + "    hue label: " + str(hue_label))
        if (x_label != hue_label) and (mapping[x_label] == "numerical"):
            dtype = "num"
            plot(plot_type="hist", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=dest_path, dtype=dtype, palette=palette, iteration=iteration)
            plot(plot_type="kde", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=dest_path, dtype=dtype, palette=palette, iteration=iteration)
            plot(plot_type="ecdf", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=dest_path, dtype=dtype, palette=palette, iteration=iteration)
        elif (x_label != hue_label) and (mapping[x_label] == "categorical"):
            plot(plot_type="hist", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=dest_path, dtype=dtype, palette=palette, iteration=iteration)
        

    else:
        print("\nx label iteration: " + str(x_label) + "    y label iteration: " + str(y_label) + "    hue label: " + str(hue_label))
        if (x_label != hue_label) and (mapping[x_label] == "categorical") and (mapping[y_label] == "numerical"):
                plot(plot_type="box", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=dest_path, dtype=dtype, y_label=y_label, palette=palette, iteration=iteration)
                plot(plot_type="bar", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=dest_path, dtype=dtype, y_label=y_label, palette=palette, iteration=iteration)
                plot(plot_type="violin", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=dest_path, dtype=dtype, y_label=y_label, palette=palette, iteration=iteration)

        elif (x_label != hue_label) and (mapping[x_label] == "numerical") and (mapping[y_label] == "numerical"):
            dtype = "num"
            plot(plot_type="hist", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=dest_path, dtype=dtype, y_label=y_label, palette=palette, iteration=iteration)
            plot(plot_type="scatter", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=dest_path, dtype=dtype, y_label=y_label, palette=palette, iteration=iteration)
            plot(plot_type="line", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=dest_path, dtype=dtype, y_label=y_label, palette=palette, iteration=iteration)

        elif (x_label != hue_label) and (mapping[x_label] == "datetime") and (mapping[y_label] == "numerical"):
            plot(plot_type="line", viz_df=viz_df, x_label=x_label, hue_label=hue_label, dest_path=dest_path, dtype=dtype, y_label=y_label, palette=palette, iteration=iteration)
