import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import copy
import os
import datetime
import itertools
from warnings import simplefilter
import time

## User Input
combined_col = []
pri_cols = ["Division", "Location"]
sec_col = ["Division"]
list_of_val_cols_calc = ["Department", "Status"]
list_of_val_cols_nocalc = ["Service Years", "Support from Company", "Personal Development (satisfaction)", "Support from Company", "Job Satisfaction", "Avg Extra time hours/week(6 mths)", "Avg Annual leave days taken/month (6 mths)"]
mapping = {
    "MSSubClass": "categorical",
    "MSZoning": "categorical",
    "OverallQual": "numerical",
    "OverallCond": "numerical",
    "YearBuilt": "datetime",
    "MiscVal": "numerical"
}

## Customised Settings
mpl.rcdefaults()
mpl.rcParams["lines.markersize"] = 2
mpl.rcParams["lines.linewidth"] = 0.5
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

## Importing DF
start = time.time()
df = pd.read_csv('.\datasets\house prices/test.csv')

## Creating Dest Folder
current_dir = os.getcwd()
current_datetime = datetime.datetime.now().strftime("%d.%m.%Y_%H%M")
dest_path = current_dir+'\\'+current_datetime + "_Visualisation"
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
readme_l = ["File Naming Conventions", "\n\n" ,"[Name of Visual] of [x axis] & [y axis] & [legend]"]
with open(dest_path + "/(1)README.txt", "w") as f:
    f.writelines(readme_l)


### START OF SAVING VISUALS FUNCTION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def save_viz(file_name):
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90, horizontalalignment = "right")
    plt.savefig(os.path.join(dest_path, file_name), dpi=300, bbox_inches='tight')
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### END OF SAVING VISUALS FUNCTION

### START OF PLOTTING FUNCTION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def label_checker(x_label: str, hue_label: str):
    if x_label != hue_label:
        return True

def str_replacing(input_str: str):
    str1 = input_str.replace(" ", "_").replace("/", "_")
    return str1

def hist_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    hist_fig = sns.displot(data=viz_df, x=x_label, y=y_label, kind="hist", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"HistPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()

def kde_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    kde_fig = sns.displot(data=viz_df, x=x_label, y=y_label, kind="kde", hue=hue_label, palette="bright") 
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"KDEPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()

def bar_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str): 
    bar_fig = sns.catplot(data=viz_df, x=x_label, y=y_label, kind="bar", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"BarPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()

def box_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    box_fig = sns.boxplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    box_fig = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"BoxPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()

def swarm_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    swarm_fig = sns.catplot(data=viz_df, x=x_label, y=y_label, kind="swarm", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"SwarmPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()

def scatter_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    scatter_fig = sns.relplot(data=viz_df, x=x_label, y=y_label, kind="scatter", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"ScatterPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()

def line_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    line_fig = sns.relplot(data=viz_df, x=x_label, y=y_label, kind="line", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"LinePlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()

def lm_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    lm_fig = sns.lmplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    hue_label = str_replacing(hue_label)
    file_name = f"LMPlot_of_{x_label}_&_{y_label}_&_{hue_label}.png"
    save_viz(file_name)
    plt.close()

def ecdf_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str):
    ecdf_fig = sns.displot(data=viz_df, x=x_label, kind="ecdf", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    file_name = f"ECDFPlot_of_{x_label}.png"
    save_viz(file_name)
    plt.close()

def viz(viz_df: pd.DataFrame, x_label_list: list, hue_label: list, y_label_list: list):
    hue_label = ''.join(hue_label)
    for (x_label, y_label) in itertools.product(x_label_list, y_label_list):

        if (mapping[x_label] == "categorical" or mapping[x_label] == "datetime") and (mapping[y_label] == "numerical") and (label_checker(x_label, hue_label)):
            bar_plot(viz_df, x_label, hue_label, y_label)
            box_plot(viz_df, x_label, hue_label, y_label)
            hist_plot(viz_df, x_label, hue_label, y_label)
            scatter_plot(viz_df, x_label, hue_label, y_label)
            swarm_plot(viz_df, x_label, hue_label, y_label)

        if (mapping[x_label] == "numerical" or mapping[x_label] == "datetime") and (mapping[y_label] == "numerical") and (label_checker(x_label, hue_label)):
            kde_plot(viz_df, x_label, hue_label, y_label)
            line_plot(viz_df, x_label, hue_label, y_label)
        
        if (mapping[x_label] == "numerical" or mapping[x_label] == "datetime") and (mapping[y_label] == "numerical" or mapping[y_label] == "datetime") and (label_checker(x_label, hue_label)):
            lm_plot(viz_df, x_label, hue_label, y_label)

        if (mapping[x_label] == "categorical" or mapping[x_label] == "numerical" or mapping[x_label] == "datetime") and (label_checker(x_label, hue_label)):
            ecdf_plot(viz_df, x_label, hue_label)
            
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### END OF PLOTTING FUNCTION

### START OF CALCULATED TABLE FUNCTION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def table_calc(input_df_calc: pd.DataFrame, pri_cols_calc: list,sec_col_calc: list, val_col_calc: list, val_col_nocalc: list = []):
    
    original_val_col_calc = copy.deepcopy(val_col_calc)
    
    try:
        group_by_col = pri_cols_calc + sec_col_calc
        group_by_col = list(set(group_by_col))
        combined_col = group_by_col + val_col_calc + val_col_nocalc
    except Exception as e:
        print(e)
    new_df = pd.DataFrame()
    temp_df = input_df_calc.loc[:, combined_col]
    temp_df = temp_df.loc[:,~temp_df.columns.duplicated()]

    val_col_calc = copy.deepcopy(original_val_col_calc)
    for val in val_col_calc: 
        new_col_name = ""
        new_col_name_list = []
        unique_vals = list(set(temp_df[val]))
        num_unique_vals = len(unique_vals)

        for i in range(num_unique_vals):
            eval_col = np.where(temp_df[val] == unique_vals[i], 1, 0)
            new_col_name = f"Count_of_{val}_{unique_vals[i]}"
            eval_col = eval_col.flatten()
            new_df[new_col_name] = eval_col
            new_col_name_list.append(new_col_name)
    
    temp_df = pd.concat([temp_df, new_df], axis=1, join="inner")
    for i in new_col_name_list:
        val_col_calc.append(i)
        mapping[i] = "numerical"

    temp_df = temp_df.drop_duplicates(group_by_col)
    val_col_calc += val_col_nocalc
    viz(temp_df, pri_cols_calc, sec_col_calc, val_col_calc)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### END OF CALCULATED TABLE FUNCTION

### START OF INITIALISATION FUNCTION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def initialisation(calc_needed: bool, input_df: pd.DataFrame, df_mapping: dict, pri_cols: list = [], sec_cols: list = [], val_cols_calc: list = [], val_cols_nocalc: list = []):
    input_df = input_df.replace("/", "_", regex=True).replace(" ", "_", regex=True)
    counter = 0
    
    if (pri_cols == []) and (sec_cols == []) and (val_cols_calc == [] and (val_cols_nocalc == [])):
        for keys,values in df_mapping.items():
            try:
                if values == "categorical":
                    pri_cols.append(keys)
                    val_cols_calc.append(keys)
                    sec_cols.append(keys)

                elif values == "numerical" or values == "datetime":
                    val_cols_nocalc.append(keys)
        
            except Exception as e:
                print(e)
                
    initial_val_cols = copy.deepcopy(val_cols_calc)

    if len(sec_cols) != 1:
        lists_of_sec_cols = [sec_cols[x: x+1] for x in range(0, len(sec_cols), 1)]
        for a in range(len(lists_of_sec_cols)):
            sec_cols = lists_of_sec_cols[a]
            if counter != 0:
                val_cols_calc = copy.deepcopy(initial_val_cols)
            if calc_needed == True:
                table_calc(input_df, pri_cols, sec_cols, val_cols_calc, val_cols_nocalc)
                counter += 1
            else:
                val_cols_calc += val_cols_nocalc
                viz(input_df, pri_cols, sec_cols, val_cols_calc)
    if calc_needed == True:
        table_calc(input_df, pri_cols, sec_cols, val_cols_calc, val_cols_nocalc)
    else:
        val_cols_calc += val_cols_nocalc
        viz(input_df, pri_cols, sec_cols, val_cols_calc)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### END OF INITIALISATION FUNCTION
initialisation(1, df, mapping)
end = time.time()
print("Time taken to run: " + str(end-start))
#initialisation(1, df, mapping, pri_cols, sec_col, list_of_val_cols_calc, list_of_val_cols_nocalc) # only this will not have an int