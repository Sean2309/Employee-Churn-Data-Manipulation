from cgi import test
from doctest import DocFileTest
from types import new_class
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import copy
import os
import time

#%matplotlib inline
mpl.rcdefaults()
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

df = pd.read_csv('.\datasets\current_data_without_preproc.csv')
df  = df.drop(columns=["Unnamed: 0.1","Unnamed: 0"])

## Fully automated function (WIP)
start = time.time()
combined_col = []
pri_cols = ["Division", "Location", "ResourceLevel"]
sec_col = ["Division", "Location", "Support from Company"]
list_of_val_cols_calc = ["Department", "Status"]
list_of_val_cols_nocalc = ["Service Years", "Age"]

# current_dir = os.getcwd()
# current_datetime = datetime.datetime.now().strftime("%d.%m.%Y_%H%M")
# dest_path = current_dir+'\\'+current_datetime + "_Visualisation"
# if not os.path.exists(dest_path):
#     os.makedirs(dest_path)

mapping = {
    "EmployeeID": "Categorical",
    "Division": "categorical",
    "Department": "categorical",
    "Location": "categorical",
    "ResourceLevel": "categorical",
    "JobTitle": "categorical",
    "JobFunction": "categorical",
    "Style": "categorical",
    "Status": "categorical",
    "OnboardDate": "datetime",
    "TerminatedDate": "datetime",
    "Gender": "categorical",
    "Birthday": "datetime",
    "Grade": "categorical",
    "LineManager": "categorical",
    "LineManager": "categorical",
    "Nationality": "categorical",
    "EmployeeType": "categorical",
    "MgrResourceLevel": "categorical",
    "K/H": "categorical",
    "Voluntary": "categorical",
    "ResignationType": "categorical",
    "Medical leave day (6 mths)": "numerical",
    "Service Years": "numerical",
    "Age": "numerical",
    "Avg Annual leave days taken/month (6 mths)": "numerical",
    "Personal Development (satisfaction)": "numerical",
    "Promotion frequency/last 2 yrs.": "numerical",
    #"Support from Company": "numerical",
    # "Job Satisfaction": "categorical",
    "Month in current level": "numerical",
    "Avg Extra time hours/week(6 mths)": "numerical",
    "Last annual performance rating": "numerical" 
}#    "ResignationReason": "categorical",
    # "Personal Development (satisfaction)": "categorical",
    # "Support from Company": "categorical",
    #     "Job Satisfaction": "categorical", => removed bc too many missing values
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
    file_name = f"HistPlot_of_{x_label}_and_{y_label}.png"
    save_viz(file_name)
    plt.close()

def kde_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    kde_fig = sns.displot(data=viz_df, x=x_label, y=y_label, kind="kde", hue=hue_label, palette="bright") 
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    file_name = f"KDEPlot_of_{x_label}_and_{y_label}.png"
    save_viz(file_name)
    plt.close()

def bar_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):  
    bar_fig = sns.catplot(data=viz_df, x=x_label, y=y_label, kind="bar", hue=hue_label, palette="bright")
    #print("x label is: " + str(x_label) + "---- y label is: " + str(y_label) + "\n")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    file_name = f"BarPlot_of_{x_label}_and_{y_label}.png"
    save_viz(file_name)
    plt.close()

def box_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    box_fig = sns.boxplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    file_name = f"BoxPlot_of_{x_label}_and_{y_label}.png"
    save_viz(file_name)
    plt.close()

def swarm_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    swarm_fig = sns.catplot(data=viz_df, x=x_label, y=y_label, kind="swarm", hue=hue_label, palette="bright")
    file_name = f"SwarmPlot_of_{x_label}_and_{y_label}.png"
    save_viz(file_name)
    plt.close()

def scatter_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    scatter_fig = sns.relplot(data=viz_df, x=x_label, y=y_label, kind="scatter", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    file_name = f"ScatterPlot_of_{x_label}_and_{y_label}.png"
    save_viz(file_name)
    plt.close()

def line_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    line_fig = sns.relplot(data=viz_df, x=x_label, y=y_label, kind="line", hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    file_name = f"LinePlot_of_{x_label}_and_{y_label}.png"
    save_viz(file_name)
    plt.close()

def lm_plot(viz_df: pd.DataFrame, x_label: str, hue_label: str, y_label: str):
    lm_fig = sns.lmplot(data=viz_df, x=x_label, y=y_label, hue=hue_label, palette="bright")
    x_label = str_replacing(x_label)
    y_label = str_replacing(y_label)
    file_name = f"LMPlot_of_{x_label}_and_{y_label}.png"
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
    start = time.time()
    for (x_label, y_label) in itertools.product(x_label_list, y_label_list):

        if (mapping[x_label] == "categorical" or mapping[x_label] == "datetime") and (mapping[y_label] == "numerical") and (label_checker(x_label, hue_label)):
            bar_plot(viz_df, x_label, hue_label, y_label)
            box_plot(viz_df, x_label, hue_label, y_label)

        elif (mapping[x_label] == "numerical" or mapping[x_label] == "datetime") and (mapping[y_label] == "numerical") and (label_checker(x_label, hue_label)):
            kde_plot(viz_df, x_label, hue_label, y_label)
            line_plot(viz_df, x_label, hue_label, y_label)
        
        elif (mapping[x_label] == "numerical" or mapping[x_label] == "datetime") and (mapping[y_label] == "numerical" or mapping[y_label] == "datetime") and (label_checker(x_label, hue_label)):
            lm_plot(viz_df, x_label, hue_label, y_label)

        elif (mapping[x_label] == "categorical" or mapping[x_label] == "datetime") and (mapping[x_label] == "numerical") and (label_checker(x_label, hue_label)):
            hist_plot(viz_df, x_label, hue_label, y_label)
            scatter_plot(viz_df, x_label, hue_label, y_label)
            swarm_plot(viz_df, x_label, hue_label, y_label)
            
        elif (mapping[x_label] == "categorical" or mapping[x_label] == "numerical" or mapping[x_label] == "datetime") and (label_checker(x_label, hue_label)):
            ecdf_plot(viz_df, x_label, hue_label)
            
    end = time.time()
    #print(end-start)
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
        print("Error was: " + str(e))
    new_df = pd.DataFrame()
    test_df = input_df_calc.loc[:, combined_col]
    test_df = test_df.T.drop_duplicates().T
    # for loop for val cols
    val_col_calc = copy.deepcopy(original_val_col_calc)
    for val in itertools.repeat(val_col_calc, len(val_col_calc)): 
        # for loop to generate cols for val cols
        unique_vals = []
        new_col_name = ""
        new_col_name_list = []
        unique_vals = list(set(test_df[val])) # gives me the names of the unique vals within the values col
        num_unique_vals = len(unique_vals)
        for i in range(num_unique_vals):
            eval_col = np.where(test_df[val] == unique_vals[i], 1, 0)# unique vals is status, voluntary words
            new_col_name = f"Count_of_{val}_{unique_vals[i]}"
            eval_col = eval_col.flatten()
            new_df[new_col_name] = eval_col
            new_col_name_list.append(new_col_name)
    
    test_df = pd.concat([test_df, new_df], axis=1, join="inner")
    #test_df = test_df.groupby(group_by_col, as_index=True).transform("sum")
    for i in new_col_name_list:
        val_col_calc.append(i)
        mapping[i] = "numerical"
            
    test_df = test_df.drop_duplicates(group_by_col)
    val_col_calc += val_col_nocalc
    viz(test_df, pri_cols_calc, sec_col_calc, val_col_calc)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### END OF CALCULATED TABLE FUNCTION

### START OF INITIALISATION FUNCTION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def initialisation(calc_needed: bool, input_df_initialisation: pd.DataFrame, df_mapping: dict, pri_cols_initialisation: list = [], sec_col_initialisation: list = [], val_col_initialisation: list = [], val_col_initialisation_nocalc: list = []):
    input_df_initialisation = input_df_initialisation.replace("/", "_", regex=True).replace(" ", "_", regex=True)
    counter = 0
    
    if (pri_cols_initialisation == []) and (sec_col_initialisation == []) and (val_col_initialisation == [] and (val_col_initialisation_nocalc == [])):
        for keys,values in df_mapping.items():
            try:
                if values == "categorical":
                    pri_cols_initialisation.append(keys)
                    val_col_initialisation.append(keys)

                elif values == "numerical" or values == "datetime":
                    val_col_initialisation_nocalc.append(keys)
        
            except Exception as e:
                print("Your error is: " + str(e))
                break
        sec_col_initialisation = pri_cols_initialisation[0]
    initial_val_cols = copy.deepcopy(val_col_initialisation)

    if counter != 0:
        val_col_initialisation = copy.deepcopy(initial_val_cols)

        ## ITERTOOLS WIP
    # for a, b in itertools.product(pri_cols_initialisation, sec_col_initialisation):
    #     if counter != 0:
    #         val_col_initialisation = copy.deepcopy(initial_val_cols)
    #     # if calc_needed == True:
    #     #     table_calc(input_df_initialisation, )
    #     print("a is:" + str(a) + "---------- b is:" + str(b) + "\n")



    if len(sec_col_initialisation) != 1:
        lists_of_sec_cols = [sec_col_initialisation[x: x+1] for x in range(0, len(sec_col_initialisation), 1)]
        for a in range(len(lists_of_sec_cols)):
            sec_col_initialisation = lists_of_sec_cols[a]
            if counter != 0:
                val_col_initialisation = copy.deepcopy(initial_val_cols)
            if calc_needed == True:
                table_calc(input_df_initialisation, pri_cols_initialisation, sec_col_initialisation, val_col_initialisation, val_col_initialisation_nocalc)
                counter += 1
            else:
                val_col_initialisation = val_col_initialisation + val_col_initialisation_nocalc
                viz(input_df_initialisation, pri_cols_initialisation, sec_col_initialisation, val_col_initialisation)
    if calc_needed == True:
        table_calc(input_df_initialisation, pri_cols_initialisation, sec_col_initialisation, val_col_initialisation, val_col_initialisation_nocalc)
    else:
        val_col_initialisation = val_col_initialisation + val_col_initialisation_nocalc
        viz(input_df_initialisation, pri_cols_initialisation, sec_col_initialisation, val_col_initialisation)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### END OF INITIALISATION FUNCTION
initialisation(1, df, mapping)
#nitialisation(1, df, mapping, pri_cols, sec_col, list_of_val_cols_calc, list_of_val_cols_nocalc) # only this will not have an int
end = time.time()
print(end-start)
