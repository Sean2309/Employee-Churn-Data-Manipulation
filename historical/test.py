
from cgi import test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('D:\Code\datasets\current_data_without_preproc.csv')
df  = df.drop(columns=["Unnamed: 0.1","Unnamed: 0"])

# df.info()

def one_dimensional_quantitative_analysis_on_status(comparison_col_name, sort_by_col_name="Number_of_People_who_left"):
    # df.grade
    test_df = df.loc[:, [f"{comparison_col_name}", "Status"]]
    test_df = test_df.assign(
        Number_of_People_who_left = np.where(test_df[["Status"]] == 2, 1, 0),
        Number_of_People_who_stayed = np.where(test_df[["Status"]] == 3, 1, 0),
        ).groupby([f"{comparison_col_name}"]).agg({f"{sort_by_col_name}":sum, "Number_of_People_who_stayed":sum}
    )
    test_df.sort_values(by=[f"{sort_by_col_name}"], ascending=False, inplace=True)
    #print(test_df.dtypes)
    #sns.catplot(kind="bar", x=f"{comparison_col_name}", y=f"{sort_by_col_name}", data=test_df)
    print(test_df) 

def f(input_df, group_by_col, val_col):
    unique_vals = list(set(df[val_col])) # gives me the names of the unique vals within the values col
    num_unique_vals = len(unique_vals)
    new_col_name = ""

    test_df = input_df.loc[:, [f"{group_by_col}", f"{val_col}"]]
    for i in range(num_unique_vals):
        eval_col = np.where(test_df[[f"{val_col}"]] == unique_vals[i], 1, 0)
        # creating the name of the col
        new_col_name = f"Count_of_{val_col}:{unique_vals[i]}"
        # creating the col in the df
        test_df[new_col_name] = eval_col.flatten()
        test_df[new_col_name] = test_df.groupby([f"{group_by_col}"])[new_col_name].transform("sum")
        
    test_df = test_df.drop_duplicates([f"{group_by_col}"])
    test_df = test_df.drop([f"{val_col}"], axis=1)
    print(test_df)

f(df, "Location", "Grade")

# def f(input_df: str, group_by_col: list, val_col: str):
#     unique_vals = list(set(df[val_col])) # gives me the names of the unique vals within the values col
#     num_unique_vals = len(unique_vals)
#     new_col_name = ""
#     combined_col = []
#     if type(group_by_col) == str and type(val_col) == str:
#         combined_col = group_by_col,val_col
#         test_df = input_df.loc[:, combined_col]

#     elif (type(group_by_col) == str and type(val_col) == list):
#         combined_col = group_by_col,val_col
#         print(combined_col)
#         test_df = input_df.loc[:, combined_col]
#         print(combined_col)

#     elif (type(group_by_col) == list and type(val_col) == str):
#         pass
    
#     for i in range(num_unique_vals):
#         eval_col = np.where(test_df[[f"{val_col}"]] == unique_vals[i], 1, 0)
#         # creating the name of the col
#         new_col_name = f"Count_of_{val_col}:{unique_vals[i]}"
#         # creating the col in the df
#         test_df[new_col_name] = eval_col.flatten()
#         test_df[new_col_name] = test_df.groupby([f"{group_by_col}"])[new_col_name].transform("sum")
        
#         #test_df[new_col_name] = test_df[new_col_name].groupby()
#         #test_df[f"{val_col}"] = pd.Series(eval_col.flatten()).groupby([f"{group_by_col}"]).agg({f"{val_col}":sum})
#         # eval col is not connected with the tesst df thats why its not working
#         # find out whether this iterates throughout the whole col before going from value 2 to 3 => it does evaluate throughout the whole col before moving on to the next
#         # test_df = test_df.assign(
#         #     eval_val = list(eval_val)
#         # ).groupby([f"{group_by_col}"]).agg({f"{val_col}":sum})
#     test_df = test_df.drop_duplicates([f"{group_by_col}"])
#     test_df = test_df.drop([f"{val_col}"], axis=1)
#     print(test_df)
# list_of_group_by_cols = ["Division", "Location"]
# list_of_val_cols = ["Status", "Voluntary"]

# #f(df, "Division", "Status")
# f(df, "Division", list_of_val_cols)
# # f(df, list_of_group_by_cols, "Status")
