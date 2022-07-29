import pandas as pd

def expand_categorical_cols(input_df_calc: pd.DataFrame, mapping: dict , pri_cols: list, sec_cols: list, cat_cols_to_expand: list, cat_cols_no_expand: list = []):
    val_out = []
    new_col_name_list = []
    try:
        group_by_col = list(set(pri_cols + sec_cols))
        combined_col = group_by_col + cat_cols_to_expand + cat_cols_no_expand
    except Exception as e:
        print(e)
    new_df = pd.DataFrame()
    temp_df = input_df_calc.loc[:, combined_col]
    temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]

    for val in cat_cols_to_expand:
        unique_vals = list(pd.Series(temp_df[val].unique()).astype("str"))
        new_col_name_list.extend(unique_vals)
        eval_col = pd.get_dummies(temp_df[val]).add_prefix(f"{val}_")
        new_df = pd.concat([new_df, eval_col], axis=1)
    new_df = new_df.add_prefix("Proportion_of_")
    temp_df = pd.concat([temp_df, new_df], axis=1, join="inner")

    for i in new_df.columns:
        val_out.append(i)
        mapping[i] = "categorical"

    temp_df = temp_df.drop_duplicates(group_by_col)
    val_out += cat_cols_no_expand
    return temp_df, mapping, pri_cols, sec_cols, val_out
    