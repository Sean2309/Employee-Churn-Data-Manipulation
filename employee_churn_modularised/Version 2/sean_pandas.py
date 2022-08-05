import pandas as pd

from sean_processing import set_x_tick_limit

def expand_categorical_cols(input_df_calc: pd.DataFrame, mapping: dict , pri_cols: list, sec_cols: list, cat_cols_to_expand: list, text_cols: list, cat_cols_no_expand: list = []):
    """
    Returns n cols from n unique vals in each categorical col input
    """
    expanded_cat_cols = []
    try:
        group_by_col = list(set(pri_cols + sec_cols))
        combined_col = group_by_col + cat_cols_to_expand + cat_cols_no_expand + text_cols
    except Exception as e:
        print(e)
    new_df = pd.DataFrame()
    temp_df = input_df_calc.loc[:, combined_col]
    temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]
    print(cat_cols_to_expand)
    cat_cols_to_expand = set_x_tick_limit(input_df_calc, cat_cols_to_expand, "categorical", mapping)
    print(cat_cols_to_expand)
    for val in cat_cols_to_expand:
        eval_col = pd.get_dummies(temp_df[val]).add_prefix(f"{val}_")
        new_df = pd.concat([new_df, eval_col], axis=1)
    #new_df = new_df.add_prefix("Proportion_of_")
    temp_df = pd.concat([temp_df, new_df], axis=1, join="inner")

    for i in new_df.columns:
        expanded_cat_cols.append(i)
        mapping[i] = "categorical"

    temp_df = temp_df.drop_duplicates(group_by_col)    
    return temp_df, mapping, expanded_cat_cols, cat_cols_no_expand
    