from src.d00_utils.Conditions import Conditions
from src.d01_data.database.db_connect import db_connect
from src.d03_processing.Features import Features
import pandas.io.sql as sqlio

def group_conditions(conditions="all", ppts="all", features="all"):
    """outputs by-participant dataframe based on condition sql table"""
    if conditions == "all":
        conditions = Conditions.list

    if ppts == "all":
        ppts_string = ""

    if features == "all":
        features = Features.conditions

    # connect to database - use separate connection class
    conn = db_connect(suppress_print=True)

    sum_features = ['n_trials', 'n_correct']
    mean_features = [i for i in features if i not in sum_features]

    agg_func_str = " "
    for feat in sum_features:
        agg_func_str += f" sum({feat}) {feat},"
    for feat in mean_features:
        agg_func_str += f" avg({feat}) {feat},"
    agg_func_str = agg_func_str[:-1]

    query = f"select ppt_id,{agg_func_str} " \
            f"from alloeye_conditions " \
            f"group by ppt_id "
    query += ";"

    df = sqlio.read_sql_query(query, conn)
    conn.close()
    return df

df = group_conditions()
print(df)
print("end")