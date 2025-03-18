import pandas as pd
from scipy.stats import spearmanr
from scipy.stats.mstats import gmean
import numpy as np

QUBA_MEAN = [0.7990557251908397, 0.18583778826016084, 0.5298791080592711, 0.5664841617855405, 0.004479927389289115, 0.783356376851669, 0.9286352601551336, 0.30548520497464765, 55.28778625954198] 
QUBA_STD = [0.034685710545946026, 0.11083062192390095, 0.23383978329969085, 0.14548627504237352, 0.0026752085291489288, 0.024047971612660052, 0.017724144886557092, 0.08354531823072534, 42.8890536095964] 

def print_result_to_excel(file_path, sheet_name, results):
    """
    Prints the results of an experiment in an Excel file specified by the given path

    :model_name: name of the model that produced the results
    :result: results of an experiment as a dictionary in the form of {"qualitydimension1": (result1, result2), ...}
    :path: path of excel file
    
    """
    df = pd.DataFrame(results)

    if check_sheet_exists(file_path, sheet_name):
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
            writer.book.remove(writer.book[sheet_name])

    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

def combine_dict(dict1, dict2):
    """
    Append dict2 on dict1
    """
    for key in dict2.keys():
        value1 = dict1.get(key)
        value2 = dict2.get(key)
        
        if value1 is None:
            dict1[key] = value2
        else:
            dict1[key] = value1 + value2
    
    return dict1

def check_sheet_exists(file_path, sheet_name):
    with pd.ExcelFile(file_path) as xls:
        return sheet_name in xls.sheet_names
    
def calculate_rank_correlation_with_p(file_path, sheet_name, usecols):
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=usecols)

    columns = df.columns
    correlation_matrix = pd.DataFrame(index=columns, columns=columns)
    for i, x_column in enumerate(columns):
        for j, y_column in enumerate(columns):
            if j > i:
                continue     
        
            correlation, p_val = spearmanr(df[x_column], df[y_column], nan_policy="omit")

            if x_column == y_column:
                correlation = 1

            correlation_matrix.loc[x_column, y_column] = f"{round(correlation, 4)} ({round(p_val, 4)})"

    if check_sheet_exists(file_path, sheet_name + '_corr_p'):
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
            writer.book.remove(writer.book[sheet_name + '_corr_p'])

    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
        correlation_matrix.to_excel(writer, sheet_name=sheet_name + '_corr_p')

def compute_normalized_values(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    normalized_df = (df.iloc[:, 1:] - QUBA_MEAN) / QUBA_STD
    normalized_df.insert(0, df.columns[0], df.iloc[:, 0])

    if check_sheet_exists(file_path, "NORM"):
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
            writer.book.remove(writer.book["NORM"])
    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
            normalized_df.to_excel(writer, sheet_name="NORM", index=False)
    return

def compute_quba_score(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    quba_matrix = pd.DataFrame(columns=["Model", "QuBA"])
    quba_list = []
    for index, row in df.iterrows():
        values = []
        weights = []
        for column in df.columns:
            if column == "Model":
                continue
            else:
                if column in ["Cal. Err.", "Params"]:
                    row[column] = row[column] * (-1)
                values.append(row[column])
                weight = 1/3 if column in ["C-Rob.", "Adv. Rob.", "OOD Rob."] else (0.5 if column in ["Shape Bias", "Obj. Foc."] else 1 )
                weights.append(weight)
        assert len(values) == len(weights)
        weights = weights / np.sum(weights)
        quba = np.average(a=values, weights=weights)
        quba_list.append(quba)
    quba_matrix["Model"] = df["Model"]
    quba_matrix["QuBA"] = quba_list

    if check_sheet_exists(file_path, "QUBA"):
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
            writer.book.remove(writer.book["QUBA"])
    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
            quba_matrix.to_excel(writer, sheet_name="QUBA", index=False)

    return quba_matrix
