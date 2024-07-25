import pandas as pd
import os
from sdv.constraints import create_custom_constraint_class
original_data = pd.read_csv(os.getcwd()+'/static/item_master.csv')

original_data[original_data['ITEM'] == 37182655582] = [37182655582,101,150,1102,"EA"]

valid_combinations = original_data.set_index('ITEM')[['DEPT', 'CLASS', 'SUBCLASS', 'STANDARD_UOM']].to_dict(orient='index')

def is_valid(column_names, data):
    item_column = column_names[0]
    combination_columns = column_names[1:]
    valid = []
    for index, row in data.iterrows():
        item = row[item_column]
        if pd.isna(item):
            if all(pd.isna(row[col]) for col in combination_columns[:-1]) and row[combination_columns[-1]] == 'EA':
                valid.append(True)
            else:
                valid.append(False)
        elif item in valid_combinations:
            valid_combination = valid_combinations[item]
            if all(row[col] == valid_combination[col] for col in combination_columns):
                valid.append(True)
            else:
                valid.append(False)
        else:
            valid.append(False)
    
    series = pd.Series(valid, index=data.index)
    return series

def transform(column_names, data):
    return data

def reverse_transform(column_names, data):
    return data

ItemConstraint = create_custom_constraint_class(
    is_valid_fn=is_valid,
    transform_fn=transform,
    reverse_transform_fn=reverse_transform
)

def qty_constraint(column_names,data):
    return data['QTY'] == data['UOM_QUANTITY']

def transform(column_names,data):
    return data

def inverse_transform(column_names,data):
    return data

QtyConstraint = create_custom_constraint_class(
    is_valid_fn=qty_constraint,
    transform_fn=transform,
    reverse_transform_fn=inverse_transform
)

def is_valid(column_names, data):
    unique_combinations = ~data[column_names].duplicated(keep=False)
    return unique_combinations

def transform(column_names, data):
    return data

def reverse_transform(column_names, data):
    return data

UniqueTransactionConstraint = create_custom_constraint_class(
    is_valid_fn=is_valid,
    transform_fn=transform,
    reverse_transform_fn=reverse_transform
)


def is_valid_sequential(column_names, data):
    unique_combinations = ~data[column_names].duplicated(keep=False)
    return unique_combinations   

def transform_sequential(column_names, data):
    return data

def reverse_transform_sequential(column_names, data):
    # tran_seq_no_col = column_names[0]
    # item_seq_no_col = column_names[1]
    # data = data.copy()
    # for tran_seq_no, group in data.groupby(tran_seq_no_col):
    #     data.loc[group.index, item_seq_no_col] = range(1, len(group) + 1)
    return data

SequentialItemConstraint = create_custom_constraint_class(
    is_valid_fn=is_valid_sequential,
    transform_fn=transform_sequential,
    reverse_transform_fn=reverse_transform_sequential
)