import os
import pandas as pd
from funnynames import get_random_file_name


def return_client_csv(high_risk_lst: list, moderate_risk_lst: list, data_frame: pd.DataFrame):
    parent_directory = 'Client_Spreadsheets/'
    while True:
        try:
            directory = get_random_file_name()
            path = os.path.join(parent_directory, directory)
            os.mkdir(path)
            break
        except FileExistsError:
            pass

    moderate_risk_df = data_frame.loc[data_frame['ID'].isin([e[0] for e in moderate_risk_lst])].copy()
    insert_column_index = moderate_risk_df.columns.to_list().index('ID') + 1
    moderate_risk_df.insert(insert_column_index, 'Probability_Satisfied', -1)
    for i in range(len(moderate_risk_df['ID'].unique())):
        try:
            # find the row with a particular ID value
            row_indexes = data_frame.index[data_frame['ID'] == moderate_risk_lst[i][0]]
            # then insert the probability the customer with that ID is satisfied in the appropriate column
            for index in row_indexes:
                moderate_risk_df.loc[index, 'Probability_Satisfied'] = moderate_risk_lst[i][1]
        except IndexError:
            print("i: ", i)

    high_risk_df = data_frame.loc[data_frame['ID'].isin([e[0] for e in high_risk_lst])].copy()
    insert_column_index = high_risk_df.columns.to_list().index('ID') + 1
    high_risk_df.insert(insert_column_index, 'Probability_Satisfied', -1)
    for i in range(len(high_risk_df['ID'].unique())):
        # find the row with a particular ID value
        row_indexes = data_frame.index[data_frame['ID'] == high_risk_lst[i][0]]
        # then insert the probability the customer with that ID is satisfied in the appropriate column
        for index in row_indexes:
            high_risk_df.loc[index, 'Probability_Satisfied'] = high_risk_lst[i][1]

    save_path = 'Client_Spreadsheets/' + directory
    file_name_1 = directory + '_moderate_risk_group.csv'
    file_name_2 = directory + '_high_risk_group.csv'
    complete_name_1 = os.path.join(save_path, file_name_1)
    complete_name_2 = os.path.join(save_path, file_name_2)
    with open(complete_name_1, 'w') as file1, open(complete_name_2, 'w') as file2:
        csv1 = moderate_risk_df.to_csv(index=False)
        file1.write(csv1)
        csv2 = high_risk_df.to_csv(index=False)
        file2.write(csv2)
    return directory
