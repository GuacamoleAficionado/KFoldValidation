import numpy as np
import pandas as pd

import cpdmaker

df = pd.read_csv('ACST_Cust_Sum.csv')

#  this makes the Deactivated column boolean with 'True' indicating that
#  the customer has deactivated their account
#  df['Deactivated'] = ~df['Deactivated'].isna()


def assign_congregant_users(n):
    if n < 50:
        return "CU < 50"
    elif n < 200:
        return "50 <= CU < 200"
    elif n >= 200:
        return "200 <= CU"
    else:
        return "N"  # indicates no value given


#df['CongregantUsers'] = df['CongregantUsers'].apply(lambda x: assign_congregant_users(x))

# print(df[['Product', 'CongregantUsers', 'Deactivated']].head(15))
# df.to_csv('ACS_modified.csv')



