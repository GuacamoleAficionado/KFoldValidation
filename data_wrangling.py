import numpy as np
import pandas as pd

import cpdmaker

df = pd.read_csv('ACS_modified.csv')

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


states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']