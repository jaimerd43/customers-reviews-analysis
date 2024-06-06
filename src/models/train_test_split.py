

# %%
from sklearn.model_selection import train_test_split

# %%
data['flag'].replace('Yes', 1, inplace = True)
data['flag'].replace('No', 0, inplace = True)

fam_income_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12}
data['fam_income'].replace(fam_income_mapping, inplace=True)

mortgage_mapping = {'1Low': 1, '2Med': 2, '3High': 3}
data['mortgage'].replace(mortgage_mapping, inplace=True)

data_encoded = pd.get_dummies(data, drop_first=True)

data_encoded.columns = data_encoded.columns.str.replace(' ', '_')

X = data_encoded.drop('flag', axis=1)
y = data_encoded['flag']
