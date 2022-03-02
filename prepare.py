
from email.errors import InvalidBase64LengthDefect
from pdb import post_mortem
from wsgiref import validate
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import acquire

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

df = acquire.get_iris_data() # utlize acquire.py to load the iris data.
df.head()

def prep_iris(df): # create a function named prep_iris that accepts the untransformed iris data, and returns the data with the transformations above applied.
    df = df.drop(columns=['species_id', 'measurement_id', 'Unnamed: 0']) # drop the species_id and measurement_id columns.
    df = df.rename(columns={'species_name': 'species'}) # rename the species_name column to just species.
    dummy_df = pd.get_dummies(df[['species']], drop_first=True) # create dummy variables of the species name and concatenate onto the iris dataframe. (This is for practice, we don't always have to encode the target, but if we used species as a feature, we would need to encode it).
    df = pd.concat([df, dummy_df], axis=1) # concatenate the iris dataframe with the dummy variables.
    return df

prep_iris(df) # call the function prep_iris to apply the transformations to the iris data.
df.head()
split_data(df)

def split_iris_data(df):
    train, test = train_test_split(df, test_size=0.2, random_state=789)
    train, validate = train_test_split(train, test_size=0.3, random_state=789)
    return train, validate, test
split_iris_data(df)


#------------------------------------------------------------------------------------------------------------

def prep_titanic(df):
    df = acquire.get_titanic_data()
    df = df.drop(columns=['deck', 'embarked', 'class', 'age', 'passenger_id', 'Unnamed: 0'])
    # fill in missing values
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=[True,True])
    df = pd.concat([df, dummy_df], axis=1)
    return df.drop(columns=['sex', 'embark_town'])
df = prep_titanic(df)

#split titanic data
def split_titanic_data(df):
    train, test = train_test_split(df, test_size=.2, random_state=789, stratify=df.survived)
    train, validate = train_test_split(train, test_size=.3, random_state=789, stratify=train.survived)
    return train, validate, test
split_titanic_data(df)

split_dataframe(df)

def prep_telco(df):
    df = acquire.get_telco_data()
    df = df.drop(columns=['Unnamed: 0', 'internet_service_type_id', 'payment_type_id', 'contract_type_id', 'multiple_lines'])
    dummy_df = pd.get_dummies(df[['gender', 'payment_type', 'contract_type', 'internet_service_type']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df.drop(columns=['gender', 'payment_type', 'contract_type', 'internet_service_type'])
df = prep_telco(df)


# function to split dataframes
def split_dataframe(df):
   train, test = train_test_split(df, test_size=0.2, random_state=789)
   train, validate = train_test_split(train, test_size=0.3, random_state=789)
   return train, validate, test 



# how many lottery tickets would you have to buy in order to win.
purchase each day
citys with the tickets
how much to Invest
    
    # how many lottery tickets would you have to buy in order to win.
    purchase each day
    citys with the tickets
    how much to Invest

beteen how many people to split the pot
everyone in the US dived by 2
in the regioun

