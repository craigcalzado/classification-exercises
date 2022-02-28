import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import acquire




def prep_iris(df): # create a function named prep_iris that accepts the untransformed iris data, and returns the data with the transformations above applied.
    df = acquire.get_iris_data() # utlize acquire.py to load the iris data.
    df = df.drop(columns=['species_id', 'Unnamed: 0']) # drop the species_id and measurement_id columns.
    df = df.rename(columns={'species_name': 'species'}) # rename the species_name column to just species.
    dummy_df = pd.get_dummies(df[['species']], drop_first=True) # create dummy variables of the species name and concatenate onto the iris dataframe. (This is for practice, we don't always have to encode the target, but if we used species as a feature, we would need to encode it).
    df = pd.concat([df, dummy_df], axis=1) # concatenate the iris dataframe with the dummy variables.
    return df.drop(columns=['species']) # drop the species column.

prep_iris(df) # call the function prep_iris to apply the transformations to the iris data.

def prep_titanic(df):
    df = acquire.get_titanic_data()
    df = df.drop(columns=['deck', 'embarked', 'class', 'age', 'passenger_id', 'Unnamed: 0'])
    # fill in missing values
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=[True,True])
    df = pd.concat([df, dummy_df], axis=1)
    return df.drop(columns=['sex', 'embark_town'])

def prep_telco(df):
    df = acquire.get_telco_data()
    df = df.drop(columns=['Unnamed: 0', 'internet_service_type_id', 'payment_type_id', 'contract_type_id'])
    dummy_df = pd.get_dummies(df[['payment_type', 'contract_type', 'internet_service_type']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df.drop(columns=['payment_type', 'contract_type', 'internet_service_type'])
