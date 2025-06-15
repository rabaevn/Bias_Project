import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from aif360.datasets import StandardDataset


def load_and_preprocess_data(filepath="data/adult.csv"):
    """
    preprocesses the adult income dataset for our experiments
    :param filepath: where the dataset is stored
    :return: X_train, y_train, X_test, y_test, prot_attr_train, prot_attr_test
    """
    # Load and assign column names
    df = pd.read_csv(filepath)
    df.columns = [
        "age", "workclass", "fnlwgt", "education", "education.num",
        "marital.status", "occupation", "relationship", "race", "sex",
        "capital.gain", "capital.loss", "hours.per.week", "native.country", "income"
    ]

    # Handle missing values
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    # Encode labels
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
    df['sex'] = df['sex'].apply(lambda x: 1 if x.strip().lower() == 'male' else 0)
    df['race'] = df['race'].apply(lambda x: 1 if x.strip().lower() == 'white' else 0)

    # Encode categorical features
    cat_cols = [col for col in df.select_dtypes(include='object').columns if col != 'income']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Convert to AIF360 dataset
    dataset = StandardDataset(
        df,
        label_name="income",
        favorable_classes=[1],
        protected_attribute_names=["sex", "race"],
        privileged_classes=[[1], [1]],  # [Male], [White]
        features_to_drop=[]
    )

    # Convert back to pandas for train/test split
    df, _ = dataset.convert_to_dataframe()
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['income'])

    # Wrap train/test in AIF360 format again
    train = StandardDataset(
        train_df,
        label_name='income',
        favorable_classes=[1],
        protected_attribute_names=['sex', 'race'],
        privileged_classes=[[1], [1]]
    )

    test = StandardDataset(
        test_df,
        label_name='income',
        favorable_classes=[1],
        protected_attribute_names=['sex', 'race'],
        privileged_classes=[[1], [1]]
    )

    # Standardize features
    scaler = StandardScaler()
    train.features = scaler.fit_transform(train.features)
    test.features = scaler.transform(test.features)

    # Extract numpy arrays
    X_train = train.features
    y_train = train.labels.ravel()
    X_test = test.features
    y_test = test.labels.ravel()
    prot_attr_train = train.protected_attributes.ravel()
    prot_attr_test = test.protected_attributes.ravel()

    return X_train, y_train, X_test, y_test, prot_attr_train, prot_attr_test
