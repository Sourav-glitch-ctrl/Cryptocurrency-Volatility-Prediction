def validate_data(df):
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")

    if len(df) == 0:
        raise ValueError("Dataset is empty")

    return True