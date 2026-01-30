import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean_data(filepath):
    """Load raw data and perform basic cleaning"""
    df = pd.read_csv(filepath)
    
    # Replace common missing value markers
    df.replace(['Unknown', 'na', 'Other'], pd.NA, inplace=True)
    
    # Drop columns with too many missing values (you can adjust threshold)
    drop_cols = ['Defect_of_vehicle', 'Fitness_of_casuality', 'Work_of_casuality']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
    
    # Fill missing values
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if col != 'Accident_severity':
            df[col] = df[col].fillna(df[col].median())
    
    return df


def engineer_features(df):
    """Create new features (mainly from Time)"""
    if 'Time' in df.columns:
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour
        df.drop(columns=['Time'], inplace=True)
    return df


def encode_and_scale(df):
    """Encode categoricals and scale numerical features"""
    le = LabelEncoder()
    scaler = StandardScaler()
    
    # Numerical columns to scale
    num_cols = ['Number_of_vehicles_involved', 'Number_of_casualties']
    if all(c in df.columns for c in num_cols):
        df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Encode all object columns except target
    for col in df.select_dtypes(include='object').columns:
        if col != 'Accident_severity':
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Map target if it's still string
    if df['Accident_severity'].dtype == 'object':
        severity_map = {'Fatal injury': 0, 'Serious Injury': 1, 'Slight Injury': 2}
        df['Accident_severity'] = df['Accident_severity'].map(severity_map)
    
    return df, le, scaler   # return encoders/scaler if you want to save them later


def get_X_y(df):
    """Split features and target"""
    X = df.drop(columns=['Accident_severity'])
    y = df['Accident_severity']
    return X, y