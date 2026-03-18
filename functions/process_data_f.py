from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#### Function to Process the data for ML modeling ####
def process_data(data, Y):

    # Remove duplicate rows from the dataset
    data = data.drop_duplicates()

    # Divide the data into X and Y
    target = data[Y]
    features = data.drop(Y, axis = 1)

    # Split the data into training and testing set
    X_train, X_test, Y_train, Y_test = train_test_split(
        features,target,
        test_size=0.2,
        random_state=42
    )
    # Preprocessing
    num_features = features.select_dtypes(include="number").columns
    char_features = features.select_dtypes(exclude="number").columns

    process = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),  # Scale Numerical Features
            ("char", OneHotEncoder(handle_unknown="ignore", sparse_output=False), char_features)  # One hot encoding (Not needed now)
        ],
        remainder="drop"  # drop any columns not listed
    )

    # Collect the datasets and process
    process_return = {
        "X_train_data" : X_train,
        "X_test_data" : X_test,
        "Y_train_data" : Y_train,
        "Y_test_data" : Y_test,
        "processing" : process
    }
    return process_return
