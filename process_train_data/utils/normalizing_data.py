import pickle
from sklearn.preprocessing import MinMaxScaler

class DataScaler:
    def __init__(self, method_scaling="minmax"):
        """
        Initializes the normalization class.

        Parameters:
        - method_scaling (str): Scaling method to use ("minmax" or "standard").
        """
        self.method_scaling = method_scaling.lower()
        if self.method_scaling == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaling method. Use 'minmax' or 'standard'.")

    def fit(self, train_data):
        """Fits the scaler on training data."""
        self.scaler.fit(train_data)

    def transform(self, data):
        """Transforms data using the fitted scaler."""
        return self.scaler.transform(data)

    def detransform(self, data):
        """Inverse transforms the normalized data back to the original scale."""
        return self.scaler.inverse_transform(data)

    def save_model(self, file_path):
        """Saves the fitted scaler to a file."""
        with open(file_path, "wb") as f:
            pickle.dump(self.scaler, f)

    def load_model(self, file_path):
        """Loads a saved scaler from a file."""
        with open(file_path, "rb") as f:
            self.scaler = pickle.load(f)
