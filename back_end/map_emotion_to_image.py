import pandas as pd
import numpy as np

class EmotionImageMapper:
    def __init__(self, csv_file):
        """
        Initialize the mapper using a normalized CSV file.

        :param csv_file: Path to the normalized CSV file containing emotion columns and 'Image URL'.
        """
        self.csv_file = csv_file
        self.emotion_columns = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        self._load_csv()

    def _load_csv(self):
        """
        Load the CSV file and validate its structure.
        Ensures the CSV contains the required columns and has no missing or empty data.
        """
        self.data = pd.read_csv(self.csv_file)
        # Validate the presence of required columns
        missing = [c for c in self.emotion_columns + ['Image URL'] if c not in self.data.columns]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        if self.data.empty:
            raise ValueError("CSV is empty")
        # Check for NaN values in emotion columns
        if self.data[self.emotion_columns].isnull().any().any():
            raise ValueError("CSV contains NaN values in emotion columns")

    def calculate_similarity(self, model_output, csv_emotions, metric='euclidean'):
        """
        Calculate similarity between model output emotions and CSV emotions using the specified metric.

        :param model_output: A numpy array representing the emotion probabilities from the model.
        :param csv_emotions: A numpy array representing the emotion probabilities from the CSV.
        :param metric: The similarity metric to use ('euclidean' or 'cosine').
        :return: A numpy array of distances or similarity scores.
        """
        if metric == 'euclidean':
            # Compute Euclidean distance
            return np.linalg.norm(csv_emotions - model_output, axis=1)
        elif metric == 'cosine':
            # Compute Cosine similarity
            dot = np.dot(csv_emotions, model_output)
            norms = np.linalg.norm(csv_emotions, axis=1) * np.linalg.norm(model_output)
            sim = dot / norms
            return 1 - sim  # Convert similarity to distance
        else:
            raise ValueError("Unsupported metric")

    def map_emotion_to_image(self, emotion_probs, metric='euclidean'):
        """
        Map the given emotion probabilities to the best matching image in the dataset.

        :param emotion_probs: A dictionary of emotion probabilities with keys matching emotion columns.
        :param metric: The similarity metric to use ('euclidean' or 'cosine').
        :return: A tuple containing:
                 - The URL of the best matching image.
                 - The emotion probabilities of the best matching image as a dictionary.
                 - The similarity score (distance).
        """
        # Convert the input emotion probabilities to a numpy array
        model_arr = np.array([emotion_probs[e] for e in self.emotion_columns])
        # Extract the emotion probabilities from the CSV as a numpy array
        csv_arr = self.data[self.emotion_columns].values
        # Calculate distances or similarities
        dists = self.calculate_similarity(model_arr, csv_arr, metric)
        # Find the index of the best match (minimum distance)
        idx = np.argmin(dists)
        # Retrieve the best matching row
        row = self.data.iloc[idx]
        # Extract the image URL, emotion probabilities, and similarity score
        url = row['Image URL']
        ems = row[self.emotion_columns].to_dict()
        score = float(dists[idx])
        return url, ems, score
    