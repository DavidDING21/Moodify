import pandas as pd
import os

class EmotionDataProcessor:
    """
    A utility class for processing emotion and image data, including mapping and normalization.
    """

    def __init__(self, input_emotion_tsv, input_image_tsv, output_dir="./outputs"):
        self.input_emotion_tsv = input_emotion_tsv
        self.input_image_tsv = input_image_tsv
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_and_map_data(self, output_csv_name):
        """
        Filter emotion data, map image URLs, save as a CSV, and return the full path.

        :param output_csv_name: Name of the output CSV file (without path)
        :return: Full path of the saved file
        :raises: FileNotFoundError, ValueError, RuntimeError
        """
        # Define column mapping
        required_columns = [
            'ID',
            'Art (image+title): anger',
            'Art (image+title): disgust',
            'Art (image+title): fear',
            'Art (image+title): happiness',
            'Art (image+title): sadness',
            'Art (image+title): surprise'
        ]
        column_mapping = {
            'Art (image+title): anger': 'anger',
            'Art (image+title): disgust': 'disgust',
            'Art (image+title): fear': 'fear',
            'Art (image+title): happiness': 'joy',
            'Art (image+title): sadness': 'sadness',
            'Art (image+title): surprise': 'surprise'
        }

        # 1) Read emotion TSV
        if not os.path.exists(self.input_emotion_tsv):
            raise FileNotFoundError(f"Emotion TSV not found: {self.input_emotion_tsv}")
        emotion_df = pd.read_csv(self.input_emotion_tsv, sep="\t", dtype={'ID': str})

        # 2) Filter columns and rename
        missing = [c for c in required_columns if c not in emotion_df.columns]
        if missing:
            raise ValueError(f"Emotion TSV is missing columns: {missing}")
        filtered = emotion_df[required_columns].rename(columns=column_mapping)

        # 3) Read image TSV
        if not os.path.exists(self.input_image_tsv):
            raise FileNotFoundError(f"Image TSV not found: {self.input_image_tsv}")
        image_df = pd.read_csv(self.input_image_tsv, sep="\t", dtype={'ID': str})
        if 'ID' not in image_df.columns or 'Image URL' not in image_df.columns:
            raise ValueError("Image TSV must contain 'ID' and 'Image URL' columns")

        # 4) Merge emotion and image data
        merged = pd.merge(filtered, image_df[['ID', 'Image URL']], on='ID', how='inner')
        if merged.empty:
            raise RuntimeError("No matching IDs found between emotion and image TSVs")

        # 5) Save merged data
        out_path = os.path.join(self.output_dir, output_csv_name)
        merged.to_csv(out_path, index=False)
        return out_path

    def normalize_emotion_probabilities(self, input_csv_name, output_csv_name):
        """
        Normalize emotion probabilities in the mapped CSV, save and return the full path.

        Rows where the sum of the six emotions is zero will be removed.

        :param input_csv_name: Name of the existing CSV file
        :param output_csv_name: Name of the output normalized CSV file
        :return: Full path of the saved file
        :raises: FileNotFoundError, ValueError
        """
        input_path = os.path.join(self.output_dir, input_csv_name)
        output_path = os.path.join(self.output_dir, output_csv_name)

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input CSV not found: {input_path}")

        df = pd.read_csv(input_path)
        emotion_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        missing = [c for c in emotion_cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing emotion columns: {missing}")

        # Compute the sum of emotion probabilities
        df['sum'] = df[emotion_cols].sum(axis=1)
        # Remove rows where the sum is zero
        df = df[df['sum'] > 0].drop(columns=['sum'])
        # Normalize emotion probabilities
        df[emotion_cols] = df[emotion_cols].div(df[emotion_cols].sum(axis=1), axis=0)

        # Save the normalized data
        df.to_csv(output_path, index=False)
        return output_path