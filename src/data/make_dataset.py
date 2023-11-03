import requests
import pandas as pd
import zipfile
import os
data_url = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"

# Define the local file path to save the ZIP file
zip_file_path = "src/data/filtered_paranmt.zip"  # Save it in the src/data directory

# Ensure that the data directory exists
os.makedirs(os.path.dirname(zip_file_path), exist_ok=True)

# Download the ZIP file
response = requests.get(data_url, stream=True)
if response.status_code == 200:
    with open(zip_file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print(f"Downloaded ZIP file to {zip_file_path}")
else:
    print("Failed to download the data.")
    exit(1)

# Extract the ZIP file
extracted_folder = "src/data/"  # Extract to src/data directory
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

# Load the data from the CSV file within the extracted folder
csv_file_path = os.path.join(extracted_folder, "filtered.tsv")
data = pd.read_csv(csv_file_path)

# Data preprocessing
data.drop(['Unnamed: 0', 'similarity', 'lenght_diff'], axis=1, inplace=True)

# Convert 'reference' and 'translation' columns to lowercase
data['reference'] = data['reference'].str.lower()
data['translation'] = data['translation'].str.lower()

# Remove non-alphabetic characters from 'reference' and 'translation' columns
data['reference'] = data['reference'].str.replace('[^a-z\s]', '', regex=True)
data['translation'] = data['translation'].str.replace('[^a-z\s]', '', regex=True)

# Filter rows where the length of 'reference' and 'translation' is less than or equal to 128
data = data[(data['reference'].str.len() <= 128) & (data['translation'].str.len() <= 128)]

# Save the preprocessed data to the same folder
output_csv_file = os.path.join(extracted_folder, "preprocessed_data.csv")
data.to_csv(output_csv_file, index=False)

print("Data preprocessing and saving completed.")
