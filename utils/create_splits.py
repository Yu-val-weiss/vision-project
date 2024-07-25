import pandas as pd
from sklearn.model_selection import train_test_split
from utils.config import COLLECTED_DATA_CSV, TRAIN_CSV, DEV_CSV, TEST_CSV

# Load the dataset
df = pd.read_csv(COLLECTED_DATA_CSV)

# Shuffle and split the data
train, test = train_test_split(df, test_size=0.2, random_state=42)
# dev, test = train_test_split(temp, test_size=0.5, random_state=42)  

# Save the splits
train.to_csv(TRAIN_CSV, index=False)
# dev.to_csv(DEV_CSV, index=False)
test.to_csv(TEST_CSV, index=False)