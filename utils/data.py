from typing import Callable, Union, NamedTuple
import numpy as np
import pandas as pd
from utils.config import CLASS_LABELS_CSV, COLLECTED_DATA_CSV
from torch.utils.data import Dataset, DataLoader
import torch


def read_class_labels():
    df = pd.read_csv(CLASS_LABELS_CSV)
    return {
        row.class_index: row.class_label for row in df.itertuples(index=False)
    }
    

def check_class_proportions():
    class_labels = read_class_labels()
    
    df = pd.read_csv(COLLECTED_DATA_CSV)
    
    df_counts = df.groupby('class_label').size().reset_index(name='count') # type: ignore
    
    # Map the class labels to class names
    df_counts['class_name'] = df_counts['class_label'].map(class_labels)
    
    # Set the 'class_label' column as the index
    df_counts.set_index('class_label', inplace=True)
    print(df_counts)
    
    
class Datapoint(NamedTuple):
    class_index: int
    landmarks: torch.Tensor
    
    
class GestureDataset(Dataset):
    def __init__(self, path: str, transform:Union[Callable, None]=None) -> None:
        super().__init__()
        self.dataset = np.loadtxt(path, delimiter=',',skiprows=1)
        self.transform = transform
    
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, index):
        row = self.dataset[index]
        class_index = int(row[0])  # Ensure class_index is an integer
        landmarks = torch.from_numpy(row[1:]).float()  # Ensure landmarks are float tensors

        if self.transform is not None:
            landmarks = self.transform(landmarks)

        return Datapoint(class_index, landmarks)
    
    
if __name__ == '__main__':
    # dl = DataLoader(GestureDataset(COLLECTED_DATA_CSV), batch_size=8, shuffle=False)
    
    # for i, batch in enumerate(dl):
    #     print(i, batch.landmarks)
    #     print(batch.landmarks.shape)
        
    # print(read_class_labels())
    
    check_class_proportions()
    