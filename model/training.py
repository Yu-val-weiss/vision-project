import os
import torch
from model.model import GestureModel
import torch.optim as optim
from torch import nn
from utils.data import GestureDataset
from utils.config import *
from torch.utils.data import DataLoader
from tqdm import tqdm

NUM_EPOCHS = 10
BATCH_SIZE = 8
MODEL_PATH = 'model/trained_models'

if __name__ == '__main__':
    mod = GestureModel()
    
    if torch.backends.mps.is_available():
        mod = mod.to('mps:0')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mod.parameters(), lr=0.001, momentum=0.9)

    ds_train = GestureDataset(TRAIN_CSV)

    dl_train = DataLoader(GestureDataset(COLLECTED_DATA_CSV), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    ds_dev = GestureDataset(DEV_CSV)
    dl_dev = DataLoader(GestureDataset(COLLECTED_DATA_CSV), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
    for epoch in tqdm(range(NUM_EPOCHS),desc='Traversing epochs'):
        running_loss = 0.0
        for i, data in tqdm(enumerate(dl_train, 0),desc='Batching',total=len(ds_train)//BATCH_SIZE,leave=False):
            # get the inputs; data is a list of [inputs, labels]
            labels, landmarks = data
            
            if torch.backends.mps.is_available():
                labels = labels.to('mps:0')
                landmarks = landmarks.to('mps:0')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = mod(landmarks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 10 mini-batches
                tqdm.write(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    
        with torch.no_grad():
            total = 0
            correct = 0
            for data in dl_dev:
                labels, landmarks = data
                if torch.backends.mps.is_available():
                    labels = labels.to('mps:0')
                    landmarks = landmarks.to('mps:0')
                # calculate outputs by running images through the network
                outputs = mod(landmarks)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
              
            print(f'Accuracy for epoch {epoch}: {correct/total:.2f}')
                
        torch.save(mod.state_dict(), os.path.join(MODEL_PATH, f'model_{epoch}'))
             
    