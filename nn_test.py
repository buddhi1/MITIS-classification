from nn import IndoorNetwork
from dataset import IndoorSceneFeatureDataset
from torch.utils.data import Dataset, DataLoader
from utils import results
import torch
from torch.optim import RMSprop
from nn_train import evaluate


indoorscene_traindataset = IndoorSceneFeatureDataset(
    text_file='Dataset/TrainImages.txt',
    feature_file='Dataset/features.h5',
    train=True)
train_loader = DataLoader(indoorscene_traindataset, batch_size=16, shuffle=True, num_workers=1)

indoorscene_testdataset = IndoorSceneFeatureDataset(
    text_file='Dataset/TestImages.txt',
    feature_file='Dataset/features.h5',
    train=False)
val_loader = DataLoader(indoorscene_testdataset, batch_size=16, shuffle=True, num_workers=1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on : {device}')

    network = IndoorNetwork()
    network.to(device)
    print(network)
    optimizer = RMSprop(network.parameters(), lr=1e-5)

    # Params
    resume_exp_name = 'network-1e5'
    resume_epoch = 900

    # Load the model
    checkpoint = f'checkpoints/{resume_exp_name}-{resume_epoch}'
    state = torch.load(checkpoint, map_location=torch.device('cpu'))
    network.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    print(f"Resuming from checkpoint : {checkpoint}")

    evaluate(network, val_loader, device)


if __name__ == '__main__':
    main()