from collections import defaultdict
import torch
import json
from data.datasets import Mpiigaze
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
import os
from tqdm import tqdm
from models.gaze_model import GazeModel
import torch.nn as nn
import torch.optim as optim

def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

def train(testlabelpathombined, gazeMpiimage_dir, transformations, batch_size, device, args):
    print(device)
    # batch size lớn quá sẽ chuyển qua sử dụng cpu
    for fold in range(args["people_num"]):
        model = GazeModel().cuda(device)
        model.train()
        criterion = nn.MSELoss().cuda(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print('Loading data.')
        dataset=Mpiigaze(testlabelpathombined,gazeMpiimage_dir, transformations, True, angle=180, fold=fold)
        train_loader_gaze = DataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=2,
            pin_memory=True)
        torch.backends.cudnn.benchmark = True
        
        for epoch in range(args["epochs_num"]):
            print('\nEpoch:', epoch)
            tbar = tqdm(train_loader_gaze, desc='Training')
            for i, (landmarks, cont_labels) in enumerate(tbar):
            # for i, (landmarks, cont_labels) in enumerate(train_loader_gaze):
                landmarks = landmarks.cuda(device)
                cont_labels = cont_labels.cuda(device)
                
                # Forward pass
                output = model(landmarks)
                loss = criterion(output, cont_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i % 2000 == 0:
                    print('\nEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args["epochs_num"], i, len(train_loader_gaze), loss.item()))
                # Save models at numbered epochs.
            if epoch % 1 == 0 and epoch < args["epochs_num"]:
                print('Taking snapshot...')
                torch.save(model.state_dict(),
                        args["output"] + '/fold' + str(fold) +
                        '_epoch_' + str(epoch+1) + '.pth')

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    file_args = "args_colab.json" if 'COLAB_GPU' in os.environ else 'args.json'
    
    # load the arguments from the json file
    with open(f'./args/{file_args}', 'r') as f:
        args = json.load(f)
    args = defaultdict_from_json(args)
    dataset = args['dataset']
    
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if dataset == "MPIIFaceGaze":
        folder = os.listdir(args["gazeMpiilabel_dir"])
        folder.sort()
        testlabelpathombined = [os.path.join(args["gazeMpiilabel_dir"], j) for j in folder]
        
        train(testlabelpathombined, args["gazeMpiimage_dir"], transformations, args["batch_size"], device, args)

    
if __name__ == "__main__":
    main()