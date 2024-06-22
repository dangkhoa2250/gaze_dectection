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

def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

def train(testlabelpathombined, gazeMpiimage_dir, transformations, batch_size, device, args):
    for fold in range(args["people_num"]):
        # model, pre_url = getArch_weights(args.arch, 28)
        # load_filtered_state_dict(model, model_zoo.load_url(pre_url))
        # model = nn.DataParallel(model)
        # model.to(gpu)
        print('Loading data.')
        dataset=Mpiigaze(testlabelpathombined,gazeMpiimage_dir, transformations, True, angle=180, fold=fold)
        train_loader_gaze = DataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=4,
            pin_memory=True)
        torch.backends.cudnn.benchmark = True
        
        for epoch in range(args["epochs_num"]):
            print('Epoch:', epoch)
            for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in tqdm(train_loader_gaze, desc="Loading Data"):
                images_gaze = Variable(images_gaze).to(device)
                # Forward pass
                # output = model(img)
                # loss = criterion(output, label)
                # Backward pass
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # if i % 100 == 0:
                #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args["epochs"], i, len(train_loader_gaze), loss.item())
        pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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