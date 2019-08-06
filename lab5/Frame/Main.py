import read_mnist as raw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import random, copy


class HyperParams:
    def __init__(self):
        self.LR = 1e-4
        self.BS = 64
        self.RUN = 'train'
        self.MAX_EPOCHS = 10
        self.USE_CUDA = torch.cuda.is_available()

    def check(self):
        assert self.RUN in ['train', 'val'], 'You should set in train or val'

    def __str__(self):
        print('LR:', self.LR)
        print('BS:', self.BS)
        print('RUN:', self.RUN)


class Dataset(Data.Dataset):
    def __init__(self, __C):
        self.__C = __C

        if __C.RUN in ['train']:
            self.imgs = raw.get_train_img()
            self.labels = raw.get_train_label()
        else:
            self.imgs = raw.get_val_img()
            self.labels = raw.get_val_label()
        
    def __getitem__(self, index):
        img = self.imgs[index].reshape([1,28,28])
        label = self.labels[index].reshape([1])
        return torch.from_numpy(img), torch.from_numpy(label)
    
    def __len__(self):
        return self.imgs.shape[0]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=(3,3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=(7,7))

        self.dropout = nn.Dropout2d(0.3)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(self.dropout(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(self.dropout(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(self.conv6(x))
        x = self.fc(x.view(-1, 256))
        return x


class Execution:
    def __init__(self, __C):
        self.__C = __C
        self.dataset_train = Dataset(__C)
        self.__C_eval = copy.deepcopy(__C)
        self.__C_eval.RUN = 'val'
        self.dataset_eval = Dataset(self.__C_eval)

    def train(self):
        net = Net()
        net.train()
        if self.__C.USE_CUDA:
            net.cuda()
        
        dataloader = Data.DataLoader(self.dataset_train, batch_size=self.__C.BS, shuffle=True, num_workers=8)
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(net.parameters(), lr=self.__C.LR)

        for epoch in range(0, self.__C.MAX_EPOCHS):
            for step, (img_batch, label_batch) in enumerate(dataloader):
                
                if self.__C.USE_CUDA:
                    img_batch = img_batch.cuda()
                    label_batch = label_batch.cuda().view(-1)

                pred = net(img_batch)
                loss = loss_fn(pred, label_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # val each epoch
            self.eval(net.state_dict())

    def eval(self, state_dict):
        net = Net()
        net.eval()
        if self.__C_eval.USE_CUDA:
            net.cuda()
        net.load_state_dict(state_dict)

        dataloader = Data.DataLoader(self.dataset_eval, batch_size=self.__C_eval.BS, shuffle=False, num_workers=8)
        correct = 0
        for step, (img_batch, label_batch) in enumerate(dataloader):
            if self.__C_eval.USE_CUDA:
                img_batch = img_batch.cuda()
            
            pred = net(img_batch)
            pred_np = pred.cpu().detach().numpy()
            pred_np_argmax = np.argmax(pred_np, axis=-1)
            label_np = label_batch.view(-1).numpy()
            correct += np.sum((pred_np_argmax==label_np).astype('float32'))
        
        # print(correct/10000)
        print(correct/self.dataset_eval.__len__())


__C = HyperParams()
execution = Execution(__C)
execution.train()
