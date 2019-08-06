from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'  # Set CUDA device
DATA_PATH = '/home/xiaowei/data/'  # Download or load datasets


class Configs:

    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 64
        self.epochs = 5
        self.lr = 0.001
        self.log_interval = 10
        self.seed = 1
        self.num_workers = 8
        self.no_cuda = False
        self.save_model = False

    def __repr__(self):
        return str(self.__dict__)


class ConvNet(nn.Module):

    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        self.img_pixels = 28
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LeNet(nn.Module):

    def __init__(self, num_class=10):
        super(LeNet, self).__init__()
        self.img_pixels = 28
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class AlexNet(nn.Module):

    def __init__(self, num_class=10):
        super(AlexNet, self).__init__()
        self.img_pixels = 224
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class VGG(nn.Module):

    def __init__(self, layer_num=13, batch_norm=False, num_class=10):
        super(VGG, self).__init__()
        self.img_pixels = 224
        self.in_channels = 1
        self.layers = []
        self.cfgs = {
            11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M',
                 512, 512, 'M', 512, 512, 'M'],
            16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                 512, 512, 512, 'M', 512, 512, 512, 'M'],
            19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
                 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

        self.features = self._make_layers(self.cfgs[layer_num], batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, batch_norm=False):
        for v in cfg:
            if v == 'M':
                self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(self.in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    self.layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    self.layers += [conv2d, nn.ReLU(inplace=True)]
                self.in_channels = v
        return nn.Sequential(*self.layers)


class Execution:
    def __init__(self, __C, model):
        self.__C = __C
        self.model = model.to(device)
        print(self.model)

        self.transform = transforms.Compose([
            transforms.Resize(self.model.img_pixels),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        self.train_loader = Data.DataLoader(torchvision.datasets.FashionMNIST(
            root=DATA_PATH, train=True,
            download=True, transform=self.transform),
            batch_size=self.__C.batch_size,
            shuffle=True,
            num_workers=self.__C.num_workers)

        self.test_loader = Data.DataLoader(torchvision.datasets.FashionMNIST(
            root=DATA_PATH, train=False,
            download=True, transform=self.transform),
            batch_size=self.__C.test_batch_size,
            shuffle=False,
            num_workers=self.__C.num_workers)

    def train(self, epoch):
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), lr=self.__C.lr)
        criterion = nn.CrossEntropyLoss()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)

            # Forward
            output = self.model(data)
            loss = criterion(output, target)

            # Backward and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % self.__C.log_interval == 0:
                print('Train Epoch: {} [{}/{}] ({:.0f}%)\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))

    def test(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss = F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)  # get max P's index
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def run(self):
        for epoch in range(1, self.__C.epochs + 1):
            self.train(epoch)
            self.test(self.model.state_dict())

        if self.__C.save_model:
            torch.save(self.model.state_dict(), "model_saved.pt")


__C = Configs()
use_cuda = not __C.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(__C.seed)
execu = Execution(__C, model=VGG(layer_num=16, batch_norm=True))  # Select Model Here
execu.run()