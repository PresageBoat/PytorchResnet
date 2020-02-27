import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
 
Layers = [3, 4, 6, 3]
 
class Bottleneck(nn.Module):
    def __init__(self, in_channels, filters, stride=1, is_downsample = False):
        super(Bottleneck, self).__init__()
        filter1, filter2, filter3 = filters
        self.conv1 = nn.Conv2d(in_channels, filter1, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter1)
        self.conv2 = nn.Conv2d(filter1, filter2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filter2)
        self.conv3 = nn.Conv2d(filter2, filter3, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filter3)
        self.relu = nn.ReLU(inplace=True)
        self.is_downsample = is_downsample
        self.parameters()
        if is_downsample:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(filter3))
 
 
    def forward(self, X):
        X_shortcut = X
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
 
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu(X)
 
        X = self.conv3(X)
        X = self.bn3(X)
 
        if self.is_downsample:
            X_shortcut = self.downsample(X_shortcut)
 
        X = X + X_shortcut
        X = self.relu(X)
        return X
 
 
class ResNetModel(nn.Module):
 
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
 
        self.layer1 = self._make_layer(64, (64, 64, 256), Layers[0])
        self.layer2 = self._make_layer(256, (128, 128, 512), Layers[1], 2)
        self.layer3 = self._make_layer(512, (256, 256, 1024), Layers[2], 2)
        self.layer4 = self._make_layer(1024, (512, 512, 2048), Layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)
        # self.named_parameters()
 
    def forward(self, input):
        # print("--ResNetModel_1--forward--input.shape={}".format(input.shape))
        X = self.conv1(input)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.maxpool(X)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
 
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        X = self.fc(X)
        return X
 
 
    def _make_layer(self, in_channels, filters, blocks, stride = 1):
        layers = []
        block_one = Bottleneck(in_channels, filters, stride=stride, is_downsample=True)
        layers.append(block_one)
        for i in range(1, blocks):
            layers.append(Bottleneck(filters[2], filters, stride=1, is_downsample=False))
 
        return nn.Sequential(*layers)
 
 
 
#对图像的预处理（固定尺寸到224， 转换成touch数据, 归一化）
tran = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])
 
if __name__ == '__main__':
    image = Image.open("tiger.jpeg")
    image = tran(image)
    image = torch.unsqueeze(image, dim=0)
 
    net = ResNetModel()
    # net = resnet50()
    # for name, parameter in net.named_parameters():
    #     print("name={},size={}".format(name, parameter.size()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    image = image.to(device)
    net.load_state_dict(torch.load("resnet50-19c8e357.pth"))  # 加载pytorch中训练好的模型参数
    net.eval()
 
    # x = torch.randn(2, 3, 32, 32)
    # out = net(x)
    # print('resnet:', out.shape)
 
    output = net(image)
    test, prop = torch.max(output, 1)
    synset = [l.strip() for l in open("synset.txt").readlines()]
    print("top1：", synset[prop.item()])
 
    preb_index = torch.argsort(output, dim=1, descending=True)[0]
    top5 = [(synset[preb_index[i]], output[0][preb_index[i]].item()) for i in range(5)]
    print(("Top5: ", top5))
 
 
