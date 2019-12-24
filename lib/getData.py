from torch.utils.data import Dataset
import torchvision.datasets as td
from torchvision import  transforms
from options import Options
import numpy
import random
from PIL import Image




opt = Options().parse()


class CifaDataset(Dataset):
    def __init__(self, index , classes, transform=None):
        self.classes = classes
        self.transform = transform
        self.index = index#  0:test  1:train
        self.all_datas = []
        self.all_labels = []
        self.labels = []
        self.all_class = []

        self.real_picture = []
        self.re_data = []
        self.dataset = td.CIFAR10(root=opt.dataroot, train=self.index, download=True)
        self.td = td.CIFAR10(root=opt.dataroot, train=1, download=True)
        if self.index ==1:
            self.Dataset_train()
            # self.selfInitDataset()

        else:
            self.Dataset_test()
            # self.selfInitDataset_test()


    def Dataset_train(self):
        for img, label in self.dataset:
            self.data = []
            if self.index == 1 and label not in self.classes: continue
            image = reTrans(img)
            self.all_datas.append(image)
            self.all_labels.append(0)
            self.all_class.append(1)

    def Dataset_test(self):
        for img, label in self.dataset:
            self.data = []
            if self.index == 1 and label not in self.classes: continue
            image = reTrans(img)
            self.all_datas.append(image)
            self.all_labels.append(0)
            if label in self.classes:
                self.all_class.append(0)
            else:
                self.all_class.append(1)

    def selfInitDataset(self):

        for img,label in self.dataset:
            self.data = []
            if self.index==1 and label not in self.classes:continue
            for k in range(len(self.transform)):
                self.real_picture.append(numpy.array(img))
                image = transforms.Compose(self.transform[k])(img)
                self.data.append(image)
            m = 0
            for i in self.data:
                n = 0
                for j in self.data:
                    if m == n:
                        n = n + 1
                        continue
                    r = numpy.array(random.random())

                    # image = r * i + (1 - r) * j
                    image = Image.blend(i, j, r)
                    image = reTrans(image)
                    self.all_datas.append(image)
                    eyes = numpy.eye(4)
                    z = (eyes[m] * (1 - r) + eyes[n] * r).astype(numpy.float32)
                    self.all_labels.append(z)
                    self.all_class.append(1)
                    n = n + 1
                m = m + 1
    def selfInitDataset_test(self):

        for img, label in self.dataset:
            if self.index == 1 and label not in self.classes: continue
            for k in range(len(self.transform)):
                self.real_picture.append(numpy.array(img))
                image = transforms.Compose(self.transform[k])(img)
                image = reTrans(image)

                self.all_datas.append(image)
                self.all_labels.append(k)
                if label in self.classes:
                    self.all_class.append(1)
                else:
                    self.all_class.append(0)
    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        return self.all_datas[idx],self.all_labels[idx],self.all_class[idx]




def reTrans(image):
      return transforms.Compose([transforms.ToTensor()]) (image)

def getTransform(index):
    com = []
    # com.append(transforms.Resize((224, 224)))
    if index == 1:
        com.append(transforms.RandomRotation(degrees=(90, 90)))
    elif index == 2:
        com.append(transforms.RandomRotation(degrees=(180, 180)))
    elif index == 3:
        com.append(transforms.RandomRotation(degrees=(270, 270)))
    # com.append(transforms.ToTensor())
    # com.append(transforms.Normalize((.5,.5,.5), (.5,.5,.5)))
    return com

def cifa10Data(normalclass):
    tf = [getTransform(i) for i in range(4)]
    print(normalclass)
    train_data = CifaDataset(index=1,classes=[normalclass],transform=tf)
    test_data = CifaDataset(index=0,classes=[normalclass],transform=tf)
    print('train num:',len(train_data))
    print('test num:',len(test_data))
    return train_data,test_data

# train_data,test_data = cifa10Data()
# print([train_data[i][1] for i in range(100)])
# print([test_data[i][1] for i in range(100)])
