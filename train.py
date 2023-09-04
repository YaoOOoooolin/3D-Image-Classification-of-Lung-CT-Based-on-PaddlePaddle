import os
import zipfile
import numpy as np
import paddle
from paddle.nn import functional as F
paddle.__version__

import nibabel as nib

from scipy import ndimage
import os
import zipfile
import numpy as np
import paddle
from paddle.nn import functional as F
paddle.__version__
import nrrd
import SimpleITK as sitk
import glob
import os
from scipy import ndimage
import random
import paddle.vision.transforms.functional as F
from paddle.io import Dataset
from paddle.vision.transforms import Compose
import paddle.nn.functional as F


# Use GPU 0 
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
if paddle.is_compiled_with_cuda():
    print("GPU is available")
else:
    print("GPU is not available")

def read_nifti_file(filepath):
    """读取和加载数据"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

print("1")
def normalize(volume):
    """数据归一化"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """跨 z 轴调整大小"""
    # 设置所需的深度
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # 获取当前深度
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # 计算深度因子
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # 旋转
    img = ndimage.rotate(img, 90, reshape=False)
    # 跨z轴调整大侠
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """读取和调整数据大小"""
    # 读取扫描文件
    volume = read_nifti_file(path)
    # 归一化
    volume = normalize(volume)
    # 调整宽度、高度和深度
    volume = resize_volume(volume)
    return volume
print("2")
data1_folder = r'e:\SZBL-test1\1'
data2_folder = r'e:\SZBL-test1\2'
# Get the paths for CT-0/1 scans in data1/2 folder
data1_CT0_paths = [
    os.path.join(data1_folder, "CT-0", x)
    for x in os.listdir(os.path.join(data1_folder, "CT-0"))
]

data1_CT1_paths = [
    os.path.join(data1_folder, "CT-1", x)
    for x in os.listdir(os.path.join(data1_folder, "CT-1"))
]

data2_CT0_paths = [
    os.path.join(data2_folder, "CT-0", x)
    for x in os.listdir(os.path.join(data2_folder, "CT-0"))
]

data2_CT1_paths = [
    os.path.join(data2_folder, "CT-1", x)
    for x in os.listdir(os.path.join(data2_folder, "CT-1"))
]
# Process the scans
print("1")
data1_CT0_scans = np.array([process_scan(path) for path in data1_CT0_paths])
data1_CT1_scans = np.array([process_scan(path) for path in data1_CT1_paths])
data2_CT0_scans = np.array([process_scan(path) for path in data2_CT0_paths])
data2_CT1_scans = np.array([process_scan(path) for path in data2_CT1_paths])
print("2")

# Create labels
data1_CT0_labels = np.zeros(len(data1_CT0_scans))
print("22")
data1_CT1_labels = np.ones(len(data1_CT1_scans))
print("222")
data2_CT0_labels = np.zeros(len(data2_CT0_scans))
print("2222")
data2_CT1_labels = np.ones(len(data2_CT1_scans))
print("22222")

x_train = np.concatenate((data1_CT0_scans, data1_CT1_scans), axis=0)
y_train = np.concatenate((data1_CT0_labels, data1_CT1_labels), axis=0)
print("3")
x_val_data2 = np.concatenate((data2_CT0_scans, data2_CT1_scans), axis=0)
y_val_data2 = np.concatenate((data2_CT0_labels, data2_CT1_labels), axis=0)

# 按照Dataset的使用规范，构建肺部数据集

from paddle.io import Dataset

class CTDataset(Dataset):
    # 肺部扫描数据集
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, x, y, transform=None):
        """
        步骤二：实现构造函数，定义数据集大小
        Args:
            x: 图像
            y: 图片存储的文件夹路径
            transform (callable, optional): 应用于图像上的数据处理方法
        """
        self.x = x
        self.y = y
        self.transform = transform # 获取 transform 方法

    def __getitem__(self, idx):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据/测试数据，对应的标签）
        """
        img = self.x[idx]
        label = self.y[idx]
        # 如果定义了transform方法，使用transform方法
        if self.transform:
            img,label = self.transform([img,label])
        # 因为上面我们已经把数据集处理好了生成了numpy形式，没必要处理了
        return img, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.y) # 返回数据集大小，即图片的数量
import paddle
import random
from scipy import ndimage
import paddle.vision.transforms.functional as F

# 将图像旋转几度
class Rotate(object):

    def __call__(self, data):
        
        image = data[0]
        key_pts = data[1]
        # 定义一些旋转角度
        angles = [-20, -10, -5, 5, 10, 20]
        # 随机挑选角度
        angle = random.choice(angles)
        # 旋转体积
        image = ndimage.rotate(image, angle, reshape=False)
        image[image < 0] = 0
        image[image > 1] = 1        
        return image, key_pts

# 将图像的格式由HWD改为CDHW
class ToCDHW(object):
    
    def __call__(self, data):
        
        image = data[0]
        key_pts = data[1]
        image = paddle.transpose(paddle.to_tensor(image),perm=[2,0,1])
        image = np.expand_dims(image,axis=0)
        return image, key_pts
    
from paddle.vision.transforms import Compose

# create the transformed dataset
train_dataset = CTDataset(x_train,y_train,transform=Compose([Rotate(),ToCDHW()]))
valid_dataset = CTDataset(x_val_data2,y_val_data2,transform=Compose([ToCDHW()]))

import paddle

class Model3D(paddle.nn.Layer):
    def __init__(self):
        super(Model3D,self).__init__()
        self.layerAll = paddle.nn.Sequential(
            paddle.nn.Conv3D(1,64,(3,3,3)),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool3D(kernel_size=2),
            paddle.nn.BatchNorm3D(64),

            paddle.nn.Conv3D(64,64,(3,3,3)),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool3D(kernel_size=2),
            paddle.nn.BatchNorm3D(64),

            paddle.nn.Conv3D(64,128,(3,3,3)),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool3D(kernel_size=2),
            paddle.nn.BatchNorm3D(128),

            paddle.nn.Conv3D(128,256,(3,3,3)),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool3D(kernel_size=2),
            paddle.nn.BatchNorm3D(256),
            
            paddle.nn.AdaptiveAvgPool3D(output_size=1),
            paddle.nn.Flatten(),
            paddle.nn.Linear(256,512),
            paddle.nn.Dropout(p=0.3),

            paddle.nn.Linear(512,1),
            paddle.nn.Sigmoid()


        )

    def forward(self, inputs):
        x = self.layerAll(inputs)
        return x

model = paddle.Model(Model3D())
model.summary((-1,1,64,128,128))

import paddle.nn.functional as F

epoch_num = 100
batch_size = 2
batch_size_valid = 10
learning_rate = 0.0001

val_acc_history = []
val_loss_history = []

def train(model):
    print('start training ... ')
    # turn into training mode
    model.train()

    #该接口提供一种学习率按指数函数衰减的策略。
    scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate= learning_rate, gamma=0.96, verbose=True)
    opt = paddle.optimizer.Adam(learning_rate=scheduler,
                                parameters=model.parameters())

    train_loader = paddle.io.DataLoader(train_dataset,
                                        shuffle=True,
                                        batch_size=batch_size)

    valid_loader = paddle.io.DataLoader(valid_dataset, batch_size=batch_size_valid)
    
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = paddle.to_tensor(data[1],dtype="float32")
            y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            bce_loss = paddle.nn.BCELoss()
            loss = bce_loss(logits, y_data)
            
            if batch_id % 10 == 0:
                print("epoch: {}/{}, batch_id: {}, loss is: {}".format(epoch,epoch_num, batch_id, loss.numpy()))
            loss.backward()
            opt.step()
            opt.clear_grad()

        # evaluate model after one epoch
        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data = data[0]
            y_data = paddle.to_tensor(data[1],dtype="float32")
            y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            bce_loss = paddle.nn.BCELoss()
            loss = bce_loss(logits, y_data)
            
            mask = np.float32(logits>=0.5) # 以0.5为阈值进行分类
            correct = np.sum(mask == np.float32(y_data))  # 计算正确预测的样本个数
            acc = correct / batch_size_valid  # 计算精度
            accuracies.append(acc)
            losses.append(loss.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        print("[validation] epoch: {}/{}, accuracy/loss: {}/{}".format(epoch,epoch_num,avg_acc, avg_loss))
        val_acc_history.append(avg_acc)
        val_loss_history.append(avg_loss)
        model.train()

model = Model3D()
train(model)
paddle.save(model.state_dict(), "net_3d.pdparams")
import matplotlib.pyplot as plt
plt.plot(val_acc_history, label = 'acc')
plt.plot(val_loss_history, label ='loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0, 1.1])
plt.legend(loc='lower left')
plt.show()
