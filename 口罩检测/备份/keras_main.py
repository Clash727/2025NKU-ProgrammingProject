###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import warnings
# 忽视警告
warnings.filterwarnings('ignore')
import os
import matplotlib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils,get_file

K.image_data_format() == 'channels_last'
from keras_py.utils import get_random_data
from keras_py.face_rec import mask_rec
from keras_py.face_rec import face_rec
from keras_py.mobileNet import MobileNet
# 数据集路径
basic_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/"
mask_num = 4
fig = plt.figure(figsize=(15, 15))
for i in range(mask_num):
    sub_img = cv.imread(basic_path + "/image/mask/mask_" + str(i + 101) + ".jpg")
    sub_img = cv.cvtColor(sub_img, cv.COLOR_RGB2BGR)
    ax = fig.add_subplot(4, 4, (i + 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("mask_" + str(i + 1))
    ax.imshow(sub_img)
nomask_num = 4
fig1 = plt.figure(figsize=(15, 15))
for i in range(nomask_num):
    sub_img = cv.imread(basic_path + "/image/nomask/nomask_" + str(i + 130) + ".jpg")
    sub_img = cv.cvtColor(sub_img, cv.COLOR_RGB2BGR)
    ax = fig1.add_subplot(4, 4, (i + 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("nomask_" + str(i + 1))
    ax.imshow(sub_img)
def letterbox_image(image, size):
    """
    调整图片尺寸
    :param image: 用于训练的图片
    :param size: 需要调整到网络输入的图片尺寸
    :return: 返回经过调整的图片
    """
    new_image = cv.resize(image, size, interpolation=cv.INTER_AREA)
    return new_image
read_img = cv.imread("test1.jpg")
print("调整前图片的尺寸:", read_img.shape)
read_img = letterbox_image(image=read_img, size=(50, 50))
print("调整前图片的尺寸:", read_img.shape)
# 导入图片生成器
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def processing_data(data_path, height, width, batch_size=32, test_split=0.1):
    """
    数据处理
    :param data_path: 带有子目录的数据集路径
    :param height: 图像形状的行数
    :param width: 图像形状的列数
    :param batch_size: batch 数据的大小，整数，默认32。
    :param test_split: 在 0 和 1 之间浮动。用作测试集的训练数据的比例，默认0.1。
    :return: train_generator, test_generator: 处理后的训练集数据、验证集数据
    """

    train_data = ImageDataGenerator(
            # 对图片的每个像素值均乘上这个放缩因子，把像素值放缩到0和1之间有利于模型的收敛
            rescale=1. / 255,  
            # 浮点数，剪切强度（逆时针方向的剪切变换角度）
            shear_range=0.1,  
            # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
            zoom_range=0.1,
            # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
            width_shift_range=0.1,
            # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
            height_shift_range=0.1,
            # 布尔值，进行随机水平翻转
            horizontal_flip=True,
            # 布尔值，进行随机竖直翻转
            vertical_flip=True,
            # 在 0 和 1 之间浮动。用作验证集的训练数据的比例
            validation_split=test_split  
    )

    # 接下来生成测试集，可以参考训练集的写法
    test_data = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=test_split)

    train_generator = train_data.flow_from_directory(
            # 提供的路径下面需要有子目录
            data_path, 
            # 整数元组 (height, width)，默认：(256, 256)。 所有的图像将被调整到的尺寸。
            target_size=(height, width),
            # 一批数据的大小
            batch_size=batch_size,
            # "categorical", "binary", "sparse", "input" 或 None 之一。
            # 默认："categorical",返回one-hot 编码标签。
            class_mode='categorical',
            # 数据子集 ("training" 或 "validation")
            subset='training', 
            seed=0)
    test_generator = test_data.flow_from_directory(
            data_path,
            target_size=(height, width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            seed=0)

    return train_generator, test_generator
# 数据路径
data_path = basic_path + 'image'

# 图像数据的行数和列数
height, width = 160, 160

# 获取训练数据和验证数据集
train_generator, test_generator = processing_data(data_path, height, width)

# 通过属性class_indices可获得文件夹名与类的序号的对应字典。 (类别的顺序将按照字母表顺序映射到标签值)。
labels = train_generator.class_indices
print(labels)

# 转换为类的序号与文件夹名对应的字典
labels = dict((v, k) for k, v in labels.items())
print(labels)
pnet_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/keras_model_data/pnet.h5"
rnet_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/keras_model_data/rnet.h5"
onet_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/keras_model_data/onet.h5"
# 读取测试图片
img = cv.imread("test.jpg")
# 转换通道
img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
# 加载模型进行识别口罩并绘制方框
detect = face_rec(pnet_path,rnet_path,onet_path)
detect.recognize(img)
# 展示结果
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('mask_1')
ax1.imshow(img)
# 加载 MobileNet 的预训练模型权重
weights_path = basic_path + 'keras_model_data/mobilenet_1_0_224_tf_no_top.h5'
# 图像数据的行数和列数
height, width = 160, 160
model = MobileNet(input_shape=[height,width,3],classes=2)
model.load_weights(weights_path,by_name=True)
print('加载完成...')
def save_model(model, checkpoint_save_path, model_dir):
    """
    保存模型，每迭代3次保存一次
    :param model: 训练的模型
    :param checkpoint_save_path: 加载历史模型
    :param model_dir: 
    :return: 
    """
    if os.path.exists(checkpoint_save_path):
        print("模型加载中")
        model.load_weights(checkpoint_save_path)
        print("模型加载完毕")
    checkpoint_period = ModelCheckpoint(
        # 模型存储路径
        model_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        # 检测的指标
        monitor='val_acc',
        # ‘auto’，‘min’，‘max’中选择
        mode='max',
        # 是否只存储模型权重
        save_weights_only=False,
        # 是否只保存最优的模型
        save_best_only=True,
        # 检测的轮数是每隔2轮
        period=2
    )
    return checkpoint_period
checkpoint_save_path = "./results/temp.h5"
model_dir = "./results/"
checkpoint_period = save_model(model, checkpoint_save_path, model_dir)
# 学习率下降的方式，acc三次不下降就下降学习率继续训练
reduce_lr = ReduceLROnPlateau(
                        monitor='acc',  # 检测的指标
                        factor=0.5,     # 当acc不下降时将学习率下调的比例
                        patience=2,     # 检测轮数是每隔两轮
                        verbose=2       # 信息展示模式
                    )
early_stopping = EarlyStopping(
                            monitor='val_loss',  # 检测的指标
                            min_delta=0,         # 增大或减小的阈值
                            patience=10,         # 检测的轮数频率
                            verbose=1            # 信息展示的模式
                        )
# 一次的训练集大小
batch_size = 8
# 图片数据路径
data_path = basic_path + 'image'
# 图片处理
train_generator,test_generator = processing_data(data_path, height=160, width=160, batch_size=batch_size, test_split=0.1)
# 编译模型
model.compile(loss='binary_crossentropy',  # 二分类损失函数   
              optimizer=Adam(lr=0.1),            # 优化器
              metrics=['accuracy'])        # 优化目标
# 训练模型
history = model.fit(train_generator,    
                    epochs=3, # epochs: 整数，数据的迭代总轮数。
                    # 一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
                    steps_per_epoch=637 // batch_size,
                    validation_data=test_generator,
                    validation_steps=70 // batch_size,
                    initial_epoch=0, # 整数。开始训练的轮次（有助于恢复之前的训练）。
                    callbacks=[checkpoint_period, reduce_lr])
# 保存模型
model.save_weights(model_dir + 'temp.h5')
plt.plot(history.history['loss'],label = 'train_loss')
plt.plot(history.history['val_loss'],'r',label = 'val_loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'],label = 'acc')
plt.plot(history.history['val_accuracy'],'r',label = 'val_acc')
plt.legend()
plt.show()
import cv2 as cv
# 读取图片
img = cv.imread("./test1.jpg")
img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

# 最佳模型路径
model_path = "results/temp.h5"

# 加载训练模型并进行口罩识别
detect = mask_rec(model_path)
img, all_num, mask_num = detect.recognize(img)

# 展示图片口罩识别结果
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('test_mask')
ax1.imshow(img)
print("图中的人数有：" + str(all_num) + "个")
print("戴口罩的人数有：" + str(mask_num) + "个")
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# 复用之前导入的模块和路径
from keras_py.mobileNet import MobileNet

# ==================== 1. 参数与路径设置 ====================
# 基础路径 (根据 Notebook 上下文)
basic_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/"
data_path = basic_path + 'image'
# 预训练权重路径
weights_path = basic_path + 'keras_model_data/mobilenet_1_0_224_tf_no_top.h5'
# 结果保存路径 (作业要求保存在 results 文件夹)
model_dir = "./results/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 图像参数
height, width = 160, 160
batch_size = 32  # 适当增加 batch_size 以加快训练速度

# ==================== 2. 数据处理 ====================
print("正在处理数据...")
# 使用 2.5 节定义的 processing_data 函数
# 注意：如果报错 'processing_data' 未定义，请务必先运行 Notebook 第 2.5 节的代码单元格
train_generator, test_generator = processing_data(
    data_path, 
    height=height, 
    width=width, 
    batch_size=batch_size, 
    test_split=0.1
)

# ==================== 3. 构建模型 ====================
print("正在构建 MobileNet 模型...")
# 初始化 MobileNet，分类数为2 (mask, nomask)
model = MobileNet(input_shape=[height, width, 3], classes=2)

# 加载预训练权重 (Transfer Learning)
if os.path.exists(weights_path):
    model.load_weights(weights_path, by_name=True)
    print("预训练权重加载成功。")
else:
    print("未找到预训练权重，将从头开始训练。")

# ==================== 4. 训练配置 (Callbacks) ====================
# 1. 保存最佳模型：监控验证集准确率，保存在 results 文件夹下的 temp.h5
checkpoint = ModelCheckpoint(
    filepath=os.path.join(model_dir, 'temp.h5'),
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=True, # 仅保存权重，减小文件体积
    mode='max'
)

# 2. 学习率调整：如果 accuracy 2轮不上升，学习率减半
reduce_lr = ReduceLROnPlateau(
    monitor='accuracy',
    factor=0.5,
    patience=2,
    verbose=1
)

# 3. 早停法：如果 val_loss 10轮不下降，提前结束训练
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1
)

# ==================== 5. 编译与训练 ====================
# 编译模型
model.compile(
    loss='binary_crossentropy',  # 二分类常用损失函数
    optimizer=Adam(learning_rate=1e-3), # 初始学习率
    metrics=['accuracy']
)

print("开始训练模型...")
# 计算每个 epoch 的步数
steps_per_epoch = max(1, train_generator.samples // batch_size)
validation_steps = max(1, test_generator.samples // batch_size)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_generator,
    validation_steps=validation_steps,
    epochs=10,  # 训练轮数，可根据需要调整 (例如 10-20)
    callbacks=[checkpoint, reduce_lr, early_stopping]
)

# ==================== 6. 结果可视化与评估 ====================
# 绘制 Loss 曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], 'r', label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制 Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], 'r', label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 最终评估
loss, acc = model.evaluate(test_generator, steps=validation_steps)
print(f"训练结束！\n验证集最终 Loss: {loss:.4f}, Accuracy: {acc:.4f}")
print(f"最佳模型已保存至: {os.path.join(model_dir, 'temp.h5')}")
from keras_py.utils import get_random_data
from keras_py.face_rec import mask_rec
from keras_py.face_rec import face_rec
from keras_py.mobileNet import MobileNet
from PIL import Image
import cv2

# -------------------------- 请加载您最满意的模型 ---------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 dnn.h5 模型，则 model_path = 'results/temp.h5'
model_path = results/temp.h5
# ---------------------------------------------------------------------------

def predict(img):
    """
    加载模型和模型预测
    :param img: cv2.imread 图像
    :return: 预测的图片中的总人数、其中佩戴口罩的人数
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    # 将 cv2.imread 图像转化为 PIL.Image 图像，用来兼容测试输入的 cv2 读取的图像（勿删！！！）
    # cv2.imread 读取图像的类型是 numpy.ndarray
    # PIL.Image.open 读取图像的类型是 PIL.JpegImagePlugin.JpegImageFile
    if isinstance(img, np.ndarray):
        # 转化为 PIL.JpegImagePlugin.JpegImageFile 类型
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    detect = mask_rec(model_path)
    img, all_num, mask_num = detect.recognize(img)
    
    # -------------------------------------------------------------------------
    return all_num,mask_num
# 输入图片路径和名称
img = cv.imread("test1.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
all_num,mask_num = predict(img)
# 打印预测该张图片中总人数以及戴口罩的人数
print(all_num, mask_num)