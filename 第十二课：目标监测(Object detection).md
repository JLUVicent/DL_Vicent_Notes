### 第十二课：目标监测(Object detection)

### 3.1 目标定位(Object localization)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.551eg4vvumk0.png)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.482po981zeg0.png)

解释上图：

对于目标监测的输出y，第一个参数$P_c$​如果图片中有目标，比如行人，车或者自行车，则输出1，如果是背景则输出0.接下来的四个参数用来定位，具体定义在第一张图片中已经给出了。最后几个参数$C_1,C_2,C_3$表示分别是行人，车，自行车，如果是自行车，则输出$C_3$为1，其他为0.

其损失函数如上图所示。

***

### 3.2 特征点检测(Landmark detection)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.2bj7dyj2m05c.png)

特征点检测通过在最后一层添加输出变量，如图所示，图二标记了64个特征点，然后让其在最后一层输出，如上面的网络最后一层，第一个参数表示是否识别到了人脸，后面的参数表示特征的具体坐标位置，根据这样我们可以判断出人的表情变化，或者判断人的姿态，比如是行走还是在奔跑等等。

***

### 3.3 目标检测(Object detection)

基于滑动窗口的目标检测算法

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.31f2gubmijk0.png)

让方框在图中从左上角按照一定步长进行遍历找到目标位置

缺点：计算量比较大，当步长太大，误差比较大。

***

### 3.4 滑动窗口的卷积实现(Convolutional implementation of sliding windows)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.411uvg3pz6e0.png)

解释：滑动窗口的卷积实现就是从左上角开始，以固定步长进行遍历，这里的步长是由最大池化的维度决定的，如图步长为2，然后得到一个$2*2*400$的输出全连接层，其中蓝色框代表左上角的矩形窗口，绿色框代表右上角，以此类推。

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.7egjpucbd8g.png)

如上图，如果以$14*14$进行滑动窗口卷积，可以得到最终$8*8$​的输出层，这里面的每一个方框对应相对位置的矩形框。

优点：效率高

缺点：不能准确预测矩形框的位置。

***

### 3.5 Bounding Box预测(Bounding box predictions)

Yolo算法效率比较高，因为它用了卷积，同时能够精确输出标准框，关于原理似懂非懂。

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.72kujug6cvk0.png)

大概过程就是，先给训练集打标签(工作量巨大)，如上图，每个框输出8个特征向量，具体含义本周第一课已经讲过了，可以确定出目标的位置。

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.2uwsl6mubzg.png)

上面这个图是关于四个参量的表示方法，其中$b_h,b_w$​表示反了，注意下。。。表示方法也和前面目标定位那块是一样的。

> Redmon, Joseph, et al. "You Only Look Once: Unified, Real-Time Object Detection." (2015):779-788.（原来大佬的yolo论文）

***

### 3.6 交并比(Intersection over union)

判断对象检测算法运作是否良好？

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4w0v3pxp59q0.png)

交并比用来表示打标签的框与预测框的交集与并集之比，如果两个框重合，则交并比为1.

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.3d88i0hqnnw0.png)

**交并比来衡量两个边界框重叠的相对大小**

***

### 3.7 非极大值抑制(Non-max suppression)

**非极大值抑制的方法作用：确保算法对每个对象只检测一次。**

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.19j1q1roryio.png)

如上图所示，对于预测的输出，假设就是针对汽车，去掉了之前的$C_1,C_2,C_3$，将$P_c$表示为概率，先去掉概率小于0.6的，然后找到最大的概率作为输出的预测，然后抛弃掉与最大概率框的交并比大于0.5的框，这就是非极大值抑制。

***

### 3.8 Anchor Boxes

Anchor Boxes：用来解决一个格子中有多个对象的问题。

对象在目标标签中的编码方式：(grid cell,anchor box)，表示实际边界框与anchor box的交并比越高，则就选择较高的anchor box。

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.70qhf571fxk0.png)

对于一个格子中有两个图像的问题：设定输出有16个 向量，比如本题，前8个表示行人，后8个表示汽车。

通常一个格子中很少有三个对象。

***

### 3.9 YOLO算法(将前面学习的综合起来)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.37fotetfhao0.png)

在训练阶段，对于每个框得到固定的输出，对于有汽车的框，注意输出。

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4tdjtvbsrta0.png)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.3cbckhocgto0.png)

预测过程如下：

1.对于每个格子，都得到两个输出预测边界框；

2.去掉概率很低的预测；

3.对于每个类别单独运行非极大值抑制。

***

### 3.10 候选区域（选修）(Region proposals)

讲了个R-CNN,了解

***

OVER!

