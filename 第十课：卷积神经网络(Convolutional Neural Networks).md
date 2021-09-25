## 第十课：卷积神经网络(Convolutional Neural Networks)

### 1.1 计算机视觉(Computer vision)

通常如果处理大图用传统的神经网络需要特别大的输入，因此需要大量内存。对于计算机视觉应用来说，要处理大图片，就需要进行卷积计算。

***

### 1.2 边缘检测示例

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.28lr0ce8u2vw.png)

**垂直边沿检测器:**

上图是一个垂直边沿检测器，注意它的计算过程。卷积过程，$6*6$​​的图形经过一个过滤器（或者叫卷积核）$3*3$变成一个$4*4$​图像。​

![image-20210910114711485](D:\typora\深度学习\img\image-20210910114711485.png)

为了更清晰看到，用上图距离，对于一个$3*3$的卷积过滤器，垂直边缘是一个$3*3$的区域。而对于$6*6$​像素的中间部分，可以被视为一个垂直边缘。

***

### 1.3 更多边缘检测内容(More edge detection)

**水平边缘检测：**

将上面的矩阵旋转90度得到：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.wetbluv3kv4.png)

当然还有其他滤波器，其中的权重不同：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.t9z3ajbk9pc.png)

第二个是Sobel filter过滤器，第三个是Scharr filter过滤器。

对于$3*3$过滤器，可以将9个数字都作为参数，下节课讨论。

***

### 1.4 Padding

![image-20210910140555807](D:\typora\深度学习\img\image-20210910140555807.png)

如果输入是$n*n$,卷积核是$f*f$，那么输出是$(n-f+1)*(n-f+1)$。

**Same卷积：**

要想使得输出与之前的输入维度相同，需要填充P个像素点，则，输出变为$(n+2p-f+1)$,令其等于$n$,得到$p=(f-1)/2$,因此当$f$是奇数时，选择相应的填充尺寸，可以得到输出相同的输出尺寸。

在计算机视觉中，通常$f$是奇数，

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.2gfr1irj8xq8.png)

Padding就是在原始输入上填充，p=1在原始输入上填充一圈，以此类推。

也有Valid卷积，也就是p=0。

***

### 1.5 卷积步长

如果定义步长为2，下图表示了计算过程：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.16y2hqj7s1x.png)



![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.61iuwdldj080.png)

stride是步长，一次移动的步长，则输出就是如上图的维度，两边的符号表示向下取整的意思。

***

### 1.6 三维卷积(Convolutions over volumes)

三维卷积过程如下：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.6wonpp1sj9o0.png)

总结一般性规律如下：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.38pva7036z80.png)

如图所示:

其中$n_c$必须相同，后面的$n^{'}_c$表示滤波器的个数，比如图中黄色表示垂直滤波器，输出为$4*4$,深黄色表示水平滤波器，输出为$4*4$​，​则将两个滤波器放一起输出就是$4*4*2$，注意这里没有考虑步长，默认步长为1，要是考虑步长，则关于输出的公式改为前一节的样子。

***

### 1.7 单层卷积网络(One layer of a convolutional network)

单层卷积网络的过程如下：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.7eicjtqppi80.png)

下面是一些符号表示，结合上图搞清楚，对于第$l$层有如下符号表示：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.2dkgnbakkgkk.png)

$f^{[l]}$表示过滤器的尺寸，如上面就是3，$p^{[l]}$表示填充的数量，填充一圈就是1，上面课说过了。$s^{[l]}$表示步长，之前也讲过，$c^{[l]}_c$表示过滤器的数量，上上图表示了有两个过滤器，一水平一个垂直。

对于输入来说，是上一层的输出，如图表示$6*6*3$维度如上(公式太长懒得写。。。)，其中H和W表示高和宽。输出表示本层的最终输出维度，如上图就是$4*4*2$,对于本层的$n^{[l]}_W$计算方式和$n^{[l]}_H$前面几节课讲过。每一个过滤器的大小、激活单元、权重、偏差的维度在图中都给出来了。​

***

### 1.8 简单卷积神经网络示例(A simple convolution network example)

![image-20210910150927131](D:\typora\深度学习\img\image-20210910150927131.png)

上图是卷积神经网络的一个示例，最终将图像处理完毕变成了$7*7*40$，展开为1960个特征，得到一个输出向量，进而使用logistic回归单元或者softmax回归单元。

规律：随着通道数的加深，高度和宽度会逐渐减少39-37-17-7，而通道数在不断增加，3-10-20-40

。

对于一个典型的神经网络通常有三层：

1.卷积层(Conv)

2.池化层(Pool)

3.全连接层(FC)

池化层和全连接层比卷积层更容易设计，后面会讲到。

***

### 1.9 池化层(Pooling layers)

**除了卷积层，卷积网络也经常使用池化层来缩减模型的大小，提高计算速度，同时提高所提取特征的鲁棒性。**

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.71cu2gqdw5w0.png)

池化层有两个超参数，f和s（滤波器大小和步长），池化层没有参数来学习。

池化分为最大池化和平均池化，最大池化用的比较多，如下图所示：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.8mo4bawjfyg.png)

看清计算过程，上面的$f=3,s=1$。

***

### 1.10 卷积神经网络实例（含有全连接层）

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.kz8giymjap8.png)

对于池化层，如果s=2,f=2,则原输入的高度和宽度都减半。

上面是一个神经网络的例子，layer1中有卷积层和池化层，然后FC3,FC4为全连接层。

***

### 1.11 为什么使用卷积

PASS

