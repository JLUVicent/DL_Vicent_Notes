## 第十一课：深度卷积网络：实例探究(Deep convolutional models:case studies)

### 2.1 为什么要进行实例探究

PASS

***

### 2.2 经典网络(Classic networks)

**三种经典的网络结构**

1.LeNet-5

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4ft63hukbze0.png)

该网络结构没有使用padding，对于池化层，如果s=2,f=2,则图像的高度和宽度都缩小2倍，随着网络层的增加，图像的高度和宽度在缩小，而通道数在增加。

用的是平均池化

2.AlexNet

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.17lq92dir4cg.png)

使用了same卷积，使用后图像的高度和宽度不变，使用了最大池化后宽度和高度减半。

3.VGG-16

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.280z2rt7chhc.png)

Conv 64表示卷积核有64个，VGG-16表示有16个网络层和全连接层。其本身结构简单，没经过一次网络层，宽度和高度都减半，通道数都翻倍。

***

### 2.3 残差网络(ResNets)

国内的何恺明大佬提出的

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.6ouglaua0k00.png)

个人理解：如上图，对于两层神经网络，若要计算$a^{[l+2]}$，需要进行一步步线性操作以及使用Relu激活函数，也就是，信息从$a^{[l]}$到$a^{[l+2]}$​需要经过上面的计算过程。​

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.8gd9bdx6vf4.png)

而残差网络相当于直接跳过了$a^{[l+1]}$直接拷贝到神经网络的深层，然后在ReLU非线性激活函数上加上$a^{[l]}$,如下：
$$
a^{[l+2]}=g(z^{[l+2]}+a^{[l]})
$$
也就是加上了$a^{[l]}$产生了残差块。

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.7hlz7rv6m8k.png)

如上图，蓝色框里面都是残差块，残差网络能使得神经网络在训练过程中误差一直减少。

***

### 2.4 残差网络有用的原因？

对于普通的网络，如果深度越深，训练效率就会变慢。

对于残差网络来说，如果残差块里面的隐层单元学到了一些东西，则它比学习恒等函数(在之前设置其权重和b都为0的时候)表现得更好。如下图

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4jsmqbg3ca60.png)

ResNets使用了很多的same卷积，保留了之前的维度。

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.1d2gkjnvnav4.png)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.6nl2he2tepc0.png)

前面是在全连接层使用残差网络，这块是在卷积层使用残差网络，跳跃连接。

***

### 2.5 网络中的网络以及1*1卷积

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.35n3hodorp60.png)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.rinuhclg7b4.png)

$1*1$网络让我们能够任意变换原输入的通道数，或者加上ReLU线性修正激活函数。

***

### 2.6 谷歌Inception网络(Inception network motivation)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.640no3sdbqw0.png)

基本思想是 Inception网络不需要人为决定使用哪个过滤器或者是否需要池化，而是由网络自行确定这些参数，你可以给网络添加这些参数的所有可能值，然后把这些输出连接起来，让网络自己学习它需要什么样的参数，采用哪些过滤器组合。

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.7dyup6bxxzo0.png)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.3wubkno5jkm0.png)

上面两个图表示了使用$1*1$​卷积之后可以减小计算量，降低计算成本。这是Inception模块的主要思想。

***

### 2.7 Inception网络

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.3ozj6ltb0iy0.png)

上面是一个Inception模块

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.5speru3dz400.png)

这是一个Inception网络，就是将很多Inception模块连接起来。

***

### 2.8 使用开源实现方案

ResNets实现的 GitHub地址 https://github.com/KaimingHe/deep-residual-networks

***

### 2.9 迁移学习(Transfer learning)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.6vmujpfqxp40.png)

将网络上的神经网络和已经训练好的权重拿来进而通过冻结某些层数来训练自己的数据。

***

### 2.10 数据增强(Data augmentation)

和之前重复了好像

PASS

***

### 2.11 计算机视觉现状

通常需要大量人工

总之，多参考别人的训练项目。

***

OVER

冲！

