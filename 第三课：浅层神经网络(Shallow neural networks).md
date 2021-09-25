## 第三课：浅层神经网络(Shallow neural networks)

### 3.1 神经网络概述

 PASS

***

### 3.2 神经网络的表示

![神经网络图](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/神经网络图.5wxc91x3nr80.jpg)

如上图，从左到右依次为<font color='red'>输入层</font>、<font color='red'>隐藏层</font>、只有一个节点的层为<font color='red'>输出层</font>，负责输出预测值。

![神经网络1](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/神经网络1.6a2uqexkwlg0.jpg)

一般称上图网络为<font color='red'>两层神经网络</font>，一般不把输入层看做一个标准层，因此该网络有一个隐藏层和输出层。

**在隐藏层有两个参数$W$​和$b$​,通常表示为$W^{[1]},b^{[1]}$​,$W$​为$4*3$​矩阵，$b$​为$4*1$​矩阵，$4$​来自于有四个节点或者隐藏层单元，$3$​表示有三个特征输入。同理我们得到输出层参数$W^{[2]},b^{[2]}$​，他们分别是$1*4$​和$1*1$​​维度矩阵​。**

***

### 3.3 计算一个神经网络的输出

![一个神经网络输出](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/一个神经网络输出.1ptzpddajsdc.jpg)

如上图，对于一个训练样本，根据给出一个单独的输入特征向量，根据上限四个公式，进而计算出一个简单神经网络的输出。

***

### 3.4 多样本向量化

$a^{[2](i)}$对于上面的网络表示的是第$i$​个训练样本的第二层输出值。

若要实现所有样本，可以使用循环方法来对上面式子进行循环，要注意所有样本要加上$(i)$,比如$z^{[1](i)}$,其他也一样，加上上标$(i)$​​​​，如下图.

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.1z3urbeoy2m.png)

然而通常使用向量化方法：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.al2kzob0oag.png)

上图中的$X,Z^{[1]},A^{[1]}$​​矩阵水平方向上代表了不同的训练样本，从竖直方向上代表了不同的隐藏单元（不同的输入特征），将训练样本横向堆叠成一个矩阵X。

**向量化方法如下：**
$$
Z^{[1]}=W^{[1]}X+b^{[1]}\\
A^{[1]}=\sigma(Z^{[1]})\\
Z^{[2]}=W^{[2]}A^{[1]}+b^{[2]}\\
A^{[2]}=\sigma(Z^{[2]})
$$

***

### 3.5 向量化实现的解释

总结：将列向量横向堆叠成矩阵通过公式计算后，得到成列堆叠的输出。

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.7e593rws0r00.png)

在此之前一直使用$sigmoid$函数，本节课内容有助于理解向量化实现，下节课介绍不同种类的激活函数。

***

### 3.6 激活函数(Activation functions)

**双曲正切函数(tanh函数):**
$$
a=tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$
如图该函数的值域是$[-1,1]$​​

<font color='red'>不同层的激活函数可以不同</font>，通常tanh函数效果比sigmoid函数要好，因此我们可以在隐藏层用tanh函数，但是输出层我们希望其输出值$\widehat{y}$在[0,1]之间，因此在输出层使用sigmoid函数比较好。

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.1rml13s6prfk.png)

上图3为线性修正单元(Relu),$a=max(0,z)$

总结：

1.对于sigmoid函数：用在二分类的输出层，几乎不用；

2.最常用的是ReLU函数

***

### 为何需要非线性激活函数

如果使用线性激活函数，则神经网络只是把输入线性组合再输出，这样的话隐藏层就没啥用了。。。

可以使用线性激活函数的地方：**输出层**（但这玩意也不常用！！！）

***

### 3.8 激活函数的导数(Derivatives of activation functions)

**(1) sigmoid函数**

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.g6tojqnqjug.png)
$$
g(z)=\frac{1}{1+e^{-z}}
$$


其导数：
$$
\frac{d}{dz}g(z)=g(z)(1-g(z))
$$
当z=10或z=-10,导数约为0，

当z=0,根据图可以得到导数约为1/4.

**通常在神经网络中：**
$$
a=g(z);\\
g(z)'=\frac{d}{dz}g(z)=a(1-a)
$$
以下几个表示方法同理，下面省略。

**(2)Tanh 函数**

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4qktlhvcbfu0.png)

其导数为：
$$
\frac{d}{dz}g(z)=1-(g(z))^2
$$
当z=10或z=-10，其导数约为0，

当z=0,导数为1。

**（3）ReLU（线性修正函数）**

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.7ctp0gjkig80.png)
$$
g(z)=max(0,z)\\
g(z)^{'} =    \left\{\begin{array}{rcl}
0 & \mbox{if}&z<0\\
1 &\mbox{if}&z>0\\
undefined & \mbox{if} & z=0
      \end{array}\right.
$$
**注**

通常在z=0时候给定其导数为1,0。但是z=0的情况非常少。

**(4)Leaky ReLU（泄露线性修正函数）**
$$
g(z)=max(0.01z,z)\\
g(z)^{'} =    \left\{\begin{array}{rcl}
0.01 & \mbox{if}&z<0\\
1 &\mbox{if}&z>0\\
undefined & \mbox{if} & z=0
      \end{array}\right.
$$
**注：**

通常在z=0的时候给定其导数为1，0.01，同上z=0的情况很少。

***

### 3.9 神经网络的梯度下降

**1.正向传播（四个式子）：**
$$
Z^{[1]}=W^{[1]}X+b^{[1]}\\
A^{[1]}=g^{[1]}(Z^{[1]})\\
Z^{[2]}=W^{[2]}A^{[1]}+b^{[2]}\\
A^{[2]}=g^{[2]}(Z^{[2]})
$$
**2.反向传播（六个式子）：**

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.47rl4e3czd40.png)

这些公式都是针对所有样本进行向量化，其中$Y$是$1*m$矩阵

$np.sum$​中$axis=1$​表示水平相加求和，$keepdims$​是防止Python输出类似$(n,)$​​，确保输出类似于$(n,1)$​​​维矩阵。还用一种输出方式，调用$reshape$。

***

### 3.10 （选修）直观理解反向传播

PASS

***

### 3.11 随机初始化(Random+Initialization)

**如果随机初始化权重或者参数都为0，梯度下降将不会起作用。**

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.5jkxn1duuh00.png)

通常做如下初始化：
$$
W^{[1]}=np.random.randn(2,2)*0.01\\
b^{[1]}=np.zeros((2,1))\\
W^{[2]}=np.random.randn(2,2)*0.01\\
b^{[2]}=0
$$
**关于参数0.01的解释：**

通常对于$Z^{[1]}=W^{[1]}X+b^{[1]},a^{[1]}=g^{[1]}(Z^{[1]})$​​,如果使用激活函数为tanh或者sigmoid函数，如果数值波动很大，则初始值会停在tanh/sigmoid函数图像平坦的地方，此时梯度很小，下降就会很慢，学习也就很慢，因此通常初始值都选择较小的值。

而对于浅层神经网络，也就是只用一层隐藏层的神经网络，设置为0.01可以使用。对于较深的神经网络，需要设置其他常数，这个在下一课出现。

***



