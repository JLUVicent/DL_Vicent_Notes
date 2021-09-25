## 第四课：深层神经网络(Deep L-layer neural network)

### 4.1 深层神经网络

主要需要掌握一些符号，如下图：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4eiw33dt3240.png)

***

### 4.2 前向传播和反向传播(Forward and backward propagation)

​    反向传播的向量化实现：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4jrcne0egk80.png)

***

### 4.3 深层网络中的前向传播(Forward propagation in a Deep Network)

对于前项传播向量化实现过程可以归纳为多次迭代如下公式:
$$
Z^{[l]}=W^{[l]}A^{[l-1]}+b{[l]}(l表示层数)\\
A^{[l]}=g^{[l]}(Z^{[l]})其中(A^{[0]}=X)
$$
该过程是在整个训练集上进行的，而且要遍历每一层，需要用到一个显式for循环，从1到L进行遍历。

***

### 4.4 核对矩阵的维度(Getting your matrix dimensions right)

对于单个训练样本：
$$
z^{[l]}=w^{[l]}a^{[l-1]}+b{[l]}(l表示层数)\\
a^{[l]}=g^{[l]}(z^{[l]})其中(a^{[0]}=x)
$$
其中对应矩阵的维度如下：
$$
z^{[l]}或a^{[l]}:(n^{[l]},1)\\
w^{[l]}或dw^{[l]}:(n^{[l]},n^{[l-1]})\\
b^{[l]}或db^{[l]}:(n^{[l]},1)
$$


对于向量化m个样本后的矩阵：
$$
Z^{[l]}=W^{[l]}A^{[l-1]}+b{[l]}(l表示层数)\\
A^{[l]}=g^{[l]}(Z^{[l]})其中(A^{[0]}=X)
$$
其中对应矩阵的维度如下：
$$
Z^{[l]}、dZ^{[l]}、A^{[l]}、dA^{[l]}:(n^{[l]},m)\\
W^{[l]}或dW^{[l]}:(n^{[l]},n^{[l-1]})\\
b^{[l]}或db^{[l]}:(n^{[l]},m)\\
l=0时，A^{[0]}=X=(n^{[l]},m)
$$

***

### 4.5 为什么使用深层表示？

PASS

***

### 4.6 搭建神经网络块

介绍整个传播步骤：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4bq2b4bwi6q0.png)

如上图，上面一行蓝色箭头表示正向传播的过程，其中得到了缓存$cache  z^{[l]}$​​​​用于反向传播,红色箭头表示反向传播的过程，方框中的参数是整个过程中所需要的参数，整个绿色箭头表示了整个神经网络的过程，得到：
$$
W^{[l]}=W^{[l]}-\alpha{d}W^{[l]}\\
b^{[l]}=b^{[l]}-\alpha{d}b^{[l]}
$$

***

### 4.7 参数 VS 超参数(Parameters Vs Hyperparameters)

要想使得神经网络起到很好的效果，必须规划参数以及超参数。

**参数：**

$W^{[l]},b^{[l]}$

**超参数：**

算法中的学习率($\alpha$​​),梯度下降法循环的迭代次数，隐藏层的数目(L),隐藏层单元数目($n^{[l]}$​,激活函数的选择，这些参数控制着最后的参数$W,b$的值，因此称为超参数。

如何寻找超参数的最优值：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4ve7tk7pqg80.png)

走Idea—Code—Experiment—Idea这个循环 尝试各种不同的参数 实现模型并观察是
否成功，然后再迭代。

***

### 4.8 深度学习和大脑的关联性

毫无关联！

OVER！

