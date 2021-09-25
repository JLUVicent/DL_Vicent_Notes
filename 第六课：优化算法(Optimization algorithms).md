## 第六课：优化算法(Optimization algorithms)

### 6.1 Mini-batch梯度下降

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.3zla01qsqpo0.png)

上图表示了整个Mini-batcha梯度下降的过程。

首先对$X^{\{t\}}$执行前项传播，$X^{\{t\}}$表示的是对于整个训练集之后的样本值，比如共有5000000个样本，每1000个划分一次，则$X^{\{t\}}$表示第t个1000个样本的x值，维度为$(n_x,1000)$,注意与X$(n_x,m)$维度的区别.$Y^{\{t\}}$同理，维度为：$(1,1000)$，注意与Y$(1,1000)$维度的区别。

mini-batch与batch区别：使用batch梯度下降法，一次遍历训练集只能做一次梯度下降，而mini-batch可以做5000个梯度下降（以本题为例）。正常来说需要多次遍历训练集，需要另外一层for循环，直到最后能收敛到一个合适的精度。

***

### 6.2 理解mini-batch梯度下降法

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.2kmp2ndxbyk0.png)

第二个图没看懂emmmm

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.7ez35esyhdc0.png)

如上图，如果考虑两种极端的情况：
1.mini-batch的大小等于 𝑚，这个时候也就是batch梯度下降法；

2.mini-batch的大小等于1，这个时候叫随机梯度下降。

batch梯度下降法的缺点：数据量太大，处理速度慢

随机梯度下降的缺点：因为没有向量化的过程，所以速度也会很慢。

样本集较小没必要采取mini-batch梯度下降法。

因此通常在实践中对于mini-batch的大小通常需要选择合适的尺寸，使得学习率达到最高。

上个视频的例子中mini-batch的大小为1000。

***

### 6.3 指数加权平均数(Exponentially weighted averages)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4t88eyixv240.png)

上图蓝色的点绘制的是日期和温度的关系，

作出如下定义：
$$
v_t=\beta v_{t-1}+(1-\beta)\theta_t
$$
其中$v_t$表示第t天的加权平均数，$\theta_t$​表示第t天的温度值。$\beta$​表示加权参数。

$\beta$的值取决所画出的图像平坦程度。如上图所示。$\beta$越大，指数加权平均值适应越缓慢，图像越平缓。

***

### 6.4 理解指数加权平均数(Understanding exponentially weighted averages)

个人理解：第t天的温度是计算之前多少天温度之和的平均值的时候，也就是离第t天越远的之前天数对于第t天的温度影响越小，而这个影响因此，需要令
$$
\beta^{(\frac{1}{1-\beta})}=\frac{1}{e}
$$
比如$\beta=0.9$​​​,则$0.9^{10}=\frac{1}{e}$​​,也就是我们计算之前10天的平均值表示当天的温度

若$\beta=0.98$​,则$0.98^{50}=\frac{1}{e}$​​,也就是我们计算之前50天的平均值表示当天的温度.

这就是个人理解的指数加权平均数。

***

### 6.5 指数加权平均的偏差修正(Bias correction in exponentially weighted averages)

偏差修正是指在估测初期，令
$$
v_t=\frac{v_t}{1-\beta^t}
$$
随着t逐渐增大，$\beta^t$​逐渐变为0，也就和之前温度估测一样了。也就是第t天的温度为$v_t$。

但是吴老师说在大多数时候都不执行偏差修正，除非我们关心初期的计算结果，就需要使用偏差修正来进行计算。

***

### 6.6 动量梯度下降法(Gradient descent with Momentum)

动量梯度下降法(Momentum)通常比梯度下降法要好，过程如下：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.29mo9ugppu68.png)

使用了指数加权平均，吴老师说在有些Momentum算法中忽略了$1-\beta$这一项，但是通常加上这一项比较好，如果忽略这一项，相应的学习率也要随之改变，通常设置$\beta$​为0.9，如上图所示，而通常不需要偏差修正，也就是图中的蓝色公式。

***

### 6.7 RMSprop

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.2mwtcuhw3xo0.png)

和之前的Momentum算法相似，上图给出了算法的具体公式(原理没怎么搞懂。。。)。

注意两点，为了和之后的$\beta$区分，这里用了$\beta_2$来表示，同时为了保证分母不为0，可以加上一个小参数$\xi$,通常$\xi=10^{-8}$。这也是加快梯度运算的算法之一。

***

### 6.8 Adam优化算法(Adam optimization algorithm)

该算法是Momentum算法和RMSprop算法的结合，如下图所示：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4h57wupvjyc0.png)

关于一些参数的选择参考下图：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.nxdnefrin5c.png)

***

### 6.9 学习率衰减(Learning rate decay)

慢慢减少$\alpha$的本质在于，在学习初期，你能承受较大的步伐，但当开始收敛的时候，小一些的学习率能让你步伐小一些。

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.2vsp2e3rxsw.png)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.3akj1pwy0iq0.png)

上图给出了$\alpha$的选择公式，其中epoch-num代表迭代次数。

***

### 6.10 局部最优的问题(The problem of local optima)

PASS