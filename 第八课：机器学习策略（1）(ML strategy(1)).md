## 第八课：机器学习策略（1）(ML strategy(1))

### 1.1 什么是ML策略

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.3vsphvrv73w0.png)

ML策略总结就是让人少走弯路，能够选择合适的方法来优化系统。

***

### 1.2 正交化(Orthogonalization)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4084df3i86k0.png)

针对不同的环节出现的问题进行不同方式的解决。判断出系统的性能瓶颈出现在那里，然后找到一组特定的旋钮来调整系统，来改善它特定的性能。

***

### 1.3 单一数字评估指标(Single number evaluation metric)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4cykijtb6zi0.png)

对于上面两个分类器，分别给出了它们的查准率和召回率，通常这两个特征必须折中，因此无法判断哪个分类器的效果更好，我们引入一个参数：
$$
F_1Score:2\frac{PR}{P+R}
$$
如上图所示，很显然分类器A的参数值FScore最大，因此选择A分类器。

选择$F_1Score$较大的那个值

>通常将算法的预测结果分为四种情况：
>
>1.正确肯定(True Positive,TP)：预测为真，实际为真；
>
>2.正确否定(True Negative,TN)：预测为假，实际为真；
>
>3.错误肯定(False Positive,FP)：预测为真，实际为假；
>
>4.错误否定(False Negative,FN)：预测为假，实际为真。
>$$
>查准率(Precision)=\frac{TP}{TP+FP}\\
>查全率/召回率(Recall)=\frac{TP}{TP+FN}
>$$
>![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4azxxh50ehc.png)
>
>查准率通常用P表示，查全率或召回率通常用R表示，则可以得到上面F_1Score的公式。

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.1jkb8qx6ahgg.png)

对于如上图可以选择每个算法在各地的误差平均值，计算之后发现平均值误差最小的是算法C，因此我们选择算法C.

这就是单一数字评估指标的基本概念，选择一个数字来评估。

***

### 1.4 满足和优化指标(Satisficing and opeimizing metrics)

当需要顾及多个指标，比如有一个优化指标以及一个或多个满足指标，对于需要满足的指标，需要达到一定门槛即可。这些评价指标必须是在训练集、开发集、测试集上求出来的，因此必须设立训练集、开发集、测试集。下节课见。

***

### 1.5 训练/开发/测试集划分(Train/dev/test distributions)

选择开发集以及评估指标，就定义了所要瞄准的目标。同时让开发集和测试集在同一分布之中。

***

### 1.6 开发集和测试集的大小(Size of dev and test sets)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4cd083qquns0.png)

划分训练集、开发集、测试集划分方法如上，如果数据量较少可以划分为7:3和6:2:2.但如果数据量比较大，可以划分为98:1:1。

在实际工作中，可能有时候不需要测试集，只有开发集和训练集两部分。测试集的目的是评估最终的成本偏差。

***

### 1.7 何时改变开发/测试集/指标

实操经验：首先构建分类器和指标，将设立目标作为第一步，而瞄准和射击目标作为第二步，也就是在设立目标之后，应该想着如何优化系统提高指标评分，比如改变神经网络的优化成本函数J。

在解决问题时候，应该首先设立评估指标和开发集。

***

### 1.8 为何比较机器学习和人类的表现

![image-20210909085433587](D:\typora\深度学习\img\image-20210909085433587.png)

贝叶斯最优错误率：指理论上可能达到的最优错误率，无论如何设置，都无法让其超过一定的准确度。

如上图蓝线为人类的精确度，绿线为贝叶斯最优错误率，紫色线表示机器学习的学习表现。

对于人类擅长的任务：比如可以让人标记数据，人工错误率分析，同时更好的分析偏差和方差。

***

### 1.9 可避免偏差(Avoidable bias)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.2d893r3zd2dc.png)

选择避免方差策略还是避免偏差策略：

如上图，当贝叶斯误差与训练集误差之差比开发集误差与训练集误差之差比较相对较大时候，选择避免偏差策略，

相反，当贝叶斯误差与训练集误差之差比开发集误差与训练集误差之差比较相对较小时候，选择避免方差策略。

***

### 1.10 理解人的表现

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.23d6wfsxhqbk.png)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.5sgxxfguuf40.png)

人类水平错误率可以用贝叶斯错误率来近似代替，在人类水平误差与训练集误差之间用来调试偏差，在训练集误差与开发集误差之间人们用来调试方差。

***

### 1.11 超过人的表现

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4zz5786f1tk0.png)

机器学习超过人的水平？？？

***

### 1.12 改善你的模型表现

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.7hs7k7s5ke00.png)

**解决高方差（过拟合）问题：**

1.获得更多的训练样本

2.减少特征的数量

3.尝试增加正则化程度$\lambda$

**解决高偏差（欠拟合）问题：**

1.增加特征的数量

2.增加多项式特征

3.减少正则化程度$\lambda$

4.训练更好的优化算法，如Rmsprop,adam,momunte等等

如上图所示

***

2021/9/9结束，冲冲冲

