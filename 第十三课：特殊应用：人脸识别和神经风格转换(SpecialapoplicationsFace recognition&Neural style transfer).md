## 第十三课：特殊应用：人脸识别和神经风格转换(Specialapoplications:Face recognition&Neural style transfer)

### 4.1 什么是人脸识别？

科普

人脸识别可能一个人的识别准确率是99%,那么100个人的识别可能需要更高的准确率，99.9%等等。

***

### 4.2 One-Shot学习

人脸识别所面临的一个挑战就是需要解决一次学习问题，要想让人脸识别做到一次学习，应该使用Similarity函数，如下图：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.45sle2686l80.png)

查看输入的两张图片(img1,img2)的差异性，如果差异性小于一个数，说明相同，差异性大于一个数，说明不相同。

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.2v8599yn6jk0.png)

***

### 4.3 Siamese网络(Siamese network)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.58uyeecf3qw0.png)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.1v9m06jeuvp.png)

Siamese网络就是首先定义了一个编码函数，对于输入的函数，能够输出一个128维编码，如果两个输入对应的输出的范数比较小，就是同一个人，相反，就是不同的人。

***

### 4.4 Triplet损失

**定义三元组损失函数然后应用梯度下降**

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.4ldu7cognwq0.png)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.2e8hjzxeutgk.png)

上面的公式是损失函数，给出3个图片，A、P、N，其中A和P是同一个人，A和N是不同的人，定义损失函数如上，$+\alpha$​是为了防止损失函数大于0，损失函数的目的是确保损失函数等于0。

只要损失函数小于0，则损失函数就是0.

上面的A、P、N就是三元组。

三元组的选择不能太随意，要选择很难训练的A、P、N。下图是解释:

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.1i7h2ikrgpc0.png)

***

### 4.5 人脸验证与二分类(Face verification and binary classification)

可以把人脸识别当做二分类问题。

定义输出$\widehat{y}$如下：

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.fyjxdfxiiug.png)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.5ji5d0c6at0.png)

如果相同输出1，相反输出0

### 4.6 神经风格迁移(Neural style transfer)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.41vox2xevqu0.png)

不得不说，第二张合成图好阴间。。。

C表示内容图像，S表示风格图像，G表示生成的图像。

***

### 4.7 深度卷积网络学习什么？(What are deep ConvNets learning?)

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.re8c2a7rwog.png)

网络第一层能检测出一些边缘或颜色阴影等，随着层数的加深，能够检测到更复杂的东西。图中举例的每个方框代表了不同的9个代表性神经元。

***

### 4.8 神经风格迁移系统的代价函数

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.7ga1z2thnb40.png)

**神经风格迁移系统的代价函数：**
$$
J(G)=\alpha {J_{content}}(C,G)+\beta J_{style}(S,G)
$$
第一个是内容代价函数，第二个是风格代价函数。

前面的系数表示权重

![image](https://cdn.jsdelivr.net/gh/JLUVicent/image-saving@master/20210731/image.6pztd8aa4t40.png)

上面定义了一个生成图片G的代价函数，并将其最小化。

***

### 4.9 内容代价函数(Content cost function)

用$a^{[l][C]}$​和$a^{[l][G]}$​来代表两个图片C和G的l层的激活函数值。如果两个激活值相似，那么就意味着两个图片的内容相似，因此：

**内容代价函数：**
$$
J_{content}(C,G)=\frac{1}{2}||a^{[l][C]}-a^{[l][G]}||^2
$$
通过超参数$\alpha$来调整代价函数。

***

### 4.10 风格代价函数(Style cost function)

没怎么看懂，给出了一个风格代价函数。

***

### 4.11 一维和三维推广

之前讲的卷积都是在2D上讨论的，当然可以以相同的方式来推广到1D和3D空间。

***

OVER！

