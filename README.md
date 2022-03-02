# CascadeR-CNN
## 总述
CascadeR-CNN是在Faster R-CNN的基础上在精度方面进行改进的一个模型。  
## 改进方面
Faster R-CNN存在问题，一个是IoU设多少，IoU是用来区分目标和背景的标准，IoU小了肯定是不好的，正样本质量不达标，但是大了就一定好么？训练和推测阶段的分布是不一样的，所以IoU定下来用于两个阶段不合适，文中作者称之为mismatch。  
通过实验发现，input的IoU与Output的IoU是成正比的，阈值的大小与生成的proposal的质量密切相关，但Faster R-CNN结构的设计限制了其阈值的大小，这也就限制了Faster R-CNN等的精度，CasCade R-CNN正是来解决这一问题的。  
## 级联结构
这种级联式的结构其实也不是作者首创的，在Faster R-CNN提出之后，诸如Iterative Bbox at inference, Integral Loss这些结构都提出了级联的模型。作者基于前边我们问题的提出背景，对Cascade这种结构为何优秀做出了描述，具体来说，我们看a，就是简单的Faster R-CNN，H0是RPN，b的特点是利用了类似RNN的结构，一次的输入做了多次循环，而c的不同在于他分开的三个网络的参数是不一样的。Cascade可谓是集合了这三个网络的优势，他每层网络的参数是变化的，输出输出进行了多次迭代。这和我们刚才说的，我们要在迭代的过程中逐步提高IoU是对应的，而本模型最大的创新也正是在于这里。  
