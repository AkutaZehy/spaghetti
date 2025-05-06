- [Spaghetti Code Collection](#spaghetti-code-collection)
- [GPS Gaussian to Live（2025-02~2025-03）](#gps-gaussian-to-live2025-022025-03)
- [Defect Detecting (2025-03~2025-05)](#defect-detecting-2025-032025-05)

> 在本repo中的`commit message`下，你会偶尔看到bruh type，即`bruh:[RANDOM SENTENSE]`。
> 
> 这个type的意思是小的未分类的commit，比如一些小的修改，或者是一些小的bugfix。
> 
> 随机语句来自[RANDOM SENTENSE](https://randomwordgenerator.com/sentence.php)，非常感谢。

# Spaghetti Code Collection
Like a pile of spaghetti.

从2025年开始收集一点自己敲的代码，~~虽然都是没有太大用的东西但是看着有敲代码什么的很幸福吧~~。

2025年以前的：

- [RoboAnalyzer-Carving(2022左右)](https://github.com/AkutaZehy/RoboAnalyzer-Carving)
- [CrackDetection-Utils-for-Python(2024-02~2024-07)](https://github.com/AkutaZehy/CrackDetection-Utils-for-Python)
- [book-sales-dbsystem(2024-11)](https://github.com/AkutaZehy/book-sales-dbsystem)

# GPS Gaussian to Live（2025-02~2025-03）

用实时的图片源产生崭新预测视角的Gaussian渲染。

原版代码来自CVPR 2024的论文，页面详情见[GPS-Gaussian](https://shunyuanzheng.github.io/GPS-Gaussian)。

源码：[GPS-Gaussian](https://github.com/aipixel/GPS-Gaussian)

源代码采用了MIT协议分发，作者为Shunyuan Zheng。

仓库中本部分代码中的gps_gaussian部分几乎全部来自于原版代码，因为脑瘫python的原因为了方便重命名把下划线去了；对其实现异常的部分进行了修改，如[issue 71](https://github.com/aipixel/GPS-Gaussian/issues/71)，另外对其结构略有调整以适应实时的图片源的数据结构。

~~我自己其实也不大清楚哪块改了懒得找了就没标，非常抱歉Orz~~

预览窗格使用了[Dear PyGui](https://github.com/hoffstadt/DearPyGui)，这个也是MIT，作者是Jonathan Hoffstadt。

# Defect Detecting (2025-03~2025-05)

基于Mask2former的多分类头缺陷检测。

源码：[Mask2former](https://github.com/facebookresearch/Mask2Former)

源代码采用了MIT协议分发，作者为"Meta, Inc."，实际上应该也就是FAIR相关。

主要的工作为为其添加了多分类头，数据不予公开。另外创建了一个工作专用的测试类test。

开发日志见morsel站上[工作总结](https://akutazehy.github.io/morsel/posts/2504%E9%A1%B9%E7%9B%AE%E6%80%BB%E7%BB%93/)的汇总。

主要的修改集中在mask2former该folder下文件。