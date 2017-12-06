# A Tour of GAN

## 概要
 GANとは, Generative Adversarial Netsの略で, ニューラルネットワークの応用である. ざっくりとした説明する際には, 想像しやすいのでネットワークのことを人工知能と呼ぶことにする. generatorとdiscriminatorと呼ばれる2つの人工知能を用意する. generatorは訓練データとできるだけ似ているデータを生成しようとする. そしてdiscriminatorは与えられたデータが訓練データなのか, generatorのものなのかをできるだけ正確に判別しようとする. 訓練を繰り返す毎にgeneratorは訓練データにより似ているデータを作れるようになり, discriminatorはより厳しく偽物を排除する能力が身につく. 最終的にはgeneratorは超本物っぽい猫の画像を出力することができるようになる.
 

 
## 参考記事
-[Generative Adversarial Networks(Gan)を勉強して、kerasで手書き文字生成する](http://yusuke-ujitoko.hatenablog.com/entry/2017/05/08/010314)

-[Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
