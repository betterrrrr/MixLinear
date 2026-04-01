把傅里叶变换更换成小波变换，结果如下：
ETTh1_720_96_MixLinear_ETTh1_ftM_sl720_pl96_test_0.95_5_0_seed2023  
mse:0.37866610288619995, mae:0.40234872698783875, rse:0.5845020413398743

ETTh1_720_96_MixLinear_ETTh1_ftM_sl720_pl96_test_0.95_1_0_seed2023  
mse:0.37667644023895264, mae:0.39952534437179565, rse:0.5829644799232483

ETTh1_720_192_MixLinear_ETTh1_ftM_sl720_pl192_test_0.95_5_0_seed2023  
mse:0.40945011377334595, mae:0.4208657741546631, rse:0.6076560616493225

ETTh1_720_192_MixLinear_ETTh1_ftM_sl720_pl192_test_0.95_1_0_seed2023  
mse:0.4120371341705322, mae:0.42110615968704224, rse:0.60957270860672

MixLinear 在96/192 的 MSE 分别为 0.351和 0.395

增加了小波变换的初始化选项，支持随机初始化、Haar小波和Daubechies 2小波
先使用haar小波初始化，结果如下：

ETTh1_720_96_MixLinear_ETTh1_ftM_sl720_pl96_test_0.95_1_0_seed2023  
mse:0.37389853596687317, mae:0.4005017578601837, rse:0.5808108448982239

ETTh1_720_96_MixLinear_ETTh1_ftM_sl720_pl96_test_0.95_5_0_seed2023  
mse:0.3730069696903229, mae:0.3988925814628601, rse:0.5801180005073547


ETTh1_720_192_MixLinear_ETTh1_ftM_sl720_pl192_test_0.95_5_0_seed2023  
mse:0.4074554145336151, mae:0.41909483075141907, rse:0.606174111366272

ETTh1_720_192_MixLinear_ETTh1_ftM_sl720_pl192_test_0.95_1_0_seed2023  
mse:0.40730178356170654, mae:0.4199638068675995, rse:0.6060597896575928

使用db2小波初始化，结果如下：
ETTh1_720_96_MixLinear_ETTh1_ftM_sl720_pl96_test_0.95_1_0_seed2023  
mse:0.3711741864681244, mae:0.39697524905204773, rse:0.5786910057067871

ETTh1_720_96_MixLinear_ETTh1_ftM_sl720_pl96_test_0.95_5_0_seed2023  
mse:0.3745533227920532, mae:0.3986344635486603, rse:0.5813192129135132

ETTh1_720_192_MixLinear_ETTh1_ftM_sl720_pl192_test_0.95_5_0_seed2023  
mse:0.40630343556404114, mae:0.41894516348838806, rse:0.6053166389465332

ETTh1_720_192_MixLinear_ETTh1_ftM_sl720_pl192_test_0.95_1_0_seed2023  
mse:0.405356764793396, mae:0.41763633489608765, rse:0.6046110391616821

Haar 小波本质上是一个阶跃函数（矩形波），它非常适合处理有锐利边缘或剧烈突变的数据。然而，ETTh1 数据集记录的是电力变压器的油温和负载，属于典型的平滑连续且具有周期性的物理信号。Db2 小波由于具有更平滑的波形特征（具备二阶消失矩），能更好地拟合这类平滑的低频趋势，因此在提取“Trend（趋势）”时更加精准

加入 RevIN (可逆实例归一化)：替换掉当前简单的 x - seq_mean 的归一化方式，改用 RevIN 来进行输入数据的归一化和反归一化。RevIN 可以更好地适应不同时间序列的分布特征，提升模型的泛化能力和预测性能。

ETTh1_720_192_MixLinear_ETTh1_ftM_sl720_pl192_test_0.95_5_0_seed2023  
mse:0.40630340576171875, mae:0.41894516348838806, rse:0.6053165793418884

ETTh1_720_96_MixLinear_ETTh1_ftM_sl720_pl96_test_0.95_5_0_seed2023  
mse:0.3745534121990204, mae:0.3986344635486603, rse:0.581319272518158

ETTh1_720_96_MixLinear_ETTh1_ftM_sl720_pl96_test_0.95_1_0_seed2023  
mse:0.3711741864681244, mae:0.39697524905204773, rse:0.5786910057067871

ETTh1_720_192_MixLinear_ETTh1_ftM_sl720_pl192_test_0.95_1_0_seed2023  
mse:0.40535667538642883, mae:0.41763633489608765, rse:0.6046109199523926

移除RevIN后，在_learnable_swt 方法中，对于不同层级提取出的高频细节，之前使用的是直接求均值：detail = torch.stack(details, dim=0).mean(dim=0)，现在引入可学习的尺度权重（Learnable Scale Weight），让模型自己决定在当前的数据集上，哪一层的高频细节更有价值：
ETTh1_720_192_MixLinear_ETTh1_ftM_sl720_pl192_test_0.95_5_0_seed2023  
mse:0.4025574326515198, mae:0.41790807247161865, rse:0.6025197505950928

ETTh1_720_96_MixLinear_ETTh1_ftM_sl720_pl96_test_0.95_1_0_seed2023  
mse:0.3723730146884918, mae:0.3973272442817688, rse:0.5796247720718384

ETTh1_720_96_MixLinear_ETTh1_ftM_sl720_pl96_test_0.95_5_0_seed2023  
mse:0.3744427561759949, mae:0.39866647124290466, rse:0.5812333822250366

ETTh1_720_192_MixLinear_ETTh1_ftM_sl720_pl192_test_0.95_1_0_seed2023  
mse:0.40617290139198303, mae:0.4177152216434479, rse:0.6052193641662598

将低频趋势交给 MixLinear 的分段线性层（Segment-based Linear）提取长程依赖，将高频细节交给带有 GELU 激活的线性层进行非线性局部去噪

ETTh1_720_96_MixLinear_ETTh1_ftM_sl720_pl96_test_0.95_5_0_seed2023  
mse:0.3655145764350891, mae:0.39596232771873474, rse:0.5742621421813965

ETTh1_720_96_MixLinear_ETTh1_ftM_sl720_pl96_test_0.95_1_0_seed2023  
mse:0.36707228422164917, mae:0.39779478311538696, rse:0.5754845142364502

ETTh1_720_192_MixLinear_ETTh1_ftM_sl720_pl192_test_0.95_1_0_seed2023  
mse:0.404374361038208, mae:0.41562986373901367, rse:0.6038779020309448

ETTh1_720_192_MixLinear_ETTh1_ftM_sl720_pl192_test_0.95_5_0_seed2023  
mse:0.39823588728904724, mae:0.4145948588848114, rse:0.5992769002914429


在DLinear的基础上，加入固定的小波变换层，并补回均值平移（Mean-Centering），构建 MaxDLinear，结果如下：

使用db2小波初始化：

ETTh1_720_96_MaxDLinear_ETTh1_ftM_sl720_pl96_test_0.95_5_0_seed2023  
mse:0.37504175305366516, mae:0.39812883734703064, rse:0.5816980600357056

ETTh1_720_96_MaxDLinear_ETTh1_ftM_sl720_pl96_test_0.95_1_0_seed2023  
mse:0.37511634826660156, mae:0.39825889468193054, rse:0.5817559957504272

ETTh1_720_192_MaxDLinear_ETTh1_ftM_sl720_pl192_test_0.95_5_0_seed2023  
mse:0.4131448566913605, mae:0.42158007621765137, rse:0.6103915572166443

ETTh1_720_192_MaxDLinear_ETTh1_ftM_sl720_pl192_test_0.95_1_0_seed2023  
mse:0.4203166663646698, mae:0.42875951528549194, rse:0.6156666874885559

对比结论：

1) 相比 DLinear（根据论文表中 ETTh1 的 RPD 反推，96/192 的 MSE 约为 0.4018/0.4312），MaxDLinear 在两个预测步长上都有提升。  
- Horizon 96：0.3750 vs 0.4018（约提升 6.66%）  
- Horizon 192：0.4131 vs 0.4312（约提升 4.19%）

2) 相比当前最优 MixLinear 版本（96: 0.3655，192: 0.3982），MaxDLinear 仍有差距。  
- Horizon 96：MaxDLinear 高 0.0095  
- Horizon 192：MaxDLinear 高 0.0149

说明固定 SWT + Mean-Centering 能稳定提升 DLinear 的频域分解能力，但若要进一步逼近 MixLinear，仍需要在趋势/细节分支表达能力上继续增强。

进一步使用 haar 小波做同配置对照实验（ETTh1, seq_len=720, alpha=0.95）：

ETTh1_720_96_MaxDLinear_ETTh1_ftM_sl720_pl96_test_0.95_1_0_seed2023  
mse:0.3750530779361725, mae:0.3981536030769348, rse:0.5817068815231323

ETTh1_720_96_MaxDLinear_ETTh1_ftM_sl720_pl96_test_0.95_5_0_seed2023  
mse:0.3750530779361725, mae:0.3981536030769348, rse:0.5817068815231323

ETTh1_720_192_MaxDLinear_ETTh1_ftM_sl720_pl192_test_0.95_1_0_seed2023  
mse:0.4132770299911499, mae:0.42169100046157837, rse:0.6104891896247864

ETTh1_720_192_MaxDLinear_ETTh1_ftM_sl720_pl192_test_0.95_5_0_seed2023  
mse:0.4132770299911499, mae:0.42169100046157837, rse:0.6104891896247864

最终总结（MaxDLinear, 固定 SWT + Mean-Centering）：

1) db2 与 haar 在本任务上都有效，但 db2 仍略优。  
- Horizon 96：db2 最优 0.3750418，haar 0.3750531（db2 略优 0.0000113）  
- Horizon 192：db2 最优 0.4131449，haar 0.4132770（db2 略优 0.0001322）

2) 在当前 MaxDLinear 配置下，haar 的 lpf=1 与 lpf=5 结果几乎一致，说明该设置对 lpf 不敏感。

3) 当前最优 MaxDLinear 配置为 db2 + lpf=5：  
- Horizon 96：mse=0.3750418  
- Horizon 192：mse=0.4131449

nn.Parameter 替代 register_buffer：原先的滤波器系数在初始化后被当做常量处理。改为 nn.Parameter 后，它们就变成了网络权重的一部分。模型在反向传播计算 Loss 时，会自动计算出这两个滤波器的梯度并进行更新，从而达到“自动调整小波基”的目的。

多层权重自适应：添加了 self.detail_weights，让模型自己去学习第 1 层、第 2 层等不同尺度下抽取出的高频细节应该各占多少比重，而不是被动地进行简单平均。这对时间序列预测精度的提升通常很有帮助。

使用haar小波初始化：

ETTh1_720_96_MaxDLinear_ETTh1_ftM_sl720_pl96_test_0.95_1_0_seed2023  
mse:0.3708111345767975, mae:0.39747384190559387, rse:0.578407883644104

ETTh1_720_192_MaxDLinear_ETTh1_ftM_sl720_pl192_test_0.95_1_0_seed2023  
mse:0.40119990706443787, mae:0.4185945689678192, rse:0.6015029549598694

ETTh1_720_96_MaxDLinear_ETTh1_ftM_sl720_pl96_test_0.95_5_0_seed2023  
mse:0.3722027540206909, mae:0.3981342911720276, rse:0.5794922113418579

ETTh1_720_192_MaxDLinear_ETTh1_ftM_sl720_pl192_test_0.95_5_0_seed2023  
mse:0.4040943384170532, mae:0.41907644271850586, rse:0.6036688089370728

使用db2小波初始化：

ETTh1_720_96_MaxDLinear_ETTh1_ftM_sl720_pl96_test_0.95_5_0_seed2023  
mse:0.3715880215167999, mae:0.3973120450973511, rse:0.5790135264396667

ETTh1_720_96_MaxDLinear_ETTh1_ftM_sl720_pl96_test_0.95_1_0_seed2023  
mse:0.3675392270088196, mae:0.3965860903263092, rse:0.5758504271507263

ETTh1_720_192_MaxDLinear_ETTh1_ftM_sl720_pl192_test_0.95_5_0_seed2023  
mse:0.408317893743515, mae:0.41850605607032776, rse:0.6068153381347656

ETTh1_720_192_MaxDLinear_ETTh1_ftM_sl720_pl192_test_0.95_1_0_seed2023  
mse:0.40223976969718933, mae:0.4177444279193878, rse:0.602281928062439

把circular padding 改成了 reflect padding，结果如下：

ETTh1_720_96_MaxDLinear_ETTh1_ftM_sl720_pl96_test_0.95_5_0_seed2023  
mse:0.37036222219467163, mae:0.3975006639957428, rse:0.5780577063560486

ETTh1_720_96_MaxDLinear_ETTh1_ftM_sl720_pl96_test_0.95_1_0_seed2023  
mse:0.37329861521720886, mae:0.40002453327178955, rse:0.5803446769714355

ETTh1_720_192_MaxDLinear_ETTh1_ftM_sl720_pl192_test_0.95_1_0_seed2023  
mse:0.4063216745853424, mae:0.42018675804138184, rse:0.605330228805542

ETTh1_720_192_MaxDLinear_ETTh1_ftM_sl720_pl192_test_0.95_5_0_seed2023  
mse:0.4086756110191345, mae:0.42019373178482056, rse:0.6070810556411743

