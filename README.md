关于cs2人物检测

# YOLOv8n
基于YOLOv8n所做目前的目标识别检测
数据集配置步骤
 
1. 把数据集存到你电脑里任意文件夹
2. 打开data.yaml文件，把第一行path后面的内容改成你存数据集的文件夹路径
3. 其他内容不用动，保存就行
 
示例（参考）：
path: 你电脑里的数据集路径（比如 D:\datasets\xl）
train: images/train
val: images/val
nc: 2
names: [head, body]
