# ============================================
# 1. 导入必要的库
# ============================================
import pandas as pd
import numpy as np
from IGTD_Functions import min_max_transform, table_to_image
from sklearn.preprocessing import LabelEncoder

# ============================================
# 2. 设置 IGTD 参数
# ============================================
num_row = 3          # 图像的行数（像素）
num_col = 3          # 图像的列数（像素）
num_pixels = num_row * num_col   # 9个像素

# 因为我们有4个特征，9个像素 > 4个特征，IGTD会自动处理（会填充伪特征）
# 你也可以设置 num_row=2, num_col=2（4个像素），刚好每个特征对应一个像素

max_step = 5000      # 最大迭代次数（Iris数据小，5000步足够了）
val_step = 100       # 每100步检查一次是否收敛
save_image_size = 3  # 保存的图像尺寸（英寸，仅显示大小）

# ============================================
# 3. 读取 Iris 数据
# ============================================
# 方法1：从本地文件读取（如果你已经下载了 iris.data）
data = pd.read_csv('../Data/iris.data', header=None)


# 打印前5行，看看数据长什么样
print("first 5：")
print(data.head())
print("\n data shape:", data.shape)  # 应该是 (150, 5)

# ============================================
# 4. 给列添加名称（方便理解）
# ============================================
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

print("\n adding head：")
print(data.head())

# ============================================
# 5. 把类别（字符串）转换成数字（标签）
# ============================================
# 解释：IGTD要求标签是数字，不能是字符串
# 'Iris-setosa' -> 0, 'Iris-versicolor' -> 1, 'Iris-virginica' -> 2

label_encoder = LabelEncoder()
data['species_code'] = label_encoder.fit_transform(data['species'])

print("\n label mapping：")
for i, species in enumerate(label_encoder.classes_):
    print(f"  {species} -> {i}")

print("\n first 5 after adding label：")
print(data.head())

# ============================================
# 6. 分离特征和标签
# ============================================
# 特征：前4列（花萼长/宽、花瓣长/宽）
features = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
# 标签：第6列（数字化的类别）
labels = data['species_code']

print(f"\n feature shape: {features.shape}")  # (150, 4)
print(f"label shape: {labels.shape}")      # (150,)

# ============================================
# 7. 对特征进行 Min-Max 归一化
# ============================================
# 解释：IGTD要求输入数据在 [0,1] 范围内
# 归一化公式：(x - min) / (max - min)
# 每个特征独立归一化，所以结果范围都是 [0,1]

norm_features = min_max_transform(features.values)
norm_features = pd.DataFrame(norm_features, columns=features.columns, index=features.index)

print("\n first 5 after normalization：")
print(norm_features.head())
print(f"\n range after normailization: [{norm_features.min().min():.2f}, \
      {norm_features.max().max():.2f}]")

# ============================================
# 8. （可选）选择特征（因为特征数 < 像素数，不需要这一步）
# ============================================
# 我们有4个特征，9个像素。IGTD会自动处理
# 如果你想精确匹配像素数，可以设置 num_row=2, num_col=2（4个像素）

# ============================================
# 9. 运行 IGTD 生成图像
# ============================================
# 创建保存结果的文件夹
result_dir = '../Results/Table_To_Image_Conversion/Test_2'
import os
os.makedirs(result_dir, exist_ok=True)

print(f"\n Staring producing images...")
print(f"image scale: {num_row} x {num_col} = {num_pixels} pixels")
print(f"image number: {features.shape[1]}")
print(f"feature number: {features.shape[0]}")

# 运行 IGTD 算法
# 参数说明：
# - norm_features: 归一化后的特征数据
# - [num_row, num_col]: 图像尺寸
# - fea_dist_method='Euclidean': 特征间用欧氏距离
# - image_dist_method='Euclidean': 像素间用欧氏距离
# - save_image_size: 保存的图片大小
# - max_step: 最大迭代次数
# - val_step: 验证步长
# - result_dir: 结果保存文件夹
# - error='abs': 误差计算方式（绝对值）
table_to_image(
    norm_features, 
    [num_row, num_col], 
    fea_dist_method='Euclidean',
    image_dist_method='Euclidean',
    save_image_size=save_image_size,
    max_step=max_step,
    val_step=val_step,
    normDir=result_dir,
    error='abs'
)

print(f"\n Done! Images saved in: {result_dir}")

# ============================================
# 10. （可选）查看生成的图像文件
# ============================================
import matplotlib.pyplot as plt
import cv2

# 读取第一张生成的图像看看
image_files = [f for f in os.listdir(result_dir) if f.endswith('.png')]
if image_files:
    first_image_path = os.path.join(result_dir, image_files[0])
    img = cv2.imread(first_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(5,5))
    plt.imshow(img_rgb)
    plt.title(f" First image (sample {image_files[0]})")
    plt.axis('off')
    plt.show()
    
    print(f"\n produced {len(image_files)} images")