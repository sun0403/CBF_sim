import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 读取CSV文件，并跳过前两行（假设前两行是header和注释行）
file_path = '/Users/yuanzhengsun/Downloads/-1500.csv'
data = pd.read_csv(file_path, skiprows=2, header=None)

# 为列命名（根据实际情况命名列）
data.columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5']

# 清理数据：替换分号为小数点并正确处理数据格式
def clean_data(value):
    if isinstance(value, str):
        value = value.replace(';', '.')
        if '0.-' in value:
            value = value.replace('0.-', '-')
        value = value.replace('0.', '')
    try:
        return int(value)
    except ValueError:
        return value

data[['Col1', 'Col2','Col3']] = data[['Col1', 'Col2','Col3']].applymap(clean_data)

# 检查数据转换后是否正确
print(data[['Col1', 'Col2','Col3']].head(10))

# 只选择前两列的前100个数据
data = data[['Col1', 'Col2','Col3']].iloc[:8000]

# 创建一个宽幅图表并绘制两个子图
fig, axs = plt.subplots(2, 1, figsize=(30, 12), sharex=True)

# 绘制第一列数据
axs[0].plot(data['Col1'], label='Phase0', marker='o')
axs[0].set_title('Phase 0')
axs[0].set_ylabel('Value')
axs[0].legend()
axs[0].grid(True)


# 绘制第二列数据
axs[1].plot(data['Col2'], label='Phase1', marker='o')
axs[1].set_title('Phase 1')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Value')
axs[1].legend()
axs[1].grid(True)




# 使用 MaxNLocator 限制 x 轴刻度的数量
axs[1].xaxis.set_major_locator(MaxNLocator(nbins=10))
axs[1].set_xticks(range(0, 8000, 100))  # 自定义 x 轴标签

# 设置整体标题
plt.suptitle('-1500——0 rpm')

# 旋转 x 轴标签以提高可读性
plt.xticks(rotation=45)
output_file_path = '/Users/yuanzhengsun/Downloads/11.png'
plt.savefig(output_file_path)

# 展示图表
plt.show()
