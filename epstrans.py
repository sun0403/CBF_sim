import os
from PIL import Image

def convert_png_to_pdf(source_folder, target_folder):
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 检查文件是否为PNG格式
        if filename.endswith(".png"):
            # 拼接源文件的完整路径
            png_path = os.path.join(source_folder, filename)
            # 打开PNG图像
            with Image.open(png_path) as img:
                # 如果图像是 RGBA 或 P 模式，转换为 RGB 模式
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                # 设置PDF文件的输出路径（在目标文件夹内）
                pdf_filename = os.path.splitext(filename)[0] + ".pdf"
                pdf_path = os.path.join(target_folder, pdf_filename)
                # 将图像转换为PDF格式并保存
                img.save(pdf_path, format='PDF')
                print(f"Converted {filename} to {pdf_filename} and saved to {target_folder}")

# 设置源文件夹和目标文件夹的路径
source_folder = "./paper_pic"  # 替换为你的PNG文件夹路径
target_folder = "./paper_pic_pdf"  # 替换为你想保存PDF文件的文件夹路径

# 调用转换函数
convert_png_to_pdf(source_folder, target_folder)
