import os
import shutil

source_dir = r'e:\数据\val\1'
destination_dir = r'e:\SZBL-test1\data2\CT-1'

# 遍历源目录中的所有文件
for filename in os.listdir(source_dir):
    if filename.endswith("_image.nrrd"):
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)
        
        # 复制文件到目标目录
        shutil.copy(source_file, destination_file)
        print(f"复制文件 {filename} 完成")
