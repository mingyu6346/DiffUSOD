import os

def delete_pth_files(folder_path):
    sum = 0
    size_sum = 0
    # 遍历文件夹内所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否以.pth结尾
            if file.endswith(".pt"):
                file_path = os.path.join(root, file)
                size_sum += os.path.getsize(file_path)
                sum += 1
                os.remove(file_path)  # 删除文件
                print(f"Deleted: {file_path}")
    print(f"Deleted {sum} files, {size_sum/(2**30):.2f} GB")

# 示例调用
folder_path = '../results/pth'  # 替换为你的文件夹路径
delete_pth_files(folder_path)
