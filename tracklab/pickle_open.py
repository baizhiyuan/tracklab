import pickle
import pandas as pd

# 指定文件路径
file_path = '/garage/projects/video-LLM/position_dataset_bzy/soccer-position/2024-11-08/08-55-59/states/soccer-position/14.pkl'

# 以二进制读取模式打开文件并加载数据
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 检查数据类型
if isinstance(data, pd.DataFrame):
    # 如果数据已经是 DataFrame，直接保存为 .csv 文件
    data.to_csv('output.csv', index=False)
else:
    # 如果数据不是 DataFrame，将其转换为 DataFrame（适用时）
    df = pd.DataFrame(data)
    df.to_csv('output.csv', index=False)

print("数据已保存为 output.csv")