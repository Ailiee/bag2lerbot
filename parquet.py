
import pandas as pd

# 第一个文件
# 读取、修改、保存
df = pd.read_parquet('/media/aliee5/QL_ros2/qingloong_Foldingclothes_20251121/meta/tasks.parquet')

print("修改前:")
print(df)

# 重命名索引
df = df.rename(index={
    'default_task': 'Lay the clothes flat on the table, then fold and arrange them'
})

print("\n修改后:")
print(df)

# 保存
df.to_parquet('/media/aliee5/QL_ros2/qingloong_Foldingclothes_20251121/meta/tasksT.parquet')
print("\n已保存到 tasksT.parquet")

# 第二个文件
# 读取文件
df = pd.read_parquet('/media/aliee5/QL_ros2/qingloong_Foldingclothes_20251121/meta/episodes/chunk-000/file-000.parquet')

print("修改前:")
print(df[['episode_index', 'tasks']].head())

# 替换 tasks 列中列表内的字符串
df['tasks'] = df['tasks'].apply(
    lambda x: [item.replace('default_task', 'Lay the clothes flat on the table, then fold and arrange them') for item in x]
)

print("\n修改后:")
print(df[['episode_index', 'tasks']].head())

# 保存文件
df.to_parquet('/media/aliee5/QL_ros2/qingloong_Foldingclothes_20251121/meta/episodes/chunk-000/file-000T.parquet')

print("\n保存完成！")