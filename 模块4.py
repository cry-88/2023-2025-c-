import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 读取原始文件 ==========
df_item = pd.read_excel('附件1.xlsx')
df_sales_raw = pd.read_excel('附件2.xlsx')
df_price = pd.read_excel('附件3.xlsx')
df_cat = pd.read_excel('附件4.xlsx', sheet_name=0)

# ========== 字段名称配置 ==========
COL_DATE = '销售日期'
COL_SALES_QTY = '销量(千克)'
COL_PRICE = '销售单价(元/千克)'
COL_SCAN_TIME = '扫码销售时间'
COL_TYPE = '销售类型'
COL_DISCOUNT = '是否打折销售'

# ========== 数据清洗函数 ==========
def clean_data(sales_df, item_df):
    data = sales_df.copy()
    data[COL_DATE] = pd.to_datetime(data[COL_DATE])
    # 剔除异常销量、单价
    data = data[data[COL_SALES_QTY] > 0]
    data = data[data[COL_PRICE] > 0]
    valid_code = item_df['单品编码'].unique()
    data = data[data['单品编码'].isin(valid_code)]
    return data

df_sales = clean_data(df_sales_raw, df_item)
# 拼接蔬菜分类名称
df_sales = df_sales.merge(df_item[["单品编码", "分类名称"]], on="单品编码", how="left")

# ========== 按日期+品类，统计每日销量 ==========
daily_category_sales = df_sales.groupby([COL_DATE, "分类名称"])[COL_SALES_QTY].sum().reset_index()

# 补齐全部日期
full_day_range = pd.date_range(start=df_sales[COL_DATE].min(), end=df_sales[COL_DATE].max(), freq="D")
pivot_table = daily_category_sales.pivot(index=COL_DATE, columns="分类名称", values=COL_SALES_QTY)
pivot_table = pivot_table.reindex(full_day_range).fillna(0)

# 固定品类顺序，保证绘图、打印顺序一致
category_list = pivot_table.columns.tolist()
corr_result = pivot_table[category_list].corr(method="spearman")

# ========== 在控制台打印相关矩阵、强相关组合 ==========
print("\n【品类间Spearman相关系数矩阵】")
print(corr_result.round(3))

strong_corr = []
for i in range(len(corr_result.columns)):
    for j in range(i + 1, len(corr_result.columns)):
        name_i = corr_result.columns[i]
        name_j = corr_result.columns[j]
        r = corr_result.iloc[i, j]
        if abs(r) > 0.3:
            strong_corr.append((name_i, name_j, r))

strong_corr.sort(key=lambda x: abs(x[2]), reverse=True)
print("\n【强关联品类配对（|r|>0.3）】")
for a,b,r in strong_corr:
    print(f"{a} ↔ {b} : r = {round(r,3)}")

# ========== 【最终无空白完整热力图】 ==========
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_result,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    vmin=-1,
    vmax=1,
    xticklabels=category_list,
    yticklabels=category_list
)
plt.gca().invert_yaxis() # Y轴和打印矩阵顺序严格对应
plt.title("六大品类日销量Spearman相关系数热力图", fontsize=14)
plt.tight_layout()
plt.savefig("模块4_完整品类热力图.png", dpi=300)
plt.show()

# ========== 正负相关时序对比绘图 ==========
top_pos = strong_corr[0]
cat_a, cat_b, r_val = top_pos
negative_group = [item for item in strong_corr if item[2] < 0]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].plot(pivot_table.index, pivot_table[cat_a], label=cat_a)
axes[0].plot(pivot_table.index, pivot_table[cat_b], label=cat_b)
axes[0].set_title(f"正相关示例：{cat_a} vs {cat_b}（r={round(r_val,2)}）")
axes[0].legend()

if len(negative_group) > 0:
    neg_a, neg_b, neg_r = negative_group[0]
    axes[1].plot(pivot_table.index, pivot_table[neg_a], label=neg_a)
    axes[1].plot(pivot_table.index, pivot_table[neg_b], label=neg_b)
    axes[1].set_title(f"负相关示例：{neg_a} vs {neg_b}（r={round(neg_r,2)}）")
    axes[1].legend()
else:
    axes[1].set_title("不存在强负相关品类组合")
plt.tight_layout()
plt.savefig("模块4_品类时序对比图.png", dpi=300)
plt.show()