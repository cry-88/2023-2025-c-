import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 加载原始数据 ==========
df_item = pd.read_excel('附件1.xlsx')
df_sales_raw = pd.read_excel('附件2.xlsx')
df_price = pd.read_excel('附件3.xlsx')
df_loss_cat = pd.read_excel('附件4.xlsx', sheet_name=0)
df_loss_item = pd.read_excel('附件4.xlsx', sheet_name=1)

# ========== 列名映射 ==========
COL_DATE = '销售日期'
COL_SALES_QTY = '销量(千克)'
COL_SALES_PRICE = '销售单价(元/千克)'
COL_SALES_TIME = '扫码销售时间'
COL_SALES_TYPE = '销售类型'
COL_DISCOUNT = '是否打折销售'

# ========== 全局基础清洗函数 ==========
def basic_clean(sales_df, item_df):
    df = sales_df.copy()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = df[df[COL_SALES_QTY] > 0]
    df = df[df[COL_SALES_PRICE] > 0]
    valid_codes = item_df['单品编码'].unique()
    df = df[df['单品编码'].isin(valid_codes)]
    return df

df_sales = basic_clean(df_sales_raw, df_item)
# 左联单品表，给销售数据添加【分类名称】，解决KeyError报错
df_sales = df_sales.merge(df_item[["单品编码", "分类名称"]], on="单品编码", how="left")

# ==================== 模块1：销售量分布规律 ====================
# 1.1 按单品聚合统计指标
item_stats = df_sales.groupby('单品编码').agg(
    销量均值=(COL_SALES_QTY, 'mean'),
    销量中位数=(COL_SALES_QTY, 'median'),
    销量方差=(COL_SALES_QTY, 'var'),
    销量标准差=(COL_SALES_QTY, 'std'),
    变异系数=(COL_SALES_QTY, lambda x: x.std() / (x.mean() + 1e-6)),
    最大销量=(COL_SALES_QTY, 'max'),
    总销量=(COL_SALES_QTY, 'sum'),
    销售天数=(COL_SALES_QTY, 'count')
).reset_index()

# 添加品类、单品名称信息
item_stats = item_stats.merge(df_item[['单品编码', '分类名称', '单品名称']], on='单品编码', how='left')

# 1.2 输出统计摘要
print("="*50)
print("【各品类销量统计摘要】")
print(item_stats.groupby('分类名称')['销量均值'].describe())
print("\n【销量TOP10爆款单品】")
print(item_stats.nlargest(10, '销量均值')[['单品名称', '分类名称', '销量均值', '销量标准差']])

print("\n【销量为0天数占比最高的滞销单品（TOP5）】")
# 构建完整日期‑单品全量面板
all_dates = pd.date_range(start=df_sales[COL_DATE].min(), end=df_sales[COL_DATE].max(), freq='D')
all_items = df_sales['单品编码'].unique()
full_idx = pd.MultiIndex.from_product([all_dates, all_items], names=['日期', '单品编码'])
tmp = pd.DataFrame(index=full_idx).reset_index()
tmp = tmp.merge(df_sales.groupby([COL_DATE, '单品编码'])[COL_SALES_QTY].sum().reset_index(),
                left_on=['日期', '单品编码'], right_on=[COL_DATE, '单品编码'], how='left')
tmp[COL_SALES_QTY] = tmp[COL_SALES_QTY].fillna(0)
zero_ratio = tmp.groupby('单品编码').apply(lambda x: (x[COL_SALES_QTY] == 0).mean()).reset_index(name='零占比')
zero_ratio = zero_ratio.merge(df_item[['单品编码', '单品名称']], on='单品编码')
print(zero_ratio.nlargest(5, '零占比'))

# 1.3 KS分布拟合检验（花叶类销量分布）
sample_category = '花叶类'
sample_data = df_sales[df_sales['分类名称'] == sample_category][COL_SALES_QTY].values
sample_pos = sample_data[sample_data > 0]

# 数据健壮性判断，防止筛选为空时报错
if len(sample_pos) > 1:
    distributions = ['norm', 'lognorm', 'gamma', 'expon']
    results = {}
    for dist_name in distributions:
        if dist_name == 'norm':
            params = stats.norm.fit(sample_pos)
            ks_stat, p_val = stats.kstest(sample_pos, 'norm', args=params)
        elif dist_name == 'lognorm':
            params = stats.lognorm.fit(sample_pos)
            ks_stat, p_val = stats.kstest(sample_pos, 'lognorm', args=params)
        elif dist_name == 'gamma':
            params = stats.gamma.fit(sample_pos)
            ks_stat, p_val = stats.kstest(sample_pos, 'gamma', args=params)
        else:
            params = stats.expon.fit(sample_pos)
            ks_stat, p_val = stats.kstest(sample_pos, 'expon', args=params)
        results[dist_name] = {'KS统计量': ks_stat, 'p值': p_val}

    print(f"\n【{sample_category}销量分布拟合检验】")
    for name, res in results.items():
        print(f"{name}: KS={res['KS统计量']:.4f}, p={res['p值']:.4f}")
else:
    print(f"\n {sample_category} 有效销量样本过少，无法执行KS检验")

# 1.4 可视化：区分淡旺季各品类销量箱线图
df_sales['月份'] = df_sales[COL_DATE].dt.month
df_sales['季节'] = df_sales['月份'].apply(lambda m: '旺季' if m in [4,5,6,7,8,9,10] else '淡季')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, season in zip(axes, ['淡季', '旺季']):
    data = df_sales[df_sales['季节'] == season]
    sns.boxplot(data=data, x='分类名称', y=COL_SALES_QTY, ax=ax, palette='Set2')
    ax.set_title(f'{season}各品类销量箱线图')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.savefig('模块1_销量分布箱线图.png', dpi=300)
plt.show()