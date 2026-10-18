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
# 关键修复：绑定分类名称字段，防止后续代码报错
df_sales = df_sales.merge(df_item[["单品编码", "分类名称"]], on="单品编码", how="left")

# ==================== 模块2：销量时间模式挖掘 ====================
from statsmodels.tsa.seasonal import STL

# 2.1 按日聚合总销量
daily_sales = df_sales.groupby(COL_DATE)[COL_SALES_QTY].sum().reset_index()
daily_sales = daily_sales.set_index(COL_DATE).asfreq('D', fill_value=0)

# 2.2 STL时序分解（自动适配数据集时间区间）
start_time = daily_sales.index.min()
end_time = daily_sales.index.max()
stl_data = daily_sales.loc[start_time:end_time]

if len(stl_data) > 14:
    stl = STL(stl_data[COL_SALES_QTY], period=7, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(result.observed, label='原始销量')
    axes[0].set_ylabel('销量')
    axes[0].legend()
    axes[1].plot(result.trend, label='趋势分量', color='orange')
    axes[1].set_ylabel('趋势')
    axes[1].legend()
    axes[2].plot(result.seasonal, label='季节分量(周)', color='green')
    axes[2].set_ylabel('季节')
    axes[2].legend()
    axes[3].plot(result.resid, label='残差(随机波动)', color='red')
    axes[3].set_ylabel('残差')
    axes[3].legend()
    plt.suptitle('STL时序分解（日总销量）')
    plt.tight_layout()
    plt.savefig('模块2_STL分解.png', dpi=300)
    plt.show()

    # 2.4 异常销量时点检测
    resid_std = result.resid.std()
    anomalies = result.resid[abs(result.resid) > 3 * resid_std].dropna()
    print(f"\n【异常销量时点（残差>3倍标准差）】共{len(anomalies)}个")
    print(anomalies.sort_values(ascending=False).head(10))
else:
    print("日销量时序样本过少，无法进行STL分解")

# 2.3 周销售波动曲线
df_sales['星期'] = df_sales[COL_DATE].dt.dayofweek
weekly_pattern = df_sales.groupby('星期')[COL_SALES_QTY].mean()
weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

plt.figure(figsize=(10, 5))
plt.bar(weekday_names, weekly_pattern.values, color='skyblue')
plt.title('周内日均销量波动曲线')
plt.ylabel('平均日销量(千克)')
for i, v in enumerate(weekly_pattern.values):
    plt.text(i, v + 0.5, f'{v:.0f}', ha='center')
plt.savefig('模块2_周波动曲线.png', dpi=300)
plt.show()

# 2.5 月度趋势图（修复period坐标轴异常）
df_sales['年月'] = df_sales[COL_DATE].dt.strftime('%Y‑%m')
monthly_sales = df_sales.groupby('年月')[COL_SALES_QTY].sum()

plt.figure(figsize=(14, 5))
monthly_sales.plot(marker='o', linestyle='-', color='darkblue')
plt.title('月度总销量趋势')
plt.ylabel('月销量(千克)')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('模块2_月度趋势.png', dpi=300)
plt.show()