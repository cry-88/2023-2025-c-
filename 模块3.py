import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
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
# 【关键修复】销售数据表绑定分类名称
df_sales = df_sales.merge(df_item[["单品编码", "分类名称"]], on="单品编码", how="left")

# ==================== 模块3：单品销售相似性聚类 ====================
# 3.1 构建完整日期‑单品面板
all_dates = pd.date_range(start=df_sales[COL_DATE].min(), end=df_sales[COL_DATE].max(), freq='D')
all_items = df_sales['单品编码'].unique()
full_idx = pd.MultiIndex.from_product([all_dates, all_items], names=['日期', '单品编码'])
panel = pd.DataFrame(index=full_idx).reset_index()
panel = panel.merge(
    df_sales.groupby([COL_DATE, '单品编码'])[COL_SALES_QTY].sum().reset_index(),
    left_on=['日期', '单品编码'], right_on=[COL_DATE, '单品编码'], how='left'
)
panel[COL_SALES_QTY] = panel[COL_SALES_QTY].fillna(0)

# 聚合单品销售特征
features = panel.groupby('单品编码').agg(
    日均销量=(COL_SALES_QTY, 'mean'),
    销量标准差=(COL_SALES_QTY, 'std'),
    变异系数=(COL_SALES_QTY, lambda x: x.std() / (x.mean() + 1e-6)),
    零占比=(COL_SALES_QTY, lambda x: (x == 0).mean()),
    最大日销量=(COL_SALES_QTY, 'max')
).reset_index()

# 合并品类、单品名称
features = features.merge(df_item[['单品编码', '分类名称','单品名称']], on='单品编码', how='left')

# 3.2 特征对数变换 + 标准化
feature_cols = ['日均销量', '销量标准差', '变异系数', '零占比', '最大日销量']
for col in feature_cols:
    features[f'{col}_log'] = np.log1p(features[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features[[f'{col}_log' for col in feature_cols]])

# 3.3 肘部法则 + 轮廓系数自动选择最优K
inertias = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# 自动选取轮廓系数最大的K值
best_k = K_range[np.argmax(sil_scores)]
print(f"\n自动识别最优聚类数目 best_k = {best_k}")

# 绘制K值选择图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, inertias, marker='o')
axes[0].set_title('肘部法则')
axes[0].set_xlabel('聚类数K')
axes[0].set_ylabel('SSE')
axes[1].plot(K_range, sil_scores, marker='o', color='red')
axes[1].axvline(x=best_k,ls='--',color='black',label=f'最优K={best_k}')
axes[1].set_title('轮廓系数')
axes[1].set_xlabel('聚类数K')
axes[1].set_ylabel('轮廓系数')
axes[1].legend()
plt.tight_layout()
plt.savefig('模块3_最优K选择.png', dpi=300)
plt.show()

# 3.4 最终聚类
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
features['聚类标签'] = km_final.fit_predict(X_scaled)

# 3.5 聚类画像输出
cluster_profile = features.groupby('聚类标签').agg(
    单品数量=('单品编码', 'count'),
    平均日均销量=('日均销量', 'mean'),
    平均变异系数=('变异系数', 'mean'),
    平均零占比=('零占比', 'mean')
).reset_index()
print("\n【聚类画像（各类典型特征）】")
print(cluster_profile)

# 3.6 PCA降维可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
features['PC1'] = X_pca[:, 0]
features['PC2'] = X_pca[:, 1]

plt.figure(figsize=(12, 8))
scatter = plt.scatter(features['PC1'], features['PC2'],
                      c=features['聚类标签'], cmap='tab10', alpha=0.7, s=50)
plt.colorbar(scatter, label='聚类标签')
plt.title('单品销售特征聚类散点图（PCA降维）')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

# 随机标注10个蔬菜单品名称，替代编码
sample_items = features.sample(10, random_state=42)
for _, row in sample_items.iterrows():
    plt.annotate(row['单品名称'], (row['PC1'], row['PC2']), fontsize=8)

plt.savefig('模块3_聚类散点图.png', dpi=300)
plt.show()

# 导出结果（新增单品名称）
out_df = features[['单品编码', '单品名称','分类名称', '聚类标签', '日均销量', '零占比']]
out_df.to_csv('单品聚类结果.csv', index=False, encoding="utf‑8‑sig")
print("聚类结果已保存至 单品聚类结果.csv")