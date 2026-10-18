import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings

# 屏蔽全部警告
warnings.filterwarnings("ignore")

from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx

# 全局中文设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 读取数据表 ==========
df_item = pd.read_excel('附件1.xlsx')
df_sales_raw = pd.read_excel('附件2.xlsx')
df_price = pd.read_excel('附件3.xlsx')
df_loss_cat = pd.read_excel('附件4.xlsx', sheet_name=0)
df_loss_item = pd.read_excel('附件4.xlsx', sheet_name=1)

# ========== 字段名称 ==========
COL_DATE = '销售日期'
COL_SALES_QTY = '销量(千克)'
COL_SALES_PRICE = '销售单价(元/千克)'
COL_SALES_TIME = '扫码销售时间'

# ========== 销售数据清洗 ==========
def basic_clean(sales_df, item_df):
    df = sales_df.copy()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = df[df[COL_SALES_QTY] > 0]
    df = df[df[COL_SALES_PRICE] > 0]
    valid_codes = item_df['单品编码'].unique()
    df = df[df['单品编码'].isin(valid_codes)]
    return df

df_sales = basic_clean(df_sales_raw, df_item)

# ========== 生成购物篮ID ==========
df_sales['time_str'] = df_sales[COL_SALES_TIME].astype(str).str[:8]
df_sales['basket_id'] = df_sales[COL_DATE].dt.strftime("%Y-%m-%d") + "_" + df_sales['time_str']
baskets = df_sales.groupby("basket_id")["单品编码"].apply(set).tolist()
print(f"原始交易购物篮总数：{len(baskets)}")

# ========== 过滤低频商品 ==========
item_counter = Counter(i for basket in baskets for i in basket)
min_appear = 5
valid_item = {code for code, count in item_counter.items() if count >= min_appear}

baskets_filtered = []
for one_basket in baskets:
    goods_list = [x for x in one_basket if x in valid_item]
    if len(goods_list) >= 2:
        baskets_filtered.append(goods_list)

print(f"筛选后有效购物篮：{len(baskets_filtered)}，参与分析单品数量：{len(valid_item)}")

# ========== bool型独热编码，消除mlxtend警告 ==========
item_list = list(valid_item)
onehot_data = []
for basket in baskets_filtered:
    row = np.zeros(len(item_list), dtype=bool)
    for goods in basket:
        row[item_list.index(goods)] = True
    onehot_data.append(row)

df_onehot = pd.DataFrame(onehot_data, columns=item_list)

# ========== 自适应最小支持度 ==========
total_basket = len(baskets_filtered)
min_sup = max(0.005, 10 / total_basket)
print(f"自适应设置最小支持度：{round(min_sup,4)}")

freq_set = apriori(df_onehot, min_support=min_sup, use_colnames=True)
print(f"挖掘得到频繁项集数目：{len(freq_set)}")

# ========== 编码转商品名称函数 ==========
def code2name(codes):
    name_list = []
    for c in list(codes):
        name = df_item[df_item["单品编码"] == c]["单品名称"].values[0]
        name_list.append(name)
    return " + ".join(name_list)

# -------- 强关联互补商品 --------
rules = association_rules(freq_set, metric="lift", min_threshold=1)
rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
rules["前件商品"] = rules["antecedents"].apply(code2name)
rules["后件商品"] = rules["consequents"].apply(code2name)

print("\n===== TOP20 商品强搭配（互补商品，按提升度排序） =====")
print(rules[["前件商品", "后件商品", "support", "confidence", "lift"]].head(20))

# -------- 替代商品 --------
rules_substitute = association_rules(freq_set, metric="lift", min_threshold=0)
rules_substitute["前件商品"] = rules_substitute["antecedents"].apply(code2name)
rules_substitute["后件商品"] = rules_substitute["consequents"].apply(code2name)
substitute_goods = rules_substitute[rules_substitute["lift"] < 0.8].sort_values("lift")

print("\n===== 容易互相替代、极少共同购买的商品组合 =====")
print(substitute_goods[["前件商品", "后件商品", "lift"]].head(10))

# 导出结果
rules.to_csv("模块5_购物篮关联规则_完整结果.csv", index=False, encoding="utf‑8‑sig")

# ========== 优化版共现网络图 ==========
single_pair = []
for _, row in rules.iterrows():
    ant = list(row["antecedents"])
    con = list(row["consequents"])
    if len(ant) == 1 and len(con) == 1:
        single_pair.append((ant[0], con[0], row["lift"]))

top_edges = single_pair[:25]

if len(top_edges) > 0:
    G = nx.Graph()
    for a, b, lift_val in top_edges:
        G.add_edge(a, b, weight=lift_val)

    # 节点大小 = 商品出现频次
    node_size_dict = {}
    for node in G.nodes:
        node_size_dict[node] = item_counter[node]

    # 线条粗细随lift提升度变化
    edge_widths = []
    for u, v in G.edges():
        edge_widths.append(G[u][v]["weight"])

    plt.figure(figsize=(14, 10), dpi=130)
    pos = nx.spring_layout(G, seed=12, k=2.3)

    # 节点标签
    node_label = {}
    for node in G.nodes:
        full_name = df_item[df_item["单品编码"] == node]["单品名称"].values[0]
        node_label[node] = full_name[:6]

    # 绘制节点、连线、文字
    nx.draw_networkx_nodes(G, pos,
                            nodelist=G.nodes,
                            node_size=np.array([node_size_dict[n] for n in G.nodes])/8,
                            node_color="#87CEEB")

    nx.draw_networkx_edges(G, pos, width=np.array(edge_widths)/2, alpha=0.68)
    nx.draw_networkx_labels(G, pos, labels=node_label, font_size=8)

    # 修复方框问题，纯干净标题，无隐形特殊字符
    plt.title("蔬菜单品购物篮共现网络图（边权重：lift提升度）", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("模块5_商品共现网络图.png", dpi=300)
    plt.show()

print("\n✅ 模块5‑购物篮关联规则全部运行完毕！")