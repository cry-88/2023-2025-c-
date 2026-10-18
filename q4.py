import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             recall_score, precision_score, f1_score)
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ========== 全局设置 ==========
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
pd.set_option('display.float_format', lambda x: '%.4f' % x)


# ===================== 1. 读取女胎数据（第二个sheet） =====================
print(">>> 1. 加载女胎数据")
df_female = pd.read_excel("附件.xlsx", sheet_name=1)  # sheet1直接就是女胎表
df_female.columns = df_female.columns.str.strip()
print(f"    女胎样本数: {len(df_female)}")

# 将“染色体的非整倍体”列的空值填充为 '0'（代表正常）
df_female["染色体的非整倍体"] = df_female["染色体的非整倍体"].fillna('0')

def make_label(val):
    s = str(val).strip()
    if s in ["0", "0.0", "nan", ""]:
        return 0
    else:
        return 1

y = df_female["染色体的非整倍体"].apply(make_label)
print(f"    异常样本: {y.sum()}，正常样本: {len(y)-y.sum()}")

# ===================== 2. 特征集（完整版） =====================
feature_cols = [
    "13号染色体的Z值",
    "18号染色体的Z值",
    "21号染色体的Z值",
    "X染色体的Z值",
    "13号染色体的GC含量",
    "18号染色体的GC含量",
    "21号染色体的GC含量",
    "X染色体浓度",           # 女胎的X染色体浓度
    "原始读段数",
    "唯一比对的读段数",
    "被过滤掉读段数的比例",
    "孕妇BMI"
]

# 检查哪些特征列存在
available_cols = [col for col in feature_cols if col in df_female.columns]
missing_cols = [col for col in feature_cols if col not in df_female.columns]
if missing_cols:
    print(f"    ⚠️ 以下特征列不存在，已跳过: {missing_cols}")

X_raw = df_female[available_cols].copy()

# 删除特征缺失的样本
mask = X_raw.notna().all(axis=1)
X = X_raw.loc[mask]
y = y.loc[mask]
print(f"    有效建模样本: {len(X)}")
print(f"    特征数: {X.shape[1]}")

# ===================== 3. 数据划分 & 标准化 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 标准化（逻辑回归需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"    训练集: {len(X_train)}，测试集: {len(X_test)}")

# ===================== 4. 主模型：L1正则逻辑回归 =====================
print("\n>>> 2. 主模型：L1正则化逻辑回归")
lr = LogisticRegression(
    penalty="l1",
    solver="saga",
    class_weight="balanced",
    max_iter=10000,
    random_state=42
)
lr.fit(X_train_scaled, y_train)

y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

# ROC曲线 & 最优阈值（约登指数）
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
youden = tpr - fpr
best_idx = np.argmax(youden)
best_thresh = thresholds[best_idx] if len(thresholds) > 0 else 0.5

print(f"    ROC最优阈值 θ = {best_thresh:.4f}")
print(f"    AUC = {auc_lr:.4f}")

y_pred_lr = (y_pred_proba_lr >= best_thresh).astype(int)
cm_lr = confusion_matrix(y_test, y_pred_lr)
tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()

sen_lr = recall_score(y_test, y_pred_lr)
spe_lr = tn_lr / (tn_lr + fp_lr) if (tn_lr + fp_lr) > 0 else 0
pre_lr = precision_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print("\n--- 逻辑回归测试集指标 ---")
print(f"  灵敏度(召回率) = {sen_lr:.4f}")
print(f"  特异度          = {spe_lr:.4f}")
print(f"  精确率          = {pre_lr:.4f}")
print(f"  F1分数          = {f1_lr:.4f}")
print(f"  混淆矩阵:\n{cm_lr}")

# 保存逻辑回归系数
coef_df = pd.DataFrame({
    "feature": available_cols,
    "coef": lr.coef_[0]
})
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False)
coef_df.to_csv("q4_lr_coef.csv", index=False, encoding="utf-8-sig")
print("    系数已保存: q4_lr_coef.csv")

# ===================== 5. 稳健性校验：XGBoost =====================
print("\n>>> 3. 稳健性校验：XGBoost")
pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

xgb_clf = xgb.XGBClassifier(
    scale_pos_weight=pos_weight,
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb_clf.fit(X_train, y_train)

y_pred_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

y_pred_xgb = (y_pred_proba_xgb >= best_thresh).astype(int)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
tn_xgb, fp_xgb, fn_xgb, tp_xgb = cm_xgb.ravel()

sen_xgb = recall_score(y_test, y_pred_xgb)
spe_xgb = tn_xgb / (tn_xgb + fp_xgb) if (tn_xgb + fp_xgb) > 0 else 0
pre_xgb = precision_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

print("\n--- XGBoost 测试集指标 ---")
print(f"  灵敏度(召回率) = {sen_xgb:.4f}")
print(f"  特异度          = {spe_xgb:.4f}")
print(f"  精确率          = {pre_xgb:.4f}")
print(f"  F1分数          = {f1_xgb:.4f}")
print(f"  AUC             = {auc_xgb:.4f}")
print(f"  混淆矩阵:\n{cm_xgb}")

# XGB特征重要性
xgb_import = pd.DataFrame({
    "feature": available_cols,
    "importance": xgb_clf.feature_importances_
}).sort_values("importance", ascending=False)
xgb_import.to_csv("q4_xgb_importance.csv", index=False, encoding="utf-8-sig")
print("    特征重要度已保存: q4_xgb_importance.csv")

# ===================== 6. 绘制ROC曲线对比 =====================
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f"L1‑Logistic (AUC={auc_lr:.4f})")

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
plt.plot(fpr_xgb, tpr_xgb, linewidth=2, label=f"XGBoost (AUC={auc_xgb:.4f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="随机猜测")
plt.xlabel("假阳性率 (FPR)", fontsize=12)
plt.ylabel("真阳性率 (TPR)", fontsize=12)
plt.title("问题4 ROC曲线对比", fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("q4_roc.png", dpi=300, bbox_inches="tight")
plt.close()
print("    ROC曲线已保存: q4_roc.png")

# ===================== 7. 结果汇总表 =====================
result_df = pd.DataFrame([
    {
        "模型": "L1‑Logistic",
        "最优阈值": round(best_thresh, 4),
        "灵敏度": round(sen_lr, 4),
        "特异度": round(spe_lr, 4),
        "精确率": round(pre_lr, 4),
        "F1": round(f1_lr, 4),
        "AUC": round(auc_lr, 4)
    },
    {
        "模型": "XGBoost",
        "最优阈值": round(best_thresh, 4),
        "灵敏度": round(sen_xgb, 4),
        "特异度": round(spe_xgb, 4),
        "精确率": round(pre_xgb, 4),
        "F1": round(f1_xgb, 4),
        "AUC": round(auc_xgb, 4)
    }
])
result_df.to_csv("q4_result_summary.csv", index=False, encoding="utf‑8‑sig")
print("\n    结果汇总已保存: q4_result_summary.csv")

# ===================== 8. 系数可视化（可选） =====================
plt.figure(figsize=(8, 5))
top5 = coef_df.head(5)
colors = ['red' if c < 0 else 'green' for c in top5['coef']]
plt.barh(top5['feature'], top5['coef'], color=colors)
plt.xlabel("回归系数")
plt.title("逻辑回归 Top5 特征系数")
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig("q4_top5_coef.png", dpi=300)
plt.close()
print("    Top5系数图已保存: q4_top5_coef.png")

# ===================== 9. 最终总结 =====================
print("\n" + "="*60)
print("✅ 问题4全部完成！生成文件清单：")
print("  - q4_lr_coef.csv        : 逻辑回归系数")
print("  - q4_xgb_importance.csv : XGBoost特征重要度")
print("  - q4_roc.png            : ROC曲线对比图")
print("  - q4_result_summary.csv : 两模型指标汇总")
print("  - q4_top5_coef.png      : Top5特征系数图")
print("="*60)