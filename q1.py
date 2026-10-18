import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')  # 忽略版本兼容的警告，保持输出清爽

# ========== 全局设置 ==========
pd.set_option('display.float_format', lambda x: '%.6f' % x)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 增加备选字体
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 加载与筛选 ==========
def step_1_load_data(file_path):
    print(">>> 1. 加载数据 & 筛选男胎")
    df_raw = pd.read_excel(file_path, sheet_name=0)
    # 【修复】一键去除所有列名首尾空格，防止KeyError
    df_raw.columns = df_raw.columns.str.strip()
    df = df_raw[df_raw['Y染色体浓度'].notna()].copy()
    print(f"    男胎记录数: {len(df)}")
    return df

# ========== 2. 孕周转换 ==========
def step_2_convert_weeks(df):
    print(">>> 2. 孕周转换 (w+d -> 小数)")
    def convert(w):
        if pd.isna(w): return np.nan
        w = str(w).strip().lower()
        if 'w' not in w:
            try:
                return float(w)
            except:
                return np.nan
        parts = w.split('w')
        week = int(parts[0])
        day = int(parts[1].split('+')[1]) if '+' in parts[1] else 0
        return week + day / 7
    df['孕周_小数'] = df['检测孕周'].apply(convert)
    print(f"    样例: {df[['检测孕周', '孕周_小数']].head(3).to_string(index=False)}")
    print(f"    孕周转换后缺失数量：{df['孕周_小数'].isna().sum()}")
    return df

# ========== 3. 同一抽血多次测序取均值（核心防伪复本） ==========
def step_3_aggregate_repeats(df):
    print(">>> 3. 按同一次采血合并重复测序")
    group_cols = ['孕妇代码', '检测抽血次数']
    num_cols = ['年龄', '身高', '体重', '孕周_小数', '孕妇BMI', 'Y染色体浓度',
                '原始读段数', '在参考基因组上比对的比例', '重复读段的比例', 'GC含量']
    df_agg = df.groupby(group_cols)[num_cols].mean().reset_index()
    print(f"    合并前 {len(df)} 行 -> 合并后 {len(df_agg)} 行")
    return df_agg

# ========== 4. 异常值探查（修复：只删缺失，不做全局3σ硬剔除） ==========
def step_4_remove_outliers(df):
    print(">>> 4. Y浓度异常值探查 (箱线图 + 仅删除缺失值)")
    # 画图保存
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df['Y染色体浓度'])
    plt.title('Y染色体浓度分布 (保留原始值)')
    plt.savefig('boxplot_y.png', dpi=150)
    plt.close()
    print("    箱线图已保存为 boxplot_y.png")
    # 【修复】只删除NaN，不做全局3σ一刀切（避免误删低孕周正常值）
    df_clean = df.dropna(subset=['Y染色体浓度','孕周_小数','孕妇BMI','年龄']).copy()
    print(f"    删除缺失值后：{len(df)} → {len(df_clean)}")
    return df_clean

# ========== 5. 特征工程（中心化 + 交互项） ==========
def step_5_feature_engineering(df):
    print(">>> 5. 中心化 & 构造交互项")
    df['孕周_中心'] = df['孕周_小数'] - df['孕周_小数'].mean()
    df['BMI_中心'] = df['孕妇BMI'] - df['孕妇BMI'].mean()
    df['年龄_中心'] = df['年龄'] - df['年龄'].mean()
    df['孕周BMI_交互'] = df['孕周_中心'] * df['BMI_中心']
    print("    中心化完成，预览:")
    print(df[['孕周_小数', '孕妇BMI', '孕周_中心', 'BMI_中心', '孕周BMI_交互']].head(3))
    return df

# ========== 6. 相关性分析 ==========
def step_6_correlation(df):
    print(">>> 6. 相关性矩阵 (Pearson & Spearman)")
    cols = ['Y染色体浓度', '孕周_小数', '孕妇BMI', '年龄']
    pearson = df[cols].corr(method='pearson')
    spearman = df[cols].corr(method='spearman')
    print("\\n--- Pearson ---")
    print(pearson.round(4))
    print("\\n--- Spearman ---")
    print(spearman.round(4))
    print(f"\\n关键: Y与孕周 r = {pearson.loc['Y染色体浓度', '孕周_小数']:.3f}")
    print(f"关键: Y与BMI  r = {pearson.loc['Y染色体浓度', '孕妇BMI']:.3f}")
    pearson.to_csv("pearson_corr.csv")
    spearman.to_csv("spearman_corr.csv")
    print("    相关性矩阵已保存 csv")
    return pearson, spearman

# ========== 7. 主模型：线性混合模型 (LMM) ==========
def step_7_lmm(df):
    print(">>> 7. 主模型：线性混合模型 (LMM)")
    df = df.sort_values(['孕妇代码', '检测抽血次数'])
    try:
        model = MixedLM.from_formula(
            "Y染色体浓度 ~ 孕周_中心 + BMI_中心 + 年龄_中心 + 孕周BMI_交互",
            data=df,
            groups=df["孕妇代码"]
        )
        result = model.fit(method='lbfgs', maxiter=2000)
        print(result.summary())
        
        # 【修复】判断版本是否支持R²计算，兼容旧版statsmodels
        if hasattr(result, 'get_marginal_r2'):
            marg_r2 = result.get_marginal_r2()
            cond_r2 = result.get_conditional_r2()
            print(f"\\n边际 R² (固定效应): {marg_r2:.4f}")
            print(f"条件 R² (含个体随机截距): {cond_r2:.4f}")
        else:
            # 旧版本回退方案：计算固定效应的近似R²
            y_mean = df['Y染色体浓度'].mean()
            ss_res = np.sum((result.fittedvalues - df['Y染色体浓度']) ** 2)
            ss_tot = np.sum((df['Y染色体浓度'] - y_mean) ** 2)
            print(f"\\n近似 R² (固定效应): {1 - ss_res/ss_tot:.4f}")
            print("提示：建议升级 statsmodels 获取精确R² (pip install --upgrade statsmodels)")
        
        # 保存系数
        res_df = result.params.to_frame(name="coef")
        res_df["std_err"] = result.bse
        res_df["pvalue"] = result.pvalues
        res_df.to_csv("lmm_result.csv")
        print("    LMM系数结果保存 lmm_result.csv")
        return result
    except Exception as e:
        print(f"!!! LMM模型收敛失败：{e}")
        print("!!! 跳过LMM，自动切换至OLS稳健模型作为主结果")
        return None

# ========== 8. 稳健性检验：OLS + 聚类稳健标准误 ==========
def step_8_ols_robust(df):
    print(">>> 8. 稳健性检验：OLS + 聚类稳健标准误")
    X = df[['孕周_中心', 'BMI_中心', '年龄_中心', '孕周BMI_交互']]
    X = add_constant(X)
    y = df['Y染色体浓度']
    try:
        model = OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['孕妇代码']})
    except:
        # 如果聚类标准误报错（旧版本），回退到普通稳健标准误
        print("   聚类标准误失败，回退到异方差稳健标准误 (HC1)")
        model = OLS(y, X).fit(cov_type='HC1')
    
    print(model.summary())
    model.params.to_frame("coef").assign(std_err=model.bse, pvalue=model.pvalues).to_csv("ols_robust_result.csv")
    print("    OLS稳健回归结果保存 ols_robust_result.csv")
    return model

# ========== 主执行流程 ==========
if __name__ == "__main__":
    # 【唯一需要修改的地方】请替换为你的附件实际路径
    FILE_PATH = "附件.xlsx"  

    df = step_1_load_data(FILE_PATH)
    df = step_2_convert_weeks(df)
    df = step_3_aggregate_repeats(df)
    df = step_4_remove_outliers(df)
    df = step_5_feature_engineering(df)
    step_6_correlation(df)
    lmm_res = step_7_lmm(df)
    ols_res = step_8_ols_robust(df)

    print("\\n✅ 全部执行完毕！请检查控制台输出与生成的 csv/png 文件。")
    print("如果LMM收敛失败，请使用 OLS 结果（已自动回退）作为问题1的关系模型。")