import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings('ignore')

# ========== 全局设置 ==========
pd.set_option('display.float_format', lambda x: '%.6f' % x)
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 加载数据 ==========
print(">>> 1. 加载并清洗数据")
file_path = "附件.xlsx"
df_raw = pd.read_excel(file_path, sheet_name=0)
df_raw.columns = df_raw.columns.str.strip()

# 男胎原始数据（用于计算测量噪声）
df_raw_male = df_raw[df_raw['Y染色体浓度'].notna()].copy()

def convert_weeks(w):
    if pd.isna(w):
        return np.nan
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

df_raw_male['孕周_小数'] = df_raw_male['检测孕周'].apply(convert_weeks)

# 合并标准差估计测量噪声
dup_groups = df_raw_male.groupby(['孕妇代码', '检测抽血次数'])['Y染色体浓度']
n_total = 0
ss_total = 0
for name, group in dup_groups:
    if len(group) > 1:
        n_total += len(group) - 1
        ss_total += np.sum((group - group.mean())**2)
sigma_meas = np.sqrt(ss_total / n_total) if n_total > 0 else 0.005
print(f"    测量噪声标准差 σ_meas (pooled) = {sigma_meas:.4f}")

# 聚合重复测序，得到建模数据集
group_cols = ['孕妇代码', '检测抽血次数']
num_cols = ['年龄', '身高', '体重', '孕周_小数', '孕妇BMI', 'Y染色体浓度']
df = df_raw_male.groupby(group_cols)[num_cols].mean().reset_index()
df = df.dropna(subset=['Y染色体浓度', '孕周_小数', '孕妇BMI','年龄','身高'])
print(f"    聚合后建模样本量: {len(df)}")

# ========== 2. 读取问题3回归系数（新增年龄、身高） ==========
try:
    lmm_coef = pd.read_csv("lmm_result_q3.csv", index_col=0)
    print("    使用 Q3 LMM模型系数（含年龄、身高）")
except:
    print("    ⚠️ 未找到q3回归文件，使用演示默认系数，请替换为你真实回归输出！")
    coef_dict = {
        'Intercept': 0.077,
        '孕周_中心': 0.003,
        'BMI_中心': -0.002,
        '年龄_中心': -0.0008,
        '身高_中心': 0.0005,
        '孕周BMI_交互': 0.0002
    }
    lmm_coef = pd.DataFrame({
        'coef': [coef_dict['Intercept'], coef_dict['孕周_中心'],
                 coef_dict['BMI_中心'], coef_dict['年龄_中心'],
                 coef_dict['身高_中心'], coef_dict['孕周BMI_交互']],
        'std_err': [0.002, 0.000, 0.001, 0.001, 0.0005, 0.000]
    }, index=['Intercept', '孕周_中心', 'BMI_中心', '年龄_中心','身高_中心','孕周BMI_交互'])

beta0 = lmm_coef.loc['Intercept', 'coef']
beta1 = lmm_coef.loc['孕周_中心', 'coef']
beta2 = lmm_coef.loc['BMI_中心', 'coef']
beta3 = lmm_coef.loc['年龄_中心', 'coef']
beta4 = lmm_coef.loc['身高_中心', 'coef']
beta5 = lmm_coef.loc['孕周BMI_交互', 'coef']

# 各变量全局均值（中心化用）
mean_ga    = df['孕周_小数'].mean()
mean_bmi   = df['孕妇BMI'].mean()
mean_age   = df['年龄'].mean()
mean_height= df['身高'].mean()

# ========== 3. 模型残差标准差 ==========
def estimate_sigma_q3(df_in, b0,b1,b2,b3,b4,b5, m_ga,m_bmi,m_age,m_h):
    ga_c  = df_in['孕周_小数'] - m_ga
    bmi_c = df_in['孕妇BMI'] - m_bmi
    age_c = df_in['年龄'] - m_age
    h_c   = df_in['身高'] - m_h
    y_pred = b0 + b1*ga_c + b2*bmi_c + b3*age_c + b4*h_c + b5*ga_c*bmi_c
    resid = df_in['Y染色体浓度'] - y_pred
    return np.std(resid, ddof=1)

sigma_resid = estimate_sigma_q3(df, beta0,beta1,beta2,beta3,beta4,beta5,
                                mean_ga,mean_bmi,mean_age,mean_height)
sigma_total = np.sqrt(sigma_resid**2 + sigma_meas**2)
print(f"    Q3模型残差标准差 σ_resid = {sigma_resid:.4f}")
print(f"    Q3总标准差 σ_total = {sigma_total:.4f}")

# ========== 4. Q3核心函数：预测、达标概率、总风险 ==========
# 单样本预测：传入t,bmi,age,height
def predict_y_q3(t, bmi, age, height, b0,b1,b2,b3,b4,b5, m_ga,m_bmi,m_age,m_h):
    ga_c  = t - m_ga
    bmi_c = bmi - m_bmi
    age_c = age - m_age
    h_c   = height - m_h
    return b0 + b1*ga_c + b2*bmi_c + b3*age_c + b4*h_c + b5 * ga_c * bmi_c

def calc_p_q3(t, bmi, age, height, b0,b1,b2,b3,b4,b5, m_ga,m_bmi,m_age,m_h, sigma):
    y_pred = predict_y_q3(t, bmi, age, height, b0,b1,b2,b3,b4,b5, m_ga,m_bmi,m_age,m_h)
    z = (0.04 - y_pred) / sigma
    p = 1 - norm.cdf(z)
    return np.clip(p,0,1)

# 组平均达标概率：传入本组全部样本的bmi/age/height数组
def p_group_q3(t, bmi_arr, age_arr, h_arr, b0,b1,b2,b3,b4,b5, m_ga,m_bmi,m_age,m_h, sigma):
    plist = []
    for b,a,hi in zip(bmi_arr, age_arr, h_arr):
        plist.append(calc_p_q3(t, b,a,hi, b0,b1,b2,b3,b4,b5, m_ga,m_bmi,m_age,m_h, sigma))
    return np.mean(plist)

def total_risk_q3(t, bmi_arr, age_arr, h_arr, b0,b1,b2,b3,b4,b5, m_ga,m_bmi,m_age,m_h, sigma, k):
    pt = p_group_q3(t, bmi_arr, age_arr, h_arr, b0,b1,b2,b3,b4,b5, m_ga,m_bmi,m_age,m_h, sigma)
    r_inacc = 1 - pt
    r_late  = 1.0 / (1.0 + np.exp(-k * (t - 12.0)))
    return r_inacc + r_late

# ========== 5. Q3优化求解 ==========
def find_optimal_t_q3(bmi_arr, age_arr, h_arr, b0,b1,b2,b3,b4,b5, m_ga,m_bmi,m_age,m_h, sigma, k):
    def obj(t):
        return total_risk_q3(t, bmi_arr, age_arr, h_arr, b0,b1,b2,b3,b4,b5, m_ga,m_bmi,m_age,m_h, sigma, k)
    try:
        res = minimize_scalar(obj, bounds=(10.0, 25.0), method='bounded')
        if res.success:
            return res.x, res.fun
        else:
            return np.nan, np.nan
    except Exception:
        return np.nan, np.nan

# ========== 6. BMI分组（和Q2完全一致，依旧BMI分位数） ==========
def get_bmi_groups(df_in, n_bins=4):
    bmi_arr = df_in['孕妇BMI'].values.reshape(-1,1)
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    labels = kbd.fit_transform(bmi_arr).flatten().astype(int)
    edges = kbd.bin_edges_[0]
    df_g = df_in.copy()
    df_g['BMI_group'] = labels
    ranges = [(edges[i], edges[i+1]) for i in range(n_bins)]
    print(f"    {n_bins}组分箱边界: {[round(e,2) for e in edges]}")
    for i,(le,re) in enumerate(ranges):
        print(f"      组{i+1}: BMI ∈ [{le:.2f}, {re:.2f})")
    return df_g, labels, ranges

# ========== 7. Q3主优化流程 ==========
def run_optimization_q3(df_in, b0,b1,b2,b3,b4,b5, m_ga,m_bmi,m_age,m_h, sigma,
                        n_bins=4, k=0.1, plot_curve=True, label_suffix=""):
    print(f"\n>>> Q3优化: {n_bins}组, k={k}")
    df_grouped, _, group_ranges = get_bmi_groups(df_in, n_bins=n_bins)

    results = []
    for g in range(n_bins):
        sub = df_grouped[df_grouped['BMI_group'] == g]
        bmi_vals = sub['孕妇BMI'].values
        age_vals = sub['年龄'].values
        h_vals   = sub['身高'].values

        t_opt, r_min = find_optimal_t_q3(bmi_vals, age_vals, h_vals,
                                         b0,b1,b2,b3,b4,b5,
                                         m_ga,m_bmi,m_age,m_h, sigma, k)
        p_opt = np.nan
        if not np.isnan(t_opt):
            p_opt = p_group_q3(t_opt, bmi_vals, age_vals, h_vals,
                               b0,b1,b2,b3,b4,b5,
                               m_ga,m_bmi,m_age,m_h, sigma)

        results.append({
            '组号': g+1,
            'BMI下限': round(group_ranges[g][0],2),
            'BMI上限': round(group_ranges[g][1],2),
            '组样本数': len(bmi_vals),
            '最优孕周(周)': round(t_opt,2) if not np.isnan(t_opt) else np.nan,
            '最小总风险': round(r_min,4) if not np.isnan(r_min) else np.nan,
            '该时点达标概率': round(p_opt,4) if not np.isnan(p_opt) else np.nan
        })

        if plot_curve and not np.isnan(t_opt):
            t_grid = np.linspace(10,25,100)
            risk_grid = [total_risk_q3(tt, bmi_vals,age_vals,h_vals,
                                       b0,b1,b2,b3,b4,b5,
                                       m_ga,m_bmi,m_age,m_h, sigma, k) for tt in t_grid]
            plt.figure(figsize=(7,4))
            plt.plot(t_grid, risk_grid, label=f'Q3组{g+1}总风险')
            plt.axvline(t_opt, ls='--', color='red', label=f'最优t*={t_opt:.2f}周')
            plt.xlabel('孕周 t (周)')
            plt.ylabel('总风险')
            plt.title(f'Q3 BMI组{g+1} 风险曲线 (k={k})')
            plt.legend()
            plt.savefig(f"q3_risk_curve_group{g+1}{label_suffix}.png", dpi=120)
            plt.close()

    df_res = pd.DataFrame(results)
    print("\n--- Q3优化结果 ---")
    print(df_res.to_string(index=False))
    return df_res, df_grouped

# ========== 8. Q3敏感性分析（σ扰动 + 蒙特卡洛，复用逻辑） ==========
def sensitivity_analysis_q3(df_in, b0,b1,b2,b3,b4,b5, m_ga,m_bmi,m_age,m_h,
                            sig_resid, sig_meas, n_bins=4, n_mc=500, k=0.1):
    print(f"\n>>> Q3敏感性分析 (蒙特卡洛 {n_mc} 次)")

    # 8.1 σ扰动扫描
    print("\n--- Q3 σ扰动扫描 ---")
    sigma_factors = [0.6,0.8,1.0,1.2,1.4]
    df_grouped, _, _ = get_bmi_groups(df_in, n_bins=n_bins)
    scan_results = []
    for fac in sigma_factors:
        sm = sig_meas * fac
        st = np.sqrt(sig_resid**2 + sm**2)
        row = {'σ因子':fac}
        for g in range(n_bins):
            sub = df_grouped[df_grouped['BMI_group'] == g]
            bmi_v = sub['孕妇BMI'].values
            age_v = sub['年龄'].values
            h_v   = sub['身高'].values
            topt,_ = find_optimal_t_q3(bmi_v,age_v,h_v, b0,b1,b2,b3,b4,b5,
                                       m_ga,m_bmi,m_age,m_h, st, k)
            row[f'组{g+1}'] = round(topt,2) if not np.isnan(topt) else np.nan
        scan_results.append(row)
        print(f"    σ因子{fac:.1f}: 各最优孕周 = {[row[f'组{i+1}'] for i in range(n_bins)]}")
    df_scan = pd.DataFrame(scan_results)
    df_scan.to_csv("q3_sigma_scan.csv", index=False)

    # 8.2 蒙特卡洛模拟，带进度打印
    print("\n--- Q3蒙特卡洛模拟 ---")
    df_grouped, _, group_ranges = get_bmi_groups(df_in, n_bins=n_bins)
    mc_results = []
    for g in range(n_bins):
        sub = df_grouped[df_grouped['BMI_group'] == g]
        bmi_vals = sub['孕妇BMI'].values
        age_vals = sub['年龄'].values
        h_vals   = sub['身高'].values
        t_samples = []
        print(f"    蒙特卡洛：正在处理第{g+1}/{n_bins}组")
        for idx,_ in enumerate(range(n_mc)):
            noise_scale = np.random.uniform(0.8,1.2)
            sigma_mc = sigma_total * noise_scale
            t_opt_mc,_ = find_optimal_t_q3(bmi_vals, age_vals, h_vals,
                                           b0,b1,b2,b3,b4,b5,
                                           m_ga,m_bmi,m_age,m_h, sigma_mc, k)
            if not np.isnan(t_opt_mc):
                t_samples.append(t_opt_mc)
            if (idx+1) % 100 == 0:
                print(f"        组{g+1}已完成 {idx+1}/{n_mc}")

        if len(t_samples)==0:
            mc_results.append({'组号':g+1,'BMI下限':group_ranges[g][0],'BMI上限':group_ranges[g][1],
                               't*均值':np.nan,'t*标准差':np.nan,'95%CI下限':np.nan,'95%CI上限':np.nan})
            continue
        t_mean = np.mean(t_samples)
        t_std  = np.std(t_samples, ddof=1)
        ci_low = np.percentile(t_samples,2.5)
        ci_high= np.percentile(t_samples,97.5)
        mc_results.append({
            '组号':g+1,
            'BMI下限':round(group_ranges[g][0],2),
            'BMI上限':round(group_ranges[g][1],2),
            't*均值(周)':round(t_mean,2),
            't*标准差(周)':round(t_std,3),
            '95%CI下限':round(ci_low,2),
            '95%CI上限':round(ci_high,2)
        })
    df_mc = pd.DataFrame(mc_results)
    print(df_mc.to_string(index=False))
    df_mc.to_csv("q3_monte_carlo.csv", index=False)

    valid = df_mc.dropna(subset=['t*标准差(周)'])
    if len(valid)>0:
        cv_list = valid['t*标准差(周)'] / valid['t*均值(周)']
        max_cv = max(cv_list)
        print(f"\nQ3最大变异系数 CV = {max_cv*100:.2f}%")
        if max_cv < 0.05:
            print("    ✅ Q3模型对检测误差稳健(CV<5%)")
        else:
            print("    ⚠️ Q3模型对检测误差存在一定敏感(CV≥5%)")
    return df_mc, df_scan

# ========== 9. Q3主入口【正式版】 ==========
if __name__ == "__main__":
    # 9.1 Q3基准主结果 k=0.1
    df_4_q3, _ = run_optimization_q3(df, beta0,beta1,beta2,beta3,beta4,beta5,
                                     mean_ga,mean_bmi,mean_age,mean_height,
                                     sigma_total, n_bins=4, k=0.1, label_suffix="_k0.1")
    df_4_q3.to_csv("q3_optimal_4bin.csv", index=False)

    # 9.2 Q3敏感性分析，正式 n_mc=500
    df_mc_q3, df_scan_q3 = sensitivity_analysis_q3(df, beta0,beta1,beta2,beta3,beta4,beta5,
                                                   mean_ga,mean_bmi,mean_age,mean_height,
                                                   sigma_resid, sigma_meas,
                                                   n_bins=4, n_mc=500, k=0.1)

    # 9.3 k参数扫描（稳健性校验）
    print("\n>>> Q3 k参数扫描(稳健性校验)")
    for kk in [0.05, 0.1, 0.20]:
        print(f"\n--- Q3 k={kk} ---")
        df_k,_ = run_optimization_q3(df, beta0,beta1,beta2,beta3,beta4,beta5,
                                    mean_ga,mean_bmi,mean_age,mean_height,
                                    sigma_total, n_bins=4, k=kk, plot_curve=False, label_suffix=f"_k{kk}")
        print(df_k[['组号','最优孕周(周)']].to_string(index=False))

    # 9.4 5组分箱稳健校验
    print("\n>>> Q3 5组方案(稳健性校验)")
    df_5_q3,_ = run_optimization_q3(df, beta0,beta1,beta2,beta3,beta4,beta5,
                                    mean_ga,mean_bmi,mean_age,mean_height,
                                    sigma_total, n_bins=5, k=0.1, plot_curve=False, label_suffix="_5bin")
    df_5_q3.to_csv("q3_optimal_5bin.csv", index=False)

    print("\n✅ 问题3【正式版】全部完成！")
    print("输出文件：")
    print("  - q3_optimal_4bin.csv    : Q3 4组主结果")
    print("  - q3_optimal_5bin.csv    : Q3 5分箱稳健性")
    print("  - q3_monte_carlo.csv     : Q3蒙特卡洛500次，95%CI")
    print("  - q3_sigma_scan.csv      : Q3 σ扰动扫描")
    print("  - q3_risk_curve_group*.png : Q3各组风险曲线图")