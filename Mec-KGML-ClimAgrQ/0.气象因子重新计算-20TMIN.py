import pandas as pd

# 读取两个 Excel 文件
df1 = pd.read_excel(r"D:\实验\毕业论文\第四章\3.模型汇总\2025年数据的模型验证\建模用数据\2025垩白度数据-算气象因子-物候期.xlsx")  # 物候期数据
df2 = pd.read_excel(r"D:\实验\毕业论文\第四章\3.模型汇总\2025年数据的模型验证\建模用数据\2025水稻气象因子.xlsx")  # 气候数据

# 第七列为 TMIN，对应 pandas 索引 6
tmin_col = 6

# 如果原始气象数据以 0.1 ℃ 为单位，则 20 ℃ 对应 200
# 如果原始气象数据已经是 ℃，则改为 20
tmin_threshold = 22

# 输出列位置
# 可根据你的 df1 表头实际情况修改
count_tmin_gt_20_col = 8          # TMIN > 20 的数量
effective_gdd_tmin_gt_20_col = 9  # TMIN 超过 20 的有效积温

for i, line1_value in enumerate(df1.iloc[:, 1]):

    # 第一步：筛选 df2 中第二列与 df1 第二列值相同的行，年份
    df2_filtered1 = df2[df2.iloc[:, 1] == line1_value]
    print(f"Step 1 for combination {i + 1}: {df2_filtered1.shape[0]} rows remaining")

    # 第二步：在筛选后的 df2 中，第一列与 df1 第三列值相同的行，站点
    df2_filtered2 = df2_filtered1[df2_filtered1.iloc[:, 0] == df1.iloc[i, 2]]
    print(f"Step 2: {df2_filtered2.shape[0]} rows remaining")

    # 第三步：在筛选后的 df2 中，第三列与 df1 第四列值相同的行，起始月份
    df2_filtered3 = df2_filtered2[df2_filtered2.iloc[:, 2] == df1.iloc[i, 3]]
    print(f"Step 3: {df2_filtered3.shape[0]} rows remaining")

    # 第四步：在筛选后的 df2 中，第四列与 df1 第五列值相同的行，起始日期
    line1 = df2_filtered3[df2_filtered3.iloc[:, 3] == df1.iloc[i, 4]]
    print(f"Step 4: {line1.shape[0]} rows in line1")

    # 第五步：在筛选后的 df2 中，第三列与 df1 第六列值相同的行，结束月份
    df2_filtered4 = df2_filtered2[df2_filtered2.iloc[:, 2] == df1.iloc[i, 5]]
    print(f"Step 5: {df2_filtered4.shape[0]} rows remaining")

    # 第六步：在筛选后的 df2 中，第四列与 df1 第七列值相同的行，结束日期
    line2 = df2_filtered4[df2_filtered4.iloc[:, 3] == df1.iloc[i, 6]]
    print(f"Step 6: {line2.shape[0]} rows in line2")

    # 获取 line1 和 line2 之间的数据区间
    if not (line1.empty or line2.empty):
        start_index = line1.index[0]
        end_index = line2.index[-1]

        # 提取第七列 TMIN
        tmin_series = pd.to_numeric(
            df2.iloc[start_index:end_index + 1, tmin_col],
            errors="coerce"
        )

        # 只保留有效数值
        tmin_series = tmin_series.dropna()

        # 计算 TMIN > 20 的数量
        count_tmin_gt_20 = (tmin_series > tmin_threshold).sum()

        # 计算 TMIN 超过 20 的有效积温
        # 公式为 sum(TMIN - 20)，仅对 TMIN > 20 的日期计算
        effective_gdd_tmin_gt_20 = (tmin_series[tmin_series > tmin_threshold] - tmin_threshold).sum()

        # 写入 df1
        df1.at[i, count_tmin_gt_20_col] = count_tmin_gt_20
        df1.at[i, effective_gdd_tmin_gt_20_col] = effective_gdd_tmin_gt_20

    else:
        print(f"未找到满足 df1 第 {i + 1} 行 line1 和 line2 条件的数据行，无法计算。请检查数据和筛选条件。")

# 保存结果
df1.to_excel(r"C:\Users\27144\Desktop\1111.xlsx", index=False)