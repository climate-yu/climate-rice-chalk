import pandas as pd
import numpy as np

# 读取两个Excel文件
df1 = pd.read_excel(r"D:\实验\毕业论文\第四章\3.模型汇总\2025年数据的模型验证\建模用数据\2025垩白度数据-算气象因子-物候期.xlsx")  # 物候期数据
df2 = pd.read_excel(r"D:\实验\毕业论文\第四章\3.模型汇总\2025年数据的模型验证\建模用数据\2025水稻气象因子.xlsx")  # 气候数据

# 定义列名称
columns = {
    5: "均温",
    6: "最高温",
    7: "最低温",
    8: "降水量",
    9: "日照时数",
    10: "相对湿度",
    11: "风速"
}

# 跳过齐穗后的前10天
skip_days_after_line1 = 10

# 跳过前10天后，剩余统计窗口至少需要10天
# 如果想更严格，可改为15
min_window_days = 1


def count_consecutive(series, condition):
    max_count = 0
    current_count = 0
    for value in series:
        if condition(value):
            current_count += 1
        else:
            max_count = max(max_count, current_count)
            current_count = 0
    return max(max_count, current_count)


# 遍历df1中的每一行进行筛选和计算
for i, (line1_value, line2_value) in enumerate(zip(df1.iloc[:, 1], df1.iloc[:, 5])):

    # Step 1-6: 筛选line1和line2
    df2_filtered1 = df2[df2.iloc[:, 1] == line1_value]
    df2_filtered2 = df2_filtered1[df2_filtered1.iloc[:, 0] == df1.iloc[i, 2]]

    # line1：齐穗时间
    df2_filtered3 = df2_filtered2[df2_filtered2.iloc[:, 2] == df1.iloc[i, 3]]
    line1 = df2_filtered3[df2_filtered3.iloc[:, 3] == df1.iloc[i, 4]]

    # line2：收获时间
    df2_filtered4 = df2_filtered2[df2_filtered2.iloc[:, 2] == df1.iloc[i, 5]]
    line2 = df2_filtered4[df2_filtered4.iloc[:, 3] == df1.iloc[i, 6]]

    if not (line1.empty or line2.empty):
        start_index = line1.index[0]
        end_index = line2.index[-1]

        # line1 到 line2 的完整区间行数，包含line1和line2
        total_rows = end_index - start_index + 1

        # 如果完整灌浆期不足以跳过齐穗后前10天，则不计算
        if total_rows <= skip_days_after_line1:
            print(
                f"df1第{i + 1}行：line1到line2总天数为{total_rows}，"
                f"不足以跳过前{skip_days_after_line1}天，结果空着。"
            )
            df1.at[i, "统计窗口天数"] = np.nan
            df1.at[i, "完整灌浆期天数"] = total_rows
            continue

        # 从line1之后第10天开始统计，一直到line2
        # line1当天不统计，line1后第1到第9天不统计，line1后第10天开始统计
        calc_start_index = start_index + skip_days_after_line1
        calc_end_index = end_index

        data_range = df2.iloc[calc_start_index:calc_end_index + 1]

        # 实际统计窗口天数
        row_count = data_range.shape[0]

        # 如果跳过前10天后剩余窗口过短，则不计算
        if row_count < min_window_days:
            print(
                f"df1第{i + 1}行：跳过齐穗后前{skip_days_after_line1}天后，"
                f"剩余统计窗口仅{row_count}天，小于{min_window_days}天，结果空着。"
            )
            df1.at[i, "统计窗口天数"] = row_count
            df1.at[i, "完整灌浆期天数"] = total_rows
            continue

        # 记录窗口信息
        df1.at[i, "完整灌浆期天数"] = total_rows
        df1.at[i, "统计窗口天数"] = row_count
        df1.at[i, "统计窗口起始偏移天数"] = skip_days_after_line1

        # 检查第10列和第11列是否为空
        if not data_range.iloc[:, 9].isnull().all() and not data_range.iloc[:, 10].isnull().all():

            # 2. 第5列到第11列的平均值
            for col in range(5, 12):
                column_name = columns[col]
                df1.at[i, f"{column_name}的平均值"] = data_range.iloc[:, col - 1].mean()

            # 3. 第5列分别大于等于0-40的数量
            for threshold in range(0, 41):
                df1.at[i, f"均温大于等于{threshold}的天数"] = (
                    data_range.iloc[:, 4] >= threshold
                ).sum()

            # 4. 第5列分别小于等于0-40的数量
            for threshold in range(0, 41):
                df1.at[i, f"均温小于等于{threshold}的天数"] = (
                    data_range.iloc[:, 4] <= threshold
                ).sum()

            # 5. 第6列分别大于等于0-40的数量
            for threshold in range(0, 41):
                df1.at[i, f"最高温大于等于{threshold}的天数"] = (
                    data_range.iloc[:, 5] >= threshold
                ).sum()

            # 6. 第6列分别小于等于0-40的数量
            for threshold in range(0, 41):
                df1.at[i, f"最高温小于等于{threshold}的天数"] = (
                    data_range.iloc[:, 5] <= threshold
                ).sum()

            # 7. 第7列分别小于等于0-40的数量
            for threshold in range(0, 41):
                df1.at[i, f"最低温小于等于{threshold}的天数"] = (
                    data_range.iloc[:, 6] <= threshold
                ).sum()

            # 8. 第7列分别大于等于0-40的数量
            for threshold in range(0, 41):
                df1.at[i, f"最低温大于等于{threshold}的天数"] = (
                    data_range.iloc[:, 6] >= threshold
                ).sum()

            # 9. 第6列和第7列差值的平均值
            temp_diff = data_range.iloc[:, 5] - data_range.iloc[:, 6]
            df1.at[i, "温差的平均值"] = temp_diff.mean()

            # 10. 第6列和第7列差值大于等于0-20的数量
            for threshold in range(0, 21):
                df1.at[i, f"温差大于等于{threshold}的天数"] = (
                    temp_diff >= threshold
                ).sum()

            # 11. 第6列和第7列差值小于等于0-20的数量
            for threshold in range(0, 21):
                df1.at[i, f"温差小于等于{threshold}的天数"] = (
                    temp_diff <= threshold
                ).sum()

            # 27. 第10列的总和
            df1.at[i, "相对湿度的总和"] = data_range.iloc[:, 9].sum()

            # 28-29. 第10列分别大于等于和小于等于0-95的数量
            for threshold in range(0, 96):
                df1.at[i, f"相对湿度大于等于{threshold}的天数"] = (
                    data_range.iloc[:, 9] >= threshold
                ).sum()
                df1.at[i, f"相对湿度小于等于{threshold}的天数"] = (
                    data_range.iloc[:, 9] <= threshold
                ).sum()

            # 30-34. 第11列的各种统计
            for threshold in range(0, 11):
                df1.at[i, f"风速大于等于{threshold}的天数"] = (
                    data_range.iloc[:, 10] >= threshold
                ).sum()
                df1.at[i, f"风速小于等于{threshold}的天数"] = (
                    data_range.iloc[:, 10] <= threshold
                ).sum()

            df1.at[i, "风速大于等于25的天数"] = (
                data_range.iloc[:, 10] >= 25
            ).sum()
            df1.at[i, "风速大于等于50的天数"] = (
                data_range.iloc[:, 10] >= 50
            ).sum()
            df1.at[i, "风速大于等于100的天数"] = (
                data_range.iloc[:, 10] >= 100
            ).sum()

        # 其他列统计逻辑

        # 12. 第9列大于等于0.1的数量
        df1.at[i, "日照时数大于等于0.1的天数"] = (
            data_range.iloc[:, 8] >= 0.1
        ).sum()

        # 13. 第9列的总和
        df1.at[i, "日照时数的总和"] = data_range.iloc[:, 8].sum()

        # 14. 第9列小于0.1的数量
        df1.at[i, "日照时数小于0.1的天数"] = (
            data_range.iloc[:, 8] < 0.1
        ).sum()

        # 15. 连续第9列小于0.1的最大连续天数
        df1.at[i, "日照时数小于0.1的最大连续天数"] = count_consecutive(
            data_range.iloc[:, 8],
            lambda x: x < 0.1
        )

        # 16. 第9列分别大于等于0-16的数量
        for threshold in range(0, 17):
            df1.at[i, f"日照时数大于等于{threshold}的天数"] = (
                data_range.iloc[:, 8] >= threshold
            ).sum()

        # 17. 第9列分别小于等于0-16的数量
        for threshold in range(0, 17):
            df1.at[i, f"日照时数小于等于{threshold}的天数"] = (
                data_range.iloc[:, 8] <= threshold
            ).sum()

        # 18. 第8列大于0.1的数量
        df1.at[i, "降水量大于0.1的天数"] = (
            data_range.iloc[:, 7] > 0.1
        ).sum()

        # 19-23. 第8列不同降水量范围的数量
        for threshold in [25, 50, 75, 100]:
            df1.at[i, f"降水量大于{threshold}的天数"] = (
                data_range.iloc[:, 7] > threshold
            ).sum()

        # 24. 第8列的总和
        df1.at[i, "降水量的总和"] = data_range.iloc[:, 7].sum()

        # 25. 连续第8列小于1的最大连续天数
        df1.at[i, "降水量小于1的最大连续天数"] = count_consecutive(
            data_range.iloc[:, 7],
            lambda x: x < 1
        )

        # 26. 第5列的分别大于0-40的数的总和
        for threshold in range(0, 41):
            df1.at[i, f"均温大于{threshold}的总和"] = (
                data_range.iloc[:, 4][data_range.iloc[:, 4] > threshold].sum()
            )

        # 35. 总行数
        # 这里记录的是齐穗后第10天到收获的统计窗口天数
        df1.at[i, "总行数"] = row_count

        print(
            f"df1第{i + 1}行计算完成：完整灌浆期{total_rows}天，"
            f"统计窗口{row_count}天。"
        )

    else:
        print(
            f"未找到满足df1第{i + 1}行line1和line2条件的数据行，"
            f"无法计算。请检查数据和筛选条件。"
        )
        df1.at[i, "完整灌浆期天数"] = np.nan
        df1.at[i, "统计窗口天数"] = np.nan
        df1.at[i, "统计窗口起始偏移天数"] = np.nan

# 将更新后的df1保存回原文件
df1.to_excel(
    r"D:\实验\毕业论文\第四章\3.模型汇总\2025年数据的模型验证\建模用数据\2025垩白度数据-算气象因子-物候期.xlsx",
    index=False
)