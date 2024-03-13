from aioptima import analys, visualize


df = analys.load_file("titanic", "csv")

type(df)

analys.overview(df)

cat_cols, num_cols, cat_but_car, num_but_cat = analys.column_detection(df)

analys.outlier_thresholds(df, "Age")

analys.check_outlier(df, "Age")

analys.outliers_themselves(df, "Age")

analys.missing_values_table(df, True)

analys.missing_values_table(df)

analys.convert_numeric_to_categorical(df, cat_but_car)

visualize.single_categorical_var_visualize(df, "Sex")

visualize.single_numeric_var_visualize(df, "Fare", "hist")

visualize.single_numeric_var_visualize(df, "Fare", "boxplot")

visualize.cat_summary(df, cat_cols)
