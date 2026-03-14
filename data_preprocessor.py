from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, isnan, when, approx_count_distinct, year, month, trim, upper
from pyspark.sql.types import DoubleType, FloatType, IntegerType, StringType
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataPreprocessor:
    def __init__(self, df: DataFrame):
        self.df = df

    def step1_general_info(self):
        """1. Загальна статистична інформація та аналіз категоріальних ознак."""
        print("\n" + "="*70)
        print("КРОК 1: Загальна інформація про набір даних (Shape & Info)")
        print("="*70)
        
        row_count = self.df.count()
        col_count = len(self.df.columns)
        print(f"Shape: ({row_count}, {col_count})")
        
        print("\nInfo (Схема даних та типи):")
        self.df.printSchema()

        print("\nУнікальні значення в категоріальних колонках:")
        categorical_cols = [f.name for f in self.df.schema.fields if isinstance(f.dataType, StringType)]
        
        for c in categorical_cols:
            # Беремо до 21 унікального значення, щоб перевірити, чи їх <= 20
            unique_rows = self.df.select(c).distinct().limit(21).collect()
            unique_vals = [r[0] for r in unique_rows]
            
            if len(unique_vals) <= 20:
                # Якщо значень 20 або менше, відображаємо всі
                print(f"\n{c} ({len(unique_vals)} унікальних):")
                print(unique_vals)
            else:
                # Якщо більше 20, рахуємо приблизну кількість і відображаємо перші 5
                total_unique = self.df.select(approx_count_distinct(c)).collect()[0][0]
                print(f"\n{c} (близько {total_unique} унікальних):")
                print(unique_vals[:5])
                print(f"...і ще ~{total_unique - 5} значень")

        return self.df

    def step2_numerical_stats(self):
        """2. Статистика щодо числових ознак, пошук аномалій та візуалізація."""
        print("\n" + "="*70)
        print("КРОК 2: Аналіз числових ознак (Describe, Аномалії, Графіки)")
        print("="*70)
        
        numeric_cols = [f.name for f in self.df.schema.fields if isinstance(f.dataType, (IntegerType, DoubleType, FloatType))]
        existing_num_cols = [c for c in numeric_cols if c in self.df.columns]
        
        print("\nDescribe (Статистика до очищення аномалій):")
        if existing_num_cols:
            self.df.select(existing_num_cols).summary().show()
        
        print("--- Пошук та ВИДАЛЕННЯ логічних аномалій ---")
        if "MAKE_YEAR" in self.df.columns:
            anomalies_year = self.df.filter((col("MAKE_YEAR") < 1900) | (col("MAKE_YEAR") > 2026)).count()
            print(f"ШУМ: Виявлено ТЗ з аномальним роком випуску (< 1900 або > 2026): {anomalies_year}")
            # Фільтруємо аномалії (залишаємо NULL або коректні роки)
            if anomalies_year > 0:
                self.df = self.df.filter(col("MAKE_YEAR").isNull() | ((col("MAKE_YEAR") >= 1900) & (col("MAKE_YEAR") <= 2026)))
                print("-> Аномальні роки випуску успішно видалено з датасету.")

        if "OWN_WEIGHT" in self.df.columns and "TOTAL_WEIGHT" in self.df.columns:
            weight_errors = self.df.filter(col("TOTAL_WEIGHT") < col("OWN_WEIGHT")).count()
            print(f"АНОМАЛІЯ: Записів, де Повна маса < Власної маси: {weight_errors}")
            # Фільтруємо аномалії
            if weight_errors > 0:
                self.df = self.df.filter(col("TOTAL_WEIGHT").isNull() | col("OWN_WEIGHT").isNull() | (col("TOTAL_WEIGHT") >= col("OWN_WEIGHT")))
                print("-> Транспортні засоби з некоректною масою успішно видалено з датасету.")

        print("\n--- Пошук та ВИДАЛЕННЯ статистичних викидів (IQR розмах) ---")
        for c in ["CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT"]:
            if c in self.df.columns:
                # Знаходимо 25 та 75 перцентилі (Q1 та Q3) з похибкою 5%
                quantiles = self.df.approxQuantile(c, [0.25, 0.75], 0.05)
                if len(quantiles) == 2:
                    q1, q3 = quantiles
                    iqr = q3 - q1
                    if iqr > 0:
                        # Беремо 3*IQR для екстремальних викидів (щоб випадково не видалити вантажівки)
                        upper_bound = q3 + 3 * iqr
                        lower_bound = max(0, q1 - 3 * iqr) # Вага/об'єм не можуть бути від'ємними
                        
                        outliers = self.df.filter((col(c) > upper_bound) | (col(c) < lower_bound)).count()
                        if outliers > 0:
                            print(f"ШУМ: Виявлено {outliers} екстремальних викидів у {c} (> {upper_bound} або < {lower_bound})")
                            self.df = self.df.filter((col(c).isNull()) | ((col(c) >= lower_bound) & (col(c) <= upper_bound)))
                            print(f"-> Викиди для {c} успішно відфільтровано.")


        # Оновлюємо список колонок після можливого створення нових у майбутньому
        numeric_cols_for_plot = [f.name for f in self.df.schema.fields if isinstance(f.dataType, (IntegerType, DoubleType, FloatType))]
        
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        pdf = self.df.select(numeric_cols_for_plot).toPandas()
        
        plt.figure(figsize=(16, 12))
        for i, column in enumerate(numeric_cols_for_plot, 1):
            plt.subplot((len(numeric_cols_for_plot) // 2) + 1, 2, i)
            sns.histplot(pdf[column].dropna(), kde=True, bins=30, color='skyblue')
            plt.title(f'Гістограма {column}')
        plt.tight_layout()
        hist_path = os.path.join(plots_dir, "histograms.png")
        plt.savefig(hist_path)
        print(f"-> Гістограми збережено у файл: '{hist_path}'")
        plt.close()

        if len(numeric_cols_for_plot) > 1:
            corr = pdf.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Кореляційна матриця числових колонок")
            corr_path = os.path.join(plots_dir, "correlation_matrix.png")
            plt.savefig(corr_path)
            print(f"-> Кореляційну матрицю збережено у файл: '{corr_path}'")
            plt.close()

        return self.df

    def step3_type_casting(self):
        """3. Стандартизація тексту та створення нових ознак (Feature Engineering)."""
        print("\n" + "="*70)
        print("КРОК 3: Стандартизація тексту, Очищення від 'сміття' та Feature Engineering")
        print("="*70)
        
        print("--- Глобальне очищення текстових ознак ---")
        string_cols = [f.name for f in self.df.schema.fields if isinstance(f.dataType, StringType)]
        
        # Список значень, які по суті є пропусками даних, введеними людьми некоректно
        garbage_values = ["", ".", "-", "ВІДСУТНЄ", "НЕ ВИЗНАЧЕНО", "НЕВИЗНАЧЕНИЙ", "НЕВІДОМО", "NULL", "NONE"]
        
        for c in string_cols:
            if c in self.df.columns:
                # 1. Прибираємо пробіли по краях і переводимо в UPPERCASE
                cleaned_col = upper(trim(col(c)))
                # 2. Якщо значення потрапляє в список 'garbage_values', перетворюємо його на справжній NULL
                self.df = self.df.withColumn(
                    c, 
                    when(cleaned_col.isin(garbage_values), None).otherwise(cleaned_col)
                )
                
        print("-> Усі текстові колонки очищено від зайвих пробілів та переведено у верхній регістр.")
        print("-> Усі некоректні псевдо-пропуски ('.', 'ВІДСУТНЄ' тощо) конвертовано в програмні NULL.")

        print("\n--- Генерація нових ознак (Feature Engineering) ---")
        if "MAKE_YEAR" in self.df.columns:
            # Обчислюємо вік авто (враховуючи, що датасет станом на 2026 рік)
            self.df = self.df.withColumn("AGE", 2026 - col("MAKE_YEAR"))
            print("-> Створено нову колонку 'AGE' (Вік автомобіля).")
            
        if "D_REG" in self.df.columns:
            # Витягуємо окремо рік та місяць реєстрації для можливого аналізу сезонності
            self.df = self.df.withColumn("REG_YEAR", year(col("D_REG"))) \
                             .withColumn("REG_MONTH", month(col("D_REG")))
            print("-> З дати 'D_REG' витягнуто 'REG_YEAR' (Рік) та 'REG_MONTH' (Місяць).")

        return self.df

    def step4_feature_selection(self):
        """4. Аналіз інформативності ознак (вилучення неінформативних та обробка рідкісних)."""
        print("\n" + "="*70)
        print("КРОК 4: Вилучення неінформативних ознак та Rare Labels")
        print("="*70)
        
        # 1. Вилучення неінформативних колонок
        cols_to_drop = ["VIN", "N_REG_NEW", "OPER_NAME", "DEP", "PERSON"]
        existing_cols_to_drop = [c for c in cols_to_drop if c in self.df.columns]
        self.df = self.df.drop(*existing_cols_to_drop)
        print(f"Вилучено колонки: {existing_cols_to_drop}")

        # 2. Обробка рідкісних категорій (Rare Labels)
        print("\n--- Обробка рідкісних категорій (Rare Labels) ---")
        if "BRAND" in self.df.columns:
            # Знаходимо марки, що зустрічаються менше 10 разів
            brand_counts = self.df.groupBy("BRAND").count()
            rare_brands_df = brand_counts.filter(col("count") < 10).select("BRAND")
            rare_brands_list = [row["BRAND"] for row in rare_brands_df.collect()]
            
            if rare_brands_list:
                # Замінюємо рідкісні значення на "ІНШЕ"
                self.df = self.df.withColumn("BRAND", when(col("BRAND").isin(rare_brands_list), "ІНШЕ").otherwise(col("BRAND")))
                print(f"-> {len(rare_brands_list)} рідкісних марок авто (менше 10 записів) об'єднано в категорію 'ІНШЕ'.")

        return self.df

    def step5_missing_and_duplicates(self):
        """5. Аналіз та опрацювання пропущених значень і дублікатів."""
        print("\n" + "="*70)
        print("КРОК 5: Missing values (Пропуски) та Duplicates (Дублікати)")
        print("="*70)
        
        print("\nMissing values (Кількість пропусків):")
        missing_exprs = []
        for f in self.df.schema.fields:
            if isinstance(f.dataType, (DoubleType, FloatType)):
                missing_exprs.append(count(when(isnan(col(f.name)) | col(f.name).isNull(), f.name)).alias(f.name))
            else:
                missing_exprs.append(count(when(col(f.name).isNull(), f.name)).alias(f.name))
                
        missing_counts = self.df.select(missing_exprs).collect()[0].asDict()
        for c_name, m_count in missing_counts.items():
            if m_count > 0:
                print(f"{c_name}: {m_count}")

        print("\n--- Опрацювання пропусків ---")
        
        # 1. Заповнюємо числові пропуски для електрокарів
        if "CAPACITY" in self.df.columns:
            self.df = self.df.fillna({"CAPACITY": 0.0})
            print("-> Пропуски в 'CAPACITY' заповнено нулями (електрокари).")
            
        # 2. Видаляємо критичні рядки, де відсутня маса (їх дуже мало)
        if "OWN_WEIGHT" in self.df.columns or "TOTAL_WEIGHT" in self.df.columns:
            self.df = self.df.dropna(subset=["OWN_WEIGHT", "TOTAL_WEIGHT"])
            print("-> Видалено невелику кількість рядків з критичними пропусками у вазі.")

        # 3. УСІ текстові колонки з NULL (включно з конвертованим "сміттям") заповнюємо як "НЕВІДОМЕ"
        string_cols = [f.name for f in self.df.schema.fields if isinstance(f.dataType, StringType)]
        existing_str_cols = [c for c in string_cols if c in self.df.columns]
        
        for c in existing_str_cols:
            self.df = self.df.fillna({c: "НЕВІДОМЕ"})
        print("-> Усі пропуски у текстових категоріях (FUEL, COLOR тощо) стандартизовано як 'НЕВІДОМЕ'.")

        # Опрацювання дублікатів
        initial_count = self.df.count()
        self.df = self.df.dropDuplicates()
        final_count = self.df.count()
        
        duplicates_sum = initial_count - final_count
        print(f"\nDuplicates: {duplicates_sum}")
        if duplicates_sum > 0:
            print(f"-> Виявлено та вилучено повних дублікатів: {duplicates_sum}")
            
        return self.df

    def run_pipeline(self):
        """Запуск усього пайплайну попередньої обробки."""
        self.step1_general_info()
        self.step3_type_casting()       # Спочатку перетворимо "сміття" на NULL
        self.step2_numerical_stats()    # Аналіз та викиди
        self.step4_feature_selection()  # Вилучення зайвого
        self.step5_missing_and_duplicates() # Заповнення NULL стандартизованим "НЕВІДОМЕ"
        
        print("\n" + "="*70)
        print("=== ПОПЕРЕДНЮ ОБРОБКУ УСПІШНО ЗАВЕРШЕНО ===")
        print("="*70 + "\n")
        return self.df