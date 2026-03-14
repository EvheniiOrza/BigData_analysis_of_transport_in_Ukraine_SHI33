from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F

def run_transformations(spark: SparkSession, df):
    """
    Виконання 24 бізнес-питань. Повертає словник {номер: DataFrame}.
    Включає етап аналізу планів виконання (explain).
    """
    results = {}
    
    # Довідники
    df_regions = spark.createDataFrame([
        ("8000000000", "м. Київ"), ("4610100000", "Львівська обл."), 
        ("6325111000", "Харківська обл."), ("0520880903", "Вінницька обл.")
    ], ["REG_ADDR_KOATUU", "REGION_NAME"])

    df_ops = spark.createDataFrame([
        ("315", "Перереєстрація на нового власника"), 
        ("70", "Первинна реєстрація"), ("254", "Належний користувач")
    ], ["OPER_CODE", "OPER_DESC"])

    print("\n[INFO] Обчислення 24 бізнес-питань...")

    # 1. Скільки реєстраційних дій відбулося в кожному відомому нам регіоні?
    results[1] = df.join(df_regions, "REG_ADDR_KOATUU", "inner") \
                   .groupBy("REGION_NAME").count()

    # 2. Які 5 брендів є лідерами в столиці (Київ)?
    results[2] = df.join(df_regions, "REG_ADDR_KOATUU", "inner") \
                   .filter(F.col("REGION_NAME") == "м. Київ") \
                   .groupBy("BRAND").count().orderBy(F.col("count").desc()).limit(5)

    # 3. Список всіх зареєстрованих електрокарів з деталізацією типу операції.
    results[3] = df.join(df_ops, "OPER_CODE", "inner") \
                   .filter(F.col("FUEL") == "ЕЛЕКТРО") \
                   .select("BRAND", "MODEL", "OPER_DESC", "MAKE_YEAR")

    # 4. Який середній вік легкових автомобілів залежно від типу операції?
    results[4] = df.join(df_ops, "OPER_CODE", "inner") \
                   .filter(F.col("KIND") == "ЛЕГКОВИЙ") \
                   .groupBy("OPER_DESC").agg(F.round(F.avg("AGE"), 1).alias("avg_age"))

    # 5. Яке сумарне логістичне навантаження (повна маса) вантажівок по регіонах?
    results[5] = df.join(df_regions, "REG_ADDR_KOATUU", "inner") \
                   .filter(F.col("KIND") == "ВАНТАЖНИЙ") \
                   .groupBy("REGION_NAME").agg(F.sum("TOTAL_WEIGHT").alias("sum_weight"))

    # 6. Як розподіляються реєстраційні дії для бренду TOYOTA?
    results[6] = df.join(df_ops, "OPER_CODE", "inner") \
                   .filter(F.col("BRAND") == "TOYOTA") \
                   .groupBy("OPER_DESC").count()

    # 7. Який середній об'єм двигуна у дизельних авто в різних областях?
    results[7] = df.join(df_regions, "REG_ADDR_KOATUU", "inner") \
                   .filter(F.col("FUEL") == "ДИЗЕЛЬНЕ ПАЛИВО") \
                   .groupBy("REGION_NAME").agg(F.round(F.avg("CAPACITY"), 0).alias("avg_cap"))

    # 8. Скільки "нових" авто (вік до 2 років) зареєстровано по типах операцій?
    results[8] = df.join(df_ops, "OPER_CODE", "inner") \
                   .filter(F.col("AGE") <= 2) \
                   .groupBy("OPER_DESC").count()

    # 9. Хто є лідером (Топ-1 марка) для кожного типу кузова?
    q9_win = Window.partitionBy("BODY").orderBy(F.col("count").desc())
    results[9] = df.groupBy("BODY", "BRAND").count() \
                   .withColumn("rank", F.rank().over(q9_win)) \
                   .filter(F.col("rank") == 1)

    # 10. Яка модель є найважчою для кожного окремого бренду?
    q10_win = Window.partitionBy("BRAND").orderBy(F.col("TOTAL_WEIGHT").desc())
    results[10] = df.withColumn("rn", F.row_number().over(q10_win)) \
                    .filter(F.col("rn") == 1).select("BRAND", "MODEL", "TOTAL_WEIGHT")

    # 11. Список автомобілів, об'єм двигуна яких вищий за середній по їхній марці.
    q11_win = Window.partitionBy("BRAND")
    results[11] = df.filter(F.col("CAPACITY") > 0) \
                    .withColumn("avg_brand", F.avg("CAPACITY").over(q11_win)) \
                    .filter(F.col("CAPACITY") > F.col("avg_brand"))

    # 12. Накопичувальна кількість реєстрацій білих та чорних авто за роками випуску.
    q12_win = Window.partitionBy("COLOR").orderBy("MAKE_YEAR")
    results[12] = df.filter(F.col("COLOR").isin("БІЛИЙ", "ЧОРНИЙ")) \
                    .groupBy("COLOR", "MAKE_YEAR").count() \
                    .withColumn("running_total", F.sum("count").over(q12_win))

    # 13. Які 3 кольори є найпопулярнішими серед мотоциклістів?
    results[13] = df.filter(F.col("KIND") == "МОТОЦИКЛ") \
                    .groupBy("COLOR").count().orderBy(F.col("count").desc()).limit(3)

    # 14. На скільки років кожне авто старіше за найновішу машину тієї ж моделі?
    q14_win = Window.partitionBy("BRAND", "MODEL")
    results[14] = df.withColumn("min_age", F.min("AGE").over(q14_win)) \
                    .withColumn("age_diff", F.col("AGE") - F.col("min_age")) \
                    .select("BRAND", "MODEL", "AGE", "age_diff")

    # 15. Яка відсоткова частка різних типів пального всередині кожного бренду?
    q15_win = Window.partitionBy("BRAND")
    results[15] = df.groupBy("BRAND", "FUEL").count() \
                    .withColumn("total", F.sum("count").over(q15_win)) \
                    .withColumn("percentage", F.round(F.col("count")/F.col("total")*100, 2))

    # 16. Яка транзакція була найпершою в реєстрі для кожної марки за датою?
    q16_win = Window.partitionBy("BRAND").orderBy("D_REG")
    results[16] = df.withColumn("rn", F.row_number().over(q16_win)) \
                    .filter(F.col("rn") == 1).select("BRAND", "MODEL", "D_REG")

    # 17. Скільки реєстрацій відбулося в січні для кожного бренду?
    results[17] = df.filter(F.col("REG_MONTH") == 1).groupBy("BRAND").count()

    # 18. Яка середня вантажопідйомність (Total - Own weight) у спеціалізованих авто?
    results[18] = df.filter(F.col("PURPOSE") == "СПЕЦІАЛІЗОВАНИЙ") \
                    .withColumn("load", F.col("TOTAL_WEIGHT") - F.col("OWN_WEIGHT")) \
                    .groupBy("BODY").agg(F.round(F.avg("load"), 1).alias("avg_load"))

    # 19. Список червоних легкових авто, випущених після 2020 року.
    results[19] = df.filter((F.col("COLOR") == "ЧЕРВОНИЙ") & 
                            (F.col("MAKE_YEAR") > 2020) & 
                            (F.col("KIND") == "ЛЕГКОВИЙ"))

    # 20. У скількох унікальних моделях брендів представлені гібридні двигуни?
    results[20] = df.filter(F.col("FUEL").contains("ЕЛЕКТРО") & F.col("FUEL").contains("БЕНЗИН")) \
                    .groupBy("BRAND").agg(F.countDistinct("MODEL").alias("hybrid_models"))

    # 21. Хто входить в Топ-10 найстаріших автомобілів реєстру?
    results[21] = df.orderBy(F.col("AGE").desc()).limit(10) \
                    .select("BRAND", "MODEL", "MAKE_YEAR", "AGE")

    # 22. По яких брендах найчастіше не вказано колір (категорія НЕВІДОМЕ)?
    results[22] = df.filter(F.col("COLOR") == "НЕВІДОМЕ").groupBy("BRAND").count()

    # 23. Який середній об'єм двигуна у трійки японських лідерів (Toyota, Nissan, Honda)?
    results[23] = df.filter(F.col("BRAND").isin("TOYOTA", "NISSAN", "HONDA")) \
                    .groupBy("BRAND").agg(F.round(F.avg("CAPACITY"), 0).alias("avg_cap"))

    # 24. Список транспортних засобів з аномально малою вагою (< 500 кг).
    results[24] = df.filter(F.col("TOTAL_WEIGHT") < 500) \
                    .select("KIND", "BRAND", "MODEL", "TOTAL_WEIGHT")

    # ЕТАП АНАЛІЗУ ПЛАНУ ВИКОНАННЯ
    print("\n" + "="*70)
    print("АНАЛІЗ ФІЗИЧНОГО ПЛАНУ ВИКОНАННЯ (EXPLAIN)")
    print("="*70)
    
    print("\n=== ПЛАН ДЛЯ ПИТАННЯ №2 (ТОП МАРКИ КИЄВА) ===")
    results[2].explain(mode="formatted")

    return results