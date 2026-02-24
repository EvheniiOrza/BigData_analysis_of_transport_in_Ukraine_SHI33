from pyspark.sql import SparkSession
import data_loader


def main():
    # Ініціалізація сесії
    spark = SparkSession.builder \
        .appName("Reestr CSV") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()

    # Замість spark.read.csv() викликаємо нашу функцію з data_loader
    df = data_loader.load_registry_data(spark, "reestrtz01.01.2026.csv")

    # Перевірка: дивимося, чи правильно визначились типи колонок
    print("--- Схема датасету ---")
    df.printSchema()

    # Перевірка: виводимо 20 рядків (дія над DataFrame)
    print("--- Перші 20 рядків ---")
    df.show(20, truncate=False)

    # Коректне завершення сесії
    spark.stop()


## third_stage
if __name__ == "__main__":
    main()