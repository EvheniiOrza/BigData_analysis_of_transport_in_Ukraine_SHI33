import os
import sys
from pyspark.sql import SparkSession
import data_loader
from data_preprocessor import DataPreprocessor
import transformations
import result_manager

def main():
    # Налаштування середовища для стабільної роботи Spark
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # Створення сесії Spark
    spark = SparkSession.builder \
        .appName("VehicleRegistryAnalysis") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()

    print("\n" + "="*70)
    print("РОБОТА КОНВЕЄРА ОБРОБКИ BIG DATA")
    print("="*70)

    try:
        # 1. ЗАВАНТАЖЕННЯ ДАНИХ
        file_path = "reestrtz01.01.2026.csv"
        print(f"\n[КРОК 1] Читання файлу: {file_path}")
        df = data_loader.load_registry_data(spark, file_path)

        # 2. ПОПЕРЕДНЯ ОБРОБКА (Очищення, IQR, Нові ознаки)
        print("\n[КРОК 2] Попередня обробка та очищення даних...")
        preprocessor = DataPreprocessor(df)
        clean_df = preprocessor.run_pipeline()

        # 3. ТРАНСФОРМАЦІЇ (Обчислення 24 бізнес-питань)
        # Отримуємо словник {номер: DataFrame}
        print("\n[КРОК 3] Запуск аналітичних трансформацій...")
        all_results = transformations.run_transformations(spark, clean_df)

        # 4. ЗАПИС РЕЗУЛЬТАТІВ У CSV
        print("\n[КРОК 4] Експорт результатів у файли CSV...")
        result_manager.save_business_results(all_results)

        print("\n" + "="*70)
        print("ПРОГРАМУ УСПІШНО ЗАВЕРШЕНО")
        print("="*70)

    except Exception as e:
        print(f"\n[КРИТИЧНА ПОМИЛКА] Під час виконання сталася помилка: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()