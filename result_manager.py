import os
import shutil

def save_business_results(results_dict, base_path="output/results"):
    """
    Отримує словник {номер_питання: DataFrame} та зберігає кожну відповідь у CSV.
    """
    print("\n" + "-"*50)
    print(f"ПРОЦЕС ЗАПИСУ РЕЗУЛЬТАТІВ У CSV (Шлях: {base_path})")
    print("-"*50)

    # Створюємо базову директорію, якщо її немає
    os.makedirs(base_path, exist_ok=True)

    for q_num, df in results_dict.items():
        output_dir = f"{base_path}/question_{q_num}"
        
        try:
            # Використовуємо метод Spark DataFrame .coalesce(1)
            # ВІН НЕ ПОТРЕБУЄ ІМПОРТУ TORCH
            df.coalesce(1).write.mode("overwrite") \
                .option("header", "true") \
                .option("delimiter", ",") \
                .csv(output_dir)
            
            print(f"[OK] Питання №{q_num} збережено.")
        except Exception as e:
            print(f"[ПОМИЛКА] Не вдалося зберегти Питання №{q_num}: {e}")

    print("-"*50)
    print(f"Всього збережено відповідей: {len(results_dict)}")
    print("-"*50)