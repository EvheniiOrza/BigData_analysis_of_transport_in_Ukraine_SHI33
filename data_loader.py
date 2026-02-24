from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col, to_date, regexp_replace


def get_spark_schema():
    # Визначаємо точну схему (замість inferSchema=True)
    return StructType([
        StructField("PERSON", StringType(), True),
        StructField("REG_ADDR_KOATUU", StringType(), True),
        StructField("OPER_CODE", StringType(), True),
        StructField("OPER_NAME", StringType(), True),
        StructField("D_REG", StringType(), True),  # Спочатку як текст, потім конвертуємо в дату
        StructField("DEP_CODE", StringType(), True),
        StructField("DEP", StringType(), True),
        StructField("BRAND", StringType(), True),
        StructField("MODEL", StringType(), True),
        StructField("VIN", StringType(), True),
        StructField("MAKE_YEAR", IntegerType(), True),
        StructField("COLOR", StringType(), True),
        StructField("KIND", StringType(), True),
        StructField("BODY", StringType(), True),
        StructField("PURPOSE", StringType(), True),
        StructField("FUEL", StringType(), True),
        StructField("CAPACITY", StringType(), True),  # Спочатку як текст (бо містить коми)
        StructField("OWN_WEIGHT", StringType(), True),  # Спочатку як текст
        StructField("TOTAL_WEIGHT", StringType(), True),  # Спочатку як текст
        StructField("N_REG_NEW", StringType(), True)
    ])


def load_registry_data(spark: SparkSession, file_path: str):
    schema = get_spark_schema()

    # Читаємо швидко з готовою схемою
    df = spark.read.csv(
        file_path,
        schema=schema,
        header=True,
        sep=";"
    )

    # Форматуємо дату
    df = df.withColumn("D_REG", to_date(col("D_REG"), "dd.MM.yy"))

    # Форматуємо числові стовпці (змінюємо кому на крапку і робимо тип Double)
    for num_col in ["CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT"]:
        df = df.withColumn(num_col, regexp_replace(col(num_col), ",", ".").cast("double"))

    return df