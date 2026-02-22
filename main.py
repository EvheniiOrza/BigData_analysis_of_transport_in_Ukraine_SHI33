from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Reestr CSV").getOrCreate()

df = spark.read.csv(
    "reestrtz01.01.2026.csv",
    header=True,
    inferSchema=True,
    sep=";"   
)

df.show(20, truncate=False)
