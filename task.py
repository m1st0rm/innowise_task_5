from pyspark.sql import SparkSession


spark = (
    SparkSession.builder.appName("Innowise Task 5")
    .config("spark.jars", "postgresql-42.7.4.jar")
    .getOrCreate()
)

url = "jdbc:postgresql://localhost:5432/postgres"

properties = {
    "user": "postgres",
    "password": "123456",
    "driver": "org.postgresql.Driver",
}

df_category = spark.read.jdbc(
    url=url, table="public.category", properties=properties
)

df_category.show(df_category.count(), truncate=False)
