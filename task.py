from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = (
    SparkSession.builder.appName("Innowise Task 5")
    .master("local[*]")
    .config("spark.jars", "postgresql-42.7.4.jar")
    .getOrCreate()
)

url = "jdbc:postgresql://localhost:5432/postgres"

properties = {
    "user": "postgres",
    "password": "123456",
    "driver": "org.postgresql.Driver",
}

df_film = spark.read.jdbc(url=url, table="public.film", properties=properties)
df_actor = spark.read.jdbc(
    url=url, table="public.actor", properties=properties
)
df_film_actor = spark.read.jdbc(
    url=url, table="public.film_actor", properties=properties
)
df_category = spark.read.jdbc(
    url=url, table="public.category", properties=properties
)
df_film_category = spark.read.jdbc(
    url=url, table="public.film_category", properties=properties
)
df_customer = spark.read.jdbc(
    url=url, table="public.customer", properties=properties
)
df_inventory = spark.read.jdbc(
    url=url, table="public.inventory", properties=properties
)
df_rental = spark.read.jdbc(
    url=url, table="public.rental", properties=properties
)
df_payment = spark.read.jdbc(
    url=url, table="public.payment", properties=properties
)
df_address = spark.read.jdbc(
    url=url, table="public.address", properties=properties
)
df_city = spark.read.jdbc(url=url, table="public.city", properties=properties)

# Task 1. Вывести количество фильмов в каждой категории, отсортировать по убыванию.

df_task1 = (
    df_category.join(
        df_film_category,
        df_category["category_id"] == df_film_category["category_id"],
        how="inner",
    )
    .groupby(df_category["name"].alias("category_name"))
    .agg(F.count("*").alias("films_in_category_count"))
    .orderBy(F.col("films_in_category_count"), ascending=False)
)

# Task 2. Вывести 10 актеров, чьи фильмы большего всего арендовали, отсортировать по убыванию.

df_task2_film_rentals = (
    df_film.join(
        df_inventory,
        df_film["film_id"] == df_inventory["film_id"],
        how="inner",
    )
    .join(
        df_rental,
        df_inventory["inventory_id"] == df_rental["inventory_id"],
        how="inner",
    )
    .groupby(df_film["film_id"])
    .agg(F.count("*").alias("film_rentals_count"))
)

df_task2 = (
    df_actor.join(
        df_film_actor,
        df_actor["actor_id"] == df_film_actor["actor_id"],
        how="inner",
    )
    .join(
        df_task2_film_rentals,
        df_film_actor["film_id"] == df_task2_film_rentals["film_id"],
        how="inner",
    )
    .groupby(
        df_actor["actor_id"], df_actor["first_name"], df_actor["last_name"]
    )
    .agg(F.sum("film_rentals_count").alias("actor_film_borrow_count"))
    .orderBy(F.col("actor_film_borrow_count"), ascending=False)
    .limit(10)
)


# Tasks output
print(
    "Task 1. Вывести количество фильмов в каждой категории, отсортировать по убыванию."
)
df_task1.show(df_task1.count(), truncate=False)

print(
    "Task 2. Вывести 10 актеров, чьи фильмы большего всего арендовали, отсортировать по убыванию."
)
df_task2.select("first_name", "last_name", "actor_film_borrow_count").show(
    truncate=False
)
