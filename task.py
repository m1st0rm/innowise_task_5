from pyspark.sql import SparkSession, Window
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

# Task 3. Вывести категорию фильмов, на которую потратили больше всего денег.

df_task3 = (
    df_category.join(
        df_film_category,
        df_category["category_id"] == df_film_category["category_id"],
        how="inner",
    )
    .join(
        df_inventory,
        df_film_category["film_id"] == df_inventory["film_id"],
        how="inner",
    )
    .join(
        df_rental,
        df_inventory["inventory_id"] == df_rental["inventory_id"],
        how="inner",
    )
    .join(
        df_payment,
        df_rental["rental_id"] == df_payment["rental_id"],
        how="inner",
    )
    .groupby(df_category["name"].alias("category_name"))
    .agg(F.sum(df_payment["amount"]).alias("spent_on_category"))
    .orderBy(F.col("spent_on_category"), ascending=False)
    .limit(1)
)

# Task 4. Вывести названия фильмов, которых нет в inventory.
# Написать запрос без использования оператора IN.

df_task4 = (
    df_film.join(
        df_inventory, df_film["film_id"] == df_inventory["film_id"], how="left"
    )
    .filter(F.col("inventory_id").isNull())
    .select("title")
)

# Task 5. Вывести топ 3 актеров, которые больше всего появлялись в фильмах в категории “Children”.
# Если у нескольких актеров одинаковое кол-во фильмов, вывести всех.

df_task5_filtered_films = (
    df_film.join(
        df_film_category,
        df_film["film_id"] == df_film_category["film_id"],
        how="inner",
    )
    .join(
        df_category,
        df_film_category["category_id"] == df_category["category_id"],
        how="inner",
    )
    .filter(df_category["name"].like("%Children%"))
    .select(df_film["film_id"])
)

df_task5_actor_appearances = (
    df_actor.join(
        df_film_actor,
        df_actor["actor_id"] == df_film_actor["actor_id"],
        how="inner",
    )
    .join(
        df_task5_filtered_films,
        df_film_actor["film_id"] == df_task5_filtered_films["film_id"],
        how="inner",
    )
    .groupBy(
        df_actor["actor_id"], df_actor["first_name"], df_actor["last_name"]
    )
    .agg(F.count("*").alias("appearances_count"))
)

windowSpec_task5 = Window.orderBy(F.desc("appearances_count"))

df_task5_ranked_actors = df_task5_actor_appearances.withColumn(
    "actor_rank", F.dense_rank().over(windowSpec_task5)
)

df_task5 = df_task5_ranked_actors.filter(F.col("actor_rank") <= 3).orderBy(
    F.col("appearances_count").desc()
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

print(
    "Task 3. Вывести категорию фильмов, на которую потратили больше всего денег."
)
df_task3.show(truncate=False)

print(
    "Task 4. Вывести названия фильмов, которых нет в inventory. "
    "Написать запрос без использования оператора IN."
)
df_task4.show(df_task4.count(), truncate=False)

print(
    "Task 5. Вывести топ 3 актеров, которые больше всего появлялись в фильмах в категории “Children”. "
    "Если у нескольких актеров одинаковое кол-во фильмов, вывести всех."
)
df_task5.select("first_name", "last_name", "appearances_count").show(
    df_task5.count(), truncate=False
)
