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

# Task 6. Вывести города с количеством активных и неактивных клиентов (активный — customer.active = 1).
# Отсортировать по количеству неактивных клиентов по убыванию.

df_task6 = (
    df_customer.join(
        df_address,
        df_customer["address_id"] == df_address["address_id"],
        how="inner",
    )
    .join(df_city, df_address["city_id"] == df_city["city_id"], how="inner")
    .groupby(df_city["city_id"], df_city["city"].alias("city_name"))
    .agg(
        F.sum(df_customer["active"]).alias("active_customers_count"),
        (F.count("*") - F.sum(df_customer["active"])).alias(
            "inactive_customers_count"
        ),
    )
    .orderBy(F.col("inactive_customers_count"), ascending=False)
)

# Вывести категорию фильмов, у которой самое большое кол-во часов суммарной аренды
# в городах (customer.address_id в этом city), и которые начинаются на букву “a”.
# То же самое сделать для городов в которых есть символ “-”. Написать все в одном запросе.

df_task7_city_a = (
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
        df_customer,
        df_rental["customer_id"] == df_customer["customer_id"],
        how="inner",
    )
    .join(
        df_address,
        df_customer["address_id"] == df_address["address_id"],
        how="inner",
    )
    .join(df_city, df_address["city_id"] == df_city["city_id"], how="inner")
    .filter(
        (df_rental["rental_date"].isNotNull())
        & (df_rental["return_date"].isNotNull())
        & (df_city["city"].like("A%"))
    )
    .groupBy(
        df_category["category_id"], df_category["name"].alias("city_name")
    )
    .agg(
        (
            F.sum(
                F.unix_timestamp(df_rental["return_date"])
                - F.unix_timestamp(df_rental["rental_date"])
            )
            / 3600
        ).alias("category_summary_rental_time")
    )
    .orderBy(F.col("category_summary_rental_time").desc())
    .limit(1)
)

df_task7_city_dash = (
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
        df_customer,
        df_rental["customer_id"] == df_customer["customer_id"],
        how="inner",
    )
    .join(
        df_address,
        df_customer["address_id"] == df_address["address_id"],
        how="inner",
    )
    .join(df_city, df_address["city_id"] == df_city["city_id"], how="inner")
    .filter(
        (df_rental["rental_date"].isNotNull())
        & (df_rental["return_date"].isNotNull())
        & (df_city["city"].like("%-%"))
    )
    .groupBy(
        df_category["category_id"], df_category["name"].alias("city_name")
    )
    .agg(
        (
            F.sum(
                F.unix_timestamp(df_rental["return_date"])
                - F.unix_timestamp(df_rental["rental_date"])
            )
            / 3600
        ).alias("category_summary_rental_time")
    )
    .orderBy(F.col("category_summary_rental_time").desc())
    .limit(1)
)

df_task7 = df_task7_city_a.unionByName(df_task7_city_dash)

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

print(
    "Task 6. Вывести города с количеством активных и неактивных клиентов (активный — customer.active = 1)."
    "Отсортировать по количеству неактивных клиентов по убыванию."
)
df_task6.select(
    "city_name", "active_customers_count", "inactive_customers_count"
).show(df_task6.count(), truncate=False)

print(
    "Task 7. Вывести категорию фильмов, у которой самое большое кол-во часов суммарной аренды"
    " в городах (customer.address_id в этом city), и которые начинаются на букву “a”."
    "То же самое сделать для городов в которых есть символ “-”. Написать все в одном запросе."
)
df_task7.select("city_name", "category_summary_rental_time").show(
    df_task7.count(), truncate=False
)
