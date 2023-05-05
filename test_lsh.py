"""
This file is NOT TO BE USED IN THE FINAL VERSION

This uses the LSH implementation of pyspark as a test for our LSH code

Source: https://stackoverflow.com/questions/43938672/efficient-string-matching-in-apache-spark/45602605#45602605
"""
import numpy as np
import pyspark
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, NGram, HashingTF, MinHashLSH


def main():

    # query = spark.createDataFrame(
    #     ["Hello there 7l | real|y like Spark!"], "string"
    # ).toDF("text")

    # db = spark.createDataFrame([
    #     "Hello there 😊! I really like Spark ❤️!", 
    #     "Can anyone suggest an efficient algorithm"
    # ], "string").toDF("text")

    # Query on the mental health subreddits,
    # db is the non-mental health

    # conf = SparkConf().setAppName("MyApp")
    # sc = SparkContext(conf=conf)
    # data = sc.textFile("filtered_data/split_data")

    spark = SparkSession.builder.appName("MyApp").getOrCreate()
    data = spark.sparkContext.textFile("filtered_data/split_data")
    
    data = data.map(lambda line: line.split(",")).map(lambda fields: (fields[0], fields[1]))

    mental_health_subs = ["adhd", "anxiety", "depression", "mentalhealth", "mentalillness", "socialanxiety", "suicidewatch"]
    non_subs = ["gaming", "guns", "music", "parenting"]

    query = data.filter(lambda x: x[0] in mental_health_subs)
    db = data.filter(lambda x: x[0] in non_subs)

    # Sample down to doable sizes
    query = query.sample(False, 10000/query.count(), seed=42)
    db = db.sample(False, 10000/db.count(), seed=42)

    print("MENTAL HEALTH RECORDS: ", query.count())
    print("NON-MENTAL HEALTH RECORDS: ", db.count())

    query = query.toDF()
    db = db.toDF()

    model = Pipeline(stages=[
        RegexTokenizer(
            pattern="", inputCol="_2", outputCol="tokens", minTokenLength=1
        ),
        NGram(n=3, inputCol="tokens", outputCol="ngrams"),
        HashingTF(inputCol="ngrams", outputCol="vectors"),
        MinHashLSH(inputCol="vectors", outputCol="lsh")
    ]).fit(db)

    db_hashed = model.transform(db)
    query_hashed = model.transform(query)

    sim = model.stages[-1].approxSimilarityJoin(db_hashed, query_hashed, 0.75)

    sim.show(n=20,truncate=80, vertical=False)

    # +--------------------+--------------------+------------------+
    # |            datasetA|            datasetB|           distCol|
    # +--------------------+--------------------+------------------+
    # |[Hello there 😊! ...|[Hello there 7l |...|0.5106382978723405|
    # +--------------------+--------------------+------------------+

    print(sim.columns)

    sim2 = sim.select("datasetA._1", "datasetA._2", "datasetB._1", "datasetB._2", "distCol")

    # https://www.geeksforgeeks.org/pyspark-dataframe-distinguish-columns-with-duplicated-name/
    df_cols = sim2.columns
    duplicate_col_index = [idx for idx, val in enumerate(df_cols) if val in df_cols[:idx]]
    for i in duplicate_col_index:
        df_cols[i] = df_cols[i] + '_duplicate_'+ str(i)
    sim2 = sim2.toDF(*df_cols)

    sim2.show(n=20,truncate=80, vertical=False)

    sim2.write.format("csv").option("header", "true").save("media/expectedOutput.csv")


if __name__ == "__main__":
    main()