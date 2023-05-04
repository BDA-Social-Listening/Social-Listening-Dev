"""
This file is NOT TO BE USED IN THE FINAL VERSION

This uses the LSH implementation of pyspark as a test for our LSH code

Source: https://stackoverflow.com/questions/43938672/efficient-string-matching-in-apache-spark/45602605#45602605
"""

import pyspark
# from pyspark import SparkConf
# from pyspark.context import SparkContext
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

    mental_health_subs = ["adhd", "anxiety", "depression", "mentalhealth", "mentalillness", "socialanxiety", "suicidewatch"]
    non_subs = ["gaming", "guns", "music", "parenting"]

    print(data.take(1))

    query = data.filter(lambda x: x[0] in mental_health_subs).toDF()
    db = data.filter(lambda x: x[0] in non_subs).toDF()

    model = Pipeline(stages=[
        RegexTokenizer(
            pattern="", inputCol="text", outputCol="tokens", minTokenLength=1
        ),
        NGram(n=3, inputCol="tokens", outputCol="ngrams"),
        HashingTF(inputCol="ngrams", outputCol="vectors"),
        MinHashLSH(inputCol="vectors", outputCol="lsh")
    ]).fit(db)

    db_hashed = model.transform(db)
    query_hashed = model.transform(query)

    model.stages[-1].approxSimilarityJoin(db_hashed, query_hashed, 0.75).show()
    # +--------------------+--------------------+------------------+
    # |            datasetA|            datasetB|           distCol|
    # +--------------------+--------------------+------------------+
    # |[Hello there 😊! ...|[Hello there 7l |...|0.5106382978723405|
    # +--------------------+--------------------+------------------+


if __name__ == "__main__":
    main()