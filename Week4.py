from pyspark.sql import SparkSession
import random

# Start Spark
spark = SparkSession.builder.appName("SentenceGenerator").getOrCreate()
sc = spark.sparkContext

# Word list
words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew"]

# Generate sentences on the driver
num_sentences = 1000
sentences = [
    " ".join(random.sample(words, random.randint(1, 6))) + "."
    for _ in range(num_sentences)
]

# Parallelize into RDD
sentences_rdd = sc.parallelize(sentences)

# Transformation (replace with your own logic)
transformed = sentences_rdd.map(lambda s: s.upper())

# Show some results (will go to YARN driver logs in cluster mode)
for line in transformed.take(100):
    print(line)

spark.stop()
