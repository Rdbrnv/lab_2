import findspark

findspark.init()

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import SnowballStemmer
import re

conf = SparkConf().setAppName("LiteraryWorkAnalysis").set("spark.executor.memory", "4g")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Loading the text data
text_rdd = sc.textFile("Bulgakov.txt")

total_word_count_before = text_rdd.flatMap(lambda line: re.sub("[^a-zA-Z\s]", "", line).lower().split()).count()

# Text cleaning
stop_words = sc.textFile("stop_words.txt").collect()
stop_words = set(stop_words)


def clean_text(line):
    line = re.sub("[^a-zA-Z\s]", "", line).lower().split()
    # Removing stop words
    line = [word for word in line if word not in stop_words]
    return line


cleaned_text_rdd = text_rdd.flatMap(clean_text)

total_word_count_after = cleaned_text_rdd.count()

word_count_rdd = cleaned_text_rdd.map(lambda word: (word, 1)).reduceByKey(lambda x, y: x + y)

top50_most_common = word_count_rdd.takeOrdered(50, key=lambda x: -x[1])
top50_least_common = word_count_rdd.takeOrdered(50, key=lambda x: x[1])

print(f"Total Word Count Before Removing Stop Words: {total_word_count_before}")
print(f"Total Word Count After Removing Stop Words: {total_word_count_after}")

print("\nTop 50 Most Common Words:")
for word, count in top50_most_common:
    print(f"{word}: {count}")

print("\nTop 50 Least Common Words:")
for word, count in top50_least_common:
    print(f"{word}: {count}")

# Stemming
stemmer = SnowballStemmer("english")


def stemming(word):
    return stemmer.stem(word)


stemmed_text_rdd = cleaned_text_rdd.map(stemming)

stemmed_word_count_rdd = stemmed_text_rdd.map(lambda word: (word, 1)).reduceByKey(lambda x, y: x + y)

top50_most_common_stemmed = stemmed_word_count_rdd.takeOrdered(50, key=lambda x: -x[1])
top50_least_common_stemmed = stemmed_word_count_rdd.takeOrdered(50, key=lambda x: x[1])

print("\nTop 50 Most Common Words After Stemming:")
for word, count in top50_most_common_stemmed:
    print(f"{word}: {count}")

print("\nTop 50 Least Common Words After Stemming:")
for word, count in top50_least_common_stemmed:
    print(f"{word}: {count}")

sc.stop()
