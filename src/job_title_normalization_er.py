from pyspark.sql import SparkSession
from pyspark.sql import types
from pyspark.sql import functions as f
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from lib.er import EntityResolution
from lib.schemas import SchemaER

import re
import sys

_punctuation_list = list(punctuation)
_punctuation_list.remove('+') # remove because of things like C++ (probably + is not in normal text
_eng_stopwords = stopwords.words('english')

# this is a copy and paste from lib/cleaners because can't get that whole module running
# on pyspark, so resort to something simpler
@f.udf(returnType=types.ArrayType(types.StringType()))
def remove_punctuations_and_stopwords(s):
	# split sentences further to individual sentences, then word tokenize
	no_bullets_sentences = [sent for sent in sent_tokenize(s)]
	keepwords = [w for s in no_bullets_sentences for w in word_tokenize(s) if w not in _eng_stopwords]
	keepwords = [word for word in keepwords if word not in _punctuation_list]
	return keepwords

@f.udf(returnType=types.ArrayType(types.StringType()))
def tokenize_job_titles(title):
        return title.apply(remove_punctuations_and_stopwords)

def main(spark, resume_path, titles_path):
	resume_df = spark.read.json(resume_path, schema=SchemaER.get_resume_schema())
	resume_df.show()
	titles_df = spark.read.json(titles_path, schema=SchemaER.get_title_schema())

	resume_df = resume_df.withColumnRenamed('id','resume_id')

	titles_df = titles_df.withColumn('id', f.monotonically_increasing_id())

	resume_df = resume_df.select('resume_id', f.explode('title').alias('job_titles'))

	# remove empty jobs
	resume_df = resume_df.where(f.col('job_titles') != '')
	resume_df = resume_df.withColumn('job_titles', remove_punctuations_and_stopwords(f.lower(f.col('job_titles'))))
	resume_df = resume_df.withColumn('id', f.monotonically_increasing_id())

	titles_df.cache()
	resume_df.cache()

	# write these files
	titles_df.write.json('titles-preprocessed-for-er', mode='overwrite')
	resume_df.write.json('resume-preprocessed-for-er', mode='overwrite')

	titles_df.show()
	resume_df.show()

	er_resol = EntityResolution()
	keep_cols1 = ['O*NET-SOC Code']
	keep_cols2 = ['resume_id']
	er_result = er_resol.jaccardJoin(f.broadcast(titles_df), resume_df, 'Alternate Title', 'job_titles', 0.5, keep_cols1, keep_cols2)
	er_result.show()
	
	er_result.write.json('job-title-normalization-er', mode='overwrite', compression='gzip')


if __name__ == "__main__":
	resume_path = sys.argv[1]
	titles_path = sys.argv[2]

	spark = SparkSession.builder.appName('job title normalize er').getOrCreate()
	spark.sparkContext.setLogLevel('WARN')

	main(spark, resume_path, titles_path)