from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import dense_rank, col
from lib.schemas import SchemaCheckingER
import sys

def main(spark, resume_path, titles_path, er_path):
	resume_df = spark.read.json(resume_path, schema=SchemaCheckingER.get_resume_er_schema())
	titles_df = spark.read.json(titles_path, schema=SchemaCheckingER.get_title_er_schema())
	er_df = spark.read.json(er_path, schema=SchemaCheckingER.get_er_schema())

	print('Total number of resume job titles: %d' % resume_df.count())
	print('Total number of titles data: %d' % titles_df.count())
	print('Total number of er result: %d' % er_df.count())

	# this ordering is based on job_title_normalization_er.py (please see it there)
	combined_df = er_df.join(titles_df, on=[titles_df.id == er_df.id1]).drop('id')
	combined_df = combined_df.join(resume_df, on=[resume_df.id == combined_df.id2]).drop('id')

	# create window partition over id1 (resume job title ID)
	window = Window.partitionBy('id1').orderBy(combined_df.jaccard.desc())

	# get top 2 per resume job title
	top_2_df = combined_df.select('*', dense_rank().over(window).alias('rank')).where(col('rank') <= 2).cache()
	top_2_df.show(100, truncate=False)
	print('Top 2 ER for each job title total is: %d' % top_2_df.count())
	top_2_df.write.json('top-2-job-er', mode='overwrite')

if __name__ == "__main__":
	resume_path = sys.argv[1]
	titles_path = sys.argv[2]
	er_path = sys.argv[3]

	spark = SparkSession.builder.appName('checking job normalization er').getOrCreate()
	spark.sparkContext.setLogLevel('WARN')
	main(spark, resume_path, titles_path, er_path)