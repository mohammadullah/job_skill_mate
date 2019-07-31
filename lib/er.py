from pyspark.sql import SparkSession
from pyspark.sql.functions import (col, collect_set, concat, flatten, lower, split,
	lit, trim, explode, array_distinct, array_intersect, array_union, size)

class EntityResolution:
	def filtering(self, df1, df2, key1, key2, keep_cols1, keep_cols2):
		""" 
			Input: $df1 and $df2 are two input DataFrames, where each of them 
				has a 'joinKey' column added by the preprocessDF function

			Output: Return a new DataFrame $candDF with four columns: 'id1', 'joinKey1', 'id2', 'joinKey2',
					where 'id1' and 'joinKey1' are from $df1, and 'id2' and 'joinKey2'are from $df2.
					Intuitively, $candDF is the joined result between $df1 and $df2 on the condition that 
					their joinKeys share at least one token. 

			Comments: Since the goal of the "filtering" function is to avoid n^2 pair comparisons, 
					you are NOT allowed to compute a catesian join between $df1 and $df2 in the function. 
					Please come up with a more efficient algorithm (see hints in Lecture 2). 
		"""
		expandedKeywordsDf1 = df1.select('id', explode(key1).alias('keywords'))
		expandedKeywordsDf2 = df2.select(col('id').alias('id2'), explode(key2).alias('keywords'))
		
		matches = expandedKeywordsDf2.join(expandedKeywordsDf1, on='keywords')\
			.dropDuplicates(['id','id2'])
		
		return df1.select('id', key1, *keep_cols1).join(matches, on='id')\
				.join(df2.select(col('id').alias('id2'), key2, *keep_cols2), on='id2').withColumnRenamed('id', 'id1')

	def verification(self, candDF, threshold, key1, key2, keep_cols1, keep_cols2):
		""" 
			Input: $candDF is the output DataFrame from the 'filtering' function. 
				   $threshold is a float value between (0, 1] 

			Output: Return a new DataFrame $resultDF that represents the ER result. 
					It has five columns: id1, joinKey1, id2, joinKey2, jaccard 

			Comments: There are two differences between $candDF and $resultDF
					  (1) $resultDF adds a new column, called jaccard, which stores the jaccard similarity 
						  between $joinKey1 and $joinKey2
					  (2) $resultDF removes the rows whose jaccard similarity is smaller than $threshold 
		"""
		return candDF.select(
			'id1', 'id2',
			(size(array_intersect(key1,key2))\
			/ size(array_union(key1,key2))).alias('jaccard'),
			# keep certain columns
			*keep_cols1, *keep_cols2
		).where(col('jaccard') >= threshold)

	def jaccardJoin(self, df1, df2, key1, key2, threshold, keep_cols1, keep_cols2):
		print ("Before filtering: %d pairs in total" % (df1.count() * df2.count())) 

		candDF = self.filtering(df1, df2, key1, key2, keep_cols1, keep_cols2)
		candDF.cache()
		print ("After Filtering: %d pairs left" %(candDF.count()))

		resultDF = self.verification(candDF, threshold, key1, key2, keep_cols1, keep_cols2)
		resultDF.cache()
		print ("After Verification: %d similar pairs" %(resultDF.count()))

		return resultDF