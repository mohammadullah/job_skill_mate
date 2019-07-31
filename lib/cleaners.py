import re
import pandas as pd
import numpy as np
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

_bullets = re.compile(r'\s*[\u2022|\u25cf|\u27a2|-|*]', flags=re.MULTILINE)
_punctuation_list = list(punctuation)
_punctuation_list.remove('+') # remove because of things like C++ (probably + is not in normal text)
_eng_stopwords = stopwords.words('english')
_non_dates = ['Present']

def expand_to_multi_rows(df, col, expanded_col_name):
	"""Expand column with lists to multiple rows
	
	Given a DataFrame, expand column with lists to multiple rows (causes duplication of data)
	
	:param df: A DataFrame
	:type df: pandas.DataFrame
	:param col: column name
	:type col: str
	:param expanded_col_name: name of column for expanded data
	:type expanded_col_name: str
	"""
	expanded = pd.DataFrame(df[col].tolist()).stack().reset_index(level=1, drop=True)
	return pd.merge(
		df,
		pd.DataFrame(expanded, columns=[expanded_col_name]),
		left_index=True,
		right_index=True,
		how='inner'
	).reset_index(level=0,drop=True)

def create_new_df_dict_col(df, col):
	"""Create a new DataFrame from column with dictionaries
	
	Given a DataFrame that holds dictionary values at col, create a new DataFrame
	with the values of the dictionaries of that column
	
	:param df: A DataFrame
	:type df: pandas.DataFrame
	:param col: Column name
	:type col: str
	:return: A new DataFrame with the keys of the dictionary as columns
	:rtype: pandas.DataFrame
	"""
	return df[col].apply(pd.Series)

def _time_duration_to_years(df, start_col, end_col):
	"""Compute delta duration in years between end date and start date
	
	From a DataFrame with start and end date columns, get the duration in years
	between them (columns are date-like strings e.g January 2014).
	
	:param df: A DataFrame with start_col and end_col
	:type df: pandas.DataFrame
	:param start_col: Start date column
	:type start_col: str, date-like
	:param end_col: End date column
	:type end_col: str, date-like
	:return: Series of deltas in years
	:rtype: pandas.Series
	"""

	where_present_string = df[start_col].isin(_non_dates) |\
		df[end_col].isin(_non_dates)
	df.loc[where_present_string,(start_col, end_col)] = np.nan

	df.loc[:,start_col] = pd.to_datetime(df[start_col])
	df.loc[:,end_col] = pd.to_datetime(df[end_col])

	delta = df[end_col] - df[start_col]
	return delta.apply(lambda x: x / np.timedelta64(1, 'Y'))

def simplify_education_information_(df, start_col, end_col):
	"""[MODIFIES] Simplifies education from DataFrame to some simple data values
	
	Given DataFrame extracts degree and the duration of the degree (in years)
	Duration of the degree may be NaN in these conditons:
	1. Beginning date is unclear (not a date)
	2. Ending date is unclear (not a date)

	Additionally it will also replace the start_col and end_col values to be
	of type datetime64 or NaT

	See: http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html#pandas.to_datetime
	
	:param df: Pandas DataFrame
	:type df: pandas.DataFrame
	:param degree_col: Column where it holds the start of the degree
	 start_col: str
	:param degree_col: Column where it holds the end of the degree
	:type end_col: str
	:return: Modified DataFrame
	:rtype: pandas.DataFrame
	"""
	df['degree_year_time'] = _time_duration_to_years(df, start_col, end_col)
	return df

def simplify_skills_information_(df, exp_col, extract_pat=r'(\d) year'):
	"""[MODIFIES] Simplifies skill information from DataFrame to simple values 
	
	Given DataFrame with an experience column of type str, exp_col, extract the experience
	(some numerical value) according to a regex extraction pattern. Replaces the experience information
	in exp_col with the numerical value

	:param df: Pandas DataFrame
	:type df: pandas.DataFrame
	:param exp_col: Column for the experience
	:type exp_col: str
	:param extract_pat: Regex str, defaults to r'(%d) year'
	:param extract_pat: str, optional
	"""
	df[exp_col] = df[exp_col].str.extract(extract_pat)
	return df

def simplify_jobs_information_(
	df,
	start_col,
	end_col,
	details_col,
	clean_punctuation=True,
	clean_stopwords=True
):
	"""[MODIFIES] Simplifies jobs information
	
	Modifies given DataFrame to include simiplified information about
	job duration and details	
	
	:param df: DataFrame with job information
	:type df: pandas.DataFrame
	:param start_col: Job start column
	:type start_col: str, date-like
	:param end_col: Job end column
	:type end_col: str, date-like
	:param end_col: Job details column
	:type end_col: str, date-like
	:param clean_punctuation: Whether to clean punctuation or not, defaults to True
	:param clean_punctuation: bool, optional
	:param clean_stopwords: Whether to remove stopwords or not, defaults to True
	:param clean_stopwords: bool, optional
	:return: Modified DataFrame with job_duration and job_detials column
	:rtype: pandas.DataFrame
	"""
	df['job_duration'] = _time_duration_to_years(df, start_col, end_col)
	df = tokenize_array_of_sentences_(
		df,
		details_col,
		clean_punctuation=clean_punctuation,
		clean_stopwords=clean_stopwords
	)
	return df.rename({details_col: 'job_details'}, axis=1, inplace=True)

def _remove_punctuations_and_stopwords(clean_punctuation):
	def _internal(s):
		# split sentences further to individual sentences, then word tokenize
		no_bullets_sentences = [re.sub(_bullets, '', sent) for sent in sent_tokenize(s)]
		keepwords = [w for s in no_bullets_sentences for w in word_tokenize(s) if w not in _eng_stopwords]
		if clean_punctuation:
			keepwords = [word for word in keepwords if word not in _punctuation_list]
		return keepwords

	return _internal

sentence_tokens_without_punctuations = _remove_punctuations_and_stopwords(True)
sentence_tokens_with_punctuations = _remove_punctuations_and_stopwords(False)

def tokenize_array_of_sentences_(df, *cols, clean_punctuation=True, clean_stopwords=True):
	"""[MODIFIES] Tokenizing array of sentences
	
	Given a column of array of sentences, tokenize each of them to words
	making an array of words over all of the sentences (duplicates possible). By default
	punctuations and stopwords are removed
	
	:param df: DataFrame
	:type df: pandas.DataFrame
	:param col: column name
	:type col: str
	:param clean_punctuation: Whether or not to remove punctuation, defaults to True
	:param clean_punctuation: bool, optional
	:param clean_stopwords: Whether or not to remove stopwords (nltk english), defaults to True
	:param clean_stopwords: bool, optional
	:return: Series with tokens from sentences
	:rtype: pandas.Series
	"""
	cols = list(cols)
	sentences = df[cols]
	# in case any is none
	sentences = sentences.dropna()
	for col in cols:
		sentences.loc[:,col] = sentences[col].str.join(' ')
		sentences.loc[:,col] = sentences[col].str.lower()
		if clean_punctuation:
			sentences[col] = sentences[col].apply(sentence_tokens_without_punctuations)
		else:
			sentences[col] = sentences[col].apply(sentence_tokens_with_punctuations)

	df[cols] = sentences
	return df