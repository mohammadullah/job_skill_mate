from lib import cleaners as c
import pandas as pd
import numpy as np
import sys
from functools import reduce

def merge_on_index(df1, df2):
	return pd.merge(df1, df2, left_index=True, right_index=True, how='inner')

def clean_school_data(df):
	edu_df = c.expand_to_multi_rows(df, 'schools', 'school')

	degree_df = c.create_new_df_dict_col(edu_df, 'school')
	degree_df = degree_df.replace('', np.nan)
	degree_df = degree_df.dropna(subset=['degree'])
	degree_df.head(10)

	c.simplify_education_information_(degree_df, 'start_date', 'end_date')
	simplified_degree_df = degree_df[['degree', 'degree_year_time']]

	edu_ids = edu_df[['id']]
	expanded_resume_degree = merge_on_index(edu_ids, simplified_degree_df)
	expanded_resume_degree.head(10)

	grouped_and_collect_by_id = expanded_resume_degree.groupby('id').agg(list)
	return grouped_and_collect_by_id.reset_index()

def clean_skills_data(df):
	skills_df = c.expand_to_multi_rows(df, 'skills', 'skill_dict')

	expanded_skill_df = c.create_new_df_dict_col(skills_df, 'skill_dict')
	expanded_skill_df = expanded_skill_df[['skill', 'experience']]

	c.simplify_skills_information_(expanded_skill_df, 'experience')

	# skills
	skills_ids = skills_df[['id']]
	expanded_resume_skills = merge_on_index(skills_ids, expanded_skill_df)

	grouped_and_collect_by_id = expanded_resume_skills.groupby('id').agg(list)
	return grouped_and_collect_by_id.reset_index()

def clean_jobs_data(df):
	# jobs
	jobs_df = c.expand_to_multi_rows(df, 'jobs', 'job')

	expanded_job_df = c.create_new_df_dict_col(jobs_df, 'job')
	expanded_job_df = expanded_job_df[['title', 'start_date', 'end_date', 'details']]

	c.simplify_jobs_information_(expanded_job_df, 'start_date', 'end_date', 'details')
	simplified_job_df = expanded_job_df[['title', 'job_duration', 'job_details']]

	jobs_ids = jobs_df[['id']]
	expanded_resume_jobs = merge_on_index(jobs_ids, simplified_job_df)
	
	grouped_and_collect_by_id = expanded_resume_jobs.groupby('id').agg(list)
	return grouped_and_collect_by_id.reset_index()

def merge_all_clean_data(*args):
	return reduce(lambda df1, df2: pd.merge(df1, df2, left_on='id', right_on='id', how='outer', copy=False), args)	

def main(input, output):
	dataset = pd.read_json(input, lines=True)
	print(dataset.head(10))
	total_data = len(dataset)
	dataset = dataset.drop_duplicates(subset=['id'])
	total_non_duplicated_data = len(dataset)

	cleaned_education_data = clean_school_data(dataset[['id', 'schools']])
	print(cleaned_education_data.head(10))

	cleaned_skills_data = clean_skills_data(dataset[['id','skills']])
	print(cleaned_skills_data.head(10))

	cleaned_jobs_data = clean_jobs_data(dataset[['id', 'jobs']])
	print(cleaned_jobs_data.head(10))

	cleaned_summary_and_additional = c.tokenize_array_of_sentences_(dataset[['id', 'summary', 'additional']], 'summary', 'additional')
	print(cleaned_summary_and_additional.head(10))

	clean_data = merge_all_clean_data(
		dataset[['id']],
		cleaned_education_data,
		cleaned_summary_and_additional,
		cleaned_skills_data,
		cleaned_jobs_data
	)

	# for everything that is missing, set it to empty list
	def turn_nan_to_empty_list(val):
		return [] if pd.isna(val) is True else val

	columns_without_id = clean_data.columns.drop('id')
	clean_data.loc[:, columns_without_id] = clean_data[columns_without_id].apply(lambda s: s.apply(turn_nan_to_empty_list))
	print(clean_data.head(10))

	return {
		'total': total_data,
		'non_duplicated': total_non_duplicated_data,
		'cleaned_data': clean_data
	}

if __name__ == "__main__":
	filename = sys.argv[1]
	output = sys.argv[2]

	results = main(filename, output)
	print('Total data initially: %d' % results['total'])
	print('Total non-duplicated data: %d' % results['non_duplicated'])
	print('Total cleaned data: %d' % len(results['cleaned_data']))
	print(results['cleaned_data'].head(100))

	results['cleaned_data'].to_json(output, orient='records', lines=True)