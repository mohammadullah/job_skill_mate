"""Preparing corpus for Doc2Vec with Gensim

There were difficulties to run Gensim by loading the data directly as a corpus, so instead here
we prepared a TaggedLineDocument for Gensim.

This requires the preprocessed files from either ER or task statements data which you can find here: https://drive.google.com/drive/folders/1Doh6YKgAri9OV8xhEVTKJzNQ_-Kl5S03
The files you need are:
- er_resume_preprocessed and onet_occupation_data.csv (for ER)
	-> Note that cleaning dataset cleaning and tokenization is/was done with NLTK stopwords and punctuations.
- tokenized_job_details_for_doc2vec and tokenized_onet_occupation_task_statements.json (for task statements)
	-> Note that tokenization is/was done with spacy without stopwords and punctuations removal

Usage:
	python -m src.prepare_corpus_for_doc2vec <directory path to er_resume_preprocessed> <onet file>

	iPython is suggested for speed
"""

from lib.load_data import read_json_from_directory
from lib.cleaners import sentence_tokens_without_punctuations
import pandas as pd
import numpy as np
import sys
import os

def corpus_from_job_descriptions(resume_dir, onet_data):
	occupation_data = pd.read_json(onet_data, orient='records', lines=True)
	resumes = read_json_from_directory(resume_dir)

	corpus = occupation_data['task']
	id_mapping = occupation_data['onet_job_id']

	print(resumes[0].head(10))

	job_details = [df['tokenized_details'] for df in resumes]
	id_maps = [df['id'] for df in resumes]
	
	corpus = corpus.append(job_details).reset_index(drop=True)
	corpus = corpus.str.join(' ')

	id_mapping = id_mapping.append(id_maps).reset_index(drop=True)
	
	if not os.path.isdir('doc2vec_data'):
		os.mkdir('doc2vec_data')
	np.savetxt('doc2vec_data/corpus_task_statements.cor', corpus.values, fmt='%s')
	id_mapping.to_csv('doc2vec_data/corpus_task_statements_id_mapping.csv')

# expected resume data to be tokenized_job_details_for_doc2vec
# expected onet data to either be:
# - tokenized_onet_occupation_data_filtered.json
# - tokenized_onet_occupation_task_statements.json
def corpus_from_onet_data(resume_dir, onet_data, onet_token_key, filename_suffix):
	# this is building corpus directly
	occupation_data = pd.read_json(onet_data, orient='records', lines=True)
	resumes = read_json_from_directory(resume_dir, lines=True, compression='gzip')

	corpus = occupation_data[onet_token_key]
	id_mapping = occupation_data['onet_job_id']

	print(resumes[0].head(10))

	job_details = [df['tokenized_details'] for df in resumes]
	id_maps = [df['id'] for df in resumes]
	
	corpus = corpus.append(job_details).reset_index(drop=True)
	corpus = corpus.str.join(' ')

	id_mapping = id_mapping.append(id_maps).reset_index(drop=True)
	
	if not os.path.isdir('doc2vec_data'):
		os.mkdir('doc2vec_data')
	np.savetxt(f'doc2vec_data/corpus_{filename_suffix}.cor', corpus.values, fmt='%s')
	id_mapping.to_csv(f'doc2vec_data/corpus_{filename_suffix}_id_mapping.csv', header=False)

def corpus_from_er(resume_dir, onet_data):
	occupation_data = pd.read_csv(onet_data)
	resumes = read_json_from_directory(resume_dir, lines=True, compression='gzip')

	for idx, df in enumerate(resumes):
		print("Number of entries in preprocessed resumes #%s is %d" % (idx, len(df)))
		df = df[df['job_details'].str.len() > 0]
		print("Number of entries after empty job details is %d" % len(df))
		resumes[idx] = df

	corpus = occupation_data['Description'].apply(lambda s: sentence_tokens_without_punctuations(s.lower()))
	id_mapping = occupation_data['O*NET-SOC Code'].rename('id')

	job_details = [df['job_details'] for df in resumes]
	id_maps = [df['id'] for df in resumes]
	
	corpus = corpus.append(job_details).reset_index(drop=True)
	corpus = corpus.str.join(' ')

	id_mapping = id_mapping.append(id_maps).reset_index(drop=True)
	
	if not os.path.isdir('doc2vec_data'):
		os.mkdir('doc2vec_data')
	np.savetxt('doc2vec_data/corpus_continuing_er.cor', corpus.values, fmt='%s')
	id_mapping.to_csv('doc2vec_data/corpus_er_id_mapping.csv')

if __name__ == "__main__":
	# get the resume directory and supporting onet data
	resume_dir = sys.argv[1]
	onet_data = sys.argv[2]
	type = sys.argv[3]

	if type == "er":
		# this one were files from ER preparation
		corpus_from_er(resume_dir, onet_data)
	elif type == "task_statements":
		# data from task statement information (or treating them as task statements)
		corpus_from_onet_data(resume_dir, onet_data, 'task', 'task_statements')
	elif type == "job_descriptions":
		corpus_from_onet_data(resume_dir, onet_data, 'desc', 'job_descriptions')
	else:
		sys.stderr.write("Unrecognizable type\n")
		sys.exit(1)