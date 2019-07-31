import pandas as pd
import sys
import re
import nltk
from lib import cleaners as c

def main(input, output):
	df = pd.read_csv(input)
	short_title_format = re.compile(r'\([\w ]+\)')

	has_short_title = df['Short Title'].notna()
	df.loc[has_short_title, 'Alternate Title'] = df['Alternate Title'].str.replace(short_title_format, '')

	short_titles_df = df.loc[df['Short Title'].notna(), ['O*NET-SOC Code', 'Title', 'Short Title']]
	short_titles_df.rename({'Short Title': 'Alternate Title'}, axis=1, inplace=True)

	# append the short titles that were made as alternate title
	df = df.append(short_titles_df, sort=False)

	# tokenize titles
	df.loc[:, 'Alternate Title'] = df['Alternate Title'].str.lower().str.strip()\
		.apply(c.sentence_tokens_without_punctuations)
	
	df.to_json(output, orient='records', lines=True)

if __name__ == "__main__":
	input = sys.argv[1]
	output = sys.argv[2]

	main(input, output)