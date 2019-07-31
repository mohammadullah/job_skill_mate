import pandas as pd
import os

def _read_from_directory(directory, func=None, **kwargs):
	reads = []
	for file in os.listdir(directory):
		res =  func(os.path.join(directory, file), **kwargs)
		reads.append(res)
	return reads

def read_json_from_directory(dir, **kwargs):
	reads = _read_from_directory(dir, func=pd.read_json, **kwargs)
	return reads