#!/usr/bin/env python3
import argparse
import re
import string
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from gensim.models.keyedvectors import KeyedVectors
import gensim.downloader as gendl
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_tags, strip_punctuation
from kneed import KneeLocator


def parse_arguments():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(description='Map course descriptions to ACM curriculum topics.')
	parser.add_argument('--acm_map', type=str, required=True,
						help='Path to the ACM curriculum map TSV file')
	parser.add_argument('--courses', type=str, default="../cs_courses.csv",
						help='Path to the courses CSV file')
	parser.add_argument('--word_vectors', type=str, default="SO_vectors_200.bin",
						help='Path to the word vectors binary file')
	parser.add_argument('--output', type=str, 
						help='Output file path (default: ACM{year}_matched_CS_courses.tsv)')
	parser.add_argument('--common_words', type=str, default="Brown_7000_common.txt",
						help='Path to common words file')
	parser.add_argument('--num_common_words', type=int, default=5000,
						help='Number of most common words to filter from text')
	parser.add_argument('--num_matches', type=int, default=0,
						help='Number of top matches to associate with course text')
	parser.add_argument('--year', type=str, default="2023_FULL",
						help='Year identifier for output filename')
	parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
						default='INFO', help='Set the logging level')
	
	return parser.parse_args()


def load_common_words(common_words_path, n_most_common=5000):
	"""Load or generate a list of common words to filter out."""
	tokenizer = RegexpTokenizer(r'\w+|\.')
	
	try:
		commons = np.loadtxt(common_words_path, delimiter=",", dtype="str").tolist()
		logging.debug(f"Loaded common words from {common_words_path}")
	except IOError:
		logging.info(f"Generating common words file at {common_words_path}")
		words = nltk.corpus.brown.sents()
		names = nltk.corpus.names.words()

		clean = tokenizer.tokenize(" ".join([i for j in words for i in j]))
		freqs = FreqDist([c for c in clean if c not in names])

		commons = freqs.most_common(n_most_common)
		commons = [i[0].lower() for i in commons]
		
		np.savetxt(common_words_path, commons, fmt="%s", delimiter=",")
	
	return commons, tokenizer


def remove_common_words(text, commons, tokenizer):
	"""Remove common words from text."""
	wlist = tokenizer.tokenize(text)
	wlist = [w for w in wlist if np.all([not i.isdigit() for i in w])]
	return " ".join([w for w in wlist if w.strip() not in commons and len(w.strip()) > 0])


def detect_acm_map_version(curriculum_topic_map):
	"""Detect if the map is 2013 or 2023 version based on column names."""
	if 'CSCore' in curriculum_topic_map.columns and 'KACore' in curriculum_topic_map.columns:
		return '2023'
	elif 'CONTEXT' in curriculum_topic_map.columns:
		return '2013'
	else:
		raise ValueError("Unknown ACM map format. Expected either 2013 or 2023 format.")


def preprocess_acm_topic(search, map_version):
	"""Extract topic description based on map version and apply filters."""
	CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, remove_stopwords]
	
	if map_version == '2023':
		# Use preference order: CSCore -> KACore -> ILOs -> NonCore
		if not pd.isna(search['CSCore']):
			topics = preprocess_string(search['CSCore'], CUSTOM_FILTERS)
		elif not pd.isna(search['KACore']):
			topics = preprocess_string(search['KACore'], CUSTOM_FILTERS)
		elif not pd.isna(search['ILOs']):
			topics = preprocess_string(search['ILOs'], CUSTOM_FILTERS)
		else:
			topics = preprocess_string(search['NonCore'], CUSTOM_FILTERS)
	else:  # 2013 map
		topics = preprocess_string(search['CONTEXT'], CUSTOM_FILTERS)
	
	return topics


def preprocess_course_description(description):
	"""Clean and preprocess course description."""
	# Define regex patterns
	prereq = re.compile("[P|p]rerequisite[s]?: .*?\.")
	hours = re.compile("([0-9] or )?[0-9] (under)?graduate hours")
	concurrent = re.compile("[S|s]ame as [A-Z]{2,4} [0-9]{3}.*?\.")
	extra_courses = re.compile("([a-z]{2,4} [1-9]{3} )+")
	
	CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, remove_stopwords]
	
	# Working from the end to the middle of the course descr fmt
	description = prereq.sub("", description)  # exclude text following prerequisite listing
	description = hours.sub("", description)   # exclude text about hours credit
	description = concurrent.sub("", description)  # remove concurrent info
	
	# preprocess_string
	description = " ".join(preprocess_string(description, CUSTOM_FILTERS))
	
	# remove extra course listings
	description = extra_courses.sub("", description)
	
	# split text into list of sentences
	description = description.split(". ")
	# remove blanks
	description = [s for s in description if s != '']
	
	return description


def main():
	args = parse_arguments()
	
	# Configure logging
	numeric_level = getattr(logging, args.log_level.upper(), None)
	if not isinstance(numeric_level, int):
		raise ValueError(f'Invalid log level: {args.log_level}')
	
	logging.basicConfig(
		level=numeric_level,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S'
	)
	logging.getLogger('gensim').setLevel(logging.WARNING)
	logging.info("Starting ACM analysis")
	
	# Set output file name if not provided
	if not args.output:
		args.output = f"ACM{args.year}_matched_CS_courses.tsv"
	
	# Load common words
	commons, tokenizer = load_common_words(args.common_words, args.num_common_words)
	
	# Define regex pattern for detecting ACM topic hierarchy levels
	level = re.compile('\.')
	parens = re.compile('\(|\)')
	
	# Load ACM text map
	curriculum_topic_map = pd.read_csv(args.acm_map, header=0, delimiter='\t')
	
	# Detect map version
	map_version = detect_acm_map_version(curriculum_topic_map)
	logging.info(f"Detected ACM map version: {map_version}")
	
	# Add level information if CODE column exists
	if 'CODE' in curriculum_topic_map.columns:
		curriculum_topic_map['Level'] = curriculum_topic_map['CODE'].apply(lambda s: len(level.findall(s)))
		# Target Level 1
		curriculum_topic_map = curriculum_topic_map[curriculum_topic_map['Level'] == 1]
	
	# Clean SEARCH column if it exists
	if 'SEARCH' in curriculum_topic_map.columns:
		curriculum_topic_map['SEARCH'] = curriculum_topic_map['SEARCH'].apply(lambda x: parens.sub('', x))
	
	# Trim any excess columns
	curriculum_topic_map.dropna(axis=1, how='all', inplace=True)
	
	# Load course list
	courses = pd.read_csv(args.courses, header=0, delimiter=',')
	
	# Set up targets (ACM topics)
	targets = curriculum_topic_map.set_index('NAME')
	
	# Load word vectors
	logging.info(f"Loading word vectors from {args.word_vectors}")
	word_vect = KeyedVectors.load_word2vec_format(args.word_vectors, binary=True)
	
	CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, remove_stopwords]
	
	# Map ACM topics to Stack Overflow (SO) terms
	logging.info("Mapping ACM topics to SO terms...")
	acm2so = {}
	for category, search in targets.iterrows():
		logging.debug(f"Processing {category}")
		
		# Process according to detected map version
		topics = preprocess_acm_topic(search, map_version)
		
		# Remove common words
		description = [remove_common_words(st, commons, tokenizer) for st in topics]
		# Strip out any remaining blanks
		description = [w for w in description if len(w.strip()) > 0]
		# Eliminate non-vocab words
		description = [w for w in description if w in word_vect]
		
		logging.debug(f"Description terms: {description}")
		logging.debug(f"Original topics: {len(topics)}, Filtered description: {len(description)}")
		
		if not description:
			logging.warning(f"No valid description terms for {category}")
			continue
		
		results = word_vect.most_similar_cosmul(positive=description, topn=20)
		# Save all results to dictionary such that ACM topics -> list of top matches to SO
		most, sim = zip(*results)
		
		acm2so[category] = list(most)
		
		logging.debug(f"{search.get('CODE', '')}: {category} had most similar: {most[:3]} (similarity: {sim[:3]})")
	
	# Write output header
	logging.info(f"Writing results to {args.output}")
	with open(args.output, "w") as outf:
		outf.write("Course\tCoursename\tDescription\tACM_Matches\tACM_Match_Score\n")
	
	# Process each course
	all_matches = {}
	uiuc2SO = {}
	for row in courses.itertuples(index=False, name="Course"):
		# Process course info
		coursetitle = row.Title  # CS 101, CS 405, etc.
		coursename = row.Course  # Name of the course
		source = row.Description  # Course description
		
		# Preprocess the course description
		source = preprocess_course_description(source)
		
		# Remove common words
		source = [remove_common_words(st, commons, tokenizer) for st in source]
		logging.debug(f"Source text after common removed for {coursetitle}: {source}")
		
		# Skip if no valid description remains
		if all(s == '' or s.isspace() for s in source):
			# Write empty results to file
			with open(args.output, "a") as outf:
				outf.write(f"{coursetitle}\t{coursename}\t \t \t \n")
			logging.warning(f"No valid source description for {coursetitle}")
			continue
		
		# Initialize for this course
		all_matches[coursetitle] = []
		uiuc2SO[coursetitle] = []
		
		# Process each sentence in the course description
		for sentence in source:
			if len(sentence) < 1:
				continue
			
			# Map Course Descriptions to SO Embedding
			description = preprocess_string(sentence, CUSTOM_FILTERS)
			# Remove common words
			description = [remove_common_words(st, commons, tokenizer) for st in description]
			# Eliminate non-vocab words
			description = [w for w in description if w in word_vect]
			
			if not description:
				continue
				
			try:
				SO_results = word_vect.most_similar_cosmul(positive=description, topn=20)
				# Unpack matches and scores
				results, scores = zip(*SO_results)
				
				# Save out to dict
				uiuc2SO[coursetitle].extend(list(results))
				
				# Compare description sentence SO matches to ACM topic SO matches
				for k, v in acm2so.items():
					distance = word_vect.wmdistance(results, v)
					all_matches[coursetitle].append((k, distance))
			except KeyError as e:
				logging.warning(f"KeyError for {coursetitle}: {e}")
				continue
		
		# If we collected any matches
		if all_matches[coursetitle]:
			# Sort matches by distance (lower is better)
			all_matches[coursetitle] = sorted(
				dict(sorted(all_matches[coursetitle], key=lambda x: x[1])).items(), 
				key=lambda x: x[1]
			)
			
			# Find the knee point to determine cutoff
			alltopics, allscores = zip(*all_matches[coursetitle])
			kn = KneeLocator(
				np.arange(len(allscores)), 
				allscores, 
				S=1.0, 
				curve="concave", 
				direction="increasing"
			)
			knee = kn.knee
			if knee == 0 or knee is None:
				
				knee = np.sum(np.array(list(allscores))<1.0)
			else:
				knee = int(knee + 3)
			
			logging.debug(f"Knee at {knee}, scores: {allscores[:knee]}")
			topn = list(alltopics[:knee])
			topscores = list(allscores[:knee])
			all_matches[coursetitle] = list(zip(topn, topscores))
			logging.debug(f"{coursetitle} best matches {all_matches[coursetitle][:5]}, knee at {knee}:{allscores[-1]}")
			
			# Write results to file
			with open(args.output, "a") as outf:
				outf.write(coursetitle)
				outf.write('\t')
				outf.write(coursename)
				outf.write('\t')
				outf.write(', '.join(source).replace(':', ';'))
				outf.write('\t')
				outf.write(':'.join([''.join(k.replace(':', ';')) for k in topn]))
				outf.write('\t')
				outf.write(np.array(topscores).__str__().replace('\n', ''))
				outf.write("\n")
		else:
			# Write empty results if no matches found
			with open(args.output, "a") as outf:
				outf.write(f"{coursetitle}\t{coursename}\t{', '.join(source).replace(':', ';')}\t\t\n")
	
	logging.info(f"Analysis complete. Results written to {args.output}")


if __name__ == "__main__":
	main()