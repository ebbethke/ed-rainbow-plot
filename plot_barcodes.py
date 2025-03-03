#!/usr/bin/env python3
import argparse
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches


def parse_arguments():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(description='Create barcode visualization of ACM curriculum topics.')
	parser.add_argument('--acm_map', type=str, default='ACM2023_Map.tsv',
						help='Path to the ACM curriculum map TSV file')
	parser.add_argument('--course_map', type=str, default='ACM2023_new_matched_CS_courses.tsv',
						help='Path to the matched courses TSV file')
	parser.add_argument('--student_data', type=str, default='../web/data.csv',
						help='Path to the student course data CSV file')
	parser.add_argument('--output', type=str, 
						help='Output file path (if not specified, plot will be displayed)')
	parser.add_argument('--student_id', type=int, default=None,
						help='Specific student ID to visualize (random student from top quintile if not specified)')
	parser.add_argument('--quintile_idx', type=int, default=None,
						help='Index from quintile to visualize (random student from top quintile if not specified)')
	parser.add_argument('--quintile', type=str, default='1', choices=['1', '2', '3', '4', '5'],
						help='GPA quintile to select from (1=top, 5=bottom)')
	parser.add_argument('--figsize', type=str, default='9,11',
						help='Figure size in inches as width,height')
	parser.add_argument('--latex', action='store_true',
						help='Use LaTeX for text rendering')
	parser.add_argument('--log_level', type=str, 
						choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
						default='INFO', 
						help='Set the logging level')
	
	return parser.parse_args()


def get_colors(num_colors):
	"""Generate a rainbow color palette."""
	# rainbow! 0=red, 0.25=yellow, 0.5=teal, 0.75=blue, 1=purple
	return plt.cm.gist_rainbow_r(np.linspace(0, 1, num_colors))


def load_acm_curriculum_map(file_path):
	"""Load the ACM curriculum map from TSV file."""
	try:
		test_data = pd.read_csv(file_path, header=0, delimiter='\t')
		logging.info(f"Loaded ACM curriculum map from {file_path}")
		
		# Create mapping from topic names to codes
		text_to_code = dict(zip(test_data['NAME'], test_data['CODE']))
		
		# Count occurrences of top-level categories
		all_cats = test_data['CODE'].apply(lambda s: s.split('.')[0]).value_counts()
		all_cats = all_cats.to_dict()
		
		return test_data, text_to_code, all_cats
	except Exception as e:
		logging.error(f"Failed to load ACM map: {e}")
		raise


def load_course_topic_map(file_path, text_to_code):
	"""Load the mapping between courses and ACM topics."""
	try:
		terms = re.compile(":")
		text_map = pd.read_csv(file_path, header=0, delimiter='\t', index_col=False)
		logging.info(f"Loaded course-to-topic map from {file_path}")
		
		# Process the map data
		text_map['ACM_Matches'] = text_map['ACM_Matches'].apply(
			lambda s: [k for k in terms.split(str(s)) if k != ''] if not pd.isna(s) else []
		)
		
		text_map['Match_Codes'] = text_map['ACM_Matches'].apply(
			lambda x: [text_to_code.get(k.strip(), '') for k in x]
		)
		
		text_map['Match_Codes'] = text_map['Match_Codes'].apply(
			lambda x: [j for j in x if j not in ('', ' ') and not pd.isna(j)]
		)
		
		# Clean course numbers by removing "CS " prefix
		text_map['Course'] = text_map['Course'].str.replace("CS ", "")
		
		# Create dictionary mapping course numbers to their ACM topics
		course_to_topic = dict(zip(text_map['Course'], text_map['Match_Codes']))
		
		# Get all unique topic codes
		all_codes = text_map['Match_Codes'].explode().unique()
		all_codes = sorted([code for code in all_codes if isinstance(code, str)])
		all_codes = dict(zip(all_codes, np.arange(len(all_codes))))
		
		return course_to_topic, all_codes
	except Exception as e:
		logging.error(f"Failed to load course-topic map: {e}")
		raise


def load_student_data(file_path, quintile='1', student_id=None, quintile_idx=None):
	"""
	Load student course data and select a student based on criteria.
	
	Args:
		file_path: Path to student data CSV
		quintile: GPA quintile to select from (1=top, 5=bottom)
		student_id: Specific student ID to select (if None, random student from quintile)
	
	Returns:
		Selected student record
	"""
	try:
		# Define columns and data types
		with open(file_path, "r") as record_file:
			header = record_file.readline().strip().split(",")
		
		dtypes = {}
		dtypes[header[0]] = int  # Index/Student ID
		dtypes[header[1]] = float  # Initial GPA
		dtypes[header[2]] = float  # Final GPA
		
		# All remaining columns should be course enrollment data
		for col in header[3:]:
			dtypes[col] = str
		
		# Load data
		coursedata = pd.read_csv(file_path, header=0, dtype=dtypes)
		logging.info(f"Loaded student data from {file_path}")
		
		# Split listings of courses from comma separated strings into lists
		coursedata[header[3:]] = coursedata[header[3:]].apply(lambda s: s.str.split(", "), axis=1)
		
		# Data cleaning: drop incomplete records
		na_lim = 2  # 7/8 semesters with no final GPA OR 6/8 semesters with final GPA
		full_data = coursedata.dropna(thresh=coursedata.shape[1]-na_lim)
		
		# Only keep data from students who have a graduating GPA
		full_data = full_data[full_data[header[2]].notna()]
		full_data = full_data.fillna("").apply(list)
		
		logging.info(f"After cleaning: {len(full_data)} student records with sufficient data")
		
		# Divide students into quintiles by final GPA
		graded_data = full_data.copy()
		graded_data = graded_data.sort_values(header[2])
		quantiles = ['1', '2', '3', '4', '5']
		graded_data['quantile'] = pd.qcut(graded_data[header[2]], 
										  q=np.linspace(0, 1, len(quantiles)+1), 
										  labels=quantiles)
		
		# Select students from specified quintile
		Q = graded_data[graded_data['quantile'] == quintile]
		logging.info(f"Selected {len(Q)} students from quintile {quintile}")
		
		if student_id is not None:
			# Select specific student if ID is provided
			record = full_data[full_data["Index"] == student_id]
			if len(record) == 0:
				logging.warning(f"Student ID {student_id} not found, selecting random student from quintile {quintile}")
				student_idx = np.random.default_rng().integers(0, len(Q), 1)[0]
				record = Q.iloc[student_idx:student_idx+1]
			else:
				logging.info(f"Selected student with ID {student_id}")
		else:
			# Select random student from quintile
			if not quintile_idx:
				quintile_idx = np.random.default_rng().integers(0, len(Q), 1)[0]
			record = Q.iloc[[quintile_idx]]
			logging.info(f"Selected student (index {record['Index'].values[0]}) from quintile {quintile}")
		
		return record, header
	except Exception as e:
		logging.error(f"Failed to load or process student data: {e}")
		raise


def process_student_courses(record, course_semester_cols, course_to_topic):
	"""Process student's courses into topics"""
	topicvals = []
	coursevals = []
	
	# For each semester, get courses and their corresponding topics
	for each in record[course_semester_cols].values[0]:
		if len(each) < 1:
			coursevals.append([""])
		else:
			coursevals.append(each)
		topicvals.append([course_to_topic.get(str(k).strip(), '') for k in each])
	
	# Flatten topics for analysis
	flat_topics = [item for course in [course for semester in topicvals for course in semester] for item in course]
	logging.debug(f"All topics: {flat_topics}")
	unique_topics, counts = np.unique(flat_topics, return_counts=True)
	logging.debug(f"Unique topics: {unique_topics}")
	logging.debug(f"Topic counts: {counts}")
	
	# Add a total row
	coursevals.append(["TOTAL"])
	
	return topicvals, coursevals


def create_barcode_plot(topicvals, coursevals, all_codes, test_data, all_cats, 
						 student_info=None, figsize=(19, 11), latex=False, 
						 output_file=None):
	"""Create the barcode visualization"""
	# Set LaTeX option if requested
	if latex:
		plt.rc('text', usetex=True)
	else:
		plt.rc('text', usetex=False)
	
	# Plotting constants
	bar_width = 3	  # line width of each bar in barcode
	bar_height = 40	# line length of barcode
	bpad = 1.5		 # padding between each bar
	spad = int(bar_height//2)  # spacing between each semester
	nbars = len(all_codes)
	hscale = 10
	
	# Calculate x positions for each code
	code_pos_x = np.transpose([np.arange(0, nbars)*bar_width, 
							   np.arange(1, nbars+1)*bar_width]) + \
				np.arange(1*bpad, (nbars+1)*bpad, bpad).reshape(-1, 1)
	code_pos_x = code_pos_x*hscale/code_pos_x.max()
	
	# Prepare colors and legend
	cat_names = np.unique([s.split('.')[0] for s in all_codes.keys()])
	legend_names = [test_data[test_data['CODE'] == cat]['NAME'].values[0] for cat in cat_names]
	legend_map = dict(zip(legend_names, get_colors(len(all_cats.keys()))))
	colrs = get_colors(len(code_pos_x))
	color_map = dict(zip(code_pos_x[:, 0].squeeze(), colrs.squeeze()))
	
	# Create line collections for each semester
	lines = []
	total = set()
	for ht, semester in enumerate(topicvals):
		y0 = ((ht+1)*spad) + (bar_height * ht)
		segs = []
		colors = []
		for course in semester:
			xs = [code_pos_x[all_codes[t]][0] for t in course if t in all_codes]
			segs.extend([[(x, y0), (x, y0+bar_height)] for x in xs])
			colors.extend([color_map[x] for x in xs])
			total.update(xs)
		if len(segs) > 0:
			lines.append(LineCollection(segs, linewidths=bar_width, colors=colors))
		else:
			lines.append(LineCollection([], linewidths=0))
	
	# Add a line for the total
	yf = ((len(lines)+2)*spad) + (bar_height * (len(lines)+2))
	lines.append(LineCollection([[(x, yf), (x, yf+bar_height)] for x in total], 
							   linewidths=bar_width, colors=[color_map[x] for x in total]))
	
	# Create patches for legend
	legends = [mpatches.Patch(facecolor=c, edgecolor='None', label=lab) 
			  for lab, c in legend_map.items()]
	
	# Create the plot
	fig, ax = plt.subplots(figsize=figsize)
	
	# Add the line collections and annotations
	for sem, collection in enumerate(lines):
		ax.add_collection(collection)
		if len(coursevals[sem][0]) > 0:
			if sem < len(lines)-1:
				text = f'Semester {sem+1}: ' + ', '.join(coursevals[sem])
			else:
				logging.debug("Inserting total label...")
				text = ' '.join(coursevals[sem])
		else:
			text = ' '
		y = ((sem+1)*spad) + (bar_height * sem) + spad//2
		x = 1.2*code_pos_x.max()
		ax.annotate(text,
					xy=(x, y), xycoords='data',
					xytext=(1.5, 1.5), textcoords='offset points',
					fontsize=24,
					annotation_clip=False)
	
	# Add title if student info is provided
	if student_info is not None:
		student_id = student_info['Index'].values[0]
		print(student_info)
		gpa = round(student_info.iloc[0,2], 3)
		ax.set_title(f"Student {student_id} topic map: GPA: {gpa}")
	
	# Set plot limits and remove ticks
	ax.set_xlim((-bpad, bpad+code_pos_x.max()))
	ax.set_ylim((0, yf + 5*bar_height/4))
	plt.xticks([])
	plt.yticks([])
	
	# Add legend
	ax = plt.gca()
	ax.legend(handles=legends, loc="upper left", bbox_to_anchor=(1.55, 1.0), fontsize=19)
	
	# Adjust layout
	plt.subplots_adjust(right=0.45, left=0.05, top=0.95, bottom=0.1)
	
	# Save or show the plot
	if output_file:
		plt.savefig(output_file)
		logging.info(f"Saved plot to {output_file}")
	else:
		plt.show()


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
	
	logging.info("Starting ACM barcode visualization")
	
	# Parse figure size
	figsize = tuple(map(float, args.figsize.split(',')))
	
	# Load ACM curriculum map
	test_data, text_to_code, all_cats = load_acm_curriculum_map(args.acm_map)
	
	# Load course to topic mapping
	course_to_topic, all_codes = load_course_topic_map(args.course_map, text_to_code)
	
	# Load student data
	student_record, cols = load_student_data(
		args.student_data, 
		quintile=args.quintile,
		student_id=args.student_id,
		quintile_idx=args.quintile_idx
	)
	
	# Process student courses
	topicvals, coursevals = process_student_courses(
		student_record, 
		cols[3:], 
		course_to_topic
	)
	
	# Create visualization
	create_barcode_plot(
		topicvals, 
		coursevals, 
		all_codes, 
		test_data, 
		all_cats,
		student_info=student_record,
		figsize=figsize,
		latex=args.latex,
		output_file=args.output
	)
	
	logging.info("Visualization complete")


if __name__ == "__main__":
	main()