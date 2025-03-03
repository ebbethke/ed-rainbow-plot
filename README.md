# Rainbow Plots
These scripts will align course description text to ACM/IEEE curriculum topic level maps, prepare anonymous student enrollment data of courses over time, and then generate rainbow barcode plots to show how student topic coverage accumulates over time. The methods used are CS specific, where a word-to-vec style embedding model trained by [Efstathiou et al](https://github.com/vefstathiou/SO_word2vec) was leveraged to establish second-order features used to match ACM topics to course description text. This model is simple and parsimonious, and works fairly well for obvious cases, but struggles with aligning topics beyond the vocabulary of the embedding. This project serves more as a template for processing and preparing these style of curriculum visualizations than for the NLP techniques.
The method used is explained in detail in our [paper](https://doi.org/00.00000/hold.000000).

## Installation
You will need the following packages to run this code on Windows 10/11, in addition to built-in libraries with Python >= 3.10:
1. Numpy (1.26.4)
2. Pandas (2.2.3)
3. NLTK (3.9.1)
4. Gensim (4.3.3)
5. Kneed (0.8.5)
6. Matplotlib (3.9.2)

## Usage
### Data Preparation
You will need the following files/formats in the ```data``` folder:

* ```SO_vectors_200.bin```: gensim KeyedVector model of Stack Overflow text. Retrieve the binary from [here](https://github.com/vefstathiou/SO_word2vec). 
* ```ACM20{X}_Map.tsv``` \[INCLUDED\]: where {X} is either 13 or 23, depending on which map you want to use.
* ```cs_courses.csv``` \[INCLUDED OR YOU PROVIDE\]: a file containing the columns:
  * ```Title```: Code of courses with space, e.g. "CS 101"
  * ```Course```: Full Name of the course, e.g. "Introduction to Computing"
  * ```Description```: Description of the course, e.g. "A broad introduction to the field of computer science..."

Each row should have a different course.

* ```data.csv```\[YOU PROVIDE\], data with the following columns:
  * ```Index```: Integer number, anonymized unique student ID (e.g. 1, 2, ... 437, 438)
  * ```GPA_INITL```: Float, the initial GPA of the student for the first semester of the data pull
  * ```GPA_FINAL```: Float, the final recorded GPA of the student for the last semester of the data pull
  * ```{SEM}_{YR}```(xN): String, comma separated course numbers for enrollment for the N semesters (e.g. Fall_2021, Spring_2022, ..., Spring_20{N}) of the data pull. If no course, leave empty 

Example:

|Index|GPA_INITL|GPA_FINAL|   Fa_2022   |   Sp_2023   |
|-----|---------|---------|-------------|-------------|
|  1  |  3.125  |  3.384  |"133,201,225"|    "376"    |
|  2  |  3.920  |  3.869  |  "486,440"  |             |     
|  3  |  2.673  |  2.841  |"101,125,201"|  "125,205"  |
| ... |   ...   |   ...   |     ...     |     ...     |

Note how student 2 has no course enrollment for **Sp_2023**. We've included some fake example data in the ```data``` folder.

### Generate map
The first step is to generate a mapping between **CS Courses** and **Stack Overflow** terminology, a mapping between **ACM Topics** and **Stack Overflow** terminology, and then compare the **Stack Overflow** neighborhoods to align **CS Courses** to **ACM Topics**. This will be managed by the script ```align_acm_to_courses.py```. See the call signature for help with parameters. This script will output a TSV file which will be used by the plotting script.

### Plot Rainbow Barcodes
Now that you've generated a map between your course descriptions and the ACM topics, you can read in student enrollment data and plot out the topic coverage over time. This is handled by ```plot_barcodes.py```. See the call signature for help with parameters. 
