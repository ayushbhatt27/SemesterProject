# Question Answering Model
This is a question answering project for squad 1.1 dataset based on a simplified model of the Bidirectional Attention Flow for Machine Comprehension paper referenced below.

## Requirements :-
Install the following packages using pip install:-
- python 2.7
- colorama 0.3.9
- nltk 3.2.5
- numpy 1.14.0
- six 1.11.0
- tensorflow-gpu 1.4.1 
- tensorflow-tensorboard 0.4.0
- tqdm 4.19.5
- download glove vectors from [here](https://nlp.stanford.edu/projects/glove/).Extract the files and paste all the files in data folder.

### About the Dataset :-
The dataset used is SQuAD 1.1 and has the following.
###### For training data:
- 442 different topics
- 18896 paragraphs or contexts
- 87599 total question answer pairs

###### For testing data:
- 48 different topics
- 2067 paragraphs or contexts
- 10570 total question answer pairs

Answers to questions are contiguous and can range anywhere from the start to end of paragraphs.
Around 75% of the answers have a word length of less than or equal to 4. There are multiple articles or contexts on several different topics and question answer pairs on these contexts. The glove vectors used were of dimension 100.

### How to run the code: 
First install all the dependecies as mentioned in the requirements and run squad_preprocess.py to generate input files.
- To train the model (from parent directory):- 
```sh
$ cd code
$ python main.py
```

- To compute the accuracy (only after training the model):-
```sh
$ cd code
$ python accuracy_check.py
```
### Reference Paper: 
The Bidirectional Attention Flow for Machine Comprehension paper can be accessed [here](https://arxiv.org/pdf/1611.01603.pdf).
### Contact: 
- Ayush Bhatt (iit2015080@iiita.ac.in)
- Soumik Chatterjee (soumik.mufc@gmail.com)
- Pranjal Sanjanwala (iit2015088@iiita.ac.in)
