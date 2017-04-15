# Machine Learning Engineer Nanodegree
## Capstone Project

Rafael Pettersen Caixeta

April 15, 2017

## I. Definition

### Project Overview

Natural language sentence matching (NLSM) is the task of comparing two sentences and identifying the relationship between them. The task is critical for many applications in natural language processing (NLP), such as information retrieval [1], question answering [2] and paraphrase identification [3].

Natural language sentences have complicated structures, both sequential and hierarchical, that are essential  for  understanding them. Sentence similarity ranges between 0% (no relationship at all) and 100% (sentences are semanticaly identical). Two similar sentences do not need to share the content, neither verbatim nor expressed in other words. They may just cover the same topic or merely be written in the same language.

Recently, deep neural network based models have been applied in this area and achieved some important progress. A lot of deep models follow the paradigm to first represent the whole sentence to a single distributed representation, and then compute similarities between the two vectors to output the matching score. Examples include DSSM [4], DeepMatch [5], CDSMM [6], ARC-I [7], CNTN [8], LSTM-RNN [9] and Bi-CNN-MI [10]. 

To develop a model for similarity detection is desired to have some pre-processed corpus available. Corpora specifically designed for this task already exist, such as the METER Corpus [17], the Microsoft Research Paraphrase Corpus [18], the PAN Plagiarism Corpus [19] and Stanford Natural Language Inference (SNLI)[20]. Recent approaches to text-pair classification have mostly been developed on the SNLI corpus. It provides over 500,000 pairs of short sentences, with human annotations indicating whether an entailment, contradiction or neutral logical relationship holds between the sentences. However, the data is also quite artificial, most of the questions aren't human generated. And that is why I think the Quora dataset is so important.

Quora is a question-and-answer site where questions are asked, answered, edited and organized by its community of users [12]. Quora have released their first public dataset to use in a Kaggle competition [11], which is related to the problem of identifying duplicate setences. And that's the problem we are going to solve in this project.

### Problem Statement

To solve this problem I'm having the same motivation as Quora had when releasing their dataset: "there should be a single question page for each logically distinct question" [13]. This type of problem is challenging because you usually can't solve it by looking at individual words. No single word is going to tell you whether two questions are duplicates. You have to look at both items together. And that is the main reason for choosing this problem, I believe that helping to solve this task might bring some enlightenment to other critical NLP tasks.

The duplicate detection problem can be defined as follows: given a pair of questions q1 and q2, train a model that learns the function[14]:
 
 f(q1, q2) → 0 or 1 
 
where 1 represents that q1 and q2 have the same intent and 0 otherwise.

The challenge of text matching lies in detecting duplicates at a intent-based, semantic level. To only use a word-based comparison approach will not give us the best results. Being inspired by recent advances in the deep learning research community, I will approach this challenge by representing sentences on their semantic and syntactic relations from different levels of abstractions with neural networks [7;15]. 

I will be using a Long Short Term Memory network (LSTM), variant of Recurrent Neural Networks (RNNs), which are better at capturing long-term dependencies. I plan to use a variation of Word2Vec [16] to convert each question into a semantic vector, and then fed those question embeddings into the neural network.

### Metrics

I propose to use the evaluation metric defined by Kaggle on the Quora competition [11], where the model will be evaluated on the log loss (logistic loss or cross-entropy loss) between the predicted values and the ground truth. For each ID in the test set, there must have a prediction on the probability that the questions are duplicates (a number between 0 and 1). The log loss looks at the actual probabilities as opposed to the order of predictions. The metric is negative the log likelihood of the model that says each test observation is chosen independently from a distribution that places the submitted probability mass on the corresponding class, for each observation [28].

<p align="center">
<img src ="https://i.stack.imgur.com/NEmt7.png)"/>
</p>

where N is the number of observations, M is the number of class labels, loglog is the natural logarithm, yi,jyi,j is 1 if observation ii is in class jj and 0 otherwise, and pi,jpi,j is the predicted probability that observation ii is in class jj.

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration

The Quora dataset is a set of question pairs, with annotations indicating whether the questions request the same information. This data set is large, real, and relevant — a rare combination. Each line contains IDs for each question in the pair, the full text for each question, and a binary value that indicates whether the line truly contains a duplicate pair. Considering that the dataset is already split into two files, training and test, here's a quick analysis of this dataset.

1) Training:
  - Question pairs: 404290
  - Questions: 537933
  - Duplicate pairs: 36.92%
  - Most questions have from 15 to 150 characters

2) Test:
  - Question pairs: 2345796
  - Questions: 4363832
  - Question pairs (Training) / Question pairs (Test): 17.0%
  - Most questions have from 15 to 150 characters  
  
From the training dataset, 37% are confirmed duplicates (positive class), which it seems reasonable in a real-life scenario since we expect that the Quora user is more likely to have looked for an answer before posting a question. All of the questions in the training are genuine examples from Quora. One source of negative examples were pairs of “related questions” which, although pertaining to similar topics, are not truly semantically equivalent [13]. Both training and test have a simliar distribution of number of characters per question, with only a few outliers out of this range.

It's worth noting that there is a lot more test data than training data, approximately 5 times more. The explanation is that Quora's original sampling method returned an imbalanced dataset with many more true examples of duplicate pairs than non-duplicates. Therefore, they supplemented the test set with negative examples (computer-generated question pairs), as an anti-cheating measure for the Kaggle competition.

### Exploratory Visualization

In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques

To tackle this problem of duplicate questions, I will use a deep learning framework based on the “Siamese” architecture [21]. In this framework, the same neural network encoder (LSTM) is applied to two input sentences individually, so that both of the two sentences are encoded into sentence vectors in the same embedding space. Then, a matching decision is made solely based on the two sentence vectors [22;23]. The advantage of this framework is that sharing parameters makes the model smaller and easier to train, and the sentence vectors can be used for visualization, sentence clustering and many other purposes [24]. The disadvantage is that there is no interaction between the two sentences during the encoding procedure, which may lose some important information.

On the Siamese framework, I will implement the “Siamese-LSTM” model. It will encode two input sentences into two sentence vectors with a neural network encoder, and make a decision based on the cosine similarity between the two sentence vectors. The LSTM model will be designed according to the architectures in [24].

### Benchmark

This is a brand-new dataset, no results have been published yet. But we do have Quora discussing about their current production model for solving this problem. They have used a random forest model with tens of handcrafted features, including the cosine similarity of the average of the word2vec embeddings of tokens, the number of common words, the number of common topics labeled on the questions, and the part-of-speech tags of the words [14]. And recently they have experimented with end-to-end deep learning solutions.

To have a benchmark so we can use as a threshold for defining success and failure, I did test a Random Forest Classifier from sklearn. As input I've converted the questions pair, from the training dataset, to a matrix of TF-IDF (Term Frequency - Inverse Document Frequency) features and used it to fit a Random Forest model. And finally I ran a prediction on the test dataset and took a log loss score of 0.6015. I intend to significantly improve this score by using a deep learning framework.

## III. Methodology

### Data Preprocessing
After doing a detailed Exploratory Data Analysis (EDA), both on training and test dataset, understanding word frequency and semantics. Now, for the data pre-processing stage of this project, we will convert the questions into semantic vectors, trying two algorithms: GloVe [26] from Stanford NLP Group and a variation of sense2vec [27] from spaCy. Both are algorithms that embed words into a vector space with 300 dimensions in general. They will capture semantics and even analogies between different words. GloVe is easy to train and it is flexible to add new words outside of its vocabulary. SpaCy has been recenlty released and is trained on Wikipedia, therefore, it might be stronger in terms of word semantics.

Then, I will use TF-IDF, which will enhance the mean vector representation. It will be applied weighted average of word vectors by using these scores, to emphasizes the importance of discriminating words and avoid useless, frequent words which are shared by many questions. In other words, this means that we weigh the terms by how uncommon they are, meaning that we care more about rare words existing in both questions than common one. This makes sense, as for example we care more about whether the word "exercise" appears in both than the word "and" - as uncommon words will be more indicative of the content. 

### Implementation
On the next step, I will build a Siamese network with 3 layers network using Euclidean distance as the measure of instance similarity. It has Batch Normalization per layer. It is particularly important since BN layers enhance the performance considerably. I believe they are able to normalize the final feature vectors and Euclidean distance performances better in this normalized space.

After having the data pre-processed and the model set, the data is split into training and validation set. And now we start training the model for 25 epochs, saving the weights from the model checkpoint with the maximum validation accuracy.

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
And finally, the sixth and last step is to evaluate the model trained. We get the best checkpointed model from the training above and run on the test set. We will be looking for the log loss between the predicted values and the ground truth.


In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
