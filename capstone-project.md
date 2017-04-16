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

<p align="center"><b>
f(q1, q2)  &#9658;  0 or 1
</b></p>

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

### Data Exploration

The Quora dataset is a set of question pairs, with annotations indicating whether the questions request the same information. This data set is large, real, and relevant — a rare combination. Each line contains IDs for each question in the pair, the full text for each question, and a binary value that indicates whether the line truly contains a duplicate pair. Considering that the dataset is already split into two files, training and test, here's a quick analysis of this dataset.

**1) Training**:
  - Question pairs: 404290
  - Questions: 537933
  - Duplicate pairs: 36.92%
  - Most questions have from 15 to 150 characters

**2) Test**:
  - Question pairs: 2345796
  - Questions: 4363832
  - Question pairs (Training) / Question pairs (Test): 17.0%
  - Most questions have from 15 to 150 characters  
  
From the training dataset, 37% are confirmed duplicates (positive class), which it seems reasonable in a real-life scenario since we expect that the Quora user is more likely to have looked for an answer before posting a question. All of the questions in the training are genuine examples from Quora. One source of negative examples were pairs of “related questions” which, although pertaining to similar topics, are not truly semantically equivalent [13]. Both training and test have a simliar distribution of number of characters per question, with only a few outliers out of this range.

It's worth noting that there is a lot more test data than training data, approximately 5 times more. The explanation is that Quora's original sampling method returned an imbalanced dataset with many more true examples of duplicate pairs than non-duplicates. Therefore, they supplemented the test set with negative examples (computer-generated question pairs), as an anti-cheating measure for the Kaggle competition.

As for a semantic analysis, let's take a look at usage of different punctuation and capital letters in questions - this may form a basis for some interesting features later on. This analysis is only on the training dataset, to avoid the auto-generated questions from the test dataset:

- Questions with question marks: 99.87%
- Questions with [math] tags: 0.12%
- Questions with full stops: 6.31%
- Questions with capitalised first letters: 99.81%
- Questions with capital letters: 99.95%
- Questions with numbers: 11.83%

### Exploratory Visualization

Let us have an idea of how the data is distributed with respect to the number of characters and words per question, both on training and test datasets.

<p align="center">
<img src ="https://raw.githubusercontent.com/rafapetter/udacity-machine-learning-capstone/master/eda/chars_distribution.png"/>
Figure 1 - Distribution of number of characters per question
</p>

We can see on the Figure 1 that the number of characters in most questions are in the range from 15 to 150. Both training and test have a very simliar distribution. The test one seems to be smoother, maybe because it has a larger dataset (5 times greater than training).

One important thing to notice is the steep cut-off at 150 characters for the training set, for most questions, while the test set slowly decreases after 150. And that's because, as of April 2016, Quora allows up to 150 for the question [29]. 

It's also worth noting that I've truncated this histogram at 250 characters, and that the max of the distribution is at just under 1200 characters for both sets - although samples with over 220 characters are very rare. We can only conclude that questions greater than 150 characters are previous to April 2016.

Let's do the same for word count on Figure 2. I'll be using a naive method for splitting words (splitting on spaces instead of using a serious tokenizer), although this should still give us a good idea of the distribution.

<p align="center">
<img src ="https://raw.githubusercontent.com/rafapetter/udacity-machine-learning-capstone/master/eda/words_distribution.png"/>
Figure 2 - Distribution of number of words per question
</p>

We see a similar distribution for word count, with most questions being about 10 words long. It looks to me like the distribution of the training set seems more "pointy", while on the test set it is wider. Nevertheless, they are quite similar.

So what are the most common words? Let's take a look at a word cloud on Figure 3.

<p align="center">
<img src ="https://raw.githubusercontent.com/rafapetter/udacity-machine-learning-capstone/master/eda/word_cloud.png"/>
Figure 3 - Word cloud
</p>

On the word cloud, we may conclude a few importat aspects of the dataset:

- **Type of Questions**: there's a lot on how questions are being emphasized: 'difference', 'best way', 'better', 'good'. That's important because, for instance, questions like 'what is the best way to learn Python' might be very similar to 'what is a better way to start learning Python'.

- **Content about recent events**: high frequency terms like 'Hillary Clinton' and 'Donald Trump', most likely related to the recent presidential election. We may conclude that we have a dataset with questions that were made over the last year.

- **India user base**: there's a lot about the country of India, with terms like 'India', 'Bangalore', 'India Best'. Which confirms the results from Alexa [30] showing that India is the most active country on the Quora website.

Before going further, let's have a look at the relationship of words being shared between the pair of questions, and their tendency on being duplicate or not.

<p align="center">
<img src ="https://raw.githubusercontent.com/rafapetter/udacity-machine-learning-capstone/master/eda/word_match_share.png"/>
Figure 4 - Distribution over word_match_share
</p>

We can see on Figure 4 that this feature has quite a lot of predictive power, as it is good at separating the duplicate questions from the non-duplicate ones. It seems very good at identifying questions which are definitely different, but is not so great at finding questions which are definitely duplicates.

### Algorithms and Techniques

To tackle this problem of duplicate questions, I will use a deep learning framework based on the “Siamese” architecture [21], the “Siamese-LSTM” framework. But before diving into this framework, let's define the Siamese and LSTM models separately. 

One of the appeals of LSTM (Long Short Term Memory) is the idea that they might be able to connect previous information to the present task, such as using previous words in a sentece might inform the understanding of the present word [http://colah.github.io/posts/2015-08-Understanding-LSTMs/], i.e. it can retain information from the begining of a question to the end of it, making it very helpful for us to solve our current problem. The LSTM are an improved version of RNN (Recurrent Neural Network), in which, it still creates an internal state of the network allowing it to exhibit dynamic temporal behavior, by using their internal memory to process arbitrary sequences of inputs [https://en.wikipedia.org/wiki/Recurrent_neural_network]. LSTM are normally augmented by recurrent gates called forget gates, preventing backpropagated errors from vanishing or exploding.

The Siamese neural network is a class of neural network architectures that contain two or more identical sub-networks, i.e. they have same parameters and weights. Parameter updating is mirrored across both subnetworks. Siamese networks are popular among tasks that involve finding similarity or a relationship between two comparable things, and in our case the input will take two sentences and the output will score how similar they are. [https://www.quora.com/What-are-Siamese-neural-networks-what-applications-are-they-good-for-and-why]

In the “Siamese-LSTM” framework, the same neural network encoder (LSTM) is applied to two input sentences individually, so that both of the two sentences are encoded into sentence vectors in the same embedding space. Then, a matching decision is made solely based on the two sentence vectors [22;23]. The advantage of this framework is that sharing parameters makes the model smaller and easier to train, and the sentence vectors can be used for visualization, sentence clustering and many other purposes [24]. The disadvantage is that there is no interaction between the two sentences during the encoding procedure, which may lose some important information.


### Benchmark

This is a brand-new dataset, no results have been published yet. But we do have Quora discussing about their current production model for solving this problem. They have used a random forest model with tens of handcrafted features, including the cosine similarity of the average of the word2vec embeddings of tokens, the number of common words, the number of common topics labeled on the questions, and the part-of-speech tags of the words [14]. And recently they have experimented with end-to-end deep learning solutions.

To have a benchmark so we can use as a threshold for defining success and failure, I did test a Random Forest Classifier from sklearn. As input I've converted the questions pair, from the training dataset, to a matrix of TF-IDF (Term Frequency - Inverse Document Frequency) features and used it to fit a Random Forest model. And finally I ran a prediction on the test dataset and took a log loss score of 0.6015. I intend to significantly improve this score by using a deep learning framework.

## III. Methodology

### Data Preprocessing

After doing a detailed Exploratory Data Analysis (EDA), both on training and test dataset, understanding characters and word frequencies, we will use TF-IDF scores, which will enhance the mean vector representation later on. It will be applied weighted average of word vectors by using these scores, to emphasizes the importance of discriminating words and avoid useless, frequent words which are shared by many questions. In other words, this means that we weigh the terms by how uncommon they are, meaning that we care more about rare words existing in both questions than common one. This makes sense, as for example we care more about whether the word "exercise" appears in both than the word "and" - as uncommon words will be more indicative of the content. 

Then, for the data pre-processing stage of this project, we will convert the questions into semantic vectors, trying two algorithms: GloVe [26] from Stanford NLP Group and a variation of sense2vec [27] from spaCy. Both are algorithms that embed words into a vector space with 300 dimensions in general. They will capture semantics and even analogies between different words. GloVe is easy to train and it is flexible to add new words outside of its vocabulary. SpaCy has been recenlty released and is trained on Wikipedia, therefore, it might be stronger in terms of word semantics.

### Implementation

After pre-processing the data, we will now train our model on the data. We're using a few functions from Keras [https://github.com/fchollet/keras], an easy and fast high-level neural network library for Python, that will help us implement a few actions on our network:

<p style="color:blue;font-weight:900">
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, merge, BatchNormalization, Activation, Input, Merge
from keras import backend as K
from keras.optimizers import RMSprop, SGD, Adam
</p>

On the next step, I will build a Siamese network with 3 layers network using Euclidean distance as the measure of instance similarity. It has Batch Normalization per layer. It is particularly important since BN layers enhance the performance considerably. I believe they are able to normalize the final feature vectors and Euclidean distance performances better in this normalized space.

After having the data pre-processed and the model set, the data is split into training and validation set. And now we start training the model for 50 epochs, saving the weights from the model checkpoint with the maximum validation accuracy.

### Refinement

For the vector representation of words, the GLOVE algorithm gives performance below our expectations. I believe, this is because the questions are short and does not induce a semantic structure that GLOVE is able to learn. When training with the word2vec from spaCy, we get a better performance. The spaCy library was just released, and it seems to be really fast.

For the optimization algorithm, we first used the simple and standard version of SGD (Stochastic Gradient Descent) [31]. And then tried with a better version o SGD called Adam (Adaptive Moment Estimation) [32], to clearly get a better result.

As for the neural network, I tried to introduce Dropout between layers, but the best I have got was 0.75. Then by replacing Dropout with concatenation of different layers, it improved the performance.

## IV. Results

### Model Evaluation and Validation

The best performing model that we have got was a 3 layers network using Euclidean distance as the measure of instance similarity, with Batch Normalization per layer. I believe, they are able to normalize the final feature vectors and Euclidean distance performances better in this normalized space. This model is reasonable and aligned with our expecations

We will use the evaluation metric defined by Kaggle on the Quora competition [11], where the model will be evaluated on the log loss (logistic loss or cross-entropy loss) between the predicted values and the ground truth. For each ID in the test set, there must have a prediction on the probability that the questions are duplicates (a number between 0 and 1). The log loss looks at the actual probabilities as opposed to the order of predictions. The metric is negative the log likelihood of the model that says each test observation is chosen independently from a distribution that places the submitted probability mass on the corresponding class, for each observation [28].

<p align="center">
<img src ="https://i.stack.imgur.com/NEmt7.png)"/>
</p>

where N is the number of observations, M is the number of class labels, loglog is the natural logarithm, yi,jyi,j is 1 if observation ii is in class jj and 0 otherwise, and pi,jpi,j is the predicted probability that observation ii is in class jj.

Here is a look at the resuls from the 3 different model architectures evaluated.

- 3 Layers + Adam : 0.22
- 3 Layers + Adam + Dropout : 0.25
- **3 Layers + Adam + Layer Concatenation : 0.21**

### Justification

This is a brand-new dataset, no results have been published yet. But we do have Quora discussing about their current production model for solving this problem. They have used a random forest model with tens of handcrafted features, including the cosine similarity of the average of the word2vec embeddings of tokens, the number of common words, the number of common topics labeled on the questions, and the part-of-speech tags of the words [14]. And recently they have experimented with end-to-end deep learning solutions.

To have a benchmark so we can use as a threshold for defining success and failure, I have trained a Random Forest Classifier from sklearn. As input I've converted the questions pair, from the training dataset, to a matrix of TF-IDF (Term Frequency - Inverse Document Frequency) features and used it to fit a Random Forest model. And finally I ran a prediction on the test dataset and took a log loss score of 0.6015. Comparing that score to the result we got from our best model we can say that we have definitely improved the result by using a deep learning framework.

## V. Conclusion

### Reflection

The process used for this project can be summarized using the following steps:

1. A detailed analysis was made about the Quora question pairs competition on Kaggle.
2. A Exploratory Data Analysis (EDA) was made, both on training and testing dataset.
3. TF-IDF scores were used to enhance the mean vector representation used on the next step.
4. The questions were converted to semantic vectors, trying two algorithms: GloVe [26] from Stanford NLP Group and a variation of sense2vec [27] from spaCy.
5. A Siamese network was built with 3 layers network. It had Batch Normalization per layer.
6. The data is split into training and validation set, with 50 epochs. And the model was set by training on this network.
7. And finally, we evaluated the model trained, by getting the best checkpointed model from the training above and run on the test set.

I had a bit more difficult on step 5, because it was hard to find a better combination of layers on the network. I did expected Dropouts to improve the accuracy performance, but it didn't.

### Improvement

Natural language sentence matching (NLSM) has been studied for many years. The early approaches were interested in designing hand-craft features to capture n-gram overlapping, word reordering and syntactic alignments phenomena. This kind of method can work well on a specific task or dataset, but it’s hard to generalize well to other tasks. With the availability of large-scale annotated datasets, many deep learning models were proposed for NLSM. 

The framework we used, based on the Siamese architecture [21], where sentences are encoded into sentence vectors based on some neural network encoder, and then the relationship between two sentences was decided solely based on the two sentence vectors. However, this kind of framework ignores the fact that the lower level interactive features between two sentences are indispensable. Therefore, many neural network models were proposed to match sentences from multiple level of granularity [10]. Experimental results on many tasks have proofed that the new framework works significantly better than the previous methods.

### References

[1] Li, H., and Xu, J. 2013. Semantic Matching in Search. Foundations and Trends in Information Retrieval 7(5):343–469.

[2] Berger, A.; Caruana, R.; Cohn, D.; Freitag, D.; and Mittal, V. 2000. Bridging the Lexical Chasm: Statistical Approaches to Answer-finding. In Proceedings of the 23rd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval , 192–199.

[3] Dolan, B.; Quirk, C.; and Brockett, C. 2004. Unsupervised construction of large paraphrase corpora: Exploiting massively parallel news sources. In Proceedings of the 20th International Conference on Computational Linguistics (Coling) , 350–356.

[4] Huang, P.-S.; He, X.; Gao, J.; Deng, L.; Acero, A.; and Heck, L. 2013. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM International Conference on Information & Knowledge Management (CIKM) , 2333– 2338.

[5] Lu, Z., and Li, H. 2013. A Deep Architecture for Matching Short Texts. In Advances in Neural Information Processing Systems (NIPS) , 1367–1375.

[6] Shen, Y.; He, X.; Gao, J.; Deng, L.; and Mesnil, G. 2014. A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval. In Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management (CIKM) , 101–110.

[7] Hu, B.; Lu, Z.; Li, H.; and Chen, Q. 2014. Convolutional neural network architectures for matching natural language sentences. In Advances in Neural Information Processing Systems (NIPS) , 2042–2050.

[8] Qiu, X., and Huang, X. 2015. Convolutional Neural Tensor Network Architecture for Community-Based Question Answering. In Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI) , 1305–1311.

[9] Palangi, H.; Deng, L.; Shen, Y.; Gao, J.; He, X.; Chen, J.; Song, X.; and Ward, R. K. 2015. Deep sentence embedding using the long short term memory network: Analysis and application to information retrieval. CoRR abs/1502.06922.

[10] Yin, W., and Schutze, H. 2015a. Convolutional Neural Network for Paraphrase Identifica- tion. In The 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL) , 901–911.

[11] Quora Question Pairs - Can you identify question pairs that have the same intent? https://www.kaggle.com/c/quora-question-pairs

[12] Quora - https://en.wikipedia.org/wiki/Quora

[13] First Quora Dataset Release: Question Pairs - https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs

[14] Semantic Question Matching with Deep Learning - https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning

[15] Socher, R.; Huang, E. H.; Pennington, J.; Ng, A. Y.; and Manning, C. D. 2011. Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection. In Advances in Neural Information Processing Systems (NIPS) , 801–809.

[16] Mikolov, T.; Sutskever, I.; Chen, K.; Corrado, G. S.; and Dean, J. 2013. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems (NIPS) , 3111–3119.

[17] http://nlp.shef.ac.uk/meter/

[18] http://research.microsoft.com/en-us/downloads/607d14d9-20cd-47e3-85bc-a2f65cd28042/

[19] http://www.webis.de/research/corpora

[20] https://nlp.stanford.edu/projects/snli/

[21] Jane Bromley, James W. Bentz, Léon Bottou, Isabelle Guyon, Yann LeCun, Cliff Moore, Eduard Sackinger, and Roopak Shah. Signature verification using a ”siamese” time delay neural network. IJPRAI , 7(4):669– 688, 1993.

[22] Samuel R Bowman, Gabor Angeli, Christopher Potts, and Christopher D Manning. A large annotated corpus for learning natural language inference. arXiv preprint arXiv:1508.05326 , 2015.

[23] Ming Tan, Cicero dos Santos, Bing Xiang, and Bowen Zhou. Lstm-based deep learning models for non-factoid answer selection. arXiv preprint arXiv:1511.04108 , 2015.

[24] Zhiguo Wang, Haitao Mi, and Abraham Ittycheriah. Semi-supervised clustering for short text via deep representation learning. In CoNLL , 2016.

[25] Ankur P Parikh, Oscar Täckström, Dipanjan Das, Jakob Uszkoreit. 2016. A Decomposable Attention Model for Natural Language Inference, arXiv 2016.

[26] GloVe - https://nlp.stanford.edu/projects/glove/

[27] Sense2vec with spaCy and Gensim - https://explosion.ai/blog/sense2vec-with-spacy

[28] https://www.kaggle.com/wiki/LogLoss

[29] https://www.quora.com/topic/Character-Limits-on-Quora/faq

[30] http://www.alexa.com/siteinfo/quora.com

[31] https://en.wikipedia.org/wiki/Stochastic_gradient_descent

[32] https://arxiv.org/abs/1412.6980
