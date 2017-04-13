# Machine Learning Engineer Nanodegree
## Capstone Proposal

Rafael Pettersen Caixeta

April 12, 2017

### Domain Background

Natural language sentence matching (NLSM) is the task of comparing two sentences and identifying the relationship between them. The task is critical for many applications in natural language processing (NLP), such as information retrieval [1], question answering [2] and paraphrase identification [3].

Recently,  deep  neural  network  based  models  have  been applied in this area and achieved some important progress. A lot of deep models follow the paradigm to first represent the whole sentence to a single distributed representation, and then compute similarities between the two vectors to output the matching score. Examples include DSSM [4], DeepMatch [5], CDSMM [6], ARC-I [7], CNTN [8], LSTM-RNN [9] and Bi-CNN-MI [10]. 

Natural language sentences have complicated structures, both sequential and hierarchical, that are essential  for  understanding them. A successful sentence-matching algorithm therefore needs to capture not only the internal structures of sentences but also the rich patterns in their interactions. Sentence similarity is a condition or property that can be measured between two sentences, which  determines  the  degree  of  similarity  between  them.  Sentence similarity ranges between 0% (no relationship at all) and 100% (sentences are semanticaly identical).  Also  note  that  two  similar  sentences  do  not  need  to  share  the  content, neither verbatim nor expressed in other words. They may just cover the same topic or merely be written in the same language.

### Problem Statement

For this project I'm proposing a binary classification problem, and I am going to use the Quora dataset from a Kaggle competition [11], which is related to the problem of identifying duplicate setences. Quora is a question-and-answer site where questions are asked, answered, edited and organized by its community of users [12]. The Quora motivation for releasing this dataset is that "there should be a single question page for each logically distinct question" [13]. This type of problem is challenging because you usually can't solve it by looking at individual words. No single word is going to tell you whether two questions are duplicates. You have to look at both items together. And that is the main reason for choosing this problem, I believe that helping to solve this task might bring some enlightenment to other critical NLP tasks.

The duplicate detection problem can be defined as follows: given a pair of questions q1 and q2, train a model that learns the function[14]:
 
 f(q1, q2) → 0 or 1 
 
where 1 represents that q1 and q2 have the same intent and 0 otherwise.

The challenge of text matching lies in detecting duplicates at a intent-based, semantic level. To only use a word-based comparison approach will not give us the best results. Being inspired by recent advances in the deep learning research community, I will approach this challenge by representing sentences on their semantic and syntactic relations from different levels of abstractions with neural networks [7;15]. 

I will be using a Long Short Term Memory network (LSTM), variant of Recurrent Neural Networks (RNNs), which are better at capturing long-term dependencies. I plan to use a variation of Word2Vec [16] to convert each question into a semantic vector, and then fed those question embeddings into the neural network.

### Datasets and Inputs

To develop a model for similarity detection is desired to have some pre-processed corpus available. Corpora specifically designed for this task already exist, such as the METER Corpus [17], the Microsoft Research Paraphrase Corpus [18], the PAN Plagiarism Corpus [19] and Stanford Natural Language Inference (SNLI)[20]. Recent approaches to text-pair classification have mostly been developed on the SNLI corpus. It provides over 500,000 pairs of short sentences, with human annotations indicating whether an entailment, contradiction or neutral logical relationship holds between the sentences. However, the data is also quite artificial, most of the questions aren't human generated. And that is why the Quora dataset is so important.

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

### Solution Statement

To tackle this problem of duplicate questions, I will use a deep learning framework based on the “Siamese” architecture [21]. In this framework, the same neural network encoder (LSTM) is applied to two input sentences individually, so that both of the two sentences are encoded into sentence vectors in the same embedding space. Then, a matching decision is made solely based on the two sentence vectors [22;23]. The advantage of this framework is that sharing parameters makes the model smaller and easier to train, and the sentence vectors can be used for visualization, sentence clustering and many other purposes [24]. The disadvantage is that there is no interaction between the two sentences during the encoding procedure, which may lose some important information.

On the Siamese framework, I will implement the “Siamese-LSTM” model. It will encode two input sentences into two sentence vectors with a neural network encoder, and make a decision based on the cosine similarity between the two sentence vectors. The LSTM model will be designed according to the architectures in [24].

### Benchmark Model

This is a brand-new dataset, no results have been published yet. But we do have Quora discussing about their current production model for solving this problem. They have used a random forest model with tens of handcrafted features, including the cosine similarity of the average of the word2vec embeddings of tokens, the number of common words, the number of common topics labeled on the questions, and the part-of-speech tags of the words [14]. And recently they have experimented with end-to-end deep learning solutions.

To have a benchmark so we can use as a threshold for defining success and failure, I did test a Random Forest Classifier from sklearn. As input I've converted the questions pair, from the training dataset, to a matrix of TF-IDF (Term Frequency - Inverse Document Frequency) features and used it to fit a Random Forest model. And finally I ran a prediction on the test dataset and took a log loss score of 0.6015. I intend to significantly improve this score by using a deep learning framework.

### Evaluation Metrics

I propose to use the evaluation metric defined by Kaggle on the Quora competition [11], where the model will be evaluated on the log loss (logistic loss or cross-entropy loss) between the predicted values and the ground truth. For each ID in the test set, there must have a prediction on the probability that the questions are duplicates (a number between 0 and 1). The log loss looks at the actual probabilities as opposed to the order of predictions. The metric is negative the log likelihood of the model that says each test observation is chosen independently from a distribution that places the submitted probability mass on the corresponding class, for each observation [28].

<p align="center">
<img src ="https://i.stack.imgur.com/NEmt7.png)"/>
</p>

where N is the number of observations, M is the number of class labels, loglog is the natural logarithm, yi,jyi,j is 1 if observation ii is in class jj and 0 otherwise, and pi,jpi,j is the predicted probability that observation ii is in class jj.

### Project Design

The first step of this project is to do a detailed Exploratory Data Analysis (EDA), both on training and testing dataset:
1) Text Analysis: a distribution on the number of characters per word and per question; the most frequent words.
2) Semantic Analysis: different punctuation in questions

The second step is to convert the questions into semantic vectors, trying two algorithms: GloVe [26] from Stanford NLP Group and a variation of sense2vec [27] from spaCy. Both are algorithms that embed words into a vector space with 300 dimensions in general. They will capture semantics and even analogies between different words. GloVe is easy to train and it is flexible to add new words outside of its vocabulary. SpaCy has been recenlty released and is trained on Wikipedia, therefore, it might be stronger in terms of word semantics.

On the third step, I will use TF-IDF, which will enhance the mean vector representation. It will be applied weighted average of word vectors by using these scores, to emphasizes the importance of discriminating words and avoid useless, frequent words which are shared by many questions. In other words, this means that we weigh the terms by how uncommon they are, meaning that we care more about rare words existing in both questions than common one. This makes sense, as for example we care more about whether the word "exercise" appears in both than the word "and" - as uncommon words will be more indicative of the content. 

On the fourth step, I will build a Siamese network with 3 layers network using Euclidean distance as the measure of instance similarity. It has Batch Normalization per layer. It is particularly important since BN layers enhance the performance considerably. I believe they are able to normalize the final feature vectors and Euclidean distance performances better in this normalized space.

On the fifth step, after having the data pre-processed and the model set, the data is split into training and validation set. And now we start training the model for 25 epochs, saving the weights from the model checkpoint with the maximum validation accuracy.

And finally, the sixth and last step is to evaluate the model trained. We get the best checkpointed model from the training above and run on the test set. We will be looking for the log loss between the predicted values and the ground truth.

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
