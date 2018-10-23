********
Overview
********

==============
Basic concepts
==============

Autoregressor uses conditional probability model to produce sequences of elements from a vocabulary.
Conditional probability model is fed with elements of a sequence one by one and is expected to produce a probability distribution of the next element. 
The model should represent history of seen elements in internally in its state. Said state may be a tensor or (possibly nested) tuple of tensors.

Vocabulary contains all possible elements that can be part of a sequence. 
Elements of the vocabulary are represented as vectors of integers. Theoretically, it can contain numerical values of some discrete features. Most of the time, however, it is just identifier of an element (for example id of word in some external lexicon) and thus elements are of a shape `[1]` and of a form `[id]`, where `id` is said identifier. 

We number steps of generation process from 1, as it is assumed that there is some zeroth step already given for conditional probability model to be able to get some input.

==================================
Assumptions about dimensionalities
==================================

It is assumed that element of a sequence is represented by a 1-D integer tensor, even if it contains only one number - that is id from the vocabulary.
Sequence of elements is represented as 2-D tensor ([time, element]) [[1],[2],[3]].

Every layer accept batch of examples as an input and examples in batch are processed in parallel but independently.


===========================================================
Representation of probability distributions over vocabulary
===========================================================

It is assumed  that probability distribution is represented as a vector of floating point values. 
Each element of the vector represents probability of one element in the vocabulary.
Often, multiple probability distributions (for example, distributions of the next word for alternative paths) are stacked into one tensor. 
Then, the first dimension always refer to the differnt distributions and the second dimension to different words.

It is not assumed that probability distribution is aligned with the order of elements in vocabulary or numerical values of their ids.
Custom mappings between position (index) in probability distribution and elements from vocabulary can be provided.
Default mapping indeed assumes that probability of an elemnent `[i]` is given by `i`-th element in probability distribution vector.
Of course, it is important to use consistent mappings in all component of the system - Autoregressor and ElementProbabilityMasking and a conditional probability model that produces probability distributions.

# TODO: sprawdzić czy to prawda
Whenever it is needed to provide a layer with mapping between sequence element and probability distribution (or in opposit direction) it is assumed that mapping function acts on just one element (not a batch or sequence). 
Example: if element is a one-element vector consisting of id number which is positive integer (1, 2, ...) and probability distribution is a vecor of probabilities of elements in the order of their id then::
    
    prob_dist_to_element_mappig = lambda index: [index+1]

