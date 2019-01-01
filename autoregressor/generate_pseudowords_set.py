import numpy as np


source_corpus_iterable = ["a", "b", "c", "d", "e", "f", "g", "h"]
#source_corpus_iterable = ["a", "b", "c",]
def generate_pseudowords_set(source_corpus_iterable):
    words = set(source_corpus_iterable)
    n_uniq_words = len(words)
    proportions = [0.8, 0.1, 0.1]

    denominator = sum(c_i * i for i, c_i in enumerate(proportions, 1))

    counts_of_pseudowords = np.array(proportions) * n_uniq_words / denominator # [p * n_uniq_words /  denominator for p in proportions]

    print(counts_of_pseudowords)
    print(counts_of_pseudowords/sum(counts_of_pseudowords))

    def some_accidental_home_brewed_elections_algorithm_that_I_should_probably_remove(counts_before_rounding):
        desired_total = np.round(sum(counts_before_rounding))
        counts = [int(c) for c in counts_before_rounding]
        remainder = desired_total - sum(counts)
        while remainder > 0:
            remainder = desired_total - sum(counts)
            underrepresented = [c < p for c, p in zip(counts, counts_before_rounding)]
            for i, is_under in enumerate(underrepresented):
                if is_under:
                    counts[i] += 1
                    remainder -= 1
                if remainder == 0:
                    break
            
        print("counts", counts)
        print("sum(counts)", sum(counts))
        print("n_uniq_words", desired_total)

        return counts


    counts_rounded = some_accidental_home_brewed_elections_algorithm_that_I_should_probably_remove(counts_of_pseudowords)
    print(counts_rounded)
    print(counts_rounded/sum(counts_of_pseudowords))
    print(np.array(counts_rounded) * (np.array(range(len(counts_rounded)))+1))
    print(sum(counts_rounded * np.arange(1,len(counts_rounded)+1)))

    words_bag = words.copy()
    pseudowords = []
    for n_meanings, count in enumerate(counts_rounded, 1):
        pseudowords.append([])
        for _ in range(count):
            pseudoword_meanings = []
            for _ in range(n_meanings):
                random_meaning = list(words_bag)[np.random.randint(0, len(words_bag))]
                pseudoword_meanings.append(random_meaning)
                words_bag.remove(random_meaning)
            pseudowords[-1].append(pseudoword_meanings)

    print(pseudowords)

    words_to_pseudowords_mapping = {}

    for n_meanings_pseudowords in pseudowords:
        for pseudoword in n_meanings_pseudowords:
            for word in pseudoword:
                words_to_pseudowords_mapping[word] = pseudoword
    
    return words_to_pseudowords_mapping

input_corpus_files = [
    "/mnt/storage/workspace_2/find_cheapest_path/autoregressor/data/simple_examples/raw/ptb.train.txt", 
    "/mnt/storage/workspace_2/find_cheapest_path/autoregressor/data/simple_examples/raw/ptb.valid.txt", 
    "/mnt/storage/workspace_2/find_cheapest_path/autoregressor/data/simple_examples/raw/ptb.test.txt",
    ]

def get_corpus(file):
    with open(file) as f:
        sentences = []
        for line in f:
            sentences.append(line.strip().split())
    return sentences

def flatten(corpus):
    return [word for sentence in corpus for word in sentence]

train = get_corpus(input_corpus_files[0])
valid = get_corpus(input_corpus_files[1])
test = get_corpus(input_corpus_files[2])

all_simple_examples = flatten(train + valid + test)

words_to_pseudowords_mapping = generate_pseudowords_set(all_simple_examples)

def transform_corpus(corpus, mapping):
    new_corpus = []
    for sentence in corpus:
        new_sentnce = ["*".join(mapping[word]) for word in sentence]
        new_corpus.append(new_sentnce)
    return new_corpus

def save_corpus(path, corpus):
    with open(path, "wt") as file:
        for sentence in corpus:
            print(" ".join(sentence), file=file)


new_train = transform_corpus(train, words_to_pseudowords_mapping)
save_corpus("ptb.pseudowords.train.txt", new_train)

new_valid = transform_corpus(valid, words_to_pseudowords_mapping)
save_corpus("ptb.pseudowords.valid.txt", new_valid)

new_test = transform_corpus(test, words_to_pseudowords_mapping)
save_corpus("ptb.pseudowords.test.txt", new_test)

