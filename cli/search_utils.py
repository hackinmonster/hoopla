# PARAMS

# controls diminishing returns of additional occurences of the search term in a document. 
# higher k1 means repetition keeps increasing score longer, lower k1 means it diminishes faster
BM25_K1 = 1.5


# length normalization factor to control how much we care about document length
# b=0  means length norm is always 1. b=1 means full normalization is applied
BM25_B = 0.75