#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer
from collections import Counter
import pickle
import os

class InvertedIndex:
    def __init__(self):
        self.index = dict() #dictionary mapping tokens to sets of document IDs
        self.docmap = dict() #dictionary mapping document IDs to their full document objects
        self.term_frequencies = dict()

    def __add_document(self, doc_id, text):
        stemmer = PorterStemmer()
        translator = str.maketrans('','',string.punctuation)
        tokens = text.translate(translator).lower().split()
        #each document corresponds to counter object. inside counter, each key is a term, and the value is a frequency
        self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            token = stemmer.stem(token)
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1


    def get_documents(self, term):
        documents = self.index.get(term, set())  
        return sorted(documents)
    
    def build(self):
        with open("data/movies.json", "r") as file:
            data = json.load(file)
            for movie in data["movies"]:
                #add movie to docmap and index
                doc_id = str(movie['id'])
                self.docmap[doc_id] = movie
                self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self):
        os.makedirs('cache', exist_ok=True)

        #pickle stores objects as binary data, not text, so we need wb for write binary
        with open('cache/index.pkl', 'wb') as file:
            pickle.dump(self.index, file)
    
        with open('cache/docmap.pkl', 'wb') as file:
            pickle.dump(self.docmap, file)

        with open('cache/term_frequencies.pkl', 'wb') as file:
            pickle.dump(self.term_frequencies, file)

    def load(self):

        try:
            with open('cache/index.pkl', 'rb') as file:
                self.index = pickle.load(file)
        
            with open('cache/docmap.pkl', 'rb') as file:
                self.docmap = pickle.load(file)

            with open('cache/term_frequencies.pkl', 'rb') as file:
                self.term_frequencies = pickle.load(file)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required cache file is not founded: {e.filename}")
        except pickle.UnpicklingError:
            raise ValueError("Cache file is corrupted or not a valid pickle file")
        

    def get_tf(self, doc_id: str, term: str) -> int:
        stemmer = PorterStemmer()
        translator = str.maketrans('','',string.punctuation)
        tokens = term.translate(translator).lower().split()

        if len(tokens) != 1:
            raise Exception(f"Expected exactly one token in term: {term}")
        
        tokenized_term = stemmer.stem(tokens[0])
        
        return self.term_frequencies[doc_id].get(tokenized_term, 0) #return frequency of term in document

        

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Return term frequency")
    tf_parser.add_argument("document_id", type=str, help="document to search in")
    tf_parser.add_argument("term", type=str, help="term to search")


    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")

            inverted_index = InvertedIndex()
            
            try:
                inverted_index.load()
            except FileNotFoundError as e:
                print(e)
                exit(1)
            except Exception as e:
                print(f"Unexpected error loading index: {e}")
                exit(1)

            #load stopwords
            stopwords = []
            with open("data/stopwords.txt", "r") as file:
                lines = file.read().splitlines()
                for line in lines:
                    stopwords.append(line)       

            #create stemmer
            stemmer = PorterStemmer()          

            #remove punctuation (third arg)
            translator = str.maketrans('', '', string.punctuation) 
            query_clean = args.query.translate(translator).lower()
            #tokenize by word
            query_tokens = query_clean.split()
            #remove stopwords
            filtered_query_tokens = [word for word in query_tokens if word not in stopwords]
            #stemmed_query_tokens
            stemmed_query_tokens = list(map(stemmer.stem, filtered_query_tokens))

            matched_doc_ids = set()

            for token in stemmed_query_tokens:

                matching_documents = inverted_index.get_documents(token)

                for doc_id in matching_documents:
                    if len(matched_doc_ids) >= 5:
                        break
                    matched_doc_ids.add(doc_id)

                if len(matched_doc_ids) >= 5:
                    break

            for doc_id in matched_doc_ids:
                movie = inverted_index.docmap[doc_id]
                print(movie['title'])

        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()

        case "tf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            term_frequency = inverted_index.get_tf(args.document_id, args.term)
            print(term_frequency)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()