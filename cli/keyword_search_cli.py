#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer
from collections import Counter
import pickle
import os
import math
from search_utils import BM25_K1

class InvertedIndex:
    def __init__(self):
        self.index = dict() #dictionary mapping tokens to sets of document IDs
        self.docmap = dict() #dictionary mapping document IDs to their full document objects
        self.term_frequencies = dict()

    def _add_document(self, doc_id, text):

        tokens = self.tokenize(text)
        #each document corresponds to counter object. inside counter, each key is a term, and the value is a frequency
        self.term_frequencies[doc_id] = Counter()

        for token in tokens:
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
                self._add_document(doc_id, f"{movie['title']} {movie['description']}")

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
            raise FileNotFoundError(f"Required cache file is not found: {e.filename}")
        except pickle.UnpicklingError:
            raise ValueError("Cache file is corrupted or not a valid pickle file")
        

    def get_tf(self, doc_id: str, term: str) -> int:

        token = self.tokenize(term)

        if len(token) != 1:
            raise Exception(f"Expected exactly one token in term: {term}")
        
        tokenized_term = token[0]
        
        return self.term_frequencies.get(doc_id, {}).get(tokenized_term, 0) #return frequency of term in document
    
    def get_bm25_idf(self, term: str) -> float:
        token = self.tokenize(term)
        if len(token) != 1:
            raise Exception(f"Expected exactly one token in term: {term}")
        
        tokenized_term = token[0]
        num_docs = len(self.docmap)
        doc_freq = len(self.index.get(tokenized_term, 0))
        bm25_idf = math.log((num_docs - doc_freq + .5) / (doc_freq + .5) + 1)

        return bm25_idf
    
    def bm25_idf_command(self, term: str) -> float:
        self.load()
        tokens = self.tokenize(term)
        tokenized_term = tokens[0]
        bm25idf = self.get_bm25_idf(tokenized_term)
        return bm25idf

    def get_bm25_tf(self, doc_id: str, term: str, k1: float = BM25_K1) -> float:
        tf = self.get_tf(doc_id, term)
        saturated_tf = (tf * (k1 + 1)) / (tf + k1)
        return saturated_tf
    
    def bm25_tf_command(self, doc_id: str, term: str, k1: float = BM25_K1) -> float:
        self.load()
        bm25_tf = self.get_bm25_tf(doc_id, term)
        return bm25_tf
    
    
    def tokenize(self, term):
        stemmer = PorterStemmer()
        translator = str.maketrans('','',string.punctuation)
        tokens = term.translate(translator).lower().split()
        tokenized_terms = [stemmer.stem(token) for token in tokens]
        return tokenized_terms

        

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Return term frequency")
    tf_parser.add_argument("document_id", type=str, help="document to search in")
    tf_parser.add_argument("term", type=str, help="term to search")

    idf_parser = subparsers.add_parser("idf", help="Return inverse document frequency")
    idf_parser.add_argument("term", type=str, help="term to search")

    tfidf_parser = subparsers.add_parser("tfidf", help="Returns tf-idf score")
    tfidf_parser.add_argument("document_id", type=str, help="document to search in")
    tfidf_parser.add_argument("term", type=str, help="term to search")

    bm25_idf_parser = subparsers.add_parser('bm25idf', help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("document_id", type=str, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")



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

            query_tokens = inverted_index.tokenize(args.query)     

            #remove stopwords
            filtered_query_tokens = [word for word in query_tokens if word not in stopwords]


            matched_doc_ids = set()

            for token in filtered_query_tokens:

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

        case "idf":
            inverted_index = InvertedIndex()

            term = args.term
            tokens = inverted_index.tokenize(term)
            if not tokens:
                print("No valid tokens found in term")
                exit(1)
            tokenized_term = tokens[0]

            inverted_index.load()
            doc_count = len(inverted_index.docmap)
            term_doc_count = len(inverted_index.index.get(tokenized_term, set()))

            idf = math.log((doc_count + 1) / (term_doc_count + 1))

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            doc_id = args.document_id
            term = args.term

            inverted_index = InvertedIndex()
            inverted_index.load()
            tokens = inverted_index.tokenize(term)

            tf_idf = 0

            for token in tokens:
                token_tf = inverted_index.get_tf(doc_id, token)

                doc_count = len(inverted_index.docmap)
                term_doc_count = len(inverted_index.index.get(token, set()))
                token_idf = math.log((doc_count + 1) / (term_doc_count + 1))

                tf_idf += token_tf * token_idf

            print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            term = args.term
            inverted_index = InvertedIndex()
            bm25_idf = inverted_index.bm25_idf_command(term)

            print(f"BM25 IDF score of '{term}': {bm25_idf:.2f}")


        case "bm25tf":
            doc_id = str(args.document_id)
            inverted_index = InvertedIndex()
            bm25tf = inverted_index.bm25_tf_command(doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document '{args.document_id}': {bm25tf:.2f}")
            
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()