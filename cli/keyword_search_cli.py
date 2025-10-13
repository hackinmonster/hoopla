#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer
import pickle
import os

class InvertedIndex:
    def __init__(self):
        self.index = dict() #dictionary mapping tokens to sets of document IDs
        self.docmap = dict() #dictionary mapping document IDs to their full document objects

    def __add_document(self, doc_id, text):
        stemmer = PorterStemmer()
        translator = str.maketrans('','',string.punctuation)
        tokens = text.translate(translator).lower().split()

        for token in tokens:
            token = stemmer.stem(token)
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        documents = self.index.get(term, set())  
        return sorted(documents)
    
    def build(self):
        with open("data/movies.json", "r") as file:
            data = json.load(file)
            for movie in data["movies"]:
                #add movie to docmap and index
                self.docmap[movie['id']] = movie
                self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")

    def save(self):
        os.makedirs('cache', exist_ok=True)

        with open('cache/index.pkl', 'wb') as file:
            pickle.dump(self.index, file)
    
        with open('cache/docmap.pkl', 'wb') as file:
            pickle.dump(self.docmap, file)

    def load(self):

        try:
            with open('cache/index.pkl', 'rb') as file:
                self.index = pickle.load(file)
        
            with open('cache/docmap.pkl', 'rb') as file:
                self.docmap = pickle.load(file)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required cache file is not founded: {e.filename}")
        except pickle.UnpicklingError:
            raise ValueError("Cache file is corrupted or not a valid pickle file")

        

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")

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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()