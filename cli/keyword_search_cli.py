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
        tokens = text.lower().split()
        for token in tokens:
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
                self.docmap[movie['id']] = f"{movie['title']} {movie['description']}"
                self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")

    def save(self):
        os.makedirs('cache', exist_ok=True)

        with open('cache/index.pkl', 'wb') as file:
            pickle.dump(self.index, file)
    
        with open('cache/docmap.pkl', 'wb') as file:
            pickle.dump(self.docmap, file)

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

            results = []

            with open("data/movies.json", "r") as file:
                data = json.load(file)
                counter = 0
                for movie in data["movies"]:
                    #transform title steps 
                    title_clean = movie["title"].translate(translator).lower()
                    title_tokens = title_clean.split()
                    filtered_title_tokens = [word for word in title_tokens if word not in stopwords]
                    stemmed_title_tokens = list(map(stemmer.stem, filtered_title_tokens))

                    #for each search token, check if its a substring of any token in the title
                    if any(token in word for word in stemmed_title_tokens for token in stemmed_query_tokens):
                        results.append(movie)
                        counter += 1

                    #stop searching when we reach 5 movies
                    if counter > 4:
                        break

                for i, result in enumerate(results):
                    print(f"{i + 1}: {result['title']}")
        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()

            docs = inverted_index.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()