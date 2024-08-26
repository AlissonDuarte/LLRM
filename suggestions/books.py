import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#this function calculate the correlation of a book dataset based in tags
#example {
#     book1: [tag1, tag2, tag3],
#     book2: [tag2, tag5, tag1]
# }

def tag_book_proccess(books:dict):
    books_relationships = {}
    
    tags = list(set(tag for tags in books.values() for tag in tags))

    matrices = np.zeros((len(books), len(tags)))

    for i, (book, books_tags) in enumerate(books.items()):
        for tag in books_tags:
            matrices[i, tags.index(tag)] = 1
            
    matrice_similarity = cosine_similarity(matrices)
    for i, book in enumerate(books.keys()):
        correlation = []
        for j, book_refer in enumerate(books.keys()):
            if i != j:
                correlation.append((book_refer, matrice_similarity[i, j]))
        books_relationships[book] = sorted(correlation, key=lambda x: x[1], reverse = True)
        
    return books_relationships
        
            
