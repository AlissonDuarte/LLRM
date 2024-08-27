import random
import numpy as np
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

#this function calculate the correlation of a book dataset based in tags
#example {
#     book1: [tag1, tag2, tag3],
#     book2: [tag2, tag5, tag1]
# }

def tag_book_proccess(books:dict) -> dict:
    """
    Calculates the similarity between books based on their associated tags.

    This function creates a similarity matrix for a list of books, where each book is represented by a binary vector
    based on the tags it has. The similarity between books is calculated using cosine similarity.

    Parameters:
    books (dict): A dictionary where the keys are book titles and the values are lists of tags associated with each book.
                  Example:
                  {
                      "book1": ["tag1", "tag2", "tag3"],
                      "book2": ["tag2", "tag5", "tag1"]
                  }

    Returns:
    dict: A dictionary where the keys are book titles and the values are lists of tuples. Each tuple contains a book title
          and the similarity score with the book from the key, sorted by similarity in descending order.
          Example:
          {
              "book1": [("book2", 0.8), ("book3", 0.5)],
              "book2": [("book1", 0.8), ("book3", 0.4)]
          }
    """
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
        

def books_by_user_history(user_books:dict, k = random.randint(20, 33)) -> dict:
    """
    Generates book recommendations for users based on their historical book preferences and similarity to other users.

    This function converts the historical book data of users into a user-item matrix, applies the K-Nearest Neighbors 
    algorithm to find similar users, and then recommends books to each user based on the preferences of similar users.

    Parameters:
    user_books (dict): A dictionary where the keys are user identifiers and the values are dictionaries of books with their associated tags.
                       Example:
                       {
                           "User1": {"Book1": ["tag1", "tag2"], "Book2": ["tag3"]},
                           "User2": {"Book2": ["tag1", "tag3"], "Book3": ["tag2"]}
                       }

    Returns:
    dict: A dictionary where the keys are user identifiers and the values are lists of tuples. Each tuple contains a book title
          and its recommendation score, sorted by the score in descending order. The score reflects the preference of the book
          based on similar users' histories.
          Example:
          {
              "User1": [("Book4", 3.5), ("Book5", 2.8)],
              "User2": [("Book6", 4.2), ("Book7", 3.1)]
          }
    """
    # Convert user_books to a user-item matrix
    users = list(user_books.keys())
    books = sorted(set(book for user in user_books.values() for book in user.keys()))
    user_item_matrix = np.zeros((len(users), len(books)))

    book_index = {book: i for i, book in enumerate(books)}

    for i, user in enumerate(users):
        for book in user_books[user]:
            user_item_matrix[i, book_index[book]] = 1  # Binary matrix

    # Fit the KNN model
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(user_item_matrix)

    recommendations = {}
    for i, user in enumerate(users):
        distances, indices = knn.kneighbors(user_item_matrix[i].reshape(1, -1))
        similar_users = [users[index] for index in indices.flatten() if index != i]
        
        # Calculate item scores based on similar users
        item_scores = {}
        for sim_user in similar_users:
            for book in user_books[sim_user]:
                if book not in user_books[user]:
                    if book not in item_scores:
                        item_scores[book] = 0
                    item_scores[book] += 1 / (distances.flatten()[indices.flatten() != i].sum() + 1e-5)  # Avoid division by zero

        # Sort items by score
        recommendations[user] = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    
    return recommendations


def books_by_friends(friends_books:dict, k = random.randint(20, 33)) -> dict:
    """
    Generates book recommendations for users based on the reading history of their friends.

    This function constructs a book-item matrix where each book is represented by binary vectors based on co-occurrence 
    within the reading history of friends. It uses the K-Nearest Neighbors algorithm to find similar books and recommends
    books that a user has not yet read, based on the preferences of their friends.

    Parameters:
    friends_books (dict): A dictionary where the keys are user identifiers and the values are dictionaries of books.
                          Example:
                          {
                              "User1": {"Book1": ["tag1", "tag2"], "Book2": ["tag3"]},
                              "User2": {"Book2": ["tag1", "tag3"], "Book3": ["tag2"]}
                          }
    k (int, optional): The number of nearest neighbors to use for the KNN model. Defaults to a random integer between 20 and 33.

    Returns:
    dict: A dictionary where the keys are user identifiers and the values are lists of tuples. Each tuple contains a book title
          and its recommendation score, sorted by the score in descending order. The score reflects the preference for the book
          based on the reading history of friends.
          Example:
          {
              "User1": [("Book4", 3.2), ("Book5", 2.7)],
              "User2": [("Book6", 4.1), ("Book7", 3.4)]
          }
    """
    books = sorted(set(book for user in friends_books.values() for book in user.keys()))
    book_index = {book: i for i, book in enumerate(books)}
    book_item_matrix = np.zeros((len(books), len(books)))

    # Representar itens como vetores binários
    for user in friends_books.values():
        for book1 in user:
            for book2 in user:
                if book1 != book2:
                    book_item_matrix[book_index[book1], book_index[book2]] += 1

    # Fit the KNN model
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(book_item_matrix)

    recommendations = {}
    for user, books_read in friends_books.items():
        # Calculate scores for items not read by the user
        item_scores = {}
        for book in books_read:
            book_idx = book_index[book]
            distances, indices = knn.kneighbors(book_item_matrix[book_idx].reshape(1, -1))
            similar_books = [books[i] for i in indices.flatten() if books[i] not in books_read]
            for sim_book in similar_books:
                if sim_book not in item_scores:
                    item_scores[sim_book] = 0
                item_scores[sim_book] += 1 / (distances.flatten().sum() + 1e-5)  # Avoid division by zero
        
        # Sort items by score
        recommendations[user] = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    
    return recommendations


# recommendation part 2
# These functions below will use few resources to create recommendations based on the rating that users give to books.

def eucliedean(base, user1, user2):
    """
    Calculates the Euclidean similarity between two users based on their ratings of common items.

    This function computes the similarity score between two users by calculating the Euclidean distance between their 
    ratings for items they both have rated. The similarity score is the reciprocal of the distance, adjusted to avoid 
    division by zero.

    Parameters:
    base (dict): A dictionary where the keys are user identifiers and the values are dictionaries of items with their ratings.
                 Example:
                 {
                     "User1": {"Item1": 4.0, "Item2": 3.5},
                     "User2": {"Item2": 3.0, "Item3": 5.0}
                 }
    user1 (str): The identifier of the first user.
    user2 (str): The identifier of the second user.

    Returns:
    float: The Euclidean similarity score between `user1` and `user2`. A score closer to 1 indicates higher similarity,
           while a score closer to 0 indicates lower similarity.
           Returns 0 if no common items are rated by both users.
    """
    si = {}
    for item in base[user1]:
        if item in base[user2]:
            si[item] = 1
            
    if len(si) == 0:
        return 0
    
    data_sum = sum(
        [pow(base[user1][item] - base[user2][item], 2) 
            for item in base[user1] 
                if item in base[user2]]
    )
    return 1/(1+sqrt(data_sum))


def get_similarity(base, user):
    """
    Calculates the similarity scores of items for a given user based on Euclidean distance.

    This function computes the similarity scores between the specified user and all other users in the dataset using the 
    Euclidean distance metric. It then calculates weighted scores for items that the user has not rated, based on the ratings 
    of similar users. The results are sorted and the top items are returned.

    Parameters:
    base (dict): A dictionary where the keys are user identifiers and the values are dictionaries of items with their ratings.
                 Example:
                 {
                     "User1": {"Item1": 4.0, "Item2": 3.5},
                     "User2": {"Item2": 3.0, "Item3": 5.0}
                 }
    user (str): The identifier of the user for whom the similarity scores are being calculated.

    Returns:
    list: A list of tuples where each tuple contains an item and its calculated similarity score, sorted in descending 
          order by the score. Only the top 30 items are returned.
    """
    total = {}
    simi_sum = {}
    for other_user in base:
        if other_user == user:
            continue
    
        sim = eucliedean(base, user, other_user)
        
        if sim <= 0:
            continue
        
        for data in base[other_user]:
            if data not in base[user] or base[user][data] == 0:
                total.setdefault(data, 0)
                total[data] += base[other_user][data] * sim
                simi_sum.setdefault(data, 0)
                simi_sum[data] += sim
                
    ranking = [(sub_total / simi_sum[data], data) for data, sub_total in total.items() if simi_sum.get(data, 0) > 0]
    ranking.sort()
    ranking.reverse()
    return ranking[0:30]


def calc_similarity(base):
    """
    Calculates similarity scores for all users in the dataset.

    This function computes similarity scores for each user in the dataset using the Euclidean distance metric. It calculates 
    the weighted similarity scores for items that each user has not rated, based on the ratings of similar users. The results 
    are aggregated for all users.

    Parameters:
    base (dict): A dictionary where the keys are user identifiers and the values are dictionaries of items with their ratings.
                 Example:
                 {
                     "User1": {"Item1": 4.0, "Item2": 3.5},
                     "User2": {"Item2": 3.0, "Item3": 5.0}
                 }

    Returns:
    dict: A dictionary where the keys are user identifiers and the values are lists of tuples. Each tuple contains an item 
          and its calculated similarity score, sorted in descending order by the score. Only the top items are returned for each user.
    """
    result = {}
    for item in base:
        rate = get_similarity(base, item)
        result[item] = rate
    return result