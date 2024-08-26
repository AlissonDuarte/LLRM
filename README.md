# LitLink Recommendation Microservice

## Overview

The LLRM (LitLink Recommendation Microservice) provides personalized recommendations for books, clubs, and users within the LitLink social network. It utilizes various recommendation algorithms to enhance user engagement and offer tailored suggestions based on user behavior and preferences.

## Features

### Book Recommendations
- **Personalized Suggestions:** Recommends books based on user’s reading history and preferences.
- **Similar Books:** Suggests books similar to those previously read or liked.
- **Tag-Based Recommendations:** Uses tags to recommend books that match user interests.

### Club Recommendations
- **Suggested Clubs:** Recommends study clubs and groups based on user’s interests and activities.
- **Related Clubs:** Identifies clubs related to those the user already follows or is a member of.

### User Recommendations
- **Friend Suggestions:** Suggests users to connect with based on shared interests and activities.
- **Activity-Based Recommendations:** Recommends users with similar reading patterns or club memberships.

### Additional Features
- **Trending Books and Clubs:** Displays popular books and clubs based on user activity.
- **Feedback Loop:** Continuously improves recommendations based on user interactions and feedback.

## Architecture

### Components

1. **Recommendation Engine:**
   - **Collaborative Filtering:** Provides recommendations based on user interactions and preferences.
   - **Content-Based Filtering:** Recommends items based on attributes of books and clubs that the user likes.

2. **Data Storage:**
   - **Database:** Stores user profiles, book tags, club information, and user interactions.
   - **Cache:** Utilizes caching to speed up recommendation queries and improve performance.

3. **API Endpoints:**
   - `GET /recommendations/books`: Retrieves book recommendations for a user.
   - `GET /recommendations/clubs`: Retrieves club recommendations for a user.
   - `GET /recommendations/users`: Retrieves user recommendations for a user.

4. **Integration:**
   - **User Data Integration:** Pulls data from user profiles and activity logs.
   - **External Data Sources:** Incorporates additional data sources if available for better recommendations.

## Technology Stack

- **Programming Language:** Python
- **Framework:** FastAPI
- **Database:** PostgreSQL
- **Cache:** Redis
- **Message Broker:** RabbitMQ


