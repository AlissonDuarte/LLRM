# main.py
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def userRecomendations():
    return {"Hello": "World"}


