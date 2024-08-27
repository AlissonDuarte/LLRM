from main import app

@app.get("/")
def userRecomendations():
    return {"Hello": "World"}

