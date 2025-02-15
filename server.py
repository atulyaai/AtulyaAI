import fastapi
app = fastapi.FastAPI()

@app.get("/")
async def root():
    return {"message": "AtulyaAI Server Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
