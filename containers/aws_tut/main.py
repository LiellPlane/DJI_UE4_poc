from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    event: str

@app.get('/')
def index():
    return 'Hello World!'

@app.post("/")
async def create_item(item: Item):
    return {item.event: "complete"}