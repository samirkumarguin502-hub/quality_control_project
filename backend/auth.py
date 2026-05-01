from fastapi import HTTPException

def login(username, password):
    if username == "admin" and password == "1234":
        return {"message": "Login success"}
    raise HTTPException(status_code=401, detail="Invalid credentials")