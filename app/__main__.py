import os

import uvicorn
from api import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.api:create_app",
        factory=True,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        reload=True,
    )

# uvicorn app.api:create_app --host 0.0.0.0 --port 8080 --reload
