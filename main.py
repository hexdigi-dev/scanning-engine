from fastapi import FastAPI

import config
from models import CheckResult, LeakEstimate, ScanRequest, ScanResponse, SupportingLeak

app = FastAPI()


@app.get("/")
def read_root():
    return {"status": "ok"}
