import os
import gc
import time
import logging

import yaml
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Response

from src.estimator import SearchEngine, Normalize
from src.schema import PredictRequest, PredictResponce

app_logger = logging.getLogger(__name__)

conf = yaml.load(open("./conf.yaml"), Loader=yaml.FullLoader)

search_engine = SearchEngine(conf["data_path"], conf["model_name"], Normalize().normalize)

app = FastAPI()

@app.get("/health_check")
def health_check():
    """Return service status"""
    return {"Status":"OK"}

@app.post("/search", response_model=PredictResponce, status_code=200)
def serch(
    response: Response,
    payload: PredictRequest,   
    search_emgine: SearchEngine = Depends(lambda: search_engine)
):  
    start = time.time()
    try:
        outputs = search_emgine.find_doc(query=payload.query, search_engine=payload.search_engine, k=payload.k)
        outputs = {"status":"success", "search_result":outputs}
    except Exception as e:
        app_logger.error(f"Something get wrong! :: Raise exception {type(e).__name__}. {e}")
        outputs = {"status":"error", "search_result":[]}
    app_logger.info(f"Outputs are: {outputs}")
    finish = time.time()
    outputs["time_search"] = (finish - start)*1000

    return outputs

uvicorn.run(app, host="0.0.0.0", port=8888)