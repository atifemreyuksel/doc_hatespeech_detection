from fastapi import FastAPI
from inference_backend import load_model, detect_hate_speech
app = FastAPI()

class MyException(Exception):
    def __init__(self, message, code=None):
        self.code = code
        super(Exception, self).__init__(message)

model, device = load_model(
    load_from="checkpoints/rulemodel/best_model.pth",
    is_gpu=0
    )

@app.post("/detect")
async def detect(text: str):
    try:
        detected_feats = detect_hate_speech(text, model, device)
        detected_feats["status"] = True
        return detected_feats
    except MyException as e:
        return {"status": False, "error": str(e)}