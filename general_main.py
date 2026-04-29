from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
from rapidocr_onnxruntime import RapidOCR
import requests

# ─── Engine globale ───────────────────────────────────────────────────────────
ocr_engine: RapidOCR | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_engine
    ocr_engine = RapidOCR()   # ← caricato UNA VOLTA SOLA all'avvio
    yield

# ─── Helpers ──────────────────────────────────────────────────────────────────

def analyze_led_circuit_from_bytes(image_bytes: bytes) -> list:
    """Detect the dominant LED colour in an image using HSV analysis."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Impossibile decodificare l'immagine")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    blurred_v = cv2.GaussianBlur(v_channel, (15, 15), 0)
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred_v)
    x, y = max_loc

    if max_val < 220:
        return [{"category": "led", "value": "off", "confidence": 0.0}]

    padding = 40
    h, w = img.shape[:2]
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(w, x + padding), min(h, y + padding)

    hsv_crop = hsv[y1:y2, x1:x2]
    color_mask = cv2.inRange(hsv_crop, (0, 100, 150), (180, 255, 255))
    valid_pixels = hsv_crop[color_mask > 0]

    if len(valid_pixels) == 0:
        return [{"category": "led", "value": "indeterminato", "confidence": 0.0}]

    color_ranges = {
        "red":    [(0, 10), (165, 180)],
        "yellow": [(15, 35)],
        "green":  [(45, 85)],
        "blue":   [(100, 130)],
        "purple": [(135, 160)],
    }

    hues = valid_pixels[:, 0]
    votes: dict[str, int] = {}
    for color, ranges in color_ranges.items():
        count = sum(np.sum((hues >= low) & (hues <= high)) for low, high in ranges)
        votes[color] = int(count)

    winner = max(votes, key=votes.get)
    confidence = votes[winner] / len(valid_pixels)

    if confidence < 0.20:
        return [{"category": "led", "value": "undefined", "confidence": round(float(confidence), 4)}]

    return [{"category": "led", "value": winner, "confidence": round(float(confidence), 4)}]

def get_local_background_category(img, bbox) -> str:
    """Classify the local background around a text bounding-box as 'document' or 'box'.
    bbox: array di 4 punti [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (formato RapidOCR)
    """
    pts = np.array(bbox, dtype=np.int32)
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    h, w, _ = img.shape
    pad = 10
    roi = img[max(0, y_min - pad):min(h, y_max + pad),
              max(0, x_min - pad):min(w, x_max + pad)]

    if roi.size == 0:
        return "document"

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    median_hsv = np.median(hsv_roi.reshape(-1, 3), axis=0)
    h_val, s_val, v_val = median_hsv

    if v_val > 170 and s_val < 60:
        return "document"
    elif 5 <= h_val <= 35 and s_val > 40:
        return "box"
    else:
        return "document"

def analyze_text_from_bytes(image_bytes: bytes) -> list:
    """Run RapidOCR on an image and return detected text with category metadata."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Impossibile decodificare l'immagine")

    raw_results, elapse = ocr_engine(img, use_cls=False)

    final_output = []
    threshold = 0.25

    if raw_results:
        for item in raw_results:
            box, text, score = item[0], item[1], item[2]
            if score > threshold:
                category = get_local_background_category(img, box)
                final_output.append({
                    "category": category,
                    "value": text,
                    "confidence": round(float(score), 4)
                })

    return final_output

# ─── Shared validation ────────────────────────────────────────────────────────

def _validate_request(file: UploadFile, x_api_key: str | None) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image or None Command String")

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)

API_KEY = os.environ["API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

@app.get("/")
def root():
    return {"message": "Computer Vision API is running."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/led")
async def predict_led(
    file: UploadFile = File(...),
    x_api_key: str = Header(None),
):
    _validate_request(file, x_api_key)
    try:
        image_bytes = await file.read()
        led_results = analyze_led_circuit_from_bytes(image_bytes)
        return led_results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/text")
async def predict_text(
    file: UploadFile = File(...),
    x_api_key: str = Header(None),
):
    _validate_request(file, x_api_key)
    try:
        image_bytes = await file.read()
        text_results = analyze_text_from_bytes(image_bytes)
        return text_results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/command")
async def predict_sentiment(
    request: Request,
    x_api_key: str = Header(None),
):

    body = await request.json()
    text = body.get("text", "").strip()

    _validate_request(text, x_api_key)

    if not text:
        raise HTTPException(status_code=400, detail="Il campo 'text' non può essere vuoto.")

    request_text = "Riceverai una frase, voglio che tu capisca se si tratti di un comando di tipo HELP,STOP,REPEAT,UNKNOWN. Devi restituire solamente una di queste 4 parole e nient'altro la frase è la seguente: " + text

    try:
        openai_response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Sei un assistente che riceve una frase in linguaggio naturale e che deve capire di che comando si tratta"},
                    {"role": "user", "content": request_text},
                ],
            },
        )
        openai_response.raise_for_status()
        result = openai_response.json()
        content = result["choices"][0]["message"]["content"]
        return JSONResponse(content={"command": content})

    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Errore OpenAI: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
