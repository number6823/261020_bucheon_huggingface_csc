"""틀린그림 찾기 서비스 - HuggingFace 모델(DINOv2, DETR, FLUX) + Groq + Gradio."""
import os
import random

import cv2
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import InferenceClient
from transformers import AutoImageProcessor, AutoModel, AutoModelForObjectDetection

load_dotenv()  # 상위 폴더(wori)의 .env까지 자동으로 찾는다
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SIDE = 640

_models = {}


def load(name):
    """모델은 처음 쓸 때 한 번만 내려받아 캐시한다."""
    if name not in _models:
        if name == "dino":
            ckpt = "facebook/dinov2-small"
            _models[name] = (AutoImageProcessor.from_pretrained(ckpt), AutoModel.from_pretrained(ckpt).to(DEVICE).eval())
        else:
            ckpt = "facebook/detr-resnet-50"
            _models[name] = (AutoImageProcessor.from_pretrained(ckpt), AutoModelForObjectDetection.from_pretrained(ckpt).to(DEVICE).eval())
    return _models[name]


def fit(img):
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def dino_diff_map(a, b):
    """두 이미지의 DINOv2 패치 특징 코사인 거리를 이미지 크기 히트맵으로 반환."""
    proc, model = load("dino")
    inputs = proc(images=[a, b], return_tensors="pt", do_resize=True, size={"height": 448, "width": 448}, do_center_crop=False).to(DEVICE)
    with torch.no_grad():
        feats = model(**inputs).last_hidden_state[:, 1:, :]  # CLS 제외, (2, 1024, C)
    dist = 1 - torch.nn.functional.cosine_similarity(feats[0], feats[1], dim=-1)
    return cv2.resize(dist.reshape(32, 32).cpu().numpy(), a.size, interpolation=cv2.INTER_CUBIC)


def draw_circle(img, box, color=(255, 0, 0)):
    x1, y1, x2, y2 = box
    cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2 + 8, (y2 - y1) // 2 + 8), 0, 0, 360, color, 3)


def find_differences(img1, img2, sensitivity, min_area):
    """두 이미지의 차이를 찾아 빨간 원과 히트맵으로 표시."""
    if img1 is None or img2 is None:
        raise gr.Error("이미지 두 장을 모두 올려주세요.")
    a, b = fit(img1), fit(img2)
    b = b.resize(a.size)
    na, nb = np.array(a), np.array(b)

    pix = cv2.absdiff(cv2.GaussianBlur(na, (5, 5), 0), cv2.GaussianBlur(nb, (5, 5), 0)).max(axis=2).astype(np.float32)
    score = 0.5 * norm(pix) + 0.5 * norm(dino_diff_map(a, b))
    mask = (score > (1 - sensitivity)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out, count = nb.copy(), 0
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        draw_circle(out, (x, y, x + w, y + h))
        count += 1
    heat = cv2.cvtColor(cv2.applyColorMap((norm(score) * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return out, heat, f"찾은 차이: {count}곳"


def overlap(p, q):
    return max(0, min(p[2], q[2]) - max(p[0], q[0])) * max(0, min(p[3], q[3]) - max(p[1], q[1]))


def edit_region(img, box, kind):
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    if kind == "remove":
        m = np.zeros(img.shape[:2], np.uint8)
        m[y1:y2, x1:x2] = 255
        return cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA)
    if kind == "color":
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + random.randint(40, 100)) % 180
        img[y1:y2, x1:x2] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    elif kind == "flip":
        img[y1:y2, x1:x2] = roi[:, ::-1]
    else:  # blur
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (0, 0), 6)
    return img


def make_puzzle(img, n):
    """DETR로 물체를 찾아 n곳을 바꿔 (원본, 틀린그림, 정답) 반환."""
    if img is None:
        raise gr.Error("이미지를 올려주세요.")
    pil = fit(img)
    proc, model = load("detr")
    inputs = proc(images=pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    res = proc.post_process_object_detection(outputs, threshold=0.7, target_sizes=[pil.size[::-1]])[0]

    w, h = pil.size
    cands = []
    for box in res["boxes"].tolist():
        box = [int(max(0, box[0])), int(max(0, box[1])), int(min(w, box[2])), int(min(h, box[3]))]
        if (box[2] - box[0]) * (box[3] - box[1]) > 0.25 * w * h or box[2] - box[0] < 20 or box[3] - box[1] < 20:
            continue
        cands.append(box)
    random.shuffle(cands)

    picked = []
    for box in cands:
        if all(overlap(box, p) == 0 for p in picked):
            picked.append(box)
        if len(picked) == n:
            break
    while len(picked) < n:  # 물체가 부족하면 임의 영역으로 보충
        bw, bh = random.randint(w // 10, w // 6), random.randint(h // 10, h // 6)
        x, y = random.randint(0, w - bw), random.randint(0, h - bh)
        box = [x, y, x + bw, y + bh]
        if all(overlap(box, p) == 0 for p in picked):
            picked.append(box)

    orig = np.array(pil)
    edited = orig.copy()
    for box in picked:
        edited = edit_region(edited, box, random.choice(["remove", "color", "flip", "blur"]))
    answer = edited.copy()
    for box in picked:
        draw_circle(answer, box)
    return orig, edited, answer, f"틀린 곳 {len(picked)}개를 만들었어요. 찾아보세요!"


def ai_puzzle(prompt, n):
    """텍스트 -> FLUX 이미지 생성 -> 틀린그림 문제로 변환."""
    if not prompt:
        raise gr.Error("만들고 싶은 그림을 설명해주세요.")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise gr.Error("HF_TOKEN이 없습니다. wori/.env를 확인해주세요.")
    image = InferenceClient(provider="nscale", api_key=token).text_to_image(prompt, model="black-forest-labs/FLUX.1-schnell")
    return make_puzzle(image, n)


def ask_llm(question):
    """Groq LLM에게 질문하고 답변 텍스트만 반환."""
    if not question:
        raise gr.Error("질문을 입력해주세요.")
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise gr.Error("GROQ_API_KEY가 없습니다. wori/.env를 확인해주세요.")
    completion = Groq(api_key=key).chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": question}],
    )
    return completion.choices[0].message.content


with gr.Blocks(title="틀린그림 찾기") as demo:
    gr.Markdown("# 🔍 틀린그림 찾기\nHuggingFace의 DINOv2 · DETR · FLUX 모델로 틀린그림을 풀고 만들어요.")
    with gr.Tab("정답 찾기"):
        with gr.Row():
            i1 = gr.Image(type="pil", label="원본")
            i2 = gr.Image(type="pil", label="틀린그림")
        with gr.Row():
            sens = gr.Slider(0.5, 0.95, value=0.75, step=0.01, label="민감도")
            area = gr.Slider(10, 2000, value=200, step=10, label="최소 영역 크기")
        btn1 = gr.Button("차이 찾기", variant="primary")
        with gr.Row():
            o1 = gr.Image(label="결과 (빨간 원)")
            o2 = gr.Image(label="차이 히트맵")
        t1 = gr.Textbox(label="결과")
        btn1.click(find_differences, [i1, i2, sens, area], [o1, o2, t1])
    with gr.Tab("문제 만들기"):
        src = gr.Image(type="pil", label="사진 업로드")
        num = gr.Slider(1, 8, value=5, step=1, label="틀린 곳 개수")
        btn2 = gr.Button("문제 만들기", variant="primary")
        with gr.Row():
            p1 = gr.Image(label="원본")
            p2 = gr.Image(label="틀린그림")
            p3 = gr.Image(label="정답")
        t2 = gr.Textbox(label="안내")
        btn2.click(make_puzzle, [src, num], [p1, p2, p3, t2])
    with gr.Tab("AI로 그림 만들기"):
        prompt = gr.Textbox(label="생성할 이미지 설명", placeholder="예: 장난감이 가득한 아이 방")
        num3 = gr.Slider(1, 8, value=5, step=1, label="틀린 곳 개수")
        btn3 = gr.Button("그림 생성 + 문제 만들기", variant="primary")
        with gr.Row():
            q1 = gr.Image(label="원본")
            q2 = gr.Image(label="틀린그림")
            q3 = gr.Image(label="정답")
        t3 = gr.Textbox(label="안내")
        btn3.click(ai_puzzle, [prompt, num3], [q1, q2, q3, t3])
    with gr.Tab("AI에게 질문"):
        question = gr.Textbox(label="질문")
        btn4 = gr.Button("질문하기", variant="primary")
        answer_box = gr.Textbox(label="[답변]", lines=12)
        btn4.click(ask_llm, question, answer_box)

if __name__ == "__main__":
    demo.launch()
