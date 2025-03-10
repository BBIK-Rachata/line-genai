import os
import csv
import json
import datetime
import traceback
import uuid
from pathlib import Path
from io import BytesIO
import hmac
import hashlib
import base64

import pytesseract
from PIL import Image
from flask import Flask, request, url_for, send_file, Response
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv

import openai
from openai import AzureOpenAI
from linebot.v3.messaging import MessagingApi
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, FileMessage,
    TextSendMessage, ImageSendMessage
)
from linebot.v3.messaging import MessagingApi
from linebot.v3.webhook import WebhookHandler
from linebot.exceptions import LineBotApiError

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain

# โหลด environment variables
load_dotenv(override=True)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
channel_secret = os.getenv("LINE_CHANNEL_SECRET")
channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
for var, name in [(deployment_name, "AZURE_OPENAI_DEPLOYMENT_NAME"),
                  (azure_endpoint, "AZURE_OPENAI_ENDPOINT"),
                  (azure_api_key, "AZURE_OPENAI_API_KEY"),
                  (channel_secret, "LINE_CHANNEL_SECRET"),
                  (channel_access_token, "LINE_CHANNEL_ACCESS_TOKEN")]:
    if not var:
        raise ValueError(f"{name} environment variable is not set")

# Configure OpenAI
openai.api_type = "azure"
openai.api_base = azure_endpoint
openai.api_version = "2024-08-01-preview"
openai.api_key = azure_api_key

client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    api_version="2024-08-01-preview"
)

def create_chat_model():
    return AzureChatOpenAI(
        openai_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        deployment_name=deployment_name,
        openai_api_version="2024-08-01-preview",
        temperature=0.5,
        max_tokens=200
    )

# Configure LINE
line_bot_api = MessagingApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# Flask App & File Paths
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

csv_file_path = "/Users/fah/Downloads/BlueBik_Intern/SRISAWAD/customers.csv"
combined_json_path = "/Users/fah/Downloads/BlueBik_Intern/SRISAWAD/combined_scraped_data.json"

# Global in-memory storage (สำหรับเก็บไฟล์ชั่วคราว)
temp_file_storage = {}

# Global vector store & chat history
GLOBAL_STATE = {
    "vector_store": None,
    "chat_history": []
}

# global user flow dictionary (key: LINE user_id, value: instance ของ flow)
user_flows = {}

# Utility Functions
def get_example_image_url(image_type):
    mapping = {
        "ThaiID": url_for("static", filename="examples/ThaiID.jpg", _external=True, _scheme="https"),
        "HouseReg": url_for("static", filename="examples/HouseReg.png", _external=True, _scheme="https"),
        "CarFront": url_for("static", filename="examples/CarFront.png", _external=True, _scheme="https"),
        "CarBack": url_for("static", filename="examples/CarBack.png", _external=True, _scheme="https"),
        "CarLeft": url_for("static", filename="examples/CarLeft.png", _external=True, _scheme="https"),
        "CarRight": url_for("static", filename="examples/CarRight.png", _external=True, _scheme="https")
    }
    return mapping.get(image_type)

def save_file_to_memory(message_id, ext=".jpg"):
    file_bytes = line_bot_api.get_message_content(message_id).content
    file_id = uuid.uuid4().hex
    temp_file_storage[file_id] = (file_bytes, ext)
    return url_for("serve_temp_file", file_id=file_id, _external=True), file_bytes

def save_file_from_message(message_id, ext=".jpg"):
    file_bytes = line_bot_api.get_message_content(message_id).content
    img = Image.open(BytesIO(file_bytes))
    max_size = (1024, 1024)
    img.thumbnail(max_size)
    output = BytesIO()
    img.save(output, format="JPEG", quality=85)
    output.seek(0)
    compressed_bytes = output.read()
    filename = f"{uuid.uuid4().hex}{ext}"
    permanent_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(permanent_path, "wb") as f:
        f.write(compressed_bytes)
    return url_for("static", filename=f"uploads/{filename}", _external=True), compressed_bytes

@app.route("/temp_file/<file_id>")
def serve_temp_file(file_id):
    if file_id in temp_file_storage:
        file_bytes, ext = temp_file_storage[file_id]
        mimetype = "application/pdf" if ext.lower() == ".pdf" else "image/jpeg"
        return send_file(BytesIO(file_bytes), mimetype=mimetype)
    return "File not found", 404

def load_data_sources():
    cache_file = Path(combined_json_path)
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            documents = []
            if isinstance(data, dict):
                if "web_data" in data and isinstance(data["web_data"], dict):
                    for source_name, content_dict in data["web_data"].items():
                        documents.append(Document(
                            page_content=content_dict.get("content", ""),
                            metadata={"source": source_name, "url": content_dict.get("url", "")}
                        ))
                for key, value in data.items():
                    if key not in ["scraped_at", "web_data"] and isinstance(value, dict):
                        documents.append(Document(
                            page_content=value.get("content", ""),
                            metadata={"source": key, "url": value.get("url", "")}
                        ))
                return documents
            else:
                raise ValueError("JSON structure is not a dictionary.")
        except Exception as e:
            print("Error loading cache file:", e)
            return []
    else:
        print(f"Cache file not found at {cache_file}")
        return []

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def create_vector_store(documents):
    if not documents:
        print("No documents provided for vector store.")
        return None
    try:
        embeddings = AzureOpenAIEmbeddings(
            openai_api_key=azure_api_key,
            azure_deployment="bbik-embedding-small",
            azure_endpoint=azure_endpoint,
            openai_api_version="2024-08-01-preview",
            chunk_size=1024
        )
        return FAISS.from_documents(documents, embeddings)
    except Exception as e:
        print("Error creating vector store:", e)
        return None

def get_next_user_id():
    next_id = 1
    if os.path.exists(csv_file_path):
        try:
            with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                max_id = 0
                for row in reader:
                    try:
                        current_id = int(row["user_id"])
                        max_id = max(max_id, current_id)
                    except Exception as e:
                        print("Error converting user_id:", e)
                next_id = max_id + 1
        except Exception as e:
            print("Error reading CSV:", e)
    return f"{next_id:04d}"

def save_to_csv(_, name, phone, id_card="", id_card_img_url="",
                house_reg_img_url="", car_front_img_url="",
                car_back_img_url="", car_left_img_url="",
                car_right_img_url="", bank_statement_img_url=""):
    new_user_id = get_next_user_id()
    file_exists = os.path.exists(csv_file_path)
    fieldnames = ["user_id", "name", "phone", "id_card", "id_card_img_url",
                  "house_reg_img_url", "car_front_img_url", "car_back_img_url",
                  "car_left_img_url", "car_right_img_url", "bank_statement_img_url", "created_at"]
    try:
        with open(csv_file_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "user_id": new_user_id,
                "name": name or "INVALID",
                "phone": phone or "INVALID",
                "id_card": id_card or "INVALID",
                "id_card_img_url": id_card_img_url or "INVALID",
                "house_reg_img_url": house_reg_img_url or "INVALID",
                "car_front_img_url": car_front_img_url or "INVALID",
                "car_back_img_url": car_back_img_url or "INVALID",
                "car_left_img_url": car_left_img_url or "INVALID",
                "car_right_img_url": car_right_img_url or "INVALID",
                "bank_statement_img_url": bank_statement_img_url or "INVALID",
                "created_at": datetime.datetime.utcnow().isoformat()
            })
    except Exception as e:
        print("Error saving customer to CSV:", e)

def create_vector_store_from_json():
    docs = load_data_sources()
    if docs:
        chunks = chunk_documents(docs)
        if chunks:
            vs = create_vector_store(chunks)
            print(f"[DEBUG] Vector store initialized with {len(chunks)} chunks.")
            return vs
        else:
            print("No document chunks created.")
    else:
        print("No documents loaded.")
    return None

def get_combined_response(user_input: str):
    if "vector_store" not in GLOBAL_STATE or not GLOBAL_STATE["vector_store"]:
        return "Vector store is not initialized."
    try:
        all_docs = list(GLOBAL_STATE["vector_store"].docstore._dict.values())
        target_file = None
        for doc in all_docs:
            if doc.metadata.get("source", "").lower() in user_input.lower():
                target_file = doc.metadata["source"]
                break
        if target_file:
            file_docs = [doc for doc in all_docs if doc.metadata.get("source", "").lower() == target_file.lower()]
            if not file_docs:
                return f"File {target_file} not found."
            mini_store = FAISS.from_documents(file_docs, GLOBAL_STATE["vector_store"].embeddings)
            retriever = mini_store.as_retriever(search_kwargs={"k": 10})
            search_context = retriever.get_relevant_documents(user_input)
        else:
            retriever = GLOBAL_STATE["vector_store"].as_retriever(search_kwargs={"k": 10})
            search_context = retriever.get_relevant_documents(user_input)
        if not search_context:
            return "No relevant context found."
        docs_json = {"documents": [{"source": doc.metadata.get("source", "Unknown"),
                                     "content": doc.page_content} for doc in search_context]}
        json_doc = Document(
            page_content=json.dumps(docs_json, ensure_ascii=False, indent=2),
            metadata={"source": "aggregated_json"}
        )
        llm = create_chat_model()
        combined_prompt = ChatPromptTemplate.from_messages([
            {"role": "system", "content": (
                "You are an AI assistant for Srisawad Corporation Public Company Limited (SAWAD). "
                "For further inquiries, please contact us at 1652. Using the context below, answer the customer's "
                "question in Thai in plain text format (max 200 tokens):\n\nContext:\n{context}"
            )},
            MessagesPlaceholder(variable_name="chat_history"),
            {"role": "human", "content": "{input}"}
        ])
        combined_chain = create_stuff_documents_chain(
            llm, combined_prompt, document_variable_name="context"
        )
        outputs = combined_chain.invoke({
            "input": user_input,
            "chat_history": GLOBAL_STATE.get("chat_history", []),
            "context": [json_doc]
        })
        return outputs.content.strip() if hasattr(outputs, "content") else str(outputs).strip()
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}"

# Initialize vector store once at startup.
GLOBAL_STATE["vector_store"] = create_vector_store_from_json()

# Flow Handlers as Classes
class CarLoanHandler:
    """
    Flow สำหรับสินเชื่อรถยนต์:
      1. รอรับชื่อ-นามสกุล (text)
      2. รอรับเบอร์โทร (text)
      3. รอรับรูปบัตรประชาชน (image) → extract หมายเลข 13 หลัก
      4. รอรับรูปทะเบียนบ้าน (image)
      5. รอรับไฟล์ PDF สเตทเมนท์ธนาคาร (file)
    """
    def __init__(self, user_id):
        self.user_id = user_id
        self.state = "waiting_for_name"
        self.data = {}
        self.initial_message = (
            "🚗สินเชื่อรถยนต์\n"
            "👉🏻ดอกเบี้ย เริ่มต้น เพียง 1.00% ต่อเตือน\n"
            "👉🏻ระยะเวลาผ่อนสูงสุด 54 งวด\n"
            "👉🏻วงเงินสูง ไม่ต้องโอนเล่ม \"กู้เท่าที่จำเป็นและชำระคืนไหว\n"
            "*อัตราดอกเบี้ยที่แท้จริง 17.85% - 24% ต่อปี เงื่อนไขอนุมัติสินเชื่อเป็นไปตามที่บริษัทฯ กำหนด\n\n"
            "เอกสารเบื้องต้น\n"
            "1. บัตรประชาชน\n"
            "2. สำเนาทะเบียนบ้าน\n"
            "3. เล่มทะเบียนตัวจริงหรือคู่ฉบับสัญญาเช่า\n"
            "💬ขออณุญาตทราบ ชื่อ-สกุล ของคุณ"
        )

    def handle_text(self, text: str):
        if self.state == "waiting_for_name":
            self.data["name"] = text.strip()
            self.state = "waiting_for_phone"
            return f"ขอบคุณค่ะ คุณ {self.data['name']} กรุณาระบุหมายเลขโทรศัพท์ของคุณ"
        elif self.state == "waiting_for_phone":
            phone_candidate = "".join(filter(str.isdigit, text))
            if len(phone_candidate) != 10:
                return "กรุณาระบุหมายเลขโทรศัพท์ที่ถูกต้อง"
            else:
                self.data["phone"] = text.strip()
                self.state = "waiting_for_idcard"
                instructions = (
                    f"ขอบคุณค่ะ คุณ {self.data.get('name')}. "
                    "กรุณาส่งรูปบัตรประชาชนของคุณ (ตัวอย่างด้านล่าง)"
                )
                return (instructions, "ThaiID")
        else:
            return "ข้อความไม่รองรับในขั้นตอนนี้สำหรับสินเชื่อรถยนต์"

    def handle_image(self, message_id: str):
        if self.state == "waiting_for_idcard":
            image_url, _ = save_file_from_message(message_id, ".jpg")
            # เรียก LLM เพื่อ extract หมายเลขบัตรประชาชน 13 หลัก
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "Extract 13-digit Thai ID number from the image provided."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Please extract and return only the 13-digit Thai ID card number as a plain string from this image:"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "\nRemove any spaces and non-digit characters. If no valid 13-digit number is found, reply with 'INVALID'."}
                    ]}
                ],
                max_tokens=2000
            )
            id_number = response.choices[0].message.content.strip()
            if id_number.upper() == "INVALID" or len(id_number) != 13:
                return "กรุณาส่งรูปที่ชัดเจนอีกครั้ง"
            else:
                self.data["id_card"] = id_number
                self.data["id_card_img_url"] = image_url
                self.state = "waiting_for_house_reg"
                instructions = "ขอบคุณค่ะ กรุณาส่งรูปทะเบียนบ้านของคุณ (ตัวอย่างด้านล่าง)"
                return (instructions, "HouseReg")
        elif self.state == "waiting_for_house_reg":
            house_reg_img_url, _ = save_file_to_memory(message_id, ".jpg")
            self.data["house_reg_img_url"] = house_reg_img_url
            self.state = "waiting_for_bank_statement"
            return "กรุณาส่งไฟล์ PDF สเตทเมนท์ธนาคารของคุณ"
        else:
            return "สถานะไม่รองรับการส่งรูปสำหรับสินเชื่อรถยนต์"

    def handle_file(self, message_id: str):
        if self.state == "waiting_for_bank_statement":
            file_url, file_bytes = save_file_to_memory(message_id, ".pdf")
            if not file_bytes.startswith(b"%PDF"):
                return "กรุณาส่งไฟล์ PDF ที่ถูกต้อง"
            else:
                self.data["bank_statement_img_url"] = file_url
                # สำหรับสินเชื่อรถยนต์ ฟิลด์ที่ไม่เกี่ยวข้องให้ส่งค่า "INVALID"
                save_to_csv(
                    self.user_id,
                    self.data.get("name"),
                    self.data.get("phone"),
                    self.data.get("id_card", ""),
                    self.data.get("id_card_img_url", ""),
                    self.data.get("house_reg_img_url", ""),
                    "INVALID",
                    "INVALID",
                    "INVALID",
                    "INVALID",
                    self.data.get("bank_statement_img_url", "")
                )
                self.state = "done"
                return f"ข้อมูลครบถ้วนค่ะ เราจะติดต่อกลับหาคุณ {self.data.get('name')}"
        else:
            return "สถานะไม่รองรับการส่งไฟล์สำหรับสินเชื่อรถยนต์"

class InsuranceClaimHandler:
    """
    Flow สำหรับเคลมประกัน:
      1. รอรับชื่อ-นามสกุล (text)
      2. รอรับเบอร์โทร (text)
      3. รอรับรูปบัตรประชาชน (image) → extract หมายเลข 13 หลัก
      4. รอรับรูปรถด้านหน้า (image)
      5. รอรับรูปรถด้านหลัง (image)
      6. รอรับรูปรถด้านซ้าย (image)
      7. รอรับรูปรถด้านขวา (image)
    """
    def __init__(self, user_id):
        self.user_id = user_id
        self.state = "waiting_for_name"
        self.data = {}
        self.initial_message = "สำหรับการเคลมประกัน กรุณาระบุ ชื่อ-สกุล ของคุณ"

    def handle_text(self, text: str):
        if self.state == "waiting_for_name":
            self.data["name"] = text.strip()
            self.state = "waiting_for_phone"
            return f"ขอบคุณค่ะ คุณ {self.data['name']} กรุณาระบุหมายเลขโทรศัพท์ของคุณ"
        elif self.state == "waiting_for_phone":
            phone_candidate = "".join(filter(str.isdigit, text))
            if len(phone_candidate) != 10:
                return "กรุณาระบุหมายเลขโทรศัพท์ที่ถูกต้อง"
            else:
                self.data["phone"] = text.strip()
                self.state = "waiting_for_idcard"
                instructions = (
                    f"ขอบคุณค่ะ คุณ {self.data.get('name')}. "
                    "กรุณาส่งรูปบัตรประชาชนของคุณ (ตัวอย่างด้านล่าง)"
                )
                return (instructions, "ThaiID")
        else:
            return "ข้อความไม่รองรับในขั้นตอนนี้สำหรับการเคลมประกัน"

    def handle_image(self, message_id: str):
        if self.state == "waiting_for_idcard":
            image_url, _ = save_file_from_message(message_id, ".jpg")
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "Extract 13-digit Thai ID number from the image provided."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Please extract and return only the 13-digit Thai ID card number as a plain string from this image:"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "\nRemove any spaces and non-digit characters. If no valid 13-digit number is found, reply with 'INVALID'."}
                    ]}
                ],
                max_tokens=2000
            )
            id_number = response.choices[0].message.content.strip()
            if id_number.upper() == "INVALID" or len(id_number) != 13:
                return "กรุณาส่งรูปที่ชัดเจนอีกครั้ง"
            else:
                self.data["id_card"] = id_number
                self.data["id_card_img_url"] = image_url
                self.state = "waiting_for_car_front"
                instructions = "ขอบคุณค่ะ กรุณาส่งรูปรถด้านหน้าของคุณ (ตัวอย่างด้านล่าง)"
                return (instructions, "CarFront")
        elif self.state == "waiting_for_car_front":
            car_front_img_url, _ = save_file_from_message(message_id, ".jpg")
            self.data["car_front_img_url"] = car_front_img_url
            self.state = "waiting_for_car_back"
            instructions = "ขอบคุณค่ะ กรุณาส่งรูปรถด้านหลังของคุณ (ตัวอย่างด้านล่าง)"
            return (instructions, "CarBack")
        elif self.state == "waiting_for_car_back":
            car_back_img_url, _ = save_file_from_message(message_id, ".jpg")
            self.data["car_back_img_url"] = car_back_img_url
            self.state = "waiting_for_car_left"
            instructions = "ขอบคุณค่ะ กรุณาส่งรูปรถด้านซ้ายของคุณ (ตัวอย่างด้านล่าง)"
            return (instructions, "CarLeft")
        elif self.state == "waiting_for_car_left":
            car_left_img_url, _ = save_file_from_message(message_id, ".jpg")
            self.data["car_left_img_url"] = car_left_img_url
            self.state = "waiting_for_car_right"
            instructions = "ขอบคุณค่ะ กรุณาส่งรูปรถด้านขวาของคุณ (ตัวอย่างด้านล่าง)"
            return (instructions, "CarRight")
        elif self.state == "waiting_for_car_right":
            car_right_img_url, _ = save_file_from_message(message_id, ".jpg")
            self.data["car_right_img_url"] = car_right_img_url
            # บันทึกข้อมูลสำหรับเคลมประกัน โดยฟิลด์ที่ไม่เกี่ยวข้องให้ส่ง "INVALID"
            save_to_csv(
                self.user_id,
                self.data.get("name"),
                self.data.get("phone"),
                self.data.get("id_card", ""),
                self.data.get("id_card_img_url", ""),
                "INVALID",  # house_reg_img_url
                self.data.get("car_front_img_url", ""),
                self.data.get("car_back_img_url", ""),
                self.data.get("car_left_img_url", ""),
                self.data.get("car_right_img_url", ""),
                "INVALID"   # bank_statement_img_url
            )
            self.state = "done"
            return f"ข้อมูลครบถ้วนค่ะ เราจะติดต่อกลับหาคุณ {self.data.get('name')}"
        else:
            return "สถานะไม่รองรับการส่งรูปสำหรับการเคลมประกัน"

    def handle_file(self, message_id: str):
        return "ไม่รองรับไฟล์ในขั้นตอนนี้สำหรับการเคลมประกัน"

# Flask Endpoints & Handlers
@app.route("/", methods=["GET", "POST"])
def callback():
    if request.method == "GET":
        return Response("OK", status=200, mimetype="text/plain")
    
    raw_body = request.get_data()
    body = raw_body.decode('utf-8')
    signature = request.headers.get("X-Line-Signature", "")
    
    # Compute expected signature using your channel secret
    secret = channel_secret.strip()
    expected_signature = base64.b64encode(
        hmac.new(secret.encode('utf-8'), raw_body, hashlib.sha256).digest()
    ).decode('utf-8')
    
    # Log for debugging
    print("[DEBUG] Expected signature:", expected_signature)
    print("[DEBUG] Received signature:", signature)
    
    # You can choose to log and continue even if the signature does not match.
    if not hmac.compare_digest(expected_signature.strip(), signature.strip()):
        print("Signature mismatch! Proceeding for debugging purposes.")
        # Optionally: return a 200 response immediately if you want to ignore the mismatch.
    
    # Process the webhook request
    try:
        handler.handle(body, signature)
    except Exception as e:
        print("Error handling webhook:", e)
    
    # Always return HTTP 200 OK to LINE
    return Response("OK", status=200, mimetype="text/plain")

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    line_user_id = event.source.user_id
    user_message = event.message.text.strip()

    # หากมี flow อยู่แล้ว ให้ส่งข้อความไปที่ instance นั้น
    if line_user_id in user_flows:
        flow = user_flows[line_user_id]
        result = flow.handle_text(user_message)
        if isinstance(result, tuple):
            reply_text, sample_type = result
            example_url = get_example_image_url(sample_type)
            reply_messages = [
                TextSendMessage(text=reply_text),
                ImageSendMessage(
                    original_content_url=example_url,
                    preview_image_url=example_url
                )
            ]
        else:
            reply_messages = [TextSendMessage(text=result)]
        try:
            line_bot_api.reply_message(event.reply_token, reply_messages)
        except LineBotApiError as e:
            print("Error replying in flow:", e)
        return

    # ถ้ายังไม่มี flow ให้เลือกเริ่มต้นตาม keyword ที่ส่งมา
    if "สินเชื่อรถยนต์" in user_message:
        # เริ่มต้น flow สำหรับสินเชื่อรถยนต์ โดยใช้ initial_message ภายในคลาส
        user_flows[line_user_id] = CarLoanHandler(line_user_id)
        reply_messages = [TextSendMessage(text=user_flows[line_user_id].initial_message)]
    elif "เคลมประกัน" in user_message:
        # เริ่มต้น flow สำหรับเคลมประกัน โดยใช้ initial_message ภายในคลาส
        user_flows[line_user_id] = InsuranceClaimHandler(line_user_id)
        reply_messages = [TextSendMessage(text=user_flows[line_user_id].initial_message)]
    else:
        # หากไม่ตรงกับ flow ใดๆ ให้ใช้ general query
        reply_text = get_combined_response(user_message)
        reply_messages = [TextSendMessage(text=reply_text)]
    try:
        line_bot_api.reply_message(event.reply_token, reply_messages)
    except LineBotApiError as e:
        print("Error replying to text message:", e)

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    line_user_id = event.source.user_id
    if line_user_id in user_flows:
        flow = user_flows[line_user_id]
        result = flow.handle_image(event.message.id)
        if isinstance(result, tuple):
            reply_text, sample_type = result
            example_url = get_example_image_url(sample_type)
            reply_messages = [
                TextSendMessage(text=reply_text),
                ImageSendMessage(
                    original_content_url=example_url,
                    preview_image_url=example_url
                )
            ]
        else:
            reply_messages = [TextSendMessage(text=result)]
    else:
        reply_messages = [TextSendMessage(text="ไม่มีการดำเนินการในตอนนี้")]
    try:
        line_bot_api.push_message(line_user_id, reply_messages)
    except LineBotApiError as e:
        print("Error pushing image response:", e)

@handler.add(MessageEvent, message=FileMessage)
def handle_file_message(event):
    line_user_id = event.source.user_id
    if line_user_id in user_flows:
        flow = user_flows[line_user_id]
        result = flow.handle_file(event.message.id)
        reply_messages = [TextSendMessage(text=result)]
    else:
        reply_messages = [TextSendMessage(text="ไม่รองรับไฟล์ในขั้นตอนนี้")]
    try:
        line_bot_api.push_message(line_user_id, reply_messages)
    except LineBotApiError as e:
        print("Error pushing file response:", e)

if __name__ == "__main__":
    app.run(debug=True)
