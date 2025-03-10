from dotenv import load_dotenv
import os
import tempfile
import json
import uuid
import requests
from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    MessageAction, ImageMessage, ImageSendMessage,
    TemplateSendMessage, ButtonsTemplate, PostbackAction,
    PostbackEvent
)
from openai import AzureOpenAI 
from pyngrok import ngrok
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ตรวจสอบ environment variables สำหรับ LINE API
if not os.getenv('LINE_CHANNEL_ACCESS_TOKEN'):
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN is not set in .env file")

if not os.getenv('LINE_CHANNEL_SECRET'):
    raise ValueError("LINE_CHANNEL_SECRET is not set in .env file")

# ตรวจสอบ environment variables สำหรับ Azure OpenAI
if not os.getenv('AZURE_OPENAI_ENDPOINT'):
    raise ValueError("AZURE_OPENAI_ENDPOINT is not set in .env file")

if not os.getenv('AZURE_OPENAI_KEY'):
    raise ValueError("AZURE_OPENAI_KEY is not set in .env file")

if not os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'):
    raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME is not set in .env file")

# Initialize Flask app
app = Flask(__name__)

# Initialize LINE API
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))

# สร้าง Azure OpenAI Client
client = AzureOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_KEY'),
    api_version="2025-01-01-preview"  # ใช้เวอร์ชันล่าสุด
)



# สร้างโฟลเดอร์สำหรับเก็บรูปภาพ (ถ้ายังไม่มี)
UPLOAD_FOLDER = 'received_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created directory for images: {UPLOAD_FOLDER}")

# เก็บ session state ของผู้ใช้
user_sessions = {}

# ประเภทการสนทนา
SESSION_TYPE_MAIN = "main"
SESSION_TYPE_INSURANCE = "insurance"
SESSION_TYPE_LOAN = "loan"

# เก็บคีย์สำหรับประเภทรูปภาพที่ต้องการ
IMAGE_TYPE_LICENSE_PLATE = "license_plate"
IMAGE_TYPE_DAMAGE = "damage"
IMAGE_TYPE_FULL_CAR = "full_car"
IMAGE_TYPE_ID_CARD = "id_card"
IMAGE_TYPE_CAR_REGISTRATION = "car_registration"
IMAGE_TYPE_DEED = "deed"
IMAGE_TYPE_FINANCIAL_STATEMENT = "financial_statement"

def create_user_session(user_id, session_type=SESSION_TYPE_MAIN):
    """สร้างหรือรีเซ็ตเซสชันของผู้ใช้"""
    user_sessions[user_id] = {
        "session_type": session_type,
        "conversation_history": [],
        "images": {
            IMAGE_TYPE_LICENSE_PLATE: None,
            IMAGE_TYPE_DAMAGE: None,
            IMAGE_TYPE_FULL_CAR: None,
            IMAGE_TYPE_ID_CARD: None,
            IMAGE_TYPE_CAR_REGISTRATION: None,
            IMAGE_TYPE_DEED: None,
            IMAGE_TYPE_FINANCIAL_STATEMENT: None
        },
        "user_info": {
            "name": None,
            "phone": None,
            "plate_number": None,
            "car_brand": None,
            "damage_area": None
        }
    }
    return user_sessions[user_id]

def get_user_session(user_id):
    """ดึงเซสชันของผู้ใช้ หากไม่มีให้สร้างใหม่"""
    if user_id not in user_sessions:
        return create_user_session(user_id)
    return user_sessions[user_id]

def update_conversation_history(user_id, role, content):
    """อัพเดทประวัติการสนทนา"""
    session = get_user_session(user_id)
    session["conversation_history"].append({"role": role, "content": content})

def get_intent(user_id, user_message):
    """ใช้ Azure OpenAI เพื่อวิเคราะห์ intent ของข้อความผู้ใช้"""
    try:
        # สร้าง system prompt เพื่อกำหนดบทบาทให้ AI
        system_prompt = """คุณคือผู้ช่วยอัจฉริยะชื่อ "น้องศรี" เพศหญิง ของบริษัทศรีสวัสดิ์ คุณต้องช่วยระบุความตั้งใจ (intent) ของลูกค้าว่าเกี่ยวข้องกับหัวข้อใดต่อไปนี้:
        1. "insurance_claim" - เกี่ยวกับการเคลมประกันหรือประกันภัยรถยนต์ หรือเกิดอุบัติเหตุต้องการความช่วยเหลือ
        2. "car_loan" - เกี่ยวกับการขอสินเชื่อรถยนต์ การขอกู้เงินซื้อรถ
        3. "others" - เรื่องอื่นๆที่ไม่เกี่ยวกับสองหัวข้อข้างต้น
        
        โปรดตอบเพียง intent เดียวเท่านั้น: "insurance_claim", "car_loan", หรือ "others" ไม่ต้องใส่คำอธิบายหรือข้อความอื่นๆ"""
        
        response = client.chat.completions.create(
            model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),  # ชื่อ deployment ที่สร้างไว้
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,  # ตั้งค่าให้ต่ำเพื่อให้คำตอบแน่นอน
        )
        
        intent = response.choices[0].message.content.strip().lower()
        logger.info(f"Detected intent: {intent}")
        
        # ตรวจสอบว่า intent อยู่ในรูปแบบที่ถูกต้อง
        if intent in ["insurance_claim", "car_loan", "others"]:
            return intent
        else:
            logger.warning(f"Unexpected intent result: {intent}. Defaulting to 'others'")
            return "others"
            
    except Exception as e:
        logger.error(f"Error detecting intent: {str(e)}")
        return "others"  # กรณีเกิดข้อผิดพลาดให้กลับไปที่ intent หลัก

def get_genai_response(user_id, user_message):
    """ขอคำตอบจาก Azure OpenAI"""
    try:
        session = get_user_session(user_id)
        history = session["conversation_history"]
        
        # ปรับ history ให้อยู่ในรูปแบบที่ Azure OpenAI ต้องการ
        messages = []
        
        # เพิ่ม system prompt ตามประเภทของ bot
        if session["session_type"] == SESSION_TYPE_MAIN:
            system_prompt = """คุณคือ "น้องศรี" ผู้ช่วยอัจฉริยะเพศหญิงของบริษัทศรีสวัสดิ์ ให้บริการข้อมูลเกี่ยวกับสินเชื่อ ประกันภัย และบริการทางการเงินอื่นๆ
            คุณต้องมีความเป็นมิตร ใส่ใจ พร้อมให้ความช่วยเหลือ และใช้ภาษาไทยในการสื่อสาร
            
            หากลูกค้าต้องการใช้บริการเคลมประกัน ให้แนะนำให้กดปุ่ม "เคลมประกัน" ในเมนูด้านล่าง
            หากลูกค้าต้องการขอสินเชื่อ ให้แนะนำให้กดปุ่ม "สินเชื่อรถยนต์" ในเมนูด้านล่าง
            
            หากลูกค้าถามเรื่องที่ไม่เกี่ยวข้องกับบริการของศรีสวัสดิ์ ให้แจ้งว่าคุณไม่สามารถให้บริการในหัวข้อดังกล่าวได้"""
        elif session["session_type"] == SESSION_TYPE_INSURANCE:
            system_prompt = """คุณคือผู้ช่วยดูแลการเคลมประกันรถยนต์ของบริษัทศรีสวัสดิ์ คุณต้องขอข้อมูลจากลูกค้าเพื่อประกอบการเคลมประกัน
            คุณต้องขอรูปภาพรถที่เกิดอุบัติเหตุ โดยต้องการรูป 3 ประเภท:
            1. รูปที่เห็นป้ายทะเบียนชัดเจน
            2. รูปที่เห็นความเสียหาย (damage) ชัดเจน
            3. รูปที่เห็นรถทั้งคัน
            
            ในการขอรูป ให้ขอทีละรูป และอธิบายว่าต้องการรูปแบบไหน เมื่อได้รับรูปแล้วให้ประเมินว่ารูปดังกล่าวชัดเจนและเพียงพอหรือไม่
            หากยังไม่ชัดเจนให้ขอใหม่ 
            
            เมื่อได้รูปครบถ้วนแล้ว ให้ยืนยันกับลูกค้าว่าได้รับข้อมูลครบถ้วนและจะดำเนินการต่อไป
            
            หากลูกค้าต้องการกลับไปหน้าหลัก ให้พิมพ์คำว่า "กลับหน้าหลัก" หรือ "กลับไปหน้าหลัก" หรือให้กดปุ่มเมนูใหม่"""
        else:  # SESSION_TYPE_LOAN
            system_prompt = """คุณคือผู้ช่วยดูแลการขอสินเชื่อรถยนต์ของบริษัทศรีสวัสดิ์ คุณต้องขอข้อมูลจากลูกค้าเพื่อประกอบการขอสินเชื่อ
            ข้อมูลที่ต้องขอประกอบด้วย:
            1. ชื่อ-นามสกุล
            2. เบอร์โทรศัพท์
            3. รูปบัตรประชาชน
            4. รูปทะเบียนรถ
            5. รูปโฉนดที่ดิน (ถ้ามี)
            6. รูปหลักฐานทางการเงิน (statement) หรือสลิปเงินเดือน
            
            ให้ขอข้อมูลทีละอย่าง และเมื่อได้รับข้อมูลครบถ้วนแล้ว ให้ยืนยันกับลูกค้าว่าจะมีเจ้าหน้าที่ติดต่อกลับไปภายใน 24 ชั่วโมง
            
            หากลูกค้าต้องการกลับไปหน้าหลัก ให้พิมพ์คำว่า "กลับหน้าหลัก" หรือ "กลับไปหน้าหลัก" หรือให้กดปุ่มเมนูใหม่"""
            
        messages.append({"role": "system", "content": system_prompt})
        
        # เพิ่มประวัติการสนทนา ไม่เกิน 10 ข้อความล่าสุด
        for message in history[-10:]:
            messages.append(message)
        
        # เพิ่มข้อความล่าสุดของผู้ใช้
        messages.append({"role": "user", "content": user_message})
        
        # ส่งคำขอไปยัง Azure OpenAI
        response = client.chat.completions.create(
            model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            messages=messages,
            temperature=0.7,
        )
        
        ai_response = response.choices[0].message.content.strip()
        logger.info(f"AI response: {ai_response[:100]}...")  # Log เฉพาะส่วนต้นของคำตอบ
        
        # อัพเดทประวัติการสนทนา
        update_conversation_history(user_id, "assistant", ai_response)
        
        # ตรวจสอบหากผู้ใช้ต้องการกลับไปหน้าหลัก
        if session["session_type"] != SESSION_TYPE_MAIN and ("กลับหน้าหลัก" in user_message.lower() or "กลับไปหน้าหลัก" in user_message.lower()):
            create_user_session(user_id, SESSION_TYPE_MAIN)
            return "ยินดีต้อนรับกลับสู่หน้าหลักค่ะ มีอะไรให้น้องศรีช่วยไหมคะ"
        
        return ai_response
            
    except Exception as e:
        logger.error(f"Error getting AI response: {str(e)}")
        return f"{e}"

def extract_user_info_from_message(user_id, message_text):
    """สกัดข้อมูลสำคัญของผู้ใช้จากข้อความ (ใช้ GenAI ช่วย)"""
    try:
        session = get_user_session(user_id)
        session_type = session["session_type"]
        
        # สร้าง system prompt เพื่อสกัดข้อมูล
        if session_type == SESSION_TYPE_LOAN:
            system_prompt = """คุณคือระบบสกัดข้อมูลเพื่อการขอสินเชื่อ ให้วิเคราะห์ข้อความและสกัดข้อมูลต่อไปนี้ (ถ้ามี):
            1. ชื่อ-นามสกุล
            2. เบอร์โทรศัพท์
            
            ให้ตอบในรูปแบบ JSON เท่านั้น ตัวอย่าง:
            {
              "name": "ชื่อ นามสกุล หรือ null ถ้าไม่พบ",
              "phone": "เบอร์โทร หรือ null ถ้าไม่พบ"
            }
            
            ไม่ต้องใส่ข้อความอื่นใดนอกเหนือจาก JSON"""
            
        elif session_type == SESSION_TYPE_INSURANCE:
            system_prompt = """คุณคือระบบสกัดข้อมูลเพื่อการเคลมประกัน ให้วิเคราะห์ข้อความและสกัดข้อมูลต่อไปนี้ (ถ้ามี):
            1. ชื่อ-นามสกุล
            2. เบอร์โทรศัพท์
            3. ทะเบียนรถ
            4. ยี่ห้อและรุ่นรถ
            5. สถานที่เกิดเหตุ
            6. วันเวลาที่เกิดเหตุ
            
            ให้ตอบในรูปแบบ JSON เท่านั้น ตัวอย่าง:
            {
              "name": "ชื่อ นามสกุล หรือ null ถ้าไม่พบ",
              "phone": "เบอร์โทร หรือ null ถ้าไม่พบ",
              "plate_number": "ทะเบียนรถ หรือ null ถ้าไม่พบ",
              "car_brand": "ยี่ห้อและรุ่นรถ หรือ null ถ้าไม่พบ",
              "location": "สถานที่เกิดเหตุ หรือ null ถ้าไม่พบ",
              "timestamp": "วันเวลาที่เกิดเหตุ หรือ null ถ้าไม่พบ"
            }
            
            ไม่ต้องใส่ข้อความอื่นใดนอกเหนือจาก JSON"""
        else:
            # ไม่ต้องสกัดข้อมูลในโหมดหลัก
            return
            
        # ส่งคำขอไปยัง Azure OpenAI
        response = client.chat.completions.create(
            model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),  # ชื่อ deployment ที่สร้างไว้
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_text}
            ],
            temperature=0.1 # ตั้งค่าให้ต่ำเพื่อให้คำตอบแน่นอน
        )
        
        try:
            # แปลงข้อความตอบกลับเป็น JSON
            result = json.loads(response.choices[0].message.content.strip())
            
            # อัพเดทข้อมูลผู้ใช้
            user_info = session["user_info"]
            for key, value in result.items():
                if value is not None and value != "null":
                    user_info[key] = value
                    
            logger.info(f"Extracted user info: {result}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from extraction: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error extracting user info: {str(e)}")
        # เราไม่ต้องการให้การสกัดล้มเหลวทำให้กระบวนการหลักหยุดทำงาน
        # ดังนั้นเราจึงไม่ throw exception ต่อ

def should_switch_to_human(user_id, message_text):
    """ตรวจสอบว่าควรโอนไปเจ้าหน้าที่หรือไม่"""
    try:
        # สร้าง system prompt เพื่อตรวจสอบว่าควรโอนไปเจ้าหน้าที่หรือไม่
        system_prompt = """คุณคือระบบตรวจสอบความต้องการของลูกค้า ให้ตรวจสอบว่าข้อความนี้เป็นกรณีที่ควรโอนไปให้เจ้าหน้าที่หรือไม่ โดยประเมินจาก:
        1. ลูกค้าขอคุยกับเจ้าหน้าที่โดยตรง
        2. ลูกค้าไม่พอใจกับระบบอัตโนมัติ
        3. ลูกค้ามีปัญหาซับซ้อนที่ไม่ใช่การเคลมประกันหรือขอสินเชื่อปกติ
        4. ลูกค้าต้องการร้องเรียนหรือแจ้งปัญหาเร่งด่วน
        
        ตอบเพียง "yes" หากควรโอนไปเจ้าหน้าที่ หรือ "no" หากไม่จำเป็น"""
        
        # ส่งคำขอไปยัง Azure OpenAI
        response = client.chat.completions.create(
            model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),  # ชื่อ deployment ที่สร้างไว้
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_text}
            ],
            temperature=0.1 # ตั้งค่าให้ต่ำเพื่อให้คำตอบแน่นอน
        )
        
        result = response.choices[0].message.content.strip().lower()
        return result == "yes"
        
    except Exception as e:
        logger.error(f"Error checking human transfer: {str(e)}")
        return False  # กรณีเกิดข้อผิดพลาด ให้กลับไปใช้ bot ต่อ

@app.route("/", methods=['GET'])
def home():
    return 'LINE Bot is running!'

@app.route("/callback", methods=['POST'])
def callback():
    """Handle LINE webhook callback"""
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    logger.info("Request body: " + body[:100])  # Log เฉพาะส่วนต้นของ body เพื่อหลีกเลี่ยงการ log ข้อมูลส่วนตัว

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("Invalid signature")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    """จัดการกับข้อความที่ผู้ใช้ส่งมา"""
    try:
        user_id = event.source.user_id
        user_message = event.message.text
        logger.info(f"Received message from {user_id}: {user_message}")
        
        # ดึงเซสชันของผู้ใช้
        session = get_user_session(user_id)
        
        # บันทึกข้อความลงในประวัติ
        update_conversation_history(user_id, "user", user_message)

        # ตรวจสอบว่าเป็นข้อความจาก Rich Menu หรือไม่
        if user_message == "car_loan":
            handle_switch_to_loan(user_id, event.reply_token)
            return
        elif user_message == "insurance_claim":
            handle_switch_to_insurance(user_id, event.reply_token)
            return
        elif user_message == "home":
            handle_switch_to_main(user_id, event.reply_token)
            return
        
        # ตรวจสอบว่าควรส่งต่อให้พนักงานหรือไม่
        if should_switch_to_human(user_id, user_message):
            # กรณีต้องการส่งต่อให้พนักงาน
            transfer_message = "ขอบคุณที่ใช้บริการค่ะ ดิฉันกำลังโอนเรื่องของคุณให้เจ้าหน้าที่ ซึ่งจะติดต่อกลับโดยเร็วที่สุดค่ะ"
            update_conversation_history(user_id, "assistant", transfer_message)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=transfer_message))
            # ในที่นี้ควรมีระบบส่งการแจ้งเตือนไปยังเจ้าหน้าที่
            # ตัวอย่างเช่น ส่งอีเมล, Line Notify, หรือบันทึกลงในฐานข้อมูล
            return
        
        # พยายามสกัดข้อมูลจากข้อความ
        #extract_user_info_from_message(user_id, user_message)
        
        # จัดการตามประเภทเซสชัน
        if session["session_type"] == SESSION_TYPE_MAIN:
            # ใช้ GenAI ตรวจจับ intent
            intent = get_intent(user_id, user_message)
            
            if intent == "insurance_claim":
                # ผู้ใช้ต้องการใช้บริการเคลมประกัน
                handle_switch_to_insurance(user_id, event.reply_token)
            elif intent == "car_loan":
                # ผู้ใช้ต้องการใช้บริการสินเชื่อรถยนต์
                handle_switch_to_loan(user_id, event.reply_token)
            else:
                # intent อื่นๆ ให้ตอบด้วย GenAI หลัก
                ai_response = get_genai_response(user_id, user_message)
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=ai_response))
        
        elif session["session_type"] == SESSION_TYPE_INSURANCE:
            # ใช้ GenAI สำหรับการเคลมประกัน
            ai_response = get_genai_response(user_id, user_message)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=ai_response))
            
        elif session["session_type"] == SESSION_TYPE_LOAN:
            # ใช้ GenAI สำหรับการขอสินเชื่อ
            ai_response = get_genai_response(user_id, user_message)
            
            # ตรวจจับข้อมูลชื่อและเบอร์โทร (ตัวอย่างง่ายๆ ในทางปฏิบัติควรใช้ GenAI ช่วยตรวจจับ)
            if session["user_info"]["name"] is None and "ชื่อ" in user_message:
                # ตัวอย่างง่ายๆ ในการจับข้อมูลชื่อ
                session["user_info"]["name"] = user_message
                
            if session["user_info"]["phone"] is None and len(user_message.strip()) >= 9 and user_message.strip().isdigit():
                # ตัวอย่างง่ายๆ ในการจับข้อมูลเบอร์โทร
                session["user_info"]["phone"] = user_message.strip()
                
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=ai_response))
            
    except Exception as e:
        logger.error(f"Error handling text message: {str(e)}")
        line_bot_api.reply_message(
            event.reply_token, 
            TextSendMessage(text=f"{e}")
        )

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    """จัดการกับรูปภาพที่ผู้ใช้ส่งมา"""
    try:
        user_id = event.source.user_id
        message_id = event.message.id
        logger.info(f"Received image from {user_id}, message_id: {message_id}")
        
        # ดึงเซสชันของผู้ใช้
        session = get_user_session(user_id)
        
        # ดาวน์โหลดเนื้อหารูปภาพ
        message_content = line_bot_api.get_message_content(message_id)
        image_data = b''
        for chunk in message_content.iter_content():
            image_data += chunk
            
        # กำหนดประเภทรูปภาพตามเซสชัน
        if session["session_type"] == SESSION_TYPE_INSURANCE:
            # ตรวจสอบว่ากำลังรอรูปประเภทใด
            if session["images"][IMAGE_TYPE_LICENSE_PLATE] is None:
                image_type = IMAGE_TYPE_LICENSE_PLATE
                next_message = "ขอบคุณค่ะ\n\nต่อไปขอรูปที่เห็นความเสียหาย (damage) ชัดเจนค่ะ"
                next_image_type = IMAGE_TYPE_DAMAGE
            elif session["images"][IMAGE_TYPE_DAMAGE] is None:
                image_type = IMAGE_TYPE_DAMAGE
                next_message = "ขอบคุณค่ะ\n\nขอรูปที่เห็นรถทั้งคันค่ะ"
                next_image_type = IMAGE_TYPE_FULL_CAR
            elif session["images"][IMAGE_TYPE_FULL_CAR] is None:
                image_type = IMAGE_TYPE_FULL_CAR
                next_message = "ขอบคุณค่ะ ได้รับรูปถ่ายครบถ้วนแล้ว\n\nจากการวิเคราะห์เบื้องต้น:"
                if session["user_info"]["plate_number"]:
                    next_message += f"\n- ทะเบียนรถ: {session['user_info']['plate_number']}"
                if session["user_info"]["car_brand"]:
                    next_message += f"\n- รถยี่ห้อ: {session['user_info']['car_brand']}"
                if session["user_info"]["damage_area"]:
                    next_message += f"\n- บริเวณที่เสียหาย: {session['user_info']['damage_area']}"
                
                next_message += "\n\nเจ้าหน้าที่จะดำเนินการตรวจสอบและติดต่อกลับโดยเร็วที่สุดค่ะ"
                next_image_type = None
            else:
                # กรณีส่งรูปเพิ่มเติมหลังจากได้รับครบแล้ว
                image_type = "additional_insurance_image"
                next_message = "ขอบคุณสำหรับรูปเพิ่มเติมค่ะ ดิฉันได้บันทึกไว้แล้ว"
                next_image_type = None
                
        elif session["session_type"] == SESSION_TYPE_LOAN:
            # ตรวจสอบว่ากำลังรอรูปประเภทใด
            if session["images"][IMAGE_TYPE_ID_CARD] is None:
                image_type = IMAGE_TYPE_ID_CARD
                next_message = "ขอบคุณสำหรับบัตรประชาชนค่ะ\n\nต่อไปขอรูปทะเบียนรถค่ะ"
                next_image_type = IMAGE_TYPE_CAR_REGISTRATION
            elif session["images"][IMAGE_TYPE_CAR_REGISTRATION] is None:
                image_type = IMAGE_TYPE_CAR_REGISTRATION
                next_message = "ขอบคุณสำหรับทะเบียนรถค่ะ\n\nต่อไปขอสลิปเงินเดือนหรือหลักฐานรายได้ค่ะ"
                next_image_type = IMAGE_TYPE_FINANCIAL_STATEMENT
            elif session["images"][IMAGE_TYPE_FINANCIAL_STATEMENT] is None:
                image_type = IMAGE_TYPE_FINANCIAL_STATEMENT
                next_message = "ขอบคุณค่ะ ได้รับเอกสารครบถ้วนแล้ว\n\nเจ้าหน้าที่จะตรวจสอบข้อมูลและติดต่อกลับภายใน 24 ชั่วโมงค่ะ"
                next_image_type = None
            else:
                # กรณีส่งรูปเพิ่มเติมหลังจากได้รับครบแล้ว
                image_type = "additional_loan_image"
                next_message = "ขอบคุณสำหรับเอกสารเพิ่มเติมค่ะ ดิฉันได้บันทึกไว้แล้ว"
                next_image_type = None
                
        else:
            # กรณีไม่ได้อยู่ในโหมดที่รอรับรูป
            image_type = "general_image"
            next_message = "ขอบคุณสำหรับรูปภาพค่ะ มีอะไรให้ดิฉันช่วยเพิ่มเติมไหมคะ"
            next_image_type = None
            
        # บันทึกรูปภาพ
        result, file_path = save_image(image_data, user_id, image_type)
        
        # ส่งข้อความตอบกลับ
        if result["success"]:
            # อัพเดทประวัติการสนทนา (เก็บรูปภาพเป็นข้อความ)
            update_conversation_history(user_id, "user", f"[ส่งรูปภาพ: {image_type}]")
            update_conversation_history(user_id, "assistant", next_message)
            
            # ส่งข้อความตอบกลับพร้อมตัวอย่างรูปถัดไป (ถ้ามี)
            if next_image_type:
                example_img_path = f"example_images/{next_image_type}_example.jpg"
                if os.path.exists(example_img_path):
                    messages = [
                        TextSendMessage(text=next_message),
                        ImageSendMessage(
                            original_content_url=f"https://yourdomain.com/{example_img_path}",
                            preview_image_url=f"https://yourdomain.com/{example_img_path}"
                        )
                    ]
                    line_bot_api.reply_message(event.reply_token, messages)
                else:
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=next_message))
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=next_message))
        else:
            # กรณีมีข้อผิดพลาดในการบันทึกรูป
            error_message = "ขออภัยค่ะ มีข้อผิดพลาดในการประมวลผลรูปภาพ กรุณาลองอีกครั้ง"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=error_message))
            
    except Exception as e:
        logger.error(f"Error handling image message: {str(e)}")
        line_bot_api.reply_message(
            event.reply_token, 
            TextSendMessage(text="ขออภัยค่ะ มีข้อผิดพลาดในการประมวลผลรูปภาพ กรุณาลองใหม่อีกครั้ง")
        )

def analyze_image(image_path, image_type):
    """วิเคราะห์รูปภาพด้วย API (ในตัวอย่างนี้เป็นแค่ Mock API)"""
    try:
        # ในตัวอย่างนี้จะเป็นการ mock ผลลัพธ์
        # ในการใช้งานจริงคุณต้องเชื่อมต่อกับ API จริงๆ
        
        if image_type == IMAGE_TYPE_LICENSE_PLATE:
            # Mock result for license plate detection
            return {
                "success": True,
                "plate_number": "กข 1234",
                "province": "กรุงเทพมหานคร",
                "confidence": 0.95
            }
        elif image_type == IMAGE_TYPE_DAMAGE:
            # Mock result for damage detection
            return {
                "success": True,
                "damage_areas": ["front bumper", "headlight"],
                "severity": "moderate",
                "confidence": 0.88
            }
        elif image_type == IMAGE_TYPE_FULL_CAR:
            # Mock result for car brand/type detection
            return {
                "success": True,
                "vehicle_type": "car",
                "brand": "Toyota",
                "model": "Camry",
                "confidence": 0.92
            }
        else:
            # สำหรับประเภทรูปภาพอื่นๆ
            return {
                "success": True,
                "document_type": image_type,
                "is_valid": True,
                "confidence": 0.9
            }
            
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def save_image(image_content, user_id, image_type):
    """บันทึกรูปภาพที่ได้รับจาก LINE"""
    try:
        # สร้างไดเรกทอรีสำหรับผู้ใช้แต่ละคน (ถ้ายังไม่มี)
        user_folder = os.path.join(UPLOAD_FOLDER, user_id)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
        
        # สร้างชื่อไฟล์ที่ไม่ซ้ำกัน
        filename = f"{image_type}_{uuid.uuid4()}.jpg"
        file_path = os.path.join(user_folder, filename)
        
        # บันทึกไฟล์
        with open(file_path, 'wb') as f:
            f.write(image_content)
            
        logger.info(f"Saved image to {file_path}")
        
        # อัพเดทข้อมูลใน session
        session = get_user_session(user_id)
        session["images"][image_type] = file_path
        
        # วิเคราะห์รูปภาพ (ถ้าเป็นประเภทที่ต้องวิเคราะห์)
        if image_type in [IMAGE_TYPE_LICENSE_PLATE, IMAGE_TYPE_DAMAGE, IMAGE_TYPE_FULL_CAR]:
            result = analyze_image(file_path, image_type)
            
            # อัพเดทข้อมูลจากการวิเคราะห์
            if result["success"]:
                if image_type == IMAGE_TYPE_LICENSE_PLATE and "plate_number" in result:
                    session["user_info"]["plate_number"] = result["plate_number"]
                elif image_type == IMAGE_TYPE_DAMAGE and "damage_areas" in result:
                    session["user_info"]["damage_area"] = ", ".join(result["damage_areas"])
                elif image_type == IMAGE_TYPE_FULL_CAR and "brand" in result:
                    session["user_info"]["car_brand"] = f"{result['brand']} {result.get('model', '')}"
                    
            return result, file_path
            
        return {"success": True}, file_path
            
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        return {"success": False, "error": str(e)}, None

def handle_switch_to_insurance(user_id, reply_token):
    """ผู้ใช้ต้องการสลับไปโหมดเคลมประกัน"""
    try:
        # สร้างเซสชันใหม่สำหรับการเคลมประกัน
        session = create_user_session(user_id, SESSION_TYPE_INSURANCE)
        
        # สร้างข้อความต้อนรับ
        initial_message = "ยินดีต้อนรับสู่บริการเคลมประกันรถยนต์ของศรีสวัสดิ์ค่ะ\n\nดิฉันจะขอรูปถ่ายเพื่อประกอบการเคลมนะคะ\n\nขอรูปที่เห็นป้ายทะเบียนรถชัดเจนก่อนเลยค่ะ"
        
        # บันทึกข้อความลงในประวัติ
        update_conversation_history(user_id, "assistant", initial_message)
        
        # ส่งข้อความพร้อมรูปตัวอย่าง
        example_img_url = "https://s3-ap-southeast-1.amazonaws.com/prakunrod-public/images/sample-car-pics/car-pic-front.jpg"
        messages = [
            TextSendMessage(text=initial_message),
            ImageSendMessage(
                original_content_url=example_img_url,
                preview_image_url=example_img_url 
            )
        ]
        line_bot_api.reply_message(reply_token, messages)
            
    except Exception as e:
        logger.error(f"Error switching to insurance: {str(e)}")
        line_bot_api.reply_message(
            reply_token, 
            TextSendMessage(text=f"{e}")
        )

def handle_switch_to_loan(user_id, reply_token):
    """ผู้ใช้ต้องการสลับไปโหมดขอสินเชื่อ"""
    try:
        # สร้างเซสชันใหม่สำหรับการขอสินเชื่อ
        session = create_user_session(user_id, SESSION_TYPE_LOAN)
        
        # สร้างข้อความต้อนรับ
        initial_message = "ยินดีต้อนรับสู่บริการสินเชื่อรถยนต์ของศรีสวัสดิ์ค่ะ\n\nดิฉันจะขอข้อมูลเพื่อประกอบการพิจารณาสินเชื่อนะคะ\n\nกรุณาแจ้งชื่อ-นามสกุลของท่านก่อนเลยค่ะ"
        
        # บันทึกข้อความลงในประวัติ
        update_conversation_history(user_id, "assistant", initial_message)
        
        # ส่งข้อความ
        line_bot_api.reply_message(reply_token, TextSendMessage(text=initial_message))
            
    except Exception as e:
        logger.error(f"Error switching to loan: {str(e)}")
        line_bot_api.reply_message(
            reply_token, 
            TextSendMessage(text=f"{e}")
        )

def handle_switch_to_main(user_id, reply_token):
    """ผู้ใช้ต้องการกลับไปหน้าหลัก"""
    try:
        # สร้างเซสชันใหม่สำหรับการสนทนาหลัก
        session = create_user_session(user_id, SESSION_TYPE_MAIN)
        
        # สร้างข้อความต้อนรับ
        initial_message = "ยินดีต้อนรับกลับมานะคะ ดิฉันน้องศรีพร้อมให้บริการคุณแล้วค่ะ\n\nท่านสามารถเลือกใช้บริการผ่าน Rich Menu ด้านล่างหรือพิมพ์ข้อความมาได้เลยค่ะ"
        
        # บันทึกข้อความลงในประวัติ
        update_conversation_history(user_id, "assistant", initial_message)
        
        # ส่งข้อความ
        line_bot_api.reply_message(reply_token, TextSendMessage(text=initial_message))
            
    except Exception as e:
        logger.error(f"Error switching to main: {str(e)}")
        line_bot_api.reply_message(
            reply_token, 
            TextSendMessage(text=f"{e}")
        )

def setup_ngrok():
    """Set up ngrok tunnel"""
    try:
        # เช็คว่า NGROK_AUTH_TOKEN ถูกตั้งค่าหรือไม่
        if os.getenv('NGROK_AUTH_TOKEN'):
            ngrok.set_auth_token(os.getenv('NGROK_AUTH_TOKEN'))
            
        public_url = ngrok.connect(5000).public_url
        logger.info(f"ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000\"")
        logger.info(f"LINE webhook URL: {public_url}/callback")
        return public_url
    except Exception as e:
        logger.error(f"Error setting up ngrok: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting Flask application...")
        app.run(port=5000,debug=True)
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")