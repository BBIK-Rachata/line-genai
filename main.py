from dotenv import load_dotenv
import os
import tempfile
from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    RichMenu, RichMenuArea, RichMenuBounds, RichMenuSize,
    MessageAction, ImageMessage
)
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

# Check if environment variables are set
if not os.getenv('LINE_CHANNEL_ACCESS_TOKEN'):
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN is not set in .env file")

if not os.getenv('LINE_CHANNEL_SECRET'):
    raise ValueError("LINE_CHANNEL_SECRET is not set in .env file")

# Initialize Flask app
app = Flask(__name__)

# Initialize LINE API
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))

# สร้างโฟลเดอร์สำหรับเก็บรูปภาพ (ถ้ายังไม่มี)
UPLOAD_FOLDER = 'received_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created directory for images: {UPLOAD_FOLDER}")

def create_rich_menu():
    """Create and upload a rich menu"""
    try:
        rich_menu_to_create = RichMenu(
            size=RichMenuSize(width=2500, height=1686),
            selected=False,
            name="Financial Services Menu",
            chat_bar_text="บริการทางการเงิน",
            areas=[
                RichMenuArea(
                    bounds=RichMenuBounds(x=0, y=0, width=1250, height=1686),
                    action=MessageAction(
                        label='Car Loan',
                        text='car_loan'
                    )
                ),
                RichMenuArea(
                    bounds=RichMenuBounds(x=1250, y=0, width=1250, height=1686),
                    action=MessageAction(
                        label='Insurance Claim',
                        text='insurance_claim'
                    )
                )
            ]
        )
        
        # Create rich menu
        rich_menu_id = line_bot_api.create_rich_menu(rich_menu_to_create)
        logger.info(f"Created rich menu with ID: {rich_menu_id}")
        
        # Check if image file exists
        image_path = "rich_menu_image.png"
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Upload rich menu image
        with open(image_path, 'rb') as f:
            content = f.read()
            line_bot_api.set_rich_menu_image(rich_menu_id, "image/png", content)
        logger.info("Uploaded rich menu image successfully")
        
        # Set as default rich menu
        line_bot_api.set_default_rich_menu(rich_menu_id)
        logger.info("Set as default rich menu")
        
        return rich_menu_id
        
    except Exception as e:
        logger.error(f"Error creating rich menu: {str(e)}")
        raise

def handle_car_loan(event):
    """Handle car loan service"""
    try:
        message = TextSendMessage(
            text="ยินดีต้อนรับสู่บริการสินเชื่อรถยนต์\nกรุณาแจ้งความประสงค์ที่ต้องการ:\n1. ขอสินเชื่อรถใหม่\n2. ขอสินเชื่อรถมือสอง\n3. ติดต่อเจ้าหน้าที่"
        )
        line_bot_api.reply_message(event.reply_token, message)
    except Exception as e:
        logger.error(f"Error handling car loan: {str(e)}")
        # ส่งข้อความแจ้งเตือนผู้ใช้แทนการ raise error
        error_message = TextSendMessage(text="ขออภัย มีข้อผิดพลาดเกิดขึ้น กรุณาลองใหม่อีกครั้ง")
        line_bot_api.reply_message(event.reply_token, error_message)

def handle_insurance_claim(event):
    """Handle insurance claim service"""
    try:
        message = TextSendMessage(
            text="ยินดีต้อนรับสู่บริการเคลมประกัน 24 ชั่วโมง\nกรุณาเลือกบริการ:\n1. แจ้งเคลมประกัน\n2. ติดตามสถานะการเคลม\n3. ติดต่อเจ้าหน้าที่ฉุกเฉิน"
        )
        line_bot_api.reply_message(event.reply_token, message)
    except Exception as e:
        logger.error(f"Error handling insurance claim: {str(e)}")
        error_message = TextSendMessage(text="ขออภัย มีข้อผิดพลาดเกิดขึ้น กรุณาลองใหม่อีกครั้ง")
        line_bot_api.reply_message(event.reply_token, error_message)

def handle_insurance_claim_image(event):
    """Request user to send image for insurance claim"""
    try:
        message = TextSendMessage(
            text="กรุณาถ่ายรูปความเสียหายและส่งมาให้เราเพื่อประกอบการเคลมประกัน"
        )
        line_bot_api.reply_message(event.reply_token, message)
    except Exception as e:
        logger.error(f"Error handling insurance claim image request: {str(e)}")
        error_message = TextSendMessage(text="ขออภัย มีข้อผิดพลาดเกิดขึ้น กรุณาลองใหม่อีกครั้ง")
        line_bot_api.reply_message(event.reply_token, error_message)

@app.route("/", methods=['GET'])
def home():
    return 'LINE Bot is running!'

@app.route("/callback", methods=['POST'])
def callback():
    """Handle LINE webhook callback"""
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("Invalid signature")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    try:
        user_id = event.source.user_id
        user_message = event.message.text
        logger.info(f"Received message from {user_id}: {user_message}")
        
        # Add debug logging
        logger.info(f"Message type: {type(user_message)}")
        logger.info(f"Message lower: {user_message.lower()}")
        
        if user_message.lower() == 'car_loan':
            logger.info("Handling car loan request")
            handle_car_loan(event)
        elif user_message.lower() == 'insurance_claim':
            logger.info("Handling insurance claim request")
            handle_insurance_claim(event)
        elif user_message == '1' and hasattr(event, 'reply_token'):
            # ตัวอย่างการจัดการกับการเลือกเมนูย่อย
            handle_insurance_claim_image(event)
        else:
            logger.info("Message didn't match any conditions")
            message = TextSendMessage(text="กรุณาเลือกบริการจาก Rich Menu ด้านล่าง")
            line_bot_api.reply_message(event.reply_token, message)
        
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}")
        message = TextSendMessage(text="ขออภัย มีข้อผิดพลาดเกิดขึ้น กรุณาลองใหม่อีกครั้ง")
        line_bot_api.reply_message(event.reply_token, message)

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    """Handle incoming image messages"""
    try:
        # รับข้อมูลของรูปภาพ
        message_id = event.message.id
        user_id = event.source.user_id
        logger.info(f"Received image from {user_id} with message_id: {message_id}")
        
        # สร้างไฟล์ชั่วคราวสำหรับบันทึกรูปภาพ
        tf = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = tf.name
        
        # ดาวน์โหลดรูปภาพจาก LINE server
        message_content = line_bot_api.get_message_content(message_id)
        with open(temp_file_path, 'wb') as f:
            for chunk in message_content.iter_content():
                f.write(chunk)
        
        # บันทึกรูปภาพในโฟลเดอร์ถาวร (อาจใช้ user_id เป็นส่วนหนึ่งของชื่อไฟล์)
        image_path = os.path.join(UPLOAD_FOLDER, f"{user_id}_{message_id}.jpg")
        os.rename(temp_file_path, image_path)
        logger.info(f"Saved image to {image_path}")
        
        # ประมวลผลรูปภาพ (ตัวอย่าง)
        # สามารถเพิ่มโค้ดสำหรับการประมวลผลรูปภาพได้ที่นี่
        # เช่น การวิเคราะห์รูปภาพด้วย Computer Vision API หรืออื่นๆ
        
        # ตอบกลับไปยัง user
        line_bot_api.reply_message(
            event.reply_token,
            [
                TextSendMessage(text="ขอบคุณที่ส่งรูปภาพมาให้ เราได้รับไว้เรียบร้อยแล้ว"),
                TextSendMessage(text="เจ้าหน้าที่จะตรวจสอบและติดต่อกลับในไม่ช้า")
            ]
        )
    except Exception as e:
        logger.error(f"Error handling image message: {str(e)}")
        message = TextSendMessage(text="ขออภัย มีข้อผิดพลาดในการรับรูปภาพ กรุณาลองใหม่อีกครั้ง")
        line_bot_api.reply_message(event.reply_token, message)

def setup_ngrok():
    """Set up ngrok tunnel"""
    try:
        public_url = ngrok.connect(5000).public_url
        logger.info(f"ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000\"")
        logger.info(f"LINE webhook URL: {public_url}/callback")
        return public_url
    except Exception as e:
        logger.error(f"Error setting up ngrok: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        public_url = setup_ngrok()
        rich_menu_id = create_rich_menu()
        logger.info("Starting Flask application...")
        app.run(port=5000)
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")