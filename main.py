import io
from typing import List, Tuple

import numpy as np
import cv2
from fastapi import FastAPI, File, Response
from PIL import Image
import uvicorn
from ultralytics import YOLO


class DetectionModel:
    def __init__(self, model_path: str, classes: List[str]):
        self.model_path = model_path
        self.classes = classes
        self.model = self._load_model()

    def _load_model(self) -> cv2.dnn_Net:
        net = cv2.dnn.readNet(self.model_path)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def _extract_output(
        self,
        preds: np.ndarray,
        image_shape: Tuple[int, int],
        input_shape: Tuple[int, int],
        score: float = 0.1,
        nms: float = 0.0,
        confidence: float = 0.0,
    ) -> dict:
        class_ids, confs, boxes = [], [], []

        image_height, image_width = image_shape
        input_height, input_width = input_shape
        x_factor = image_width / input_width
        y_factor = image_height / input_height

        rows = preds[0].shape[0]
        for i in range(rows):
            row = preds[0][i]
            conf = row[4]

            classes_score = row[4:]
            _, _, _, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]

            if classes_score[class_id] > score:
                confs.append(conf)
                label = self.classes[int(class_id)]
                class_ids.append(label)

                # Extract boxes
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

        r_class_ids, r_confs, r_boxes = [], [], []
        indexes = cv2.dnn.NMSBoxes(boxes, confs, confidence, nms)
        for i in indexes:
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i] * 100)
            r_boxes.append(boxes[i].tolist())

        return {
            'boxes': r_boxes,
            'confidences': r_confs,
            'classes': r_class_ids,
        }

    def __call__(
        self,
        image: np.ndarray,
        width: int = 640,
        height: int = 640,
        score: float = 0.1,
        nms: float = 0.0,
        confidence: float = 0.0,
    ) -> dict:
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (width, height), swapRB=True, crop=False
        )
        self.model.setInput(blob)
        preds = self.model.forward()
        preds = preds.transpose((0, 2, 1))

        # Extract output
        results = self._extract_output(
            preds=preds,
            image_shape=image.shape[:2],
            input_shape=(height, width),
            score=score,
            nms=nms,
            confidence=confidence,
        )
        return results


# Initialize detection models
detection = DetectionModel(
    model_path=r"C:\Users\SunanthineePiyarat\Desktop\SRISAWAD_detection\model\car_motor\car_motor.onnx",
    classes=['car', 'motorbike'],
)

detection2 = DetectionModel(
    model_path=r"C:\Users\SunanthineePiyarat\Desktop\SRISAWAD_detection\model\car_brand\car_brand.onnx",
    classes=['BMW', 'BYD', 'Honda', 'Mazda', 'MercedesBenz', 'Perodua', 'Proton', 'Tesla', 'Toyota'],
)

detection3 = DetectionModel(
    model_path=r"C:\Users\SunanthineePiyarat\Desktop\SRISAWAD_detection\model\car_license\car_license.onnx",
    classes=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A01', 'A02', 'A04', 'A06', 'A07', 'A08', 'A09', 'A10', 'A12', 'A13', 'A14', 'A16', 'A18',
        'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A28', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37',
        'A38', 'A39', 'A40', 'A41', 'A42', 'A43', 'A44', 'ATG', 'AYA', 'BKK', 'BKN', 'BRM', 'CBI', 'CCO', 'CMI', 'CNT', 'CPM', 'CPN',
        'CRI', 'CTI', 'KBI', 'KKN', 'KPT', 'KRI', 'KSN', 'LEI', 'LPG', 'LPN', 'LRI', 'MDH', 'MKM', 'NAN', 'NBI', 'NBP', 'NKI', 'NMA',
        'NPM', 'NPT', 'NRT', 'NSN', 'NYK', 'PBI', 'PCK', 'PKN', 'PKT', 'PLG', 'PLK', 'PNB', 'PRE', 'PRI', 'PTE', 'PYO', 'RBR', 'RET',
        'RYG', 'SBR', 'SKA', 'SKM', 'SKN', 'SKW', 'SNI', 'SNK', 'SPB', 'SPK', 'SRI', 'SRN', 'SSK', 'STI', 'TAK', 'TRT', 'UBN', 'UDN',
        'UTI', 'YLA', 'YST',
    ]
)

detection4 = DetectionModel(
    model_path=r"C:\Users\SunanthineePiyarat\Desktop\SRISAWAD_detection\model\car_license_plate\car_license_plate.onnx",
    classes=['license-plate'],
)


thai_characters = {
    "A01": "ก", "A02": "ข", "A03": "ฃ", "A04": "ค", "A05": "ฅ", "A06": "ฆ", "A07": "ง", "A08": "จ",
    "A09": "ฉ", "A10": "ช", "A11": "ซ", "A12": "ฌ", "A13": "ญ", "A14": "ฎ", "A15": "ฏ", "A16": "ฐ",
    "A17": "ฑ", "A18": "ฒ", "A19": "ณ", "A20": "ด", "A21": "ต", "A22": "ถ", "A23": "ท", "A24": "ธ",
    "A25": "น", "A26": "บ", "A27": "ป", "A28": "ผ", "A29": "ฝ", "A30": "พ", "A31": "ฟ", "A32": "ภ",
    "A33": "ม", "A34": "ย", "A35": "ร", "A36": "ล", "A37": "ว", "A38": "ศ", "A39": "ษ", "A40": "ส",
    "A41": "ห", "A42": "ฬ", "A43": "อ", "A44": "ฮ"
}

province_fullnames = {
    "ATG": "อ่างทอง", "AYA": "อยุธยา", "BKK": "กรุงเทพมหานคร", "BKN": "บึงกาฬ", "BRM": "บุรีรัมย์",
    "CBI": "ชัยนาท", "CCO": "ชลบุรี", "CMI": "เชียงใหม่", "CNT": "เชียงราย", "CPM": "ชัยภูมิ",
    "CPN": "ขอนแก่น", "CRI": "กระบี่", "CTI": "กาญจนบุรี", "KBI": "กาฬสินธุ์", "KKN": "ขอนแก่น",
    "KPT": "กระบี่", "KRI": "กาญจนบุรี", "KSN": "กาฬสินธุ์", "LEI": "เลย", "LPG": "ลพบุรี",
    "LPN": "ลำปาง", "LRI": "ลำพูน", "MDH": "มหาสารคาม", "MKM": "มุกดาหาร", "NAN": "น่าน",
    "NBI": "นครบาล", "NBP": "นครปฐม", "NKI": "นครราชสีมา", "NMA": "นครสวรรค์", "NPM": "น่าน",
    "NPT": "น่าน", "NRT": "นราธิวาส", "NSN": "นราธิวาส", "NYK": "น่าน", "PBI": "พะเยา",
    "PCK": "เพชรบุรี", "PKN": "พิษณุโลก", "PKT": "พิษณุโลก", "PLG": "ประจวบคีรีขันธ์", "PLK": "ปทุมธานี",
    "PNB": "พัทลุง", "PRE": "เพชรบูรณ์", "PRI": "ปราจีนบุรี", "PTE": "พัทลุง", "PYO": "ปัตตานี",
    "RBR": "ระยอง", "RET": "ราชบุรี", "RYG": "ร้อยเอ็ด", "SBR": "สระบุรี", "SKA": "สกลนคร",
    "SKM": "สมุทรปราการ", "SKN": "สมุทรสงคราม", "SKW": "สมุทรสาคร", "SNI": "สตูล", "SNK": "สุพรรณบุรี",
    "SPB": "สุราษฎร์ธานี", "SPK": "สงขลา", "SRI": "สุรินทร์", "SRN": "สุพรรณบุรี", "SSK": "สงขลา",
    "STI": "สตูล", "TAK": "ตาก", "TRT": "ตราด", "UBN": "อุบลราชธานี", "UDN": "อุดรธานี",
    "UTI": "อุตรดิตถ์", "YLA": "ยะลา", "YST": "ยโสธร",
}

province_list = [
    "ATG", "AYA", "BKK", "BKN", "BRM", "CBI", "CCO", "CMI", "CNT", "CPM",
    "CPN", "CRI", "CTI", "KBI", "KKN", "KPT", "KRI", "KSN", "LEI", "LPG",
    "LPN", "LRI", "MDH", "MKM", "NAN", "NBI", "NBP", "NKI", "NMA", "NPM",
    "NPT", "NRT", "NSN", "NYK", "PBI", "PCK", "PKN", "PKT", "PLG", "PLK", 
    "PNB", "PRE", "PRI", "PTE", "PYO", "RBR", "RET", "RYG", "SBR", "SKA", 
    "SKM", "SKN", "SKW", "SNI", "SNK", "SPB", "SPK", "SRI", "SRN", "SSK",
    "STI", "TAK", "TRT", "UBN", "UDN", "UTI", "YLA", "YST"
]

# Convert image array to PNG format
def convert_image(arr_img: np.ndarray) -> tuple:
    im = Image.fromarray(arr_img)
    with io.BytesIO() as buf:
        im.save(buf, format='PNG')
        im_bytes = buf.getvalue()
    headers = {"Content-Disposition": "inline; filename='test.png'"}
    return im_bytes, headers


# Initialize FastAPI app
app = FastAPI()


@app.post('/detection')
def post_detection(file: bytes = File(...)):
    """
    Endpoint for detecting objects in an image.
    Returns a processed image with detections.
    """
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = np.array(image)
    image = image[:, :, ::-1].copy()

    # Load YOLO model
    model = YOLO(r"C:\Users\SunanthineePiyarat\Desktop\SRISAWAD_detection\model\car_demage\car_demage.pt")
    results = model(image, conf=0.25)  
    res_img = results[0].plot()
    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

    res_img_bytes, headers = convert_image(res_img)
    return Response(content=res_img_bytes, headers=headers, media_type='image/png')


@app.post('/vehicle')
def vehicle(file: bytes = File(...)):
    """
    Endpoint for vehicle detection.
    Returns the results of vehicle and brand detection.
    """
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = np.array(image)
    image = image[:, :, ::-1].copy()

    # Detect vehicles and brands
    result = detection(image)
    result2 = detection2(image)

    return result, result2


@app.post('/license')
def license(file: bytes = File(...)):
    """
    Endpoint for license plate detection.
    Returns sorted license plate information.
    """
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = np.array(image)
    image = image[:, :, ::-1].copy()

    result4 = detection4(image)
    boxes = result4["boxes"]

    if not boxes:
        return {"message": "No license plate detected"}

    # Assume only one license plate is detected
    x, y, w, h = boxes[0]
    cropped = image[y:y + h, x:x + w]  # Crop the image with the box dimensions

    # Run detection on the cropped license plate
    result3 = detection3(cropped)

    boxes = result3['boxes']
    confs = result3['confidences']
    classes = result3['classes']

    y_positions = [box[1] for box in boxes]
    y_threshold = (max(y_positions) + min(y_positions)) / 2  

    top_row = []  
    bottom_row = []  

    for box, conf, cls in zip(boxes, confs, classes):
        if cls in thai_characters:
            cls = thai_characters[cls]
        
        if cls in province_list:
            full_province_name = province_fullnames.get(cls, cls)
            bottom_row.append((box, conf, full_province_name))  
        else:
            top_row.append((box, conf, cls))  

    # Sort rows based on the x-coordinate of the bounding boxes
    top_row_sorted = sorted(top_row, key=lambda x: x[0][0])
    bottom_row_sorted = sorted(bottom_row, key=lambda x: x[0][0])

    sorted_result3 = {
        'top_row': {
            'boxes': [box for box, _, _ in top_row_sorted],
            'confidences': [conf for _, conf, _ in top_row_sorted],
            'classes': [cls for _, _, cls in top_row_sorted]
        },
        'bottom_row': {
            'boxes': [box for box, _, _ in bottom_row_sorted],
            'confidences': [conf for _, conf, _ in bottom_row_sorted],
            'classes': [cls for _, _, cls in bottom_row_sorted]
        }
    }

    return sorted_result3

if __name__ == '__main__':

    # Start the FastAPI server
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
