import streamlit as st
from ultralytics import YOLO
import cv2
import base64
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from streamlit_option_menu import option_menu

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

st.set_page_config(
    layout="wide",
    page_title="TrafficScogg"
)

model = YOLO('Model/train_40epoch_3/detect/train/weights/best.pt')

class_names = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
    "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
    "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons", 
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop", "No vehicles", 
    "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution", "Dangerous curve to the left", 
    "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", 
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", 
    "Beware of ice/snow", "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead", 
    "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left", 
    "Roundabout mandatory", "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
]

traffic_sign_descriptions = {
    "Speed limit (20km/h)": "Tanda ini menunjukkan batas kecepatan 20 kilometer per jam.",
    "Speed limit (30km/h)": "Tanda ini menunjukkan batas kecepatan 30 kilometer per jam.",
    "Speed limit (50km/h)": "Tanda ini menunjukkan batas kecepatan 50 kilometer per jam.",
    "Speed limit (60km/h)": "Tanda ini menunjukkan batas kecepatan 60 kilometer per jam.",
    "Speed limit (70km/h)": "Tanda ini menunjukkan batas kecepatan 70 kilometer per jam.",
    "Speed limit (80km/h)": "Tanda ini menunjukkan batas kecepatan 80 kilometer per jam.",
    "End of speed limit (80km/h)": "Tanda ini menandakan akhir dari batas kecepatan 80 kilometer per jam.",
    "Speed limit (100km/h)": "Tanda ini menunjukkan batas kecepatan 100 kilometer per jam.",
    "Speed limit (120km/h)": "Tanda ini menunjukkan batas kecepatan 120 kilometer per jam.",
    "No passing": "Tanda ini melarang pengendara untuk mendahului.",
    "No passing for vehicles over 3.5 metric tons": "Tanda ini melarang kendaraan dengan berat lebih dari 3,5 ton untuk mendahului.",
    "Right-of-way at the next intersection": "Tanda ini memberikan hak utama kepada pengendara di persimpangan berikutnya.",
    "Priority road": "Tanda ini menunjukkan bahwa jalan ini adalah jalan utama.",
    "Yield": "Tanda ini mengharuskan pengendara untuk memberikan jalan kepada pengendara lain.",
    "Stop": "Tanda ini mengharuskan pengendara untuk berhenti.",
    "No vehicles": "Tanda ini melarang semua jenis kendaraan.",
    "Vehicles over 3.5 metric tons prohibited": "Tanda ini melarang kendaraan dengan berat lebih dari 3,5 ton.",
    "No entry": "Tanda ini melarang pengendara untuk memasuki area tertentu.",
    "General caution": "Tanda ini menunjukkan peringatan umum untuk berhati-hati.",
    "Dangerous curve to the left": "Tanda ini menunjukkan adanya tikungan berbahaya ke kiri.",
    "Dangerous curve to the right": "Tanda ini menunjukkan adanya tikungan berbahaya ke kanan.",
    "Double curve": "Tanda ini menunjukkan adanya dua tikungan berturut-turut.",
    "Bumpy road": "Tanda ini menunjukkan jalan yang bergelombang.",
    "Slippery road": "Tanda ini menunjukkan jalan yang licin.",
    "Road narrows on the right": "Tanda ini menunjukkan jalan menyempit di sebelah kanan.",
    "Road work": "Tanda ini menunjukkan adanya pekerjaan jalan.",
    "Traffic signals": "Tanda ini menunjukkan adanya lampu lalu lintas.",
    "Pedestrians": "Tanda ini menunjukkan adanya pejalan kaki.",
    "Children crossing": "Tanda ini menunjukkan adanya anak-anak yang menyeberang.",
    "Bicycles crossing": "Tanda ini menunjukkan adanya sepeda yang menyeberang.",
    "Beware of ice/snow": "Tanda ini menunjukkan adanya es atau salju.",
    "Wild animals crossing": "Tanda ini menunjukkan adanya hewan liar yang menyeberang.",
    "End of all speed and passing limits": "Tanda ini menandakan akhir dari semua batas kecepatan dan larangan mendahului.",
    "Turn right ahead": "Tanda ini menunjukkan adanya belokan ke kanan di depan.",
    "Turn left ahead": "Tanda ini menunjukkan adanya belokan ke kiri di depan.",
    "Ahead only": "Tanda ini menunjukkan bahwa pengendara hanya bisa melanjutkan lurus ke depan.",
    "Go straight or right": "Tanda ini menunjukkan bahwa pengendara bisa melanjutkan lurus atau belok ke kanan.",
    "Go straight or left": "Tanda ini menunjukkan bahwa pengendara bisa melanjutkan lurus atau belok ke kiri.",
    "Keep right": "Tanda ini menunjukkan bahwa pengendara harus tetap di jalur kanan.",
    "Keep left": "Tanda ini menunjukkan bahwa pengendara harus tetap di jalur kiri.",
    "Roundabout mandatory": "Tanda ini menunjukkan adanya bundaran yang harus dilalui.",
    "End of no passing": "Tanda ini menandakan akhir dari zona larangan mendahului.",
    "End of no passing by vehicles over 3.5 metric tons": "Tanda ini menandakan akhir dari zona larangan mendahului untuk kendaraan dengan berat lebih dari 3,5 ton."
}

selected = option_menu(
    menu_title=None,  
    options=["Home", "Projects", "Contact"],  
    icons=["house", "book", "envelope"], 
    menu_icon="cast",
    default_index=0, 
    orientation="horizontal",
)

# Kelas untuk memproses frame video
THRESHOLD = 0.5  # Definisikan threshold yang diinginkan di sini

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_text = "Click result button to see the result."
        self.description = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Logika pengenalan rambu lalu lintas
        results = model(img)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])

                # Cek apakah confidence melebihi threshold
                if conf > THRESHOLD:
                    label = f'{class_names[cls]}: {conf:.2f}'

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    self.result_text = label
                    self.description = f'This sign indicates {class_names[cls].lower()}.'

        return frame.from_ndarray(img, format="bgr24")

    def get_result(self):
        return self.result_text

    def get_description(self):
        return self.description

# Inisialisasi state untuk video processor
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = None

# Fungsi untuk menampilkan custom CSS
def load_css():
    css = """
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #0e1117;
            color: #f4f4f4;
        }
        .container {
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        #traffic-signs {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
        }
        .marquee {
            width: 100%;
            overflow: hidden;
            position: relative;
            margin-bottom: 20px;
        }
        .marquee-content {
            display: flex;
            width: 200%;
            animation: marquee 30s linear infinite;
        }
        .marquee img {
            width: 50px;
            margin: 10px;
        }
        .reverse .marquee-content {
            animation-direction: reverse;
        }
        .centered-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 60%;
            margin: 0 auto;
        }
        @keyframes marquee {
            0% {
                transform: translateX(0);
            }
            100% {
                transform: translateX(-50%);
            }
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Fungsi untuk mengonversi gambar ke base64
def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
    
def img_to_base(img):
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

# Fungsi untuk menampilkan custom HTML
def load_html():
    row1_images = ''.join([f'<img src="data:image/png;base64,{img_to_base64(f"static/images/{i}.png")}" alt="Traffic Sign">' for i in range(1, 22)])
    row2_images = ''.join([f'<img src="data:image/png;base64,{img_to_base64(f"static/images/{i}.png")}" alt="Traffic Sign">' for i in range(22, 43)])

    # Menggabungkan baris gambar untuk infinite loop
    row1_images += row1_images
    row2_images += row2_images

    import streamlit as st

    st.markdown(
        "<div style='text-align: center; color: white; font-size: 42px; font-weight: 700; max-width: 65%; margin: 70px auto 0;'>\
        Traffic Sign Recognition</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='text-align: center; color: white; font-size: 17px; font-weight: 400; max-width: 65%; margin: 20px auto;'>\
        Welcome to our website, a center for traffic sign recognition. We utilize advanced AI technology to quickly and accurately identify and interpret various types and shapes of traffic signs. Our system is designed to learn and adapt to different types of traffic signs, allowing us to provide fast and precise information. Thank you.\
        </div>",
        unsafe_allow_html=True
    )


    html = f"""
    <div class="container"> 
        <!-- 
        <section class="intro">
            <h1>Traffic Sign Recognitions</h1>
            <p>Welcome to our website, a center for traffic sign recognition. We utilize advanced AI technology to quickly and accurately identify and interpret various types and shapes of traffic signs. Our system is designed to learn and adapt to different types of traffic signs, allowing us to provide fast and precise information. To try our application, you can press the button below. Thank you.</p>
            <button>Try it now</button>
        </section>
        -->
        <section id="traffic-signs">
            <div class="marquee">
                <div class="marquee-content">
                    <!-- Baris pertama gambar -->
                    {row1_images}
                </div>
            </div>
            <div class="marquee reverse">
                <div class="marquee-content">
                    <!-- Baris kedua gambar -->
                    {row2_images}
                </div>
            </div>
        </section>
    </div>
    """
    # Menampilkan HTML dengan gambar
    st.markdown(html, unsafe_allow_html=True)

# Layout untuk halaman Home
if selected == "Home":
    load_css()
    load_html()

# Layout untuk halaman Projects
if selected == "Projects":
    st.markdown('<div id="video_feed_section" class="centered-content">', unsafe_allow_html=True)

    cols = st.columns([1, 2, 1])  # Kolom pertama dan ketiga untuk memberikan jarak, kolom kedua untuk konten

    with cols[1]:  # Mengatur agar konten berada di tengah
        st.markdown("<p style='text-align: center; color: white; font-size: 24px; font-weight:500'>Recognition Test</p>", unsafe_allow_html=True)
        
        # Dropdown untuk memilih jenis inputan data
        input_type = st.selectbox("Pilih jenis input:", ["Camera", "File/Gambar"])
        
        if input_type == "Camera":
            # ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_processor_factory=VideoProcessor)
            ctx = webrtc_streamer(
                key="example",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=VideoProcessor,
                async_processing=True,
                translations={
                    "start": "Start",
                    "stop": "Result",
                }
            )
            st.markdown("<p style='text-align: center; color: white; font-size: 24px; font-weight:500'>Recognition Result</p>", unsafe_allow_html=True)

            if ctx.video_processor:
                st.session_state.video_processor = ctx.video_processor
        elif input_type == "File/Gambar":
            st.session_state.video_processor = None
            uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
            
            st.markdown("<p style='text-align: center; color: white; font-size: 24px; font-weight:500'>Recognition Results</p>", unsafe_allow_html=True)

            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                # st.image(img, channels="BGR")
                
                # Logika pengenalan rambu lalu lintas pada gambar
                results = model(img)
                label = ""
                description = ""
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        cls = int(box.cls[0])

                        label = f'{class_names[cls]}: {conf:.2f}'
                        description = f'{traffic_sign_descriptions.get(class_names[cls])}'
                        # description = f'This sign indicates {class_names[cls].lower()}.'

                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # st.image(img, channels="BGR")
                
                if label:
                    image_base = img_to_base(img)
                    st.markdown(f"<div style='text-align: center'><img src='data:image/png;base64,{image_base}' width='100'></div>", unsafe_allow_html=True)
                    # st.write(f"Detected Traffic Sign: {label}")
                    # st.write(f"Description: {description}")
                    st.markdown(f"<p style='text-align: center; color: white;'>{label}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: white;'>{description}</p>", unsafe_allow_html=True)

        if st.session_state.video_processor:
            result = st.session_state.video_processor.get_result()
            description = st.session_state.video_processor.get_description()
            # st.write(f"**Nama Rambu:** {result}")
            # st.write(f"**Penjelasan:** {description}")
            st.markdown(f"<p style='text-align: center; color: white;'>{result}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: white;'>{description}</p>", unsafe_allow_html=True)
        elif input_type == "File/Gambar" and uploaded_file is None:
            # st.write("Please upload an image to see the recognition results.")
            st.markdown(f"<p style='text-align: center; margin-top: 5; '>Please upload an image to see the recognition results.</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Layout untuk halaman Contact
if selected == "Contact":
    st.markdown("<h4 style='text-align: center; color: white; margin-top:70px;'>Contact Information</h4>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white; margin-bottom: 50px;'>Berikut adalah detail informasi dari kami selaku pembuat aplikasi ini</h6>", unsafe_allow_html=True)
    
    team_members = [
        {"name": "Rio Irawan", "email": "rio22003@mail.unpad.ac.id", "image": "static/images/rio.jpg"},
        {"name": "Angga Prasetyo", "email": "angga22004@mail.unpad.ac.id", "image": "static/images/angga.png"},
        {"name": "Alif Al Husaini", "email": "alif22001@mail.unpad.ac.id", "image": "static/images/alif.jpg"},
        {"name": "Giast Ahmad", "email": "giast22001@mail.unpad.ac.id", "image": "static/images/giast.jpg"},
        {"name": "M. Danendra", "email": "danen22001@mail.unpad.ac.id", "image": "static/images/danen.jpg"},
    ]

    cols = st.columns(len(team_members))

    for i, member in enumerate(team_members):
        image_base64 = img_to_base64(member["image"])
        with cols[i]:
            st.markdown(f"<div style='text-align: center'><img src='data:image/png;base64,{image_base64}' width='100' height='150'></div>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; margin: 0.5rem 0 '> {member['name']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; margin: 0 '> {member['email']}</p>", unsafe_allow_html=True)
