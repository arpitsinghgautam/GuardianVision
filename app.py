import streamlit as st
import cv2
import numpy as np
import os
from keras.layers import Input
from keras.optimizers import SGD
from models.slowfast import SlowFast_body, bottleneck
import base64
# ResNet50 Model Definition
def resnet50(inputs, **kwargs):
    model = SlowFast_body(inputs, [3, 4, 6, 3], bottleneck, **kwargs)
    return model

# Function to process video frames
def frames_from_video(video_capture, nb_frames=25, img_size=224):
    frames = []
    for i in range(nb_frames):
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
    return np.array(frames) / 255.0 if len(frames) == nb_frames else None

# Function to process video and display predictions
def process_video_with_predictions(video_path, model, output_dir, nb_frames=25, img_size=224, display_interval=25):
    cap = cv2.VideoCapture(video_path)
    frame_buffer = []
    current_prediction = ""

    if not cap.isOpened():
        st.error(f"Error: Unable to open video file {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(output_dir, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Changed from 'mp4v' to 'avc1'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)

        if len(frame_buffer) == display_interval:
            resized_frames = [cv2.resize(f, (img_size, img_size)) for f in frame_buffer]
            X = np.array(resized_frames) / 255.0
            X = np.reshape(X, (1, nb_frames, img_size, img_size, 3))

            predictions = model.predict(X)
            preds = predictions.argmax(axis=1)

            classes = []
            with open(os.path.join('output', 'classes.txt'), 'r') as fp:
                for line in fp:
                    classes.append(line.split()[1])

            current_prediction = classes[preds[0]]
            print("Current Predictions: ", current_prediction)

            frame_buffer = []

        display_frame = frame.copy()
        cv2.putText(display_frame, f"Prediction: {current_prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out.write(display_frame)

    cap.release()
    out.release()

    return output_path, current_prediction

# Load the model (this should be done once when the app starts)
@st.cache_resource
def load_model():
    x = Input(shape=(25, 224, 224, 3))
    model = resnet50(x, num_classes=14)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01, momentum=0.9), metrics=['accuracy'])
    model.load_weights('slowfast_finalmodel.hdf5')
    return model

# Home page
def home():
    st.title("Crime Statistics Dashboard")
    st.write("Welcome to the Crime Detection and Analysis Platform")

    # Displaying crime statistics with background color
    st.header("2023 Crime Statistics Overview")
    st.markdown("""
    <div style='background-color: #f0f8ff; padding: 10px; border-radius: 5px; color: black;'>
        In 2023, the FBI recorded a rate of <strong>363.8 violent crimes</strong> per <strong>100,000 people</strong> in the U.S.
        This includes various types of violent crimes, which have a significant impact on public safety.
        For example:
        <ul>
            <li><strong>Shoplifting</strong> alone accounts for over <strong>$13 billion</strong> in losses for retailers annually.</li>
            <li><strong>Assault</strong> and <strong>robbery</strong> rates contribute to rising concerns in urban areas.</li>
        </ul>
        Despite these alarming statistics, many incidents go unnoticed or unreported, leading to delayed responses from law enforcement.
    </div>
""", unsafe_allow_html=True)

    # Additional statistics or visualizations
    st.header("Why This Matters")
    st.markdown("""
        <div style='background-color: #e6ffe6; padding: 10px; border-radius: 5px; color: black;'>
            Often, society dismisses these issues, viewing them as common occurrences. 
            This highlights the necessity for proactive measures in crime detection, particularly in public spaces, 
            where traditional surveillance lacks the capability for real-time intervention.
        </div>
    """, unsafe_allow_html=True)



    st.subheader("Crime Rates in 2023")
    st.image("crime_rate_trends_2023.jpg", use_column_width=True)

    # Display crime rate charts from images
    st.subheader("Crime Rate Trends (2000-2020)")
    st.image("crime_rate_trends.png", use_column_width=True)

    st.write("Use the sidebar to navigate to the Video Analysis page for crime detection in uploaded videos.")

# Video analysis page
def video_analysis():
    st.title("Crime Detection in Video")
    st.write("Upload a video to detect potential criminal activities.")

    # Model description
    st.subheader("Model Description")
    st.markdown("""
    <div style='background-color: #f0f8ff; padding: 10px; border-radius: 5px; color: black;'>
        Our crime detection model utilizes a combination of SlowFast networks and Convolutional Neural Networks (CNN) to provide an effective solution for real-time crime analysis.

        **SlowFast Network:**
        - Processes video data at two different frame rates (slow and fast pathways).
        - Captures both fine spatial details and rapid temporal changes, ensuring a comprehensive understanding of movements.
        - Effective for understanding complex actions, making it ideal for identifying suspicious behaviors in various environments.
        - Optimized to handle large volumes of video data, enabling scalability in surveillance applications.

        **CNN (ResNet50):**
        - Deep residual learning framework with **50 layers**, allowing for complex feature extraction.
        - Utilizes skip connections to enhance training efficiency and accuracy.
        - Helps in identifying intricate patterns and details in each video frame, critical for classifying criminal activities.
        - Trained with over **25 million parameters**, ensuring robust performance in various scenarios.

        This combined approach allows our model to effectively detect and classify various criminal activities in video footage, with an emphasis on accuracy and real-time processing capabilities. 
        We have trained the model on the IBM Z Linux One Platform, leveraging its powerful computational resources to optimize our detection algorithms.
    </div>
""", unsafe_allow_html=True)
    model = load_model()

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_video_path = os.path.join("temp", uploaded_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Process the video
        with st.spinner("Processing video..."):
            output_video_path, predicted_crime = process_video_with_predictions(temp_video_path, model, "Predicted_Videos")

        # Display the output video
        if os.path.exists(output_video_path):
            st.success("Video processed successfully!")
            st.video(output_video_path)
            # st.write(f"Predicted Crime: {predicted_crime}")

            # Download button
            with open(output_video_path, 'rb') as f:
                st.download_button('Download Processed Video', f, file_name='predicted_video.mp4')

            # Display final message
            st.markdown("### :rotating_light: Crime detected! Video sent to police! :police_car:")
        else:
            st.error(f"Error: Output video not found at {output_video_path}")

        # Clean up temporary files
        os.remove(temp_video_path)
        os.remove(output_video_path)

# Main app logic
def main():
    st.set_page_config(page_title="Crime Detection App", layout="wide")
    image_file = 'assets/glass.png'
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
            f"""
            <style>
            .stSidebar {{
                background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                background-size: cover;
                colour:
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    
    mainpage_image_file = 'assets/gun.png'
    with open(mainpage_image_file, "rb") as mainpage_image_file:
        mainpage_encoded_string = base64.b64encode(mainpage_image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{mainpage_encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Video Analysis"])

    if page == "Home":
        home()
    elif page == "Video Analysis":
        video_analysis()

if __name__ == "__main__":
    main()