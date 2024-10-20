# GuardianVision

## Overview
**GuardianVision** is a real-time crime detection system that leverages advanced AI techniques, including SlowFast networks and Convolutional Neural Networks (CNN), to analyze video footage for identifying and classifying various criminal activities. The system is designed to enhance public safety and provide timely alerts to law enforcement agencies.

## Problem Statement
In 2023, the FBI recorded a rate of **363.8 violent crimes per 100,000 people**, highlighting the pressing need for effective crime detection solutions. Despite being a significant issue, many instances of crime often go unnoticed or unreported. **GuardianVision** aims to address this gap by providing a proactive approach to crime detection, ensuring swift responses to potential threats.

## Solution Overview
Our model utilizes a combination of:

- **SlowFast Network**: 
  - Processes video data at two different frame rates (slow and fast pathways).
  - Captures both fine spatial details and rapid temporal changes.
  - Optimized for identifying suspicious behaviors in various environments.

- **Convolutional Neural Network (ResNet50)**:
  - Deep residual learning framework with **50 layers**.
  - Utilizes skip connections for enhanced training efficiency.
  - Trained with over **25 million parameters** for robust performance.

The model has been trained on the **IBM Z Linux One Platform**, leveraging powerful computational resources to optimize detection algorithms.

## Features
- Real-time video analysis for crime detection.
- Classification of various criminal activities, including:
  - Abuse
  - Arrest
  - Arson
  - Assault
  - Burglary
  - Explosion
  - Fighting
  - Normal
  - Road Accident
  - Robbery
  - Shooting
  - Shoplifting
  - Stealing

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GuardianVision.git
   cd GuardianVision
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
- Upload a video file through the dashboard.
- The system processes the video and displays real-time predictions for identified criminal activities.

## Team Members
Our project was made possible by the collaborative efforts of a dedicated team:

- **Arpit Singh Gautam**: Led the development of AI and machine learning algorithms to enhance crime detection capabilities.
- **Swastik Vaish**: Spearheaded the implementation of the Streamlit application, creating an intuitive user interface for the dashboard.
- **Raghav Kapoor**: Conducted thorough data collection and preprocessing to ensure high-quality input for the model training.
- **Lakshya Goel**: Designed and developed the PowerPoint presentation, effectively communicating our project's vision and findings.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
