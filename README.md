*LeafGuard AI is an AI-powered web application that screens plant leaf images for potential diseases using a Convolutional Neural Network (CNN) with MobileNetV2 transfer learning. It provides predicted disease classes, confidence scores, educational treatment suggestions, and scan history through a responsive web interface.
*Features
  1.AI-Based Disease Screening — Classifies uploaded leaf images using TensorFlow/Keras and MobileNetV2.
  2.Image Upload & Preview — Supports leaf image uploads with drag-and-drop functionality.
  3.Prediction Results — Displays the predicted class and model confidence score.
  4.Smart Guidance — Provides general follow-up and educational treatment suggestions.
  5.Scan History — Stores previous scans using an SQLite database.
  6.Input Validation — Validates uploaded file types and file sizes.
  7.Health Monitoring — Includes a /health endpoint to check model availability.
  8.Responsive UI — Designed for a clean user experience across different screen sizes.
*Tech Stack
  1.Frontend: HTML, CSS, JavaScript
  2.Backend: Python, Flask
  3.AI/ML: TensorFlow, Keras, MobileNetV2, CNN, Transfer Learning
  4.Database: SQLite
  5.Image Processing: Pillow , MobileNetV2-based CNN
*How It Works
  1.Upload a plant leaf image through the web interface.
  2.The application preprocesses the image into RGB format and resizes it to 224 × 224 pixels.
  3.The MobileNetV2-based CNN processes the image and predicts a class.
  4.The application displays the predicted class, confidence score, and relevant next-step guidance.
  5.The scan is saved in the SQLite database for future reference.
