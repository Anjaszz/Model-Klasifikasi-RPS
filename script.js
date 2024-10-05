let model;

async function loadModel() {
    // Load the TensorFlow.js model from the 'model_js' folder
    model = await tf.loadLayersModel('model_js/model.json');
    console.log('Model loaded');
}

async function predict() {
    const imageElement = document.getElementById('imagePreview');
    
    // Preprocess the image to the required size
    let tensorImg = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([100, 150]) // Resize to model's input shape
        .toFloat()
        .div(tf.scalar(255.0)) // Normalize the image
        .expandDims(); // Add batch dimension

    // Perform the prediction
    let predictions = await model.predict(tensorImg).data();
    let classes = ['Rock', 'Paper', 'Scissors'];
    
    // Get the highest prediction
    let maxPredictionIndex = predictions.indexOf(Math.max(...predictions));
    
    // Display the result
    document.getElementById('predictionResult').innerText = `Prediction: ${classes[maxPredictionIndex]}`;
}

function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function(){
        const output = document.getElementById('imagePreview');
        output.src = reader.result;
    };
    reader.readAsDataURL(event.target.files[0]);
}

// Load model when page loads
window.onload = loadModel;
