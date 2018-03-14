
import {KNNImageClassifier} from 'deeplearn-knn-image-classifier';
import * as dl from 'deeplearn';

// properties
var content, webcam, tracker, raf, eyeRect, interval, oldData, curData, cData, currentCorrelation, blinks;

// canvas and contexts
var originalCanvas, originalContext, trackerCanvas, trackerContext, eyeCanvas, eyeContext, bwCanvas, bwContext, thCanvas, thContext, oldCanvas, oldContext, curCanvas, curContext, cCanvas, cContext;

// dom elements
var correlationPercentage, blinksDetected;

var settings = {
    padding: 3,
    contrast: 3,
    brightness: 0.3,
    threshold: 80,
    minCorrelation: 0.17,
};

// Number of classes to classify
const NUM_CLASSES = 2;
// Webcam Image size. Must be 227. 
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;

var knn, training=-1;
var infoTexts = [];

function init() {
    
    content = document.getElementById('content');

    // adds listeners to activate and deactivate on iframe focus
    window.addEventListener('focus', start, false);
    window.addEventListener('blur', stop, false);

    // instanciate our Webcam class
    webcam = new Webcam(320, 240);

    // tracker
    tracker = new clm.tracker();
    tracker.init(window.pModel);

    // eye rect
    eyeRect = {
        x: 0, y: 0,
        w: 0, h: 0,
    };

    // original canvas and context
    originalCanvas = document.getElementById('originalCanvas');
    originalContext = originalCanvas.getContext('2d');

    // tracker canvas and context
    trackerCanvas = document.getElementById('trackerCanvas');
    trackerContext = trackerCanvas.getContext('2d');

    // eye canvas and context
    eyeCanvas = document.getElementById('eyeCanvas');
    eyeContext = eyeCanvas.getContext('2d');

    // Initiate deeplearn.js math and knn classifier objects
    knn = new KNNImageClassifier(NUM_CLASSES, TOPK);
    knn.load()
    createButtons();
    setInterval(train, 200);
}

function createButtons(){
    // Create training buttons and info texts    
    for(let i=0;i<NUM_CLASSES; i++){
      const div = document.getElementById('content');
      // document.body.appendChild(div);
      // div.style.marginBottom = '10px';

      // Create training button
      const button = document.createElement('button')
      button.innerText = "Train "+i;
      div.appendChild(button);

      // Listen for mouse events when clicking the button
      button.addEventListener('mousedown', () => training = i);
      button.addEventListener('mouseup'  , () => training = -1);
      
      // Create info text
      const infoText = document.createElement('span')
      infoText.innerText = " No examples added ";
      div.appendChild(infoText);
      infoTexts.push(infoText);
    }
}


function start(e) {
    e.preventDefault();
    document.body.className = 'active';

    webcam.start()
    tracker.start(webcam.domElement);

    raf = requestAnimationFrame(update);
    //interval = setInterval(correlation, 100);

    blinks = 0;
}

function stop(e) {
    e.preventDefault();
    document.body.className = '';

    webcam.stop();
    tracker.stop();

    cancelAnimationFrame(raf);
    clearInterval(interval);

    blinks = 0;
}

function update() {
    
    raf = requestAnimationFrame(update);

    originalContext.clearRect(0, 0, originalContext.canvas.width, originalContext.canvas.height);
    trackerContext.clearRect(0, 0, trackerContext.canvas.width, trackerContext.canvas.height);

    // draw video element to canvas
    originalContext.drawImage(webcam.domElement, 0, 0, originalContext.canvas.width, originalContext.canvas.height);

    // draw tracker to canvas
    trackerContext.drawImage(webcam.domElement, 0, 0, trackerContext.canvas.width, trackerContext.canvas.height);
    tracker.draw(trackerCanvas);

    // extract right eye data
    var pos = tracker.getCurrentPosition();
    if (pos) {
        var angle = Math.atan2(pos[25][1]-pos[23][1], pos[25][0]-pos[23][0]);
        eyeRect.x = pos[23][0];
        eyeRect.y = pos[24][1];
        eyeRect.w = pos[25][0] - pos[23][0];
        eyeRect.h = pos[26][1] - pos[24][1];
        
        var d = Math.max(eyeRect.w, eyeRect.h)+settings.padding;
        var cx = (pos[23][0]+pos[25][0])/2.0;
        var cy = (pos[24][1]+pos[26][1])/2.0;
        
        // draw eye
        eyeContext.save();
        var w = eyeContext.canvas.width;
        var h = eyeContext.canvas.height;
        eyeContext.translate(w/2.0, w/2.0);
        eyeContext.rotate(-angle);

        var kx = webcam.videoWidth  /  originalContext.canvas.width ;
        var ky = webcam.videoHeight /  originalContext.canvas.height;
        eyeContext.drawImage(webcam.domElement,
            (cx-d/2.0)*kx, (cy-d/2.0)*ky,
            d*kx, d*ky,
            -w/2.0, -h/2.0,
            eyeContext.canvas.width,
            eyeContext.canvas.height);
        eyeContext.restore();
    }
}

function train(){
    
  // Get image data from video element
  if(!eyeCanvas) return;

  const image = dl.fromPixels(eyeContext.canvas);
  
  // Train class if one of the buttons is held down
  if(training != -1){
    // Add current image to classifier
    console.log(image, training);
    knn.addImage(image, training);
  }
  
  // If any examples have been added, run predict
  const exampleCount = knn.getClassExampleCount();
  if(Math.max(...exampleCount) > 0){
    knn.predictClass(image)
    .then((res)=>{
      for(let i=0;i<NUM_CLASSES; i++){
        // Make the predicted class bold
        if(res.classIndex == i){
          infoTexts[i].style.fontWeight = 'bold';
          infoTexts[i].style.fontColor = '#0C5';
        } else {
          infoTexts[i].style.fontWeight = 'normal';
          infoTexts[i].style.fontColor = '#FFF';
        }

        // Update info text
        if(exampleCount[i] > 0){
          infoTexts[i].innerText = ` ${exampleCount[i]} examples - ${res.confidences[i]*100}%`
        }
      }
    })
    // Dispose image when done
    .then(()=> image.dispose())
  } else {
    image.dispose()
  }
}
init();
