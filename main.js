// properties
var content, webcam, tracker, raf, eyeRect, interval, oldData, curData, cData, currentCorrelation, blinks;

// canvas and contexts
var originalCanvas, originalContext, trackerCanvas, trackerContext, eyeCanvas, eyeContext, bwCanvas, bwContext, thCanvas, thContext, oldCanvas, oldContext, curCanvas, curContext, cCanvas, cContext;

// dom elements
var correlationPercentage, blinksDetected;

var settings = {
    padding: 5,
    contrast: 3,
    brightness: 0.3,
    threshold: 80,
    minCorrelation: 0.17,
};


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

    // black & white canvas and context
    bwCanvas = document.getElementById('bwCanvas');
    bwContext = bwCanvas.getContext('2d');

    // threshold canvas and context
    thCanvas = document.getElementById('thCanvas');
    thContext = thCanvas.getContext('2d');

    // old canvas and context
    oldCanvas = document.getElementById('oldCanvas');
    oldContext = oldCanvas.getContext('2d');

    // cur canvas and context
    curCanvas = document.getElementById('curCanvas');
    curContext = curCanvas.getContext('2d');

    // correlation canvas and context
    cCanvas = document.getElementById('cCanvas');
    cContext = cCanvas.getContext('2d');

    // correlation percentage dom
    correlationPercentage = document.getElementById('correlationPercentage');

    // blinks detected dom
    blinksDetected = document.getElementById('blinksDetected');
}

function start(e) {
    e.preventDefault();
    document.body.className = 'active';

    webcam.start()
    tracker.start(webcam.domElement);

    raf = requestAnimationFrame(update);
    interval = setInterval(correlation, 100);

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
    bwContext.clearRect(0, 0, bwContext.canvas.width, bwContext.canvas.height);
    thContext.clearRect(0, 0, thContext.canvas.width, thContext.canvas.height);
    oldContext.clearRect(0, 0, oldContext.canvas.width, oldContext.canvas.height);
    curContext.clearRect(0, 0, curContext.canvas.width, curContext.canvas.height);
    cContext.clearRect(0, 0, cContext.canvas.width, cContext.canvas.height);

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

        // black and white
        var data = CanvasFilters.getPixels(eyeCanvas);
        var grayscale = CanvasFilters.grayscale(data, settings.contrast, settings.brightness);

        bwContext.putImageData(grayscale, 0, 0);

        // threshold
        var data = CanvasFilters.getPixels(eyeCanvas);
        var grayscale = CanvasFilters.grayscale(data, settings.contrast, settings.brightness);
        var threshold = CanvasFilters.threshold(grayscale, settings.threshold);

        thContext.putImageData(threshold, 0, 0);

        // draw old data set
        if (oldData) {
            oldContext.putImageData(oldData, 0, 0);
        }

        // draw cur data set
        if (curData) {
            curContext.putImageData(curData, 0, 0);
        }

        // draw correlation
        if (cData) {
            cContext.putImageData(cData, 0, 0);
        }
    }
}

function correlation() {

    if (curData) {
        oldData = curData;
    }

    curData = thContext.getImageData(0, 0, thContext.canvas.width, thContext.canvas.height);

    // correlation data
    cData = cContext.createImageData(cContext.canvas.width, cContext.canvas.height);

    var count = 0;
    if (oldData && curData) {
        var total = curData.data.length;
        for (var i = 0; i < total; i += 4) {
            cData.data[i + 3] = 255;
            if (curData.data[i] !== oldData.data[i]) {
                cData.data[i] = 255;
                count++;
            }
        }
    }

    currentCorrelation = count / (cContext.canvas.width * cContext.canvas.height);

    correlationPercentage.innerHTML = parseFloat(currentCorrelation).toFixed(2) + '%';

    if (currentCorrelation > settings.minCorrelation) {
        blinks++;
    }

    blinksDetected.innerHTML = blinks + ' blinks detected';
}

init();

