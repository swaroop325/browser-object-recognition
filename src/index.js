const tf = require('@tensorflow/tfjs')
const labels = require('./../models/coco_model/labels.json')
const ColorHash = require('color-hash')
const colorHash = new ColorHash()
var synth = window.speechSynthesis;
var available_voices;
var utter = new SpeechSynthesisUtterance();
const MODEL_URL = 'models/coco_model/web_model/tensorflowjs_model.pb'
const WEIGHTS_URL = 'models/coco_model/web_model/weights_manifest.json'

const THRESHOLD = 0.3
const MAX_OBJECT_COUNT = Infinity

function VideoObjectDetection(opts = {}) {
  if (!(this instanceof VideoObjectDetection)) return new VideoObjectDetection(opts)

  if (!opts.inputVideo) throw new Error('inputVideo option is required')

  this._inputVideo = opts.inputVideo

  tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL).then(async (model) => {
    this._model = model
    this._execute()
  })
}

function videoToTensor(inputVideo) {
  if (!inputVideo.HAVE_METADATA) {
    return new Promise((resolve, reject) => {
      inputVideo.addEventListener('loadedmetadata', () => { // wait for the video to load
        resolve(videoToTensor(inputVideo))
      })
    })
  }

  // Tensorflow uses the wrong width/height properties
  Object.defineProperty(inputVideo, 'width', {
    configurable: true,
    get: function () { return inputVideo.videoWidth }
  })
  Object.defineProperty(inputVideo, 'height', {
    configurable: true,
    get: function () { return inputVideo.videoHeight }
  })
  return tf.tidy(() => {
    let img = tf.fromPixels(inputVideo)
    return img.expandDims(0)
  })
}

VideoObjectDetection.prototype._execute = async function () {
  // take a snapshot of the video, turning into a tensor
  const imageTensor = await videoToTensor(this._inputVideo)

  // execute the model, getting our result tensors
  const result = await this._model.executeAsync({ image_tensor: imageTensor })

  // get the number of detections
  const objectCount = Math.min(await result[3].data(), MAX_OBJECT_COUNT)

  // fetch the data for each tensor, skipping excess data
  const [boxes, confidences, classes] = await Promise.all(result.slice(0, 3).map(tensor => {
    let sliceRange = tensor.rank > 2 ? [1, objectCount, 4] : [1, objectCount] // boxes tensor is a different size
    return tensor.slice(0, sliceRange).data()
  }))

  // remove loading message
  const loading = document.querySelector('#loading')
  if (loading) loading.parentElement.removeChild(loading)

  // clear boxes from last frame
  this._clearBoxes()

  // for each object, show a bounding box and text
  for (var i = 0; i < objectCount; i++) {
    if (confidences[i] > THRESHOLD) {
      this._showBox(
        labels[classes[i]],
        confidences[i],
        boxes[i * 4],
        boxes[i * 4 + 1],
        boxes[i * 4 + 2],
        boxes[i * 4 + 3]
      )
    }
  }

  // dispose of the tensors we created
  result.map(tensor => tensor.dispose())
  imageTensor.dispose()

  // start the next frame!
  this._execute()
}

VideoObjectDetection.prototype._clearBoxes = function () {
  const boxes = document.querySelectorAll('.object-box')
    ;[].forEach.call(boxes, (box) => {
      box.parentElement.removeChild(box)
    })
}

VideoObjectDetection.prototype._showBox = function (label, confidence, y1, x1, y2, x2) {


  // list of languages is probably not loaded, wait for it
  if (window.speechSynthesis.getVoices().length == 0) {
    window.speechSynthesis.addEventListener('voiceschanged', function () {
      available_voices = window.speechSynthesis.getVoices();
    });
  }
  else {
    available_voices = window.speechSynthesis.getVoices();
  }
  const div = document.createElement('div')
  div.className = 'object-box'
  const color = colorHash.hex(label)
  div.style.border = 'solid 5px ' + color
  div.style.color = color

  const rect = this._inputVideo.getBoundingClientRect()

  div.style.left = rect.x + x1 * rect.width
  div.style.top = rect.y + y1 * rect.height
  div.style.width = (x2 - x1) * rect.width
  div.style.height = (y2 - y1) * rect.height
  function textToSpeech(label) {
    // get all voices that browser offers
    var available_voices = window.speechSynthesis.getVoices();
  
    // this will hold an english voice
    var english_voice = '';
  
    // find voice by language locale "en-US"
    // if not then select the first voice
    for (var i = 0; i < available_voices.length; i++) {
      if (available_voices[i].lang === 'en-US') {
        english_voice = available_voices[i];
        break;
      }
    }
    if (english_voice === '')
      english_voice = available_voices[0];
  
    // new SpeechSynthesisUtterance object
    var utter = new SpeechSynthesisUtterance();
    utter.rate = 1;
    utter.pitch = 0.5;
    utter.text = label;
    utter.voice = english_voice;
  
    // event after text has been spoken
    utter.onend = function () {
      alert('Speech has finished');
    }
  
    // speak
    window.speechSynthesis.speak(utter);
  }
  const text = document.createElement('p')
  text.innerHTML = label + '<br>' + Math.floor(confidence * 100) + '%'
  div.appendChild(text)
  if (parseInt(div.style.width) < 50) {
    text.style.fontSize = '10px'
    div.style.border = 'solid 2px ' + color
  }

  this._inputVideo.parentElement.appendChild(div)
}

module.exports = VideoObjectDetection
