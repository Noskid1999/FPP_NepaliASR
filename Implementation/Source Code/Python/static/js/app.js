// set up basic variables for app

const record = document.querySelector("#record-btn");
const stop = document.querySelector("#stop-btn");
const preview = document.querySelector("#preview-btn");
const transcribe = document.querySelector("#transcribe-btn");
const voiceQuote = document.querySelector("#voice-quote");
const imgContainer = document.querySelector("#img-recorder");
const canvas = document.querySelector("#dictaphone-visualizer");
const recordInfo = document.querySelector(".record-info");
const restart = document.querySelector("#restart-btn");
var reqBlob = null;

// disable stop button while not recording

// stop.disabled = true;

// visualiser setup - create web audio api context and canvas

let audioCtx;
const canvasCtx = canvas.getContext("2d");

//main block for doing the audio recording

if (navigator.mediaDevices.getUserMedia) {
  console.log("getUserMedia supported.");

  const constraints = { audio: true };

  let onSuccess = function (stream) {
    var audioContext;
    /*  assign to gumStream for later use  */
    var gumStream;
    /* use the stream */
    var input;
    var rec;

    visualize(stream);

    record.onclick = startRecording;

    function startRecording() {
      audioContext = new AudioContext();
      /*  assign to gumStream for later use  */
      gumStream = stream;
      /* use the stream */
      input = audioContext.createMediaStreamSource(stream);
      rec = new Recorder(input, { numChannels: 1 });

      rec.record();

      console.log("recorder started");

      record.classList.add("d-none");
      stop.classList.remove("d-none");
      transcribe.classList.add("d-none");
      preview.classList.add("d-none");
      voiceQuote.classList.add("d-none");
      imgContainer.classList.add("d-none");
      canvas.classList.remove("d-none");
    }
    stop.onclick = function () {
      stop.classList.add("d-none");
      preview.classList.remove("d-none");
      transcribe.classList.remove("d-none");

      // Stop recording
      rec.stop();
      //stop microphone access
      gumStream.getAudioTracks()[0].stop();
      rec.exportWAV(createDownloadLink);

      console.log("recorder stopped");
    };
    function createDownloadLink(blob) {
      console.log("recorder stopped");

      reqBlob = blob;

      const audio = document.createElement("audio");
      audio.controls = true;
      const audioURL = window.URL.createObjectURL(blob);
      audio.src = audioURL;

      document.querySelector("body").appendChild(audio);
    }
    transcribe.onclick = function () {
      var fd = new FormData();

      fd.append("file", reqBlob, "test.wav");

      var xhr = new XMLHttpRequest();
      xhr.onreadystatechange = function () {};

      xhr.open("POST", "http://127.0.0.1:5000/speechrecognitiondefault", true);
      xhr.onload = function () {
        var response = JSON.parse(xhr.response);
        Object.keys(response).forEach((key) => {
          recordInfo.innerHTML += `<div>
                                  <span>${key} :</span><strong>${response[key]}</strong>
                              </div>`;
        });
        restart.classList.remove("d-none");
      };
      xhr.send(fd);
    };

    restart.onclick = function () {
      window.location = window.location;
    };
  };

  let onError = function (err) {
    console.log("The following error occured: " + err);
  };

  navigator.mediaDevices.getUserMedia(constraints).then(onSuccess, onError);
} else {
  console.log("getUserMedia not supported on your browser!");
}

function visualize(stream) {
  if (!audioCtx) {
    audioCtx = new AudioContext();
  }

  const source = audioCtx.createMediaStreamSource(stream);

  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  source.connect(analyser);
  //analyser.connect(audioCtx.destination);

  draw();

  function draw() {
    const WIDTH = canvas.width;
    const HEIGHT = canvas.height;

    requestAnimationFrame(draw);

    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = "#F7EEFF";
    canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = "rgb(0, 0, 0)";

    canvasCtx.beginPath();

    let sliceWidth = (WIDTH * 1.0) / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      let v = dataArray[i] / 128.0;
      let y = (v * HEIGHT) / 2;

      if (i === 0) {
        canvasCtx.moveTo(x, y);
      } else {
        canvasCtx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
  }
}
