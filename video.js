const tamano = 400;
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const otrocanvas = document.getElementById("otrocanvas");
const ctx = canvas.getContext("2d");
let currentStream = null;
const facingMode = "user";

let modelo = null;

(async() => {
  console.log("Cargando modelo...");
  modelo = await tf.loadLayersModel("model.json");
  console.log("Modelo cargado");
})();

window.onload = function() {
  mostrarCamara();
}

function mostrarCamara() {
  const opciones = {
    audio: false,
    video: {width: tamano, height: tamano}
  }

  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia(opciones)
             .then(function(stream) {
                 currentStream = stream;
                 video.srcObject = currentStream;
                 procesarCamara();
                 predecir();
             })
             .catch(function(err) {
                 alert("No se pudo utilizar la camara :(");
                 console.log(err);
                 alert(err);
             })
  } else {
    alert("No existe la funcion getUserMedia")
  }
}

function _cambiarCamara() {
  if (currentStream) {
    currentStream.getTracks().forEach(track => {
      track.stop();
    });
  }

  facingMode = facingMode == "user" ? "environment" : "user";

  const opciones = {
    audio: false,
    video: {
      facingMode: facingMode,
      width: tamano,
      height: tamano,
    }
  }

  navigator.mediaDevices.getUserMedia(opciones)
           .then(function(stream) {
             currentStream = stream;
             video.srcObject = currentStream;
           })
           .catch(function(err) {
             console.log("Hubo un error: ", err);

           })
}

function procesarCamara() {
  ctx.drawImage(video, 0, 0, tamano, tamano, 0, 0, tamano, tamano);
  setTimeout(procesarCamara, 20);
}

function predecir() {
  if (modelo != null) {
    resample_single(canvas, 100, 100, otrocanvas);

    // hacer prediccion
    const ctx2 = otrocanvas.getContext("2d");
    const imgData = ctx2.getImageData(0, 0, 100, 100);

    let arr = [];
    let arr100 = [];

    for (let p=0; p < imgData.data.length; p+=4) {
      const rojo = imgData.data[p] / 255;
      const verde = imgData[p+1] / 255;
      const azul = imgData[p+2] / 255;
      const gris = (rojo+verde+azul) / 3;

      arr100.push([gris]);
      if (arr100.length == 100) {
        arr.push(arr100);
        arr100 = [];
      }
    }

    arr = [arr];

    const tensor = tf.tensor4d(arr);
    const resultado = modelo.predict(tensor).dataSync();

    let respuesta;
    if (resultado <= .5) {
      respuesta = "Gato";
    } else {
      respuesta = "Perro";
    }
    document.getElementById("resultado").innerHTML = respuesta;
  }
  setTimeout(predecir, 150);
}

/*
 * Hermit resize - fast image resize/resample using hermite filter. 1 cpu version!
 *
 * @param {HtmlElement} canvas
 * @param {int} width
 * @param {int} height
 * @param {boolean} resize_canvas if true, canvas will be resized. Opcional.
 * Cambiado por RT, resize canvas ahora es donde se pone el ....
 */
function resample_single(canvas, width, height, resize_canvas) {
  const width_source = canvas.width;
  const height_source = canvas.height;
  width = Math.round(width);
  height = Math.round(height);

  const ratio_w = width_source / width;
  const ratio_h = height_source / height;
  const ratio_w_half = Math.ceil(ratio_w / 2);
  const ratio_h_half = Math.ceil(ratio_h / 2);

  const ctx = canvas.getContext("2d");
  const ctx2 = resize_canvas.getContext("2d");
  const img = ctx.getImageData(0, 0, width_source, height_source);
  const img2 = ctx2.createImageData(width, height);
  const data = img.data;
  const data2 = img2.data;

  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      const x2 = (i + j * width) * 4;
      let weight = 0;
      let weights = 0;
      let weights_alpha = 0;
      let gx_r = 0;
      let gx_g = 0;
      let gx_b = 0;
      let gx_a = 0;
      const center_y = (j + 0.5) * ratio_h;
      const yy_start = Math.floor(j * ratio_h);
      const yy_stop = Math.ceil((j + 1) * ratio_h);

      for (let yy = yy_start; yy < yy_stop; yy++) {
        const dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
        const center_x = (i + 0.5) * ratio_w;
        const w0 = dy * dy;
        const xx_start = Math.floor(i * ratio_w);
        const xx_stop = Math.ceil((i + 1) * ratio_w);

        for (let xx = xx_start; xx < xx_stop; xx++) {
          const dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
          const w = Math.sqrt(w0 + dx * dx);
          if (w >= 1) {
            // pixel too far
            continue;
          }
          // hermite filter
          weight = 2 * w * w * w - 3 * w * w + 1;
          const pos_x = 4 * (xx + yy * width_source);
          // alpha
          gx_a += weight * data[pos_x + 3];
          weights_alpha += weight;
          // colors
          if (data[pos_x + 3] < 255)
            weight = weight * data[pos_x + 3] / 250;

          gx_r += weight * data[pos_x];
          gx_g += weight * data[pos_x + 1];
          gx_b += weight * data[pos_x + 2];

          weights += weight;
        }
      }
      data2[x2] = gx_r / weights;
      data2[x2 + 1] = gx_g / weights;
      data2[x2 + 2] = gx_b / weights;
      data2[x2 + 3] = gx_a / weights_alpha;
    }
  }
  ctx2.putImageData(img2, 0, 0);
}
