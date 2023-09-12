

// to build it run :
// npm run build

// use an async context to call onnxruntime functions.
const ort = require('onnxruntime-web');
const png = require('fast-png');
const FileSaver = require('file-saver');
let session
let rgbData,rgbBlob
let imageName = null
let fileName

   


async function initAPP() 
{
  // GENERATE THE DEMO PANOS
  const bricks = [];


  const ids = ['001a8396','00ac9bb3','00bdfea7','00c19968','00cbed51','00cdb6a5','00d38b83','00d474fc','00d9b058','00e6fbb4','00ec2e00','00f2da53','00f96a4e','00fb9b01','00ff55b9','00ffc857','01a41fda','01a55cc6','01a9fb2d','01b20386','01b6d406','01b76159','01b93f83','0a084cd5','0a1076ba','0a124a24','0a18ae0d','0a2112a6','0a257fa9','0a2c67ee','0a3538d6','0a3733ae','0a3bfc77','0a3cf03d','0a4c2b11','0a529eb9','0a5efd33','0a5fc1b0','0a7d282d','0a7e75ac','0a7ff741','0a8b6eba','0a8f4047','0a91dd5d','0a95ccc1','0a96d377','0aa19b47','0aa54d4c','0aacb7f3','0aae26bb','0ab0425a','0abb3aa6','0adb9c75','0ae35877','0ae7bdcf','0af815f7','0afddbf7','0b0e21ad','0b1564d0','0b1d7144','0b1f178f','0b22b957','0b442cfe','0b447fd2','0b49274d','0b4c9e36','0b4e0aff','0b4e3f11','0b51f1a9','0b565cd3','0b626ad0','0b6e2ccc','0b6ead6d','0b732ba7','0b7d08eb','0b859fa8','0b869dc9','0b8f4ea9','0b953c90','0ba0e86d','0ba12086','0bb30584','0bc09d6c','0bcc5e2b','0bd0c679','0bdc535b','0be6a389','0beb0f9a','0bf0bcbb','0bf409da','0bf63077','0bf8fc00','0c02c18c','0c06ec13','0c080a12','0c0bfb20','0c0fc91a','0c12c237','0c15cec4','0c24da2f','0c2d6523','0c33c0b2','0c37dea6','0c406a90','0c42fb44','0c47b708','0c6384a3','0c666fb8','0c687446','0c813168','0c844e31','0c84c4ec','0c87c274','0c89c095','0c8aa5b9','0c946961','0ca0bd13','0ca64f1e','0cabac91','0cb389b8','0cb48377','0cb5c51e','0cbb0706','0cc50011','0cca5143','0ccb54f4','0ccd24df','0ce37351','0ce7794c','0ce923d5','0cf02fcf','0cf2cd62','0cf3aa73','0cf9e6f3','0cfba116','0cfee702','0d17d871','0d191aac','0d1aa32a','0d240e61','0d259257','0d296128','0d31658b','0d38440d','0d3cdb93','0d3ede49','0d544d11','0d5bb157','0d5f8cbe','0d6f8225','0d79eea9','0d7c0efd','0d7c3d8c','0d8b7b21','0d8f95c2','0d8fa52f','0da0140c','0db27098','0db805e0','0db80b34','0dbb3da3','0dc4ce81','0dd8a2c3','0ddc066c','0de1d864','0de2237e','0de57e0a','0df08c19','0df90d62','0dfa8991','0dfcbd5d','0e053c08','0e0f6298','0e0fcd65','0e1136fc','0e1ddd0f','0e2105f5','0e22b913','0e31b31b','0e48f239','0e4a97e2','0e4ab77b','0e4f23ff','0e4f8377','0e60c93b','0e63fdff','0e6a6eea','0e6c2744','0e70067e','0e712105','0e73b8fd','0e7b2778','0e86dfbf','0e8b8d1c','0ea30e2a','0eacf8c4','0ead5cfb','0eb58f76','0ec75d72','0ec78d54','0eca6d3f','0ecdbc70','0ece0496','0ed9316d','0edfa95e','0ee443e5','0ee64c3b','0eec730a','0eed35d9','0ef567fc','0f01d6e6','0f053931','0f074841','0f0a21a6','0f100206','0f18790c','0f18a778','0f2f5815','0f3eb913','0f41bd12','0f43e4ca','0f45a16c','0f562d6a','0f575216','0f5adbaa','0f5d08f5','0f635a8b','0f64bf55','0f6c76c8','0f6ec552','0f6fef15','0f74c0e7','0f853e04','0f8feb42','0f935f20','0f939384','0f982722','0f9bdaf9','0f9cc070','0fa04b7e','0fa127fe','0fa447c2','0fa8fa6b','0faf9494','0fb119fc','0fb8ca31','0fc9d8a7','0fcd7c13','0fcebfe0','0fd49884','0fd9f747','0fda3517','0fe44c86','0fe9d5a0','0ff2f96d','1a007f29','1a044552','1a0dd439','1a10d674','1a15d39e','1a1a0cab','1a1fe705','1a2cc5cf','1a2f2676','1a3fd017','1a402f68','1a406433','1a463e35','1a5c2b1d','1a8551ec','1a9e3728','1aa16d84','1aa3d2d5','1aa59ac4','1aa5cc62','1ab16c39','1ab19036','1ab514fc','1ac0523e','1acead3a','1ada1aac','1ae4bbe2','1ae8e08e','1affb454','1b04e0a5','1b0757e1','1b09a551','1b0b444b','1b1ab7aa','1b2b4743','1b3ace41','1b416a4d','1b434dc0','1b4684a8','1b5fff41','1b6600ed','1b69f485','1b72eb41','1b73d2ff','1b7721a4','1b78479f','1b7a374f','1b7a7592','1b8b6e91','1b948ae5','1b971160','1b9904f4','1b9c215b','1b9c371e','1b9dabd3']
  ids.forEach(id => {
    bricks.push(`brick_%{id}.png`)
  })
  
  for (let i = 0; i < bricks.length; i++) {
    let btn = document.createElement("button");
    btn.innerHTML =
      '<img src="./images/' + bricks[i] + '" width="85" height ="100" />';
    btn.onclick = function () {
      everpano.getModelInference("./images/" + bricks[i]);
    };
    document.body.appendChild(btn);
  }

  // setup the dnd listeners so we can drag panos to the div
  var dropZone = document.getElementById("brick");
  // console.log('dropZone',dropZone);
  dropZone.addEventListener("dragover", handleDragOver, false);
  dropZone.addEventListener("drop", handleFileSelect, false);


  await loadOnnxModel();
}



function handleFileSelect(evt)
{
    evt.stopPropagation();
    evt.preventDefault();

    var files = evt.dataTransfer.files; // FileList object
    
    // Loop through the FileList
    for (var i = 0, f; f = files[i]; i++)
    {
        // Only process image files.
        if (!f.type.match('image.*'))
            continue;

        var reader = new FileReader();
        imageName = f.name
        // Read in the image file as a data URL.
        reader.onloadend = handleReaderLoadEnd;
        reader.readAsDataURL(f);
    }
}

function handleReaderLoadEnd(e)
{
    var imageurl = e.target.result;
    getModelInference(imageurl)
}

function handleDragOver(evt)
{
    evt.stopPropagation();
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'copy';
}

async function loadOnnxModel() {

    // LOAD MODEL
    let start = window.performance.now();

    // START SESSION
    // we can not use the webgl but the web assembly (wasm) otherwise we get an error
	// const session = await ort.InferenceSession.create(modelFile,{ executionProviders: ['webgl'] });
    
    session = await ort.InferenceSession.create('./capture/model.onnx',{ executionProviders: ['wasm'] });
    console.log('create session : ' + time_diff(start) + ' ms');
    document.getElementById("createsession").innerHTML = 'create session : <b>' + time_diff(start) + '</b> ms <br>------------------------'
    
    
}

async function cleanImageValues(rgbPath) {
    
    if (imageName != null) {
        rgbPath = imageName;
        imageName = null
    } else {
        rgbPath = rgbPath.replace('./faces/','')
    }

    fileName = rgbPath

    document.getElementById("imageID").innerHTML = 'processing : <b>' + rgbPath + '</b>';
    document.getElementById("loadimage").innerHTML = 'loading image : ';
    document.getElementById("scaleimage").innerHTML = 'scale image :';
    document.getElementById("inferencetime").innerHTML = 'inference : ';


}



async function getModelInference(rgbPath,topk = 3, debug = true) {
    try {
      console.log("getModelInference");

      await cleanImageValues(rgbPath);

      // create a new session and load the specific model.
      let start = window.performance.now();

      // Load image.
      const imageTensor = await getImageTensorFromPath(rgbPath);

      console.log("loading image : " + time_diff(start) + " ms");
      document.getElementById("loadimage").innerHTML =
        "loading image : <b>" + time_diff(start) + "</b> ms";
      start = window.performance.now();

      // Preprocess the image data to match input dimension requirement
      const feeds = { input: imageTensor };

      // feed inputs and run
      const modelResults = await session.run(feeds);

      console.log("inference :" + time_diff(start) + " ms");
      document.getElementById("inferencetime").innerHTML =
        "inference : <b>" + time_diff(start) + "</b> ms";

      console.log("modelResults", modelResults);
      let sortedResult = prediction2Array(modelResults.gender_output.data)
        let gender = 'female'
        if(sortedResult[0][0] == 0) gender = 'male'

      console.log("sorted gender Result", sortedResult);
      console.log("gender :", gender);
      const age = parseInt(modelResults.age_output.data * 100);
      console.log("age :", age);
      
 
      const eye_pos = modelResults.pos_output.data;
      console.log(`eye pos \nLEFT [${eye_pos[0]} ${eye_pos[1]}] \nRIGHT [${eye_pos[2]} ${eye_pos[3]}]`);

      printResults(rgbPath, age, gender, [eye_pos[0], eye_pos[1], eye_pos[2],eye_pos[3]], [178 , 218]);


    } catch (e) {
		console.error(`failed to inference ONNX model: ${e}.`);
	}
}


function printResults(url, age, gender, eye_pos, dims) {
  return new Promise((resolve, reject) => {
    // Convert image to base64 string
    var canvas = document.createElement("canvas");
    var ctx = canvas.getContext("2d");
    var img = new Image();

    // img.crossOrigin = 'Anonymous';

    function drawPoint(x, y, radius, color) {
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI, false);
      ctx.fillStyle = color;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(x, y, radius / 2, 0, 2 * Math.PI, false);
      ctx.fillStyle = "#fafafa";
      ctx.fill();
    }

    function text(
      x,
      y,
      txt,
      size = 15,
      font = "Courier New",
      color = "#00FF00",
      align = "left"
    ) {
      ctx.font = size + "px " + font;
      ctx.textAlign = align;
      ctx.fillStyle = color;
      ctx.fillText(txt, x, y);
    }

    img.onload = () => {
      const width = dims[0];
      const height = dims[1];
      canvas.width = width;
      canvas.height = height;

      var start = window.performance.now();

      ctx.drawImage(img, 0, 0, width, height);
      drawPoint(eye_pos[0] * width, eye_pos[1] * height, 5, "#ea496b");
      drawPoint(eye_pos[2] * width, eye_pos[3] * height, 5, "#0b7a9a");
      text(20, 20, `Age : ${age}`);
      text(20, 40, `Gender : ${gender}`);

      // create a Blob to save later the rgb image
      canvas.toBlob(function (blob) {
        rgbBlob = blob;
      });


      // Convert the canvas content to a data URL
      const dataURL = canvas.toDataURL();

      var rgbImg = document.getElementById("rgb");
      rgbImg.src = dataURL;
      rgbImg.width = width;
      rgbImg.height = height;
      rgbImg.onclick = function () {
        saveImage(rgbBlob, "rgb.png");
      };

      document.getElementById("p_rgb").innerHTML = `
        Gender : ${gender}<br>
        Age : ${age}<br>
        LEFT x: ${parseInt(eye_pos[0] * width)} y: ${parseInt(
        eye_pos[1] * height
      )}
        RIGHT x: ${parseInt(eye_pos[2] * width)} y: ${parseInt(
        eye_pos[3] * height
      )}`;

     
    };

    img.onerror = (err) => {
      reject(err);
    };

    img.src = url;
  });
}


/**
 * 
 * @param {*} array 
 * @returns an array of objects where the original index becomes the id 
 */
function prediction2Array (array)
{
    const newArray = []

    array.forEach((result,i) => {
        newArray.push({id:i,confidence:result})
    })


    newArray.sort((a, b) => b.confidence - a.confidence);


    const sortedArray =[]

    newArray.forEach( result => {
        sortedArray.push([result.id,result.confidence])
    })

    return sortedArray
}



function cleanFileName() {

    fileName = fileName.replace('.png','');
    fileName = fileName.replace('.jpg','');

    return fileName
}

function saveImage(blob,name) {
    FileSaver.saveAs(blob, cleanFileName() + '_' + name);
}



function blobToDataURL(blob, callback) {
    var a = new FileReader();
    a.onload = function(e) {callback(e.target.result);}
    a.readAsDataURL(blob);
}




async function getImageTensorFromPath(rgbPath, dims =  [1, 3, 224, 224]) 
{
	// 1. load the image
	var rgba = await getImagePixelData(rgbPath, dims);
    
    rgbData = rgba // THIS IS USED TO ENCODE the rgbDepth Image
    // console.log('rgbData',rgbData);

	// 2. convert to tensor
	var imageTensor = imageDataToTensor(rgba, dims);
	// 3. return the tensor
	return imageTensor;
}
  


function getImagePixelData(url,dims) {
    
    return new Promise( (resolve, reject) => {
    
        // Convert image to base64 string
        var canvas = document.createElement('canvas');
        var ctx = canvas.getContext('2d');
        var img = new Image;
        
        // img.crossOrigin = 'Anonymous';
        
        img.onload = () => {

            const width = dims[2]
            const height = dims[3]
            canvas.width = width;
            canvas.height = height;
            
            var start = window.performance.now();

            ctx.drawImage(img, 0, 0,width,height);
            
            const print2canvas = false
            if (print2canvas) {
                var HTMLcanvas = document.getElementById('panorama');
                var HTMLctx = HTMLcanvas.getContext('2d');
                HTMLcanvas.width = width;
                HTMLcanvas.height = height;
                HTMLctx.drawImage(img, 0, 0,width,height);
            }

            // create a Blob to save later the rgb image
            canvas.toBlob(function(blob) {
                rgbBlob = blob
            });

            var imgData = ctx.getImageData(0, 0, width, height);
            // console.log('imgData',imgData);
            // console.log('scaling :', time_diff(start) + ' ms');
            document.getElementById("scaleimage").innerHTML = 'scale image :<b>' + time_diff(start) + ' ms'
            resolve(imgData.data);
        };
        
        img.onerror = (err) => {
            reject(err);
        };
        
        img.src = url;

    });
    
}


function imageDataToTensor(imageBufferData, dims) {
	
    // 1. Get buffer data from image and create R, G, and B arrays.
	
	const [redArray, greenArray, blueArray] = new Array(new Array(), new Array(), new Array());
	let i,l,m
	// 2. Loop through the image buffer and extract the R, G, and B channels and skip the alpha channel
	for (i = 0; i < imageBufferData.length; i += 4) {
        // if(imageBufferData[i + 0] == 0 && imageBufferData[i + 1] == 0 && imageBufferData[i + 2] == 0 ) console.warn(i,'=> 000')
		redArray.push(imageBufferData[i + 0]);
		greenArray.push(imageBufferData[i + 1]);
		blueArray.push(imageBufferData[i + 2]);
	}

  
	// 3. Concatenate RGB to transpose [224, 224, 3] -> [3, 224, 224] to a number array
	const transposedData = redArray.concat(greenArray).concat(blueArray);
	
	// console.log('transposedData',transposedData);

	// 4. convert to float32
	i, l = transposedData.length; // length, we need this for the loop
	// create the Float32Array size 3 * 224 * 224 for these dimensions output
	const float32Data = new Float32Array(dims[1] * dims[3] * dims[2]);
	for (i = 0; i < l; i++) {
		float32Data[i] = transposedData[i] / 255.0; // convert to float
	}

	// console.log('float32Data',float32Data);

	const transposedDims = [dims[0],dims[1],dims[3],dims[2]] // Note that with and height are inverted
	
	// 5. create the tensor object from onnxruntime-web.
	const inputTensor = new ort.Tensor("float32", float32Data, transposedDims);
	
    // console.log('inputTensor',inputTensor);

	return inputTensor;

}




/*
Time scale at microsecond resolution but returned as seconds
*/
function GetRunTime()
{
	return Math.trunc(Date.now()/ 1000);
}

function time_diff(time) {
    return Math.round((window.performance.now() - time) * 10000) / 10000
  }


initAPP();

export { getModelInference };
// export { ort as onnx, vis, png, aws, three, pica};

