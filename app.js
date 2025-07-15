let model;
let minInput, maxInput, minOutput, maxOutput;
let lossChart, compareChart;

// Normalize helper
function normalize(tensor, min, max) {
  const MIN = min || tf.min(tensor);
  const MAX = max || tf.max(tensor);
  const range = tf.sub(MAX, MIN);
  const normalized = tf.div(tf.sub(tensor, MIN), range);
  return {
    NORMALIZED_VALUES: normalized,
    MIN_VALUES: MIN,
    MAX_VALUES: MAX
  };
}

// Denormalize helper
function denormalize(tensor, min, max) {
  return tf.add(tf.mul(tensor, tf.sub(max, min)), min);
}

// Generate data for function
function generateData(selectedFn) {
  const Xs = [];
  const Ys = [];
  for (let i = 1; i <= 20; i++) {
    const x = i;
    let y;
    try {
      y = eval(selectedFn);
    } catch {
      y = 0;
    }
    Xs.push(x);
    Ys.push(y);
  }
  return { Xs, Ys };
}

// Create model with dynamic layers
function createModel(neuronCounts) {
  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [1], units: neuronCounts[0], activation: 'relu' }));
  for (let i = 1; i < neuronCounts.length; i++) {
    model.add(tf.layers.dense({ units: neuronCounts[i], activation: 'relu' }));
  }
  model.add(tf.layers.dense({ units: 1 }));
}

// Train the model
async function trainModel() {
  const selectedFn = document.getElementById("functionSelect").value;
  const { Xs, Ys } = generateData(selectedFn);
  const inputTensor = tf.tensor1d(Xs);
  const outputTensor = tf.tensor1d(Ys);

  // Get neurons per layer
  const neuronsInputs = document.querySelectorAll(".layerNeuronInput");
  const neuronCounts = Array.from(neuronsInputs).map(inp => parseInt(inp.value)).filter(n => !isNaN(n));

  if (neuronCounts.length === 0) {
    alert("Please enter valid neurons for each layer.");
    return;
  }

  const epochs = parseInt(document.getElementById("epochsInput").value) || 200;

  createModel(neuronCounts);

  const inputNorm = normalize(inputTensor);
  const outputNorm = normalize(outputTensor);

  minInput = inputNorm.MIN_VALUES;
  maxInput = inputNorm.MAX_VALUES;
  minOutput = outputNorm.MIN_VALUES;
  maxOutput = outputNorm.MAX_VALUES;

  model.compile({
    optimizer: tf.train.sgd(0.01),
    loss: 'meanSquaredError'
  });

  if (lossChart) lossChart.destroy();
  lossChart = new Chart(document.getElementById('lossChart'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Loss',
        data: [],
        borderColor: 'red',
        fill: false
      }]
    }
  });

  await model.fit(inputNorm.NORMALIZED_VALUES, outputNorm.NORMALIZED_VALUES, {
    epochs,
    batchSize: 2,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const loss = Math.sqrt(logs.loss);
        lossChart.data.labels.push(epoch);
        lossChart.data.datasets[0].data.push(loss);
        lossChart.update();
      }
    }
  });

  drawPredictionVsTrue();
}

// Predict a single value
function predictValue() {
  const inputX = parseFloat(document.getElementById('predictInput').value);
  if (isNaN(inputX)) return;

  tf.tidy(() => {
    const normInput = normalize(tf.tensor1d([inputX]), minInput, maxInput);
    const prediction = model.predict(normInput.NORMALIZED_VALUES);
    const denorm = denormalize(prediction, minOutput, maxOutput);

    denorm.data().then(value => {
      document.getElementById('predictResult').innerText =
        `Predicted y for x=${inputX} is ${value[0].toFixed(4)}`;
    });
  });
}

// Plot predicted vs true values
function drawPredictionVsTrue() {
  const xs = [];
  const ysTrue = [];
  const ysPred = [];

  for (let x = 0; x <= 20; x += 0.5) {
    xs.push(x);
    ysTrue.push(eval(document.getElementById("functionSelect").value));
  }

  const inputTensor = tf.tensor1d(xs);
  const norm = normalize(inputTensor, minInput, maxInput);
  const predictions = model.predict(norm.NORMALIZED_VALUES);
  const denorm = denormalize(predictions, minOutput, maxOutput);

  denorm.data().then(predVals => {
    ysPred.push(...predVals);

    if (compareChart) compareChart.destroy();
    compareChart = new Chart(document.getElementById('compareChart'), {
      type: 'line',
      data: {
        labels: xs,
        datasets: [
          {
            label: 'True y',
            data: ysTrue,
            borderColor: 'orange',
            fill: false
          },
          {
            label: 'Predicted y',
            data: ysPred,
            borderColor: 'blue',
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        scales: {
          x: { title: { display: true, text: 'x' } },
          y: { title: { display: true, text: 'y' } }
        }
      }
    });

    inputTensor.dispose();
    norm.NORMALIZED_VALUES.dispose();
    denorm.dispose();
  });
}

