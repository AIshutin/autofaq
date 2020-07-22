async function createModel() {
  const model = await tf.loadGraphModel('./mobilebert/web_model/model.json');
  return model;
}

function prepare_input(text) {
  // {input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids}
  let input_ids = tokenizer.encode(text)
  let input_mask = [];
  let segment_ids = []
  while (input_mask.length < input_ids.length) {
    input_mask.push(1);
    segment_ids.push(0);
  }
  tf.tensor(input_mask)
  const len = input_ids.length
  input_ids = tf.tensor(input_ids, shape=[1, len], dtype="int32"); // , dtype="int32"
  input_mask = tf.tensor(input_mask, shape=[1, len], dtype="int32");
  segment_ids = tf.tensor(segment_ids, shape=[1, len], dtype="int32");
  //console.log(input_ids)
  return {
    "input_ids": input_ids,
    "input_mask": input_mask,
    "segment_ids": segment_ids
  }
}

function get_embeddings(outputs) {
  //console.log(outputs)
  var norm = tf.norm(outputs, ord='euclidean', axis=1, keepdims=true);
  var result = outputs / norm
  norm.dispose()
  return result;
}

var MODEL = undefined;

function run_model(text) {
  console.log('model')
  console.log(MODEL);
  const inputs = prepare_input(text)
  console.log(inputs);
  const results = MODEL.execute()[0];
  console.log(results);
  const embeddings = get_embeddings(results);
  console.log(embeddings);
  results.dispose();
  return embeddings;
}

createModel().then(model => {
  MODEL = model;
  inputs = prepare_input("Hello");
  bert_outputs = model.execute(inputs)
  return get_embeddings(bert_outputs[0]);
  // return tf.tidy(() => get_embeddings(final_hidden));
});

console.log('continue')
