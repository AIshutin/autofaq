

createModel().then(model => {
  MODEL = model;
  inputs = prepare_input("Hello");
  bert_outputs = model.execute(inputs)
  return get_embeddings(bert_outputs[0]);
  // return tf.tidy(() => get_embeddings(final_hidden));
});

console.log('continue')
