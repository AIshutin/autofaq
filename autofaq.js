class BertTokenizer {
  constructor(vocab, special_tokens_map, tokenizer_config) {
      this.tokenizer_config = tokenizer_config;
      this.special_tokens_map = special_tokens_map;
      this.vocab = vocab;
      this.max_len = 0;
      for (var key in vocab) {
        if (this.max_len < key.length) this.max_len = key.length;
      }
      this.whitespaces = new Set([" ", "\n", "\t"]);
  }

  isLemma(token) {
    if (token.length < 3) {
      return false;
    }
    return token[0] === '#' && token[1] === '#';
  }

  formatToken(token) {
    if (this.isLemma(token)) {
      return token.slice(2);
    }
    return token;
  }

  encode(text, add_special_tokens=true) {
    if (this.tokenizer_config['do_lower_case']) {
      text = text.toLowerCase();
    }

    var tokens_ids = [];
    if (add_special_tokens) {
      /*console.log(this.special_tokens_map)
      console.log(this.special_tokens_map.cls_token in this.vocab);*/
      tokens_ids = [this.vocab[this.special_tokens_map.cls_token]]
    }
    var start = 0;
    var is_new_word = true;

    while (start < text.length) {
      var longest_key = "";
      // check if it's whitespace;
      if (this.whitespaces.has(text[start])) {
        is_new_word = true;
        start += 1;
        continue;
      }
      // search longest applicable token
      //console.log('start: '+ start + " tokens_ids: " + tokens_ids)

      var L = 0;
      var R = this.max_len + 1;
      //console.log('this.max_len ' + this.max_len)
      while (R - L > 1) {
        var med = (L + R) / 2;
        var str = text.slice(start, start + med + 1);
        // console.log(med + ' ' + str);

        var flag = is_new_word && this.vocab.hasOwnProperty(str) ||
                  !is_new_word && this.vocab.hasOwnProperty('##' + str);
        if (flag) {
          L = med;
          longest_key = str;
        } else {
          R = med;
        }
      }

      start += this.formatToken(longest_key).length;

      if (longest_key === "") {
        start += 1;
        longest_key = this.special_tokens_map.unk_token;
      }
      //console.log(longest_key);
      tokens_ids.push(this.vocab[longest_key]);

    //  console.log('START ' + start);
    }
    tokens_ids.push(this.vocab[this.special_tokens_map.sep_token]);
    //console.log(tokens_ids);
    return tokens_ids;
  }
}

function get_vocab(url) {
  var text = "";
  $.ajax({
    'url': url,
    'async': false,
    'success': function (text1) {
      text = text1;
      return text1;
      },
  })
  let arr = text.split('\n');
  console.log(arr);
  let vocab = {};
  let ind = 0;
  for (var i = 0; i < arr.length; i++) { // arr.length;
    vocab[arr[i]] = i;
  }
  return vocab;
}

var defaultSpecialTokensMap = {"unk_token": "[UNK]", "sep_token": "[SEP]",
                              "pad_token": "[PAD]", "cls_token": "[CLS]",
                              "mask_token": "[MASK]"};
var defaultTokenizerConfig = {"do_lower_case": true, "model_max_length": 512};

function find_vocab() {
  return get_vocab("./distilbert/vocab.txt");
}
let vocab = find_vocab();
var tokenizer = new BertTokenizer(vocab, defaultSpecialTokensMap, defaultTokenizerConfig);

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
  // console.log('embedding')
  // console.log(outputs)
  var norm = tf.norm(outputs, ord='euclidean', axis=1, keepdims=true);
  // console.log(norm)
  var result = tf.div(outputs, norm);
  // console.log(result);
  norm.dispose()
  return result;
}

var MODEL = undefined;

function run_model(text) {
  // console.log('modeling ' + text);
  const inputs = prepare_input(text)
  const results = MODEL.execute(inputs)[0];
  const embeddings = get_embeddings(results);
  // console.log(embeddings);
  results.dispose();
  return embeddings;
}

document.TEXT_CACHE = {}

function addAllTextToCache() {
  $("p").each(function() {
    const text = $(this).text();
    if (document.TEXT_CACHE.hasOwnProperty(text)) {
      return;
    }
    document.TEXT_CACHE[text] = run_model(text);
  });
}

function findRelevant(text, n=1) {
  var vector = run_model(text);
  results = []
  for (var key in document.TEXT_CACHE) {
    if (document.TEXT_CACHE.hasOwnProperty(key)) {
      var product_tensor = tf.mul(vector, document.TEXT_CACHE[key]);
      var score_tensor = product_tensor.sum();
      results.push([score_tensor.dataSync()[0], key]);
      product_tensor.dispose();
      score_tensor.dispose();
    }
  }
  console.log('results');
  console.log(results);
  console.log(vector);
  vector.dispose();
  results.sort(function(a, b){return b[0] - a[0]});
  return results.slice(0, n);
}

function switchToChat() {
  var x = document.getElementById("autofaq-widget");
  x.style.display = "none";
  x = document.getElementById("autofaq-chat");
  x.style.display = "initial"
}

function switchToLogo() {
  var x = document.getElementById("autofaq-widget");
  x.style.display = "initial";
  x = document.getElementById("autofaq-chat");
  x.style.display = "none";
}

function addResponseToAutoFAQ(text, score) {
  var suggestions = document.getElementById("autofaq-suggestions")
  var current = document.createElement('div')
  current.style = "border-radius: 5px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); transition: 0.3s; margin: 5px; padding: 5px; background-color: #FFFFFF;";
  current.innerHTML = "<p>" + text + "</p>"
  suggestions.appendChild(current);
}

function addAutoFAQLogo() {
  var elem = document.createElement('img');
  elem.setAttribute('onclick','switchToChat()')
  elem.src = "./logo.png"
  let float_style = "padding: 5px; width: 130px; z-index: 100; position: fixed; bottom: 10px; right: 10px;"
  elem.style += float_style;
  elem.id = "autofaq-widget";
  console.log(elem);
  document.body.appendChild(elem);
}

function autoAnswerQuestion(question) {
  var suggestions = document.getElementById('autofaq-suggestions');
  suggestions.innerHTML = "...";
  var relevant = findRelevant(question, 3);
  suggestions.innerHTML = "";
  for (var i = 0; i < relevant.length; i += 1) {
    addResponseToAutoFAQ(relevant[i][1], relevant[i][0]);
  }
}

function autoFAQInputListener() {
  var input = document.getElementById('autofaq-input');
  console.log(input);
  autoAnswerQuestion(input.value);
}

function addAutoFAQChat() {
  let faq = document.createElement('div');
  faq.id = "autofaq-chat"
  faq.style = "border-width: 2px; border-radius: 5px; box-shadow: 16px 16px 16px 16px rgba(0,0,0,0.4); transition: 0.3s; background-color: #eb7d46; border-style: solid; padding: 5px; z-index: 100; width: 250px; position: fixed; bottom: 10px; right: 10px;";
  faq.innerHTML = `<div>
                    <div style="padding-bottom: 10px; display: flex;">
                      <h3 style="text-align: center; position: relative;"><a style="color: inherit; text-decoration: inherit;" href="https://github.com/AIshutin/autofaq">AutoFAQ</a></h3>
                      <img src="./CloseIcon.png" onclick="switchToLogo()" style="width: 30px; position: absolute; right: 5px; top: 5px;"></img>
                    </div>
                    <input id="autofaq-input" onchange="autoFAQInputListener()"placeholder=" Ask me something.. (then press enter)" style="width: 244px; border-radius: 5px;"></input>
                   </div>
                   <div id="autofaq-suggestions" style="padding: 3px;">
                   </div>
                   <p style="text-align: center; padding: 3px;">by <a style="color: inherit; text-decoration: inherit;" href="https://t.me/aishutin">Andrew Ishutin</a></p>`
  faq.style.display = "none"
  document.body.appendChild(faq);
}

function setupAutoFAQ() {
  createModel().then(model => {
    MODEL = model;
    addAllTextToCache();
    addAutoFAQLogo();
    addAutoFAQChat();
  });
}

setupAutoFAQ();
