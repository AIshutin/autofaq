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
      console.log(this.special_tokens_map)
      console.log(this.special_tokens_map.cls_token in this.vocab);
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

function sleep(milliseconds) {
  const date = Date.now();
  let currentDate = null;
  do {
    currentDate = Date.now();
  } while (currentDate - date < milliseconds);
}

let vocab = find_vocab();
var tokenizer = new BertTokenizer(vocab, defaultSpecialTokensMap, defaultTokenizerConfig);
