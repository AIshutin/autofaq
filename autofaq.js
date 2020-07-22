document.TEXT_CACHE = {}

function addAllTextToCache() {
  console.log('real scanning')
  $("p").each(function() {
    const text = $(this).text();
    console.log("next <p> element with text: " + text);
    console.log(document.TEXT_CACHE.hasOwnProperty(text));
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
      results.push([-score_tensor.dataSync()[0], key]);
      product_tensor.dispose();
      score_tensor.dispose();
    }
  }
  console.log(vector);
  vector.dispose();
  results.sort();
  for (var i = 0; i < n && i < results.length; i++) {
    results[i][0] *= -1;
  }
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
  addAllTextToCache();
  addAutoFAQLogo();
  addAutoFAQChat();
}

setupAutoFAQ();
