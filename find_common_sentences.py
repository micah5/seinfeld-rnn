var fs = require('fs');
var obj = JSON.parse(fs.readFileSync('utterance.json', 'utf8'));
var str = ""
for(var i = 0; i < obj.features.length; i++) {
  var properties = obj.features[i].properties
  str += properties.speaker + ": " + properties.text.toLowerCase().replace(/[^\w\s]|_/g, "").replace(/\s+/g, " ") + " <newline> "
}
fs.writeFile('seinfeld_modified_cut.txt', str, 'utf8');
