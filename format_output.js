var fs = require('fs');
var str = fs.readFileSync('unformatted_output.txt', 'utf8');
fs.writeFile('formatted_output.txt', str.replace(/<newline> /g, "\n"), 'utf8');
