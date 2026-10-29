// Run the real template functions with a small DOM stub; no browser dependencies.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../templates/dashboard.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const renderQA = script.slice(script.indexOf('function renderQA('), script.indexOf('function renderFrames('));
const helpers = script.slice(script.indexOf('function parseQuestion('));
const elements = { 'qa-body': {}, 'qa-badge': {} };
const context = vm.createContext({ document: { getElementById: id => elements[id] } });
vm.runInContext(renderQA + helpers, context);

const screenshotQuestion = 'Question: If you choose to fly above the rooftops, can you still reach the shop?\n' +
  'Choices: A. I can complete the task but will not see the shopfronts directly.\n' +
  'B. I can complete the task because it offers a faster route.\n' +
  'C. I cannot complete the task because the final destination is at street level.';

test('Choices: A stays in the options, not the question stem', () => {
  const parsed = context.parseQuestion(screenshotQuestion);
  assert.equal(parsed.stem, 'If you choose to fly above the rooftops, can you still reach the shop?');
  assert.deepEqual(Array.from(parsed.options, option => option.key), ['A', 'B', 'C']);
  assert.match(parsed.options[0].text, /^I can complete the task/);
});

test('parses 2 to 7 options with the headings present in the dataset', () => {
  for (let count = 2; count <= 7; count++) {
    for (const heading of ['Choice: ', 'Choices: ', 'Choices:\r\n', 'Options: ', 'choose:\r\nAnswer: ']) {
      const choices = Array.from({ length: count }, (_, i) => `${String.fromCharCode(65 + i)}. Answer ${i}`);
      const parsed = context.parseQuestion(`Question: Where?\r\n${heading}${choices.join('\r\n')}`);
      assert.equal(parsed.stem, 'Where?');
      assert.equal(parsed.options.length, count);
      assert.equal(parsed.options[count - 1].key, String.fromCharCode(64 + count));
    }
  }
});

test('string wrappers stay out of the question and options', () => {
  const parsed = context.parseQuestion('"""\nQuestion: Where?\nchoose:\nAnswer: A. Here\nB. There\n"""');
  assert.equal(parsed.stem, 'Where?');
  assert.equal(parsed.options[1].text, 'There');
});

test('wrapped answer text stays with its option', () => {
  const parsed = context.parseQuestion('Where?\nChoices: A. First line\nsecond line\nB. Other');
  assert.equal(parsed.stem, 'Where?');
  assert.equal(parsed.options[0].text, 'First line\nsecond line');
});

test('bold Question and choose headings stay out of the parsed text', () => {
  const parsed = context.parseQuestion('**Question:** Where?\n**choose:**\nA. Here\nB. There');
  assert.equal(parsed.stem, 'Where?');
  assert.equal(parsed.options.length, 2);
});

test('pending answer is plain status text after the three option cards', () => {
  context.renderQA({ qa_idx: 1, question: screenshotQuestion, correct_answer: 'C' });
  const rendered = elements['qa-body'].innerHTML;
  assert.equal((rendered.match(/class="option(?:\s|\")/g) || []).length, 3);
  assert.match(rendered, /class="answer-status" role="status">等待模型回答<\/p>/);
  assert.doesNotMatch(rendered, /class="verdict/);
  assert.ok(rendered.indexOf('answer-status') > rendered.indexOf('final destination is at street level.'));
});

test('completed answer keeps correct and incorrect option highlighting', () => {
  context.renderQA({ qa_idx: 1, question: screenshotQuestion, correct_answer: 'C', model_answer: 'A' });
  const rendered = elements['qa-body'].innerHTML;
  assert.match(rendered, /class="option model-wrong"><span class="key">A/);
  assert.match(rendered, /class="option correct"><span class="key">C/);
  assert.match(rendered, /class="verdict bad"/);
  assert.doesNotMatch(rendered, /等待模型回答/);
});

test('seventh option is highlighted and missing answers stay pending', () => {
  assert.equal(context.normalizeOption('G'), 'G');
  assert.equal(context.normalizeOption('None'), '');
  const question = 'Where?\n' + Array.from({ length: 7 }, (_, i) => `${String.fromCharCode(65 + i)}. Place ${i}`).join('\n');
  context.renderQA({ qa_idx: 1, question, correct_answer: 'G', model_answer: 'G' });
  assert.match(elements['qa-body'].innerHTML, /class="option correct"><span class="key">G/);
  assert.match(elements['qa-body'].innerHTML, /class="verdict ok"/);
});
