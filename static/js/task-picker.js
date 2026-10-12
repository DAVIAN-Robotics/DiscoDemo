/* Shared task choices: only presentation changes; existing player handlers stay attached. */
(function () {
  'use strict';
  var titles = {banana:'Banana', cube:'Cube stacking', round:'Round peg', sqcirc:'Square & circle'};
  window.decorateTaskPicker = function (container, tasks) {
    var group = document.createElement('div'); group.className = 'task-picker';
    var label = document.createElement('div'); label.className = 'task-picker-label';
    label.textContent = 'Choose a task';
    container.before(group); group.append(label, container);
    container.classList.add('task-choices');
    var buttons = Array.from(container.querySelectorAll('button'));
    buttons.forEach(function (button, i) {
      var task = tasks[i];
      button.dataset.task = task.value;
      button.setAttribute('aria-label', task.label);
      button.tabIndex = button.getAttribute('aria-selected') === 'true' ? 0 : -1;
      var img = document.createElement('img');
      img.className = 'task-thumbnail'; img.alt = ''; img.loading = 'lazy';
      img.src = 'static/images/tasks/' + task.value + '.jpg';
      var copy = document.createElement('span'); copy.className = 'task-copy';
      var title = document.createElement('strong'); title.textContent = titles[task.value];
      var subtitle = document.createElement('small'); subtitle.textContent = task.label;
      copy.append(title,subtitle);
      button.replaceChildren(img,copy);
      button.addEventListener('click',function () { buttons.forEach(function (b) { b.tabIndex = b===button ? 0 : -1; }); });
      button.addEventListener('keydown', function (e) {
        var j;
        if (e.key==='ArrowRight') j=(i+1)%buttons.length;
        else if (e.key==='ArrowLeft') j=(i+buttons.length-1)%buttons.length;
        else if (e.key==='Home') j=0;
        else if (e.key==='End') j=buttons.length-1;
        else return;
        e.preventDefault(); buttons[j].focus(); buttons[j].click();
      });
    });
  };
})();
