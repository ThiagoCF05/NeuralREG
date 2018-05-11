function startTimer(duration, display) {
    var timer = duration, minutes, seconds;
    var interval = setInterval(function () {
        minutes = parseInt(timer / 60, 10);
        seconds = parseInt(timer % 60, 10);

        minutes = minutes < 10 ? "0" + minutes : minutes;
        seconds = seconds < 10 ? "0" + seconds : seconds;

        display.textContent = minutes + ":" + seconds;

        if (--timer < 0) {
            timer = duration;
            document.getElementById('button').removeAttribute('disabled');
            document.getElementById('timer').style.display = 'none';
        }
    }, 1000);
}

window.onload = function () {
    document.getElementById('button').setAttribute('disabled', 'true');
    var fiveMinutes = 20,
        display = document.querySelector('#timer');
    startTimer(fiveMinutes, display);
};