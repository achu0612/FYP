document.addEventListener('DOMContentLoaded', function() {
    var backButton = document.getElementById('back-btn');
    backButton.addEventListener('click', function() {
        history.back();
    });
});
