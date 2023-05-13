console.log("hii")
window.onload = function() {
    function playVideo(name) {
      var player = document.getElementById('player');
      var source = document.getElementById('source');
      source.src = '../videos/' + name + '.mp4';
      player.load();
      player.play();
    }
  }
  