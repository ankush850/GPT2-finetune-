document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("user-input").addEventListener("keypress", function(e) {
  if (e.key === "Enter") sendMessage();
});

document.getElementById("toggle-theme").addEventListener("click", function() {
  document.body.classList.toggle("dark-mode");
});

document.getElementById("mic-btn").addEventListener("click", startVoiceInput);

function appendMessage(message, sender) {
  const chatBox = document.getElementById("chat-box");
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message", sender === "user" ? "user-message" : "bot-message");
  msgDiv.textContent = message;
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function sendMessage(messageFromVoice = null) {
  const input = document.getElementById("user-input");
  const message = messageFromVoice || input.value.trim();
  if (message === "") return;

  appendMessage(message, "user");
  input.value = "";

  fetch("/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: message })
  })
  .then(res => res.json())
  .then(data => {
    appendMessage(data.response, "bot");
    speakText(data.response); // 🔊 speak reply
  });
}

function startVoiceInput() {
  if (!("webkitSpeechRecognition" in window)) {
    alert("Your browser does not support Speech Recognition. Try Chrome!");
    return;
  }

  const recognition = new webkitSpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.start();

  recognition.onresult = function(event) {
    const voiceText = event.results[0][0].transcript;
    sendMessage(voiceText);
  };

  recognition.onerror = function(event) {
    console.error("Speech recognition error:", event.error);
  };
}

function speakText(text) {
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-US";
  window.speechSynthesis.speak(utterance);
}
