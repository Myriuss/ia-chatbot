function sendMessage() {
  const input = document.getElementById("user-input");
  const message = input.value.trim();
  if (!message) return;

  appendMessage("Vous", message, "user-msg", "https://i.pravatar.cc/32?img=5");
  input.value = "";

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: message })
  })
    .then(res => res.json())
    .then(data => {
      appendMessage("SportBot", data.response, "bot-msg", "https://cdn-icons-png.flaticon.com/512/4712/4712037.png");
    })
    .catch(() => {
      appendMessage("Erreur", "Une erreur s’est produite.", "bot-msg", "https://cdn-icons-png.flaticon.com/512/4712/4712037.png");
    });
}

function appendMessage(sender, text, className, avatarUrl) {
  const chatBox = document.getElementById("chat-box");
  const wrapper = document.createElement("div");
  wrapper.className = `chat-msg ${className}`;

  const avatar = document.createElement("img");
  avatar.src = avatarUrl;
  avatar.className = "avatar";

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";
  bubble.innerHTML = `<strong>${sender}:</strong> ${text}`;

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;
}
