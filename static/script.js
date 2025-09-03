async function generateText() {
  const prompt = document.getElementById("prompt").value;
  const responseBox = document.getElementById("response");
  responseBox.innerText = "⏳ Generating...";

  const res = await fetch("/generate", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ prompt })
  });

  const data = await res.json();
  responseBox.innerText = data.response;
}
