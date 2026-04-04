const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("videoFile");
const statusBox = document.getElementById("statusBox");

function getCsrfToken() {
  const input = document.querySelector("input[name='csrfmiddlewaretoken']");
  return input ? input.value : null;
}

let currentFileKey = null;

// Video Upload
const videoForm = document.getElementById("videoForm");
const videoResult = document.getElementById("videoResult");

videoForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    statusBox.textContent = "Please select a file.";
    return;
  }

  const csrfToken = getCsrfToken();
  if (!csrfToken) {
    statusBox.textContent = "CSRF token not found in page.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/videos/", {
      method: "POST",
      body: formData,
      headers: {
        "X-CSRFToken": csrfToken
      },
      credentials: "same-origin"
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    currentFileKey = data.file_key;

    videoResult.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    videoResult.textContent = "Error: " + error.message;
  }
});

// Whitelist Upload

const whitelistForm = document.getElementById("whitelistForm");
const whitelistResult = document.getElementById("whitelistResult");

whitelistForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const files = document.getElementById("whitelistFiles").files;

  const formData = new FormData();
  formData.append("file_key", currentFileKey);

  for (let i = 0; i < files.length; i++) {
    formData.append("files", files[i]);
  }

  try {
    const res = await fetch("/api/whitelist-images/", {
      method: "POST",
      body: formData,
      headers: {
        "X-CSRFToken": getCsrfToken(),
      },
    });

    const data = await res.json();

    if (!res.ok) throw new Error(data.error);

    whitelistResult.textContent = JSON.stringify(data, null, 2);

  } catch (err) {
    whitelistResult.textContent = "Error: " + err.message;
  }
});