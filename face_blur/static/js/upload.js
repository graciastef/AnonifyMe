const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("videoFile");
const statusBox = document.getElementById("statusBox");

function getCsrfToken() {
  const input = document.querySelector("input[name='csrfmiddlewaretoken']");
  return input ? input.value : null;
}

form.addEventListener("submit", async (event) => {
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
    const response = await fetch("/api/upload/", {
      method: "POST",
      body: formData,
      headers: {
        "X-CSRFToken": csrfToken
      },
      credentials: "same-origin"
    });

    const data = await response.json();
    statusBox.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    statusBox.textContent = error.message;
  }
});