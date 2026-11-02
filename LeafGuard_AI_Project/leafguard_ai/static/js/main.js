(() => {
  const input = document.getElementById("leafImage");
  const dropzone = document.getElementById("dropzone");
  const preview = document.getElementById("imagePreview");
  const content = document.getElementById("dropContent");
  const form = document.getElementById("uploadForm");
  const button = document.getElementById("analyseBtn");
  if (!input || !dropzone) return;

  function showFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      alert("Please choose an image file.");
      input.value = "";
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      alert("Please choose an image smaller than 10 MB.");
      input.value = "";
      return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
      preview.src = event.target.result;
      preview.classList.remove("hidden");
      content.classList.add("hidden");
    };
    reader.readAsDataURL(file);
  }

  input.addEventListener("change", () => showFile(input.files[0]));
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    showFile(file);
  });
  form?.addEventListener("submit", () => {
    if (!input.files.length) return;
    button.disabled = true;
    button.innerHTML = 'Analysing leaf <span class="spinner"></span>';
  });
})();
