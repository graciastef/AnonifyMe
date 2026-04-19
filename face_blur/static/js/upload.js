let currentFileKey = null
let faceFiles = []
let currentVideoFile = null

const tasks = [
  { label: 'Uploading Video to AI', doneAt: 10 },
  { label: 'Detecting All Faces', doneAt: 30 },
  { label: 'Matching Whitelisted Face(s)', doneAt: 50 },
  { label: 'Applying Blur', doneAt: 70 },
  { label: 'Rendering Output', doneAt: 90 },
]

// CSRF 
function getCsrfToken() {
  const input = document.querySelector("input[name='csrfmiddlewaretoken']")
  return input ? input.value : null
}

// PAGES
function showPage(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'))
  document.getElementById(id).classList.add('active')
}

// VIDEO UPLOAD 
const uploadBox = document.getElementById('uploadBox')
const videoInput = document.getElementById('videoFile')

uploadBox.addEventListener('click', () => videoInput.click())

uploadBox.addEventListener('dragover', (e) => {
  e.preventDefault()
  uploadBox.classList.add('dragging')
})

uploadBox.addEventListener('dragleave', () => {
  uploadBox.classList.remove('dragging')
})

uploadBox.addEventListener('drop', (e) => {
  e.preventDefault()
  uploadBox.classList.remove('dragging')
  const file = e.dataTransfer.files[0]
  if (file) setVideoFile(file)
})

videoInput.addEventListener('change', (e) => {
  if (e.target.files[0]) setVideoFile(e.target.files[0])
})

function setVideoFile(file) {
  currentVideoFile = file
  const preview = document.getElementById('videoPreview')
  preview.src = URL.createObjectURL(file)
  document.getElementById('upload-section').style.display = 'none'
  document.getElementById('preview-section').style.display = 'block'
  document.getElementById('processBtn').disabled = false
}

function removeVideo() {
  currentVideoFile = null
  document.getElementById('videoPreview').src = ''
  document.getElementById('upload-section').style.display = 'block'
  document.getElementById('preview-section').style.display = 'none'
  document.getElementById('processBtn').disabled = true
}

// FACE UPLOAD 
function handleFaceUpload(e) {
  const files = Array.from(e.target.files)
  faceFiles = [...faceFiles, ...files]
  renderFaces()
}

function removeFace(index) {
  faceFiles = faceFiles.filter((_, i) => i !== index)
  renderFaces()
}

function renderFaces() {
  const grid = document.getElementById('faceGrid')
  grid.innerHTML = ''

  faceFiles.forEach((face, i) => {
    const div = document.createElement('div')
    div.className = 'face-thumb'
    div.innerHTML = `
      <img src="${URL.createObjectURL(face)}" alt="face" />
      <button class="remove-btn" onclick="removeFace(${i})">×</button>
    `
    grid.appendChild(div)
  })

  // Add face button
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = 'image/*'
  input.multiple = true
  input.style.display = 'none'
  input.id = 'faceFilesInput'
  input.onchange = handleFaceUpload

  const addBtn = document.createElement('div')
  addBtn.className = 'add-face-btn'
  addBtn.innerHTML = `+ <span>Add Face</span>`
  addBtn.onclick = () => input.click()

  grid.appendChild(input)
  grid.appendChild(addBtn)
}

// TASKS
function renderTasks(progress) {
  const grid = document.getElementById('taskGrid')
  grid.innerHTML = ''
  tasks.forEach(task => {
    const done = progress >= task.doneAt
    const div = document.createElement('div')
    div.className = 'task-item'
    div.innerHTML = `
      <i class="bi ${done ? 'bi-check-circle-fill' : 'bi-circle'}" style="font-size: 22px; color: ${done ? '#6C8EF5' : '#9ca3af'};"></i>
      <span>${task.label}</span>
    `
    grid.appendChild(div)
  })
}

// PROCESSING
async function startProcessing() {
  showPage('page-processing')
  document.getElementById('restartBtn').style.display = 'block'
  document.getElementById('error-state').style.display = 'none'
  document.getElementById('done-state').style.display = 'none'
  document.getElementById('tasksCard').style.display = 'block'
  updateProgress(0)

  try {
    // Fetch CSRF cookie first
    const csrf = getCsrfToken()

    // Step 1 — Upload video
    const videoFormData = new FormData()
    videoFormData.append('file', currentVideoFile)

    const videoRes = await fetch('/api/videos/', {
      method: 'POST',
      body: videoFormData,
      headers: { 'X-CSRFToken': csrf },
      credentials: 'same-origin'
    })
    const videoData = await videoRes.json()
    if (!videoRes.ok) throw new Error(videoData.error)
    currentFileKey = videoData.file_key

    // Step 2 — Upload whitelist faces
    const whitelistFormData = new FormData()
    whitelistFormData.append('file_key', currentFileKey)
    faceFiles.forEach(face => whitelistFormData.append('files', face))

    const whitelistRes = await fetch('/api/whitelist-images/', {
      method: 'POST',
      body: whitelistFormData,
      headers: { 'X-CSRFToken': csrf },
      credentials: 'same-origin'
    })
    const whitelistData = await whitelistRes.json()
    if (!whitelistRes.ok) throw new Error(whitelistData.error)

    // Step 3 — Progress stream
    const source = new EventSource(`/api/progress/${currentFileKey}/`)
    source.onmessage = (e) => {
      const { percentage, eta } = JSON.parse(e.data)
      updateProgress(percentage, eta)
      if (percentage >= 100) {
        source.close()
        showDone()
      }
    }
    source.onerror = () => {
      source.close()
      showError('Connection to server lost. Please try again.')
    }

  } catch (err) {
    showError(err.message)
  }
}

function updateProgress(percentage, eta) {
  document.getElementById('progressFill').style.width = `${percentage}%`
  document.getElementById('progressFill').style.backgroundColor = percentage >= 100 ? '#22c55e' : '#EAD637'
  document.getElementById('progressLabel').textContent = `Processing Video... (${percentage}%)`
  renderTasks(percentage)
  const etaEl = document.getElementById('etaText')
  if (eta) {
    etaEl.textContent = `ETA: ${eta}`
    etaEl.style.display = 'block'
  } else {
    etaEl.style.display = 'none'
  }
}

function showDone() {
  document.getElementById('progressLabel').textContent = 'Processing Complete!'
  document.getElementById('progressFill').style.backgroundColor = '#22c55e'
  document.getElementById('done-state').style.display = 'block'
  document.getElementById('resultPreview').src = URL.createObjectURL(currentVideoFile)
}

function showError(message) {
  document.getElementById('tasksCard').style.display = 'none'
  document.getElementById('progressFill').style.width = '100%'
  document.getElementById('progressFill').style.backgroundColor = '#ef4444'
  document.getElementById('progressLabel').textContent = 'Processing Failed'
  document.getElementById('errorMessage').textContent = message
  document.getElementById('error-state').style.display = 'block'
}

// DOWNLOAD
async function handleDownload() {
  try {
    const res = await fetch(`/api/download/${encodeURIComponent(currentFileKey)}/`)
    const data = await res.json()
    if (!res.ok) throw new Error(data.error || 'Download failed')
    window.location.href = data.download_url
  } catch (err) {
    showError(err.message)
  }
}

// RESTART 
function handleRestart() {
  currentFileKey = null
  faceFiles = []
  currentVideoFile = null
  document.getElementById('restartBtn').style.display = 'none'
  removeVideo()
  renderFaces()
  showPage('page-upload')
}


document.addEventListener('DOMContentLoaded', () => {
  renderFaces()
})