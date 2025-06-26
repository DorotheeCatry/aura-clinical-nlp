// audio_recorder.js – enregistrement, arrêt fiable + transcription réelle via endpoint Django
// Place ce fichier dans theme/static/js/audio_recorder.js
// BACK‑END : prévois une vue Django POST /api/transcribe/ qui renvoie { "text": "..." }

(function () {
    // Sélecteurs DOM ------------------------------------------------------
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const transcribeBtn = document.getElementById('transcribeBtn');
    const audioPlayer = document.getElementById('audioPlayer');
    const audioSection = document.getElementById('audioSection');
    const timerDisplay = document.getElementById('timer');
    const recordingInfo = document.getElementById('recordingInfo');
    const charCount = document.getElementById('charCount');
    const texteField = document.querySelector('textarea');
    const audioInput = document.querySelector('input[type="file"]');

    // États ---------------------------------------------------------------
    let mediaRecorder = null;
    let mediaStream = null;
    let audioChunks = [];
    let isRecording = false;
    let timerInterval = null;
    let startTimestamp = 0;

    // Helpers -------------------------------------------------------------
    const updateCharCount = () => {
        charCount.textContent = `${texteField.value.length} caractères`;
    };

    const formatTime = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

    const resetUI = () => {
        clearInterval(timerInterval);
        timerInterval = null;
        timerDisplay.textContent = '00:00';
        recordingInfo.classList.add('hidden');
        recordBtn.disabled = false;
        stopBtn.disabled = true;
        stopBtn.classList.add('cursor-not-allowed');
        stopBtn.classList.replace('bg-red-600', 'bg-gray-400');
        isRecording = false;
    };

    // Enregistrement ------------------------------------------------------
    async function startRecording() {
        try {
            if (isRecording) return;
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const opts = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? { mimeType: 'audio/webm;codecs=opus' } : { mimeType: 'audio/webm' };
            mediaRecorder = new MediaRecorder(mediaStream, opts);
            audioChunks = [];
            mediaRecorder.ondataavailable = e => e.data.size && audioChunks.push(e.data);
            mediaRecorder.onstop = buildAudioFile;
            mediaRecorder.start();
            // UI
            isRecording = true;
            recordBtn.disabled = true;
            stopBtn.disabled = false;
            stopBtn.classList.remove('cursor-not-allowed');
            stopBtn.classList.replace('bg-gray-400', 'bg-red-600');
            recordingInfo.classList.remove('hidden');
            startTimestamp = Date.now();
            timerInterval = setInterval(() => { timerDisplay.textContent = formatTime(Math.floor((Date.now() - startTimestamp) / 1000)); }, 1000);
        } catch (err) {
            alert('Accès microphone impossible : ' + err.message);
            resetUI();
        }
    }

    function stopRecording() {
        if (!isRecording) return;
        if (mediaRecorder?.state === 'recording') mediaRecorder.stop();
        mediaStream?.getTracks().forEach(t => t.stop());
        mediaStream = null;
        resetUI();
    }

    // Création du blob + injection dans input --------------------------------
    function buildAudioFile() {
        if (!audioChunks.length) return;
        const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
        const url = URL.createObjectURL(blob);
        audioPlayer.src = url;
        audioSection.classList.remove('hidden');
        const file = new File([blob], 'recording.webm', { type: blob.type });
        const dt = new DataTransfer();
        dt.items.add(file);
        audioInput.files = dt.files;
    }

    // Transcription réelle -------------------------------------------------
    async function transcribeAudio() {
        if (!audioInput.files.length) {
            alert('Aucun fichier audio à transcrire.');
            return;
        }
        const file = audioInput.files[0];
        transcribeBtn.disabled = true;
        transcribeBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i> Transcription...';
        try {
            const formData = new FormData();
            formData.append('audio', file, file.name);
            const csrftoken = document.querySelector('input[name="csrfmiddlewaretoken"]').value;
            const resp = await fetch('/api/transcribe/', {
                method: 'POST',
                headers: { 'X-CSRFToken': csrftoken },
                body: formData
            });
            if (!resp.ok) throw new Error('Erreur serveur ' + resp.status);
            const data = await resp.json();
            if (!data.text) throw new Error('Réponse invalide');
            texteField.value += `\n\n--- Transcription ---\n${data.text}`;
            updateCharCount();
            transcribeBtn.innerHTML = '<i class="fas fa-check mr-1"></i> Transcrit';
        } catch (err) {
            console.error(err);
            alert('Transcription échouée : ' + err.message);
            transcribeBtn.innerHTML = '<i class="fas fa-times mr-1"></i> Erreur';
        } finally {
            transcribeBtn.disabled = false;
        }
    }

    // Events ---------------------------------------------------------------
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
    if (transcribeBtn) transcribeBtn.addEventListener('click', transcribeAudio);
    texteField.addEventListener('input', updateCharCount);
    document.addEventListener('keydown', e => {
        if (e.code === 'Space' && !e.target.matches('input,textarea')) { e.preventDefault(); isRecording ? stopRecording() : startRecording(); }
        if (e.code === 'Escape' && isRecording) { e.preventDefault(); stopRecording(); }
    });
    window.addEventListener('beforeunload', stopRecording);

    updateCharCount();
})();
