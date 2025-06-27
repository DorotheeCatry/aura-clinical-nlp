(function () {
    // 🎯 Sélection des éléments HTML
    const recordBtn = document.getElementById('recordBtn');         // bouton micro (start/stop toggle)
    const transcribeBtn = document.getElementById('transcribeBtn'); // bouton transcrire
    const audioPlayer = document.getElementById('audioPlayer');     // lecteur audio
    const audioSection = document.getElementById('audioSection');   // bloc audio visible après enregistrement
    const timerDisplay = document.getElementById('timer');          // minuteur
    const recordingInfo = document.getElementById('recordingInfo'); // "Enregistrement..." + timer
    const charCount = document.getElementById('charCount');         // compteur caractères
    const texteField = document.querySelector('textarea');          // champ texte observation
    const audioInput = document.querySelector('input[type="file"]');// champ input fichier audio

    // 🔧 Variables d'état
    let mediaRecorder = null;
    let mediaStream = null;
    let audioChunks = [];
    let isRecording = false;
    let timerInterval = null;
    let startTimestamp = 0;

    // 🔠 Affiche le nombre de caractères du champ texte
    const updateCharCount = () => {
        charCount.textContent = `${texteField.value.length} caractères`;
    };

    // ⏱️ Format du timer (ex: 01:24)
    const formatTime = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

    // 🧹 Reset UI après enregistrement
    const resetUI = () => {
        clearInterval(timerInterval);
        timerInterval = null;
        timerDisplay.textContent = '00:00';
        recordingInfo.classList.add('hidden');

        // Bouton redevient "record" (rouge)
        recordBtn.innerHTML = '<i class="fas fa-microphone text-lg"></i>';
        recordBtn.classList.remove('bg-red-700', 'bg-gray-600');
        recordBtn.classList.add('bg-red-500', 'hover:bg-red-600');
        recordBtn.disabled = false;
        recordBtn.title = 'Démarrer l\'enregistrement';
        
        isRecording = false;
    };

    // 🔴 Démarrer l'enregistrement
    async function startRecording() {
        try {
            if (isRecording) return;

            // 🔊 Demande d'accès au micro
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // 🧠 Choix du format MIME (webm/opus si supporté)
            const opts = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? { mimeType: 'audio/webm;codecs=opus' }
                : { mimeType: 'audio/webm' };

            mediaRecorder = new MediaRecorder(mediaStream, opts);
            audioChunks = [];

            mediaRecorder.ondataavailable = e => {
                if (e.data.size) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = buildAudioFile;
            mediaRecorder.start();

            // 🎨 MAJ interface - Bouton devient "stop" (gris foncé)
            isRecording = true;
            recordBtn.innerHTML = '<i class="fas fa-stop text-lg"></i>';
            recordBtn.classList.remove('bg-red-500', 'hover:bg-red-600');
            recordBtn.classList.add('bg-red-700', 'hover:bg-red-800');
            recordBtn.title = 'Arrêter l\'enregistrement';

            // Timer actif
            recordingInfo.classList.remove('hidden');
            startTimestamp = Date.now();
            timerInterval = setInterval(() => {
                timerDisplay.textContent = formatTime(Math.floor((Date.now() - startTimestamp) / 1000));
            }, 1000);
        } catch (err) {
            alert('Accès microphone impossible : ' + err.message);
            resetUI();
        }
    }

    // ⏹️ Arrêter l'enregistrement proprement
    function stopRecording() {
        if (!isRecording) return;
        if (mediaRecorder?.state === 'recording') mediaRecorder.stop();
        mediaStream?.getTracks().forEach(t => t.stop());
        mediaStream = null;
        resetUI();
    }

    // 🎧 Créer le fichier audio à partir du blob et l'injecter dans l'input file
    function buildAudioFile() {
        if (!audioChunks.length) return;
        const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
        const url = URL.createObjectURL(blob);

        // Prévisualisation
        audioPlayer.src = url;
        audioSection.classList.remove('hidden');

        // Injection dans input[type="file"]
        const file = new File([blob], 'recording.webm', { type: blob.type });
        const dt = new DataTransfer();
        dt.items.add(file);
        audioInput.files = dt.files;
    }

    // ✨ Transcrire l'audio via l'API Django
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

            const resp = await fetch('/aura/api/transcribe/', {
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

    // 🎮 Toggle record/stop avec un seul bouton
    function toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }

    // 🎮 Bind des événements
    recordBtn.addEventListener('click', toggleRecording);
    if (transcribeBtn) transcribeBtn.addEventListener('click', transcribeAudio);

    texteField.addEventListener('input', updateCharCount);

    // ⌨️ Raccourcis clavier (espace pour toggle, Échap pour stop)
    document.addEventListener('keydown', e => {
        if (e.code === 'Space' && !e.target.matches('input,textarea')) {
            e.preventDefault();
            toggleRecording();
        }
        if (e.code === 'Escape' && isRecording) {
            e.preventDefault();
            stopRecording();
        }
    });

    // 🔁 On quitte la page = on coupe proprement
    window.addEventListener('beforeunload', stopRecording);

    // 🚀 Initialisation au chargement
    function initializeUI() {
        // S'assurer que le bouton est en mode "record" au démarrage
        recordBtn.innerHTML = '<i class="fas fa-microphone text-lg"></i>';
        recordBtn.classList.remove('bg-red-700', 'bg-gray-600', 'bg-gray-400', 'cursor-not-allowed');
        recordBtn.classList.add('bg-red-500', 'hover:bg-red-600', 'cursor-pointer');
        recordBtn.disabled = false;
        recordBtn.title = 'Démarrer l\'enregistrement';
        
        updateCharCount(); // maj dès le chargement
    }

    // Initialiser l'interface au chargement
    initializeUI();
})();