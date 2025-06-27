(function () {
    // 🎯 Sélection des éléments HTML
    const recordBtn = document.getElementById('recordBtn');         // bouton micro (start)
    const stopBtn = document.getElementById('stopBtn');             // bouton stop (arrêt)
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

    // 🚀 Initialisation - Vérifier que tous les éléments existent
    function initializeUI() {
        console.log('🚀 Initialisation de l\'interface audio...');
        
        // Vérifier que tous les éléments essentiels existent
        if (!recordBtn) {
            console.error('❌ Bouton d\'enregistrement non trouvé dans le DOM');
            return;
        }
        
        if (!stopBtn) {
            console.error('❌ Bouton d\'arrêt non trouvé dans le DOM');
            return;
        }
        
        if (!texteField) {
            console.error('❌ Champ textarea non trouvé dans le DOM');
            return;
        }
        
        if (!audioInput) {
            console.error('❌ Input fichier audio non trouvé dans le DOM');
            return;
        }
        
        // Bind des événements
        recordBtn.addEventListener('click', startRecording);
        stopBtn.addEventListener('click', stopRecording);
        texteField.addEventListener('input', updateCharCount);
        
        // Bind du bouton de transcription (sera disponible après enregistrement)
        if (transcribeBtn) {
            transcribeBtn.addEventListener('click', transcribeAudio);
        }
        
        // Initialiser le compteur
        updateCharCount();
        
        console.log('✅ Interface audio initialisée avec succès');
        console.log('🔍 État initial - Record:', recordBtn.disabled, 'Stop:', stopBtn.disabled);
    }

    // 🔠 Affiche le nombre de caractères du champ texte
    const updateCharCount = () => {
        if (charCount && texteField) {
            charCount.textContent = `${texteField.value.length} caractères`;
        }
    };

    // ⏱️ Format du timer (ex: 01:24)
    const formatTime = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

    // 🧹 Reset UI après enregistrement (boutons, timer, etc.)
    const resetUI = () => {
        console.log('🧹 Reset de l\'interface...');
        
        clearInterval(timerInterval);
        timerInterval = null;
        if (timerDisplay) timerDisplay.textContent = '00:00';
        if (recordingInfo) recordingInfo.classList.add('hidden');

        // Remettre le bouton stop en état désactivé
        if (stopBtn) {
            stopBtn.disabled = true;
            stopBtn.classList.add('opacity-50', 'cursor-not-allowed');
            stopBtn.classList.remove('bg-red-700', 'hover:bg-red-800');
            stopBtn.classList.add('bg-gray-400');
        }

        // Réactiver le bouton record
        if (recordBtn) {
            recordBtn.disabled = false;
            recordBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        }

        isRecording = false;
        console.log('✅ Interface remise à zéro');
    };

    // 🔴 Démarrer l'enregistrement
    async function startRecording() {
        try {
            if (isRecording) return;

            console.log('🎤 Démarrage de l\'enregistrement...');

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

            // 🎨 MAJ interface
            isRecording = true;
            
            // Désactiver le bouton record
            if (recordBtn) {
                recordBtn.disabled = true;
                recordBtn.classList.add('opacity-50', 'cursor-not-allowed');
            }

            // Activer le bouton stop
            if (stopBtn) {
                stopBtn.disabled = false;
                stopBtn.classList.remove('opacity-50', 'cursor-not-allowed', 'bg-gray-400');
                stopBtn.classList.add('bg-red-700', 'hover:bg-red-800');
            }

            // Timer actif
            if (recordingInfo) recordingInfo.classList.remove('hidden');
            startTimestamp = Date.now();
            timerInterval = setInterval(() => {
                if (timerDisplay) {
                    timerDisplay.textContent = formatTime(Math.floor((Date.now() - startTimestamp) / 1000));
                }
            }, 1000);

            console.log('✅ Enregistrement démarré - Stop button activé');
        } catch (err) {
            console.error('❌ Erreur enregistrement:', err);
            alert('Accès microphone impossible : ' + err.message);
            resetUI();
        }
    }

    // ⏹️ Arrêter l'enregistrement proprement
    function stopRecording() {
        if (!isRecording) return;
        
        console.log('⏹️ Arrêt de l\'enregistrement...');
        
        if (mediaRecorder?.state === 'recording') mediaRecorder.stop();
        if (mediaStream) {
            mediaStream.getTracks().forEach(t => t.stop());
            mediaStream = null;
        }
        resetUI();
        
        console.log('✅ Enregistrement arrêté');
    }

    // 🎧 Créer le fichier audio à partir du blob et l'injecter dans l'input file
    function buildAudioFile() {
        if (!audioChunks.length) return;
        
        console.log('🎧 Construction du fichier audio...');
        
        const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
        const url = URL.createObjectURL(blob);

        // Prévisualisation
        if (audioPlayer) {
            audioPlayer.src = url;
        }
        if (audioSection) {
            audioSection.classList.remove('hidden');
        }

        // Injection dans input[type="file"]
        if (audioInput) {
            const file = new File([blob], 'recording.webm', { type: blob.type });
            const dt = new DataTransfer();
            dt.items.add(file);
            audioInput.files = dt.files;
        }

        // Bind du bouton de transcription maintenant qu'il existe
        const transcribeBtn = document.getElementById('transcribeBtn');
        if (transcribeBtn) {
            transcribeBtn.addEventListener('click', transcribeAudio);
        }
        
        console.log('✅ Fichier audio créé et injecté');
    }

    // ✨ Transcrire l'audio via l'API Django
    async function transcribeAudio() {
        if (!audioInput?.files.length) {
            alert('Aucun fichier audio à transcrire.');
            return;
        }

        const file = audioInput.files[0];
        console.log('🔤 Début transcription...', file.name);
        
        const transcribeBtn = document.getElementById('transcribeBtn');
        if (transcribeBtn) {
            transcribeBtn.disabled = true;
            transcribeBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i> Transcription...';
        }

        try {
            const formData = new FormData();
            formData.append('audio', file, file.name);
            const csrftoken = document.querySelector('input[name="csrfmiddlewaretoken"]')?.value;

            // 🔧 URL CORRIGÉE : /aura/api/transcribe/
            const resp = await fetch('/aura/api/transcribe/', {
                method: 'POST',
                headers: { 'X-CSRFToken': csrftoken },
                body: formData
            });

            if (!resp.ok) throw new Error('Erreur serveur ' + resp.status);

            const data = await resp.json();
            if (!data.text) throw new Error('Réponse invalide');

            if (texteField) {
                texteField.value += `\n\n--- Transcription ---\n${data.text}`;
                updateCharCount();
            }

            if (transcribeBtn) {
                transcribeBtn.innerHTML = '<i class="fas fa-check mr-1"></i> Transcrit';
            }
            
            console.log('✅ Transcription réussie');
        } catch (err) {
            console.error('❌ Erreur transcription:', err);
            alert('Transcription échouée : ' + err.message);
            if (transcribeBtn) {
                transcribeBtn.innerHTML = '<i class="fas fa-times mr-1"></i> Erreur';
            }
        } finally {
            if (transcribeBtn) {
                transcribeBtn.disabled = false;
            }
        }
    }

    // ⌨️ Raccourcis clavier (espace pour start/stop, Échap pour stop)
    document.addEventListener('keydown', e => {
        if (e.code === 'Space' && !e.target.matches('input,textarea')) {
            e.preventDefault();
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }
        if (e.code === 'Escape' && isRecording) {
            e.preventDefault();
            stopRecording();
        }
    });

    // 🔁 On quitte la page = on coupe proprement
    window.addEventListener('beforeunload', stopRecording);

    // 🚀 Initialiser quand le DOM est prêt
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeUI);
    } else {
        initializeUI();
    }
})();