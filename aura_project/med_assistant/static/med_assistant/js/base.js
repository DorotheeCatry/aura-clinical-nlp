/**
 * Base JavaScript - Fonctionnalités communes à toutes les pages
 * Gestion de la sidebar, messages, et interactions générales
 */

document.addEventListener('DOMContentLoaded', function() {
    // Sidebar toggle pour mobile
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');

    if (sidebarToggle && sidebar && overlay) {
        sidebarToggle.addEventListener('click', function () {
            sidebar.classList.toggle('open');
            overlay.classList.toggle('hidden');
        });

        overlay.addEventListener('click', function () {
            sidebar.classList.remove('open');
            overlay.classList.add('hidden');
        });
    }

    // Auto-hide des messages après 5 secondes
    setTimeout(function () {
        const messages = document.querySelectorAll('.fade-in');
        messages.forEach(function (message) {
            if (message.classList.contains('border-l-4')) {
                message.style.transition = 'all 0.5s ease-out';
                message.style.opacity = '0';
                message.style.transform = 'translateY(-20px)';
                setTimeout(function () {
                    message.remove();
                }, 500);
            }
        });
    }, 5000);

    // Animation des barres de progression génériques
    const progressBars = document.querySelectorAll('.progress-bar');
    progressBars.forEach(bar => {
        const width = bar.getAttribute('data-width') || bar.style.width;
        if (width) {
            bar.style.width = '0%';
            setTimeout(() => {
                bar.style.width = width;
            }, 500);
        }
    });
});