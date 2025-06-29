/**
 * Statistics JavaScript - Animations pour les statistiques
 * Gestion des barres de progression et effets visuels
 */

// Animation des barres de progression au chargement
document.addEventListener('DOMContentLoaded', function() {
    const progressBars = document.querySelectorAll('[data-width]');
    progressBars.forEach(bar => {
        const width = bar.getAttribute('data-width');
        bar.style.width = '0%';
        setTimeout(() => {
            bar.style.transition = 'width 1.5s ease-out';
            bar.style.width = width;
        }, 500);
    });
});