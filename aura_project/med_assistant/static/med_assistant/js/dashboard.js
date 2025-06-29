/**
 * Dashboard JavaScript - Gestion des graphiques et animations
 * Utilise ApexCharts pour les visualisations de données
 */

// Animation des barres de progression au chargement
document.addEventListener('DOMContentLoaded', function() {
    const progressBars = document.querySelectorAll('.progress-bar');
    progressBars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0%';
        setTimeout(() => {
            bar.style.width = width;
        }, 500);
    });

    // Graphique des consultations hebdomadaires - VERSION COMPACTE
    const weeklyOptions = {
        colors: ["#3B82F6"],
        series: [{
            name: "Consultations",
            data: [
                {% for day, count in weekly_consultations.items %}
                { x: "{{ day|slice:':3' }}", y: {{ count }} },
                {% endfor %}
            ],
        }],
        chart: {
            type: "bar",
            height: 200,
            fontFamily: "Inter, sans-serif",
            toolbar: { show: false },
            sparkline: { enabled: false }
        },
        plotOptions: {
            bar: {
                horizontal: false,
                columnWidth: "60%",
                borderRadiusApplication: "end",
                borderRadius: 4,
            },
        },
        tooltip: {
            shared: true,
            intersect: false,
            style: { fontFamily: "Inter, sans-serif" },
        },
        stroke: {
            show: true,
            width: 0,
            colors: ["transparent"],
        },
        grid: {
            show: true,
            strokeDashArray: 2,
            borderColor: '#f1f5f9',
            padding: { left: 10, right: 10, top: 0, bottom: 0 }
        },
        dataLabels: { enabled: false },
        legend: { show: false },
        xaxis: {
            labels: {
                show: true,
                style: {
                    fontFamily: "Inter, sans-serif",
                    fontSize: '11px',
                    colors: '#64748b'
                }
            },
            axisBorder: { show: false },
            axisTicks: { show: false },
        },
        yaxis: {
            show: true,
            labels: {
                style: {
                    fontFamily: "Inter, sans-serif",
                    fontSize: '10px',
                    colors: '#64748b'
                }
            }
        },
        fill: { opacity: 0.9 },
    };

    if(document.getElementById("weekly-chart") && typeof ApexCharts !== 'undefined') {
        const weeklyChart = new ApexCharts(document.getElementById("weekly-chart"), weeklyOptions);
        weeklyChart.render();
    }

    // Graphique des services hospitaliers - VERSION COMPACTE
    const servicesData = window.servicesData || [];
    if (servicesData.length > 0) {
        const servicesOptions = {
            colors: ["#10B981"],
            series: [{
                name: "Patients",
                data: servicesData,
            }],
            chart: {
                type: "bar",
                height: 200,
                fontFamily: "Inter, sans-serif",
                toolbar: { show: false },
                sparkline: { enabled: false }
            },
            plotOptions: {
                bar: {
                    horizontal: false,
                    columnWidth: "60%",
                    borderRadiusApplication: "end",
                    borderRadius: 4,
                },
            },
            tooltip: {
                shared: true,
                intersect: false,
                style: { fontFamily: "Inter, sans-serif" },
            },
            stroke: {
                show: true,
                width: 0,
                colors: ["transparent"],
            },
            grid: {
                show: true,
                strokeDashArray: 2,
                borderColor: '#f1f5f9',
                padding: { left: 10, right: 10, top: 0, bottom: 0 }
            },
            dataLabels: { enabled: false },
            legend: { show: false },
            xaxis: {
                labels: {
                    show: true,
                    style: {
                        fontFamily: "Inter, sans-serif",
                        fontSize: '11px',
                        colors: '#64748b'
                    }
                },
                axisBorder: { show: false },
                axisTicks: { show: false },
            },
            yaxis: {
                show: true,
                labels: {
                    style: {
                        fontFamily: "Inter, sans-serif",
                        fontSize: '10px',
                        colors: '#64748b'
                    }
                }
            },
            fill: { opacity: 0.9 },
        };

        if(document.getElementById("services-chart") && typeof ApexCharts !== 'undefined') {
            const servicesChart = new ApexCharts(document.getElementById("services-chart"), servicesOptions);
            servicesChart.render();
        }
    }
});