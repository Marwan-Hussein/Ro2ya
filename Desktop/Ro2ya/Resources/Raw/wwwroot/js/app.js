document.addEventListener('DOMContentLoaded', () => {
    const iframe = document.getElementById('content-frame');
    const navLinks = document.querySelectorAll('.nav-link');

    const loadPage = (page) => {
        // Only update src if it's different to prevent unnecessary reloads
        const newSrc = `pages/${page}.html`;
        if (iframe && !iframe.src.endsWith(newSrc)) {
            iframe.src = newSrc;
        }
        updateActiveNav(page);
    };

    const updateActiveNav = (page) => {
        navLinks.forEach(link => {
            const linkPage = link.getAttribute('href').substring(1);
            if (linkPage === page) {
                link.className = "flex items-center gap-md text-primary-container font-semibold border-l-2 border-primary-container bg-surface-container-high px-lg py-md nav-link active";
            } else {
                link.className = "flex items-center gap-md text-on-surface-variant font-body-md px-lg py-md hover:bg-surface-variant transition-colors nav-link";
            }
        });
    };

    window.addEventListener('hashchange', () => {
        const page = window.location.hash.substring(1) || 'dashboard';
        loadPage(page);
    });

    // Initial load
    const initialPage = window.location.hash.substring(1) || 'dashboard';
    loadPage(initialPage);
});