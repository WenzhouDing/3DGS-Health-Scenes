// The header stays outside the viewer's camera controls.
document.documentElement.classList.toggle('viewer-no-ui', new URL(location.href).searchParams.has('noui'));
const siteHeader = document.querySelector('.site-header');
if (siteHeader) {
    // Let keyup reach the viewer so a previously held movement key is released.
    for (const type of ['keydown', 'pointerdown', 'wheel']) {
        siteHeader.addEventListener(type, event => event.stopPropagation());
    }
}
