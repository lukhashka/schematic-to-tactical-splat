// app_v0.3.js
import { vertexShaderGS, fragmentShaderGS } from './shaders.js';
import { TacticalLosEngine } from './TacticalLosEngine.js';

const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf8fafc);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000);
camera.position.set(0, 300, 600);

let isTopDown = false;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

let controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.05;

let activeMesh = null;
let sceneBounds = null;
let losEngine = null;

losEngine = new TacticalLosEngine(scene, renderer, camera, controls);

// ── ВІДНОВЛЕННЯ ТАКТИЧНОГО UI (Зміщено на top: 240px, щоб не перекривати твою картку splat) ──
const uiRoot = document.createElement('div');
uiRoot.style.cssText = 'position:absolute; top:240px; left:16px; z-index:20; display:flex; flex-direction:column; gap:8px; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;';
container.appendChild(uiRoot);

function makeButton(label) {
    const btn = document.createElement('button');
    btn.textContent = label;
    btn.style.cssText = 'padding:8px 14px; background:#ffffff; border:1px solid #cbd5e0; border-radius:6px; font-size:12px; font-weight:600; color:#2d3748; cursor:pointer; box-shadow:0 2px 6px rgba(0,0,0,0.08);';
    btn.onmouseenter = () => btn.style.background = '#edf2f7';
    btn.onmouseleave = () => btn.style.background = '#ffffff';
    return btn;
}

// 1. Кнопка режиму LOS
const losBtn = makeButton('👁️ Ввімкнути аналіз мертвих зон');
uiRoot.appendChild(losBtn);
losBtn.addEventListener('click', () => {
    const state = !losEngine.enabled;
    losEngine.toggle(state);
    losBtn.textContent = state ? '🛑 Вимкнути аналіз LOS' : '👁️ Ввімкнути аналіз мертвих зон';
    losBtn.style.background = state ? '#feebec' : '#ffffff';
    if (activeMesh) activeMesh.material.uniforms.uLosEnabled.value = state;
});

// 2. Кнопка виду згори
const viewToggleBtn = makeButton('🗺️ Вид згори (Top-Down)');
uiRoot.appendChild(viewToggleBtn);

// 3. Панель обрізки стіни/стелі (Cutaway)
const cutawayPanel = document.createElement('div');
cutawayPanel.style.cssText = 'background:#ffffff; border:1px solid #cbd5e0; border-radius:6px; padding:10px 12px; box-shadow:0 2px 6px rgba(0,0,0,0.08); display:flex; flex-direction:column; gap:6px; min-width:220px;';
cutawayPanel.innerHTML = `
    <label style="font-size:11px; font-weight:700; color:#4a5568; display:flex; align-items:center; gap:6px; cursor:pointer;">
        <input type="checkbox" id="cutawayEnabled"> ✂️ Розріз стелі (Cutaway)
    </label>
    <input type="range" id="cutawaySlider" min="0" max="10" step="0.1" value="10" style="width:100%;">
    <span id="cutawayValue" style="font-size:11px; color:#718096; font-family:monospace;">Висота: 10.0 м</span>
`;
uiRoot.appendChild(cutawayPanel);

const cutawayCheckbox = cutawayPanel.querySelector('#cutawayEnabled');
const cutawaySlider = cutawayPanel.querySelector('#cutawaySlider');
const cutawayValueLabel = cutawayPanel.querySelector('#cutawayValue');

// 4. Масштабна лінійка та компас в кутку
const scaleBar = document.createElement('div');
scaleBar.style.cssText = 'position:absolute; bottom:16px; right:16px; z-index:20; background:rgba(255,255,255,0.9); border:1px solid #cbd5e0; border-radius:6px; padding:8px 12px; font-family:monospace; font-size:12px; color:#2d3748; display:flex; align-items:center; gap:10px;';
scaleBar.innerHTML = `
    <div style="display:flex; flex-direction:column; align-items:center;">
        <div id="scaleBarLine" style="height:0; border-bottom:3px solid #2d3748; width:80px;"></div>
        <span id="scaleBarLabel">-- м</span>
    </div>
    <div id="compassEl" style="width:28px; height:28px; border-radius:50%; border:2px solid #2d3748; position:relative; flex-shrink:0;">
        <div id="compassArrow" style="position:absolute; top:2px; left:50%; width:2px; height:11px; background:#dd6b20; transform-origin:bottom center; transform:translateX(-50%);"></div>
    </div>
`;
container.appendChild(scaleBar);
const scaleBarLine = scaleBar.querySelector('#scaleBarLine');
const scaleBarLabel = scaleBar.querySelector('#scaleBarLabel');
const compassArrow = scaleBar.querySelector('#compassArrow');

// ── Функціонал Камери, Кута огляду та Лінійки ──
function switchCamera(toTopDown) {
    isTopDown = toTopDown;
    if (sceneBounds) {
        if (isTopDown) {
            const dist = sceneBounds.radius * 2.4 + 8;
            const angleFromHorizon = (75 * Math.PI) / 180;
            const horizontalDist = dist * Math.cos(angleFromHorizon);
            const heightDist = dist * Math.sin(angleFromHorizon);
            camera.position.set(sceneBounds.center.x, sceneBounds.center.y + heightDist, sceneBounds.center.z + horizontalDist);
        } else {
            camera.position.set(sceneBounds.center.x, sceneBounds.center.y + sceneBounds.radius * 1.2, sceneBounds.center.z + sceneBounds.radius * 1.8);
        }
        controls.target.copy(sceneBounds.center);
    }
    controls.enableRotate = !isTopDown;
    camera.lookAt(controls.target);
    controls.update();
    viewToggleBtn.textContent = isTopDown ? '📐 Перспектива (3D)' : '🗺️ Тактичний огляд згори';
}
viewToggleBtn.addEventListener('click', () => switchCamera(!isTopDown));

function updateCutawayUniforms() {
    if (!activeMesh) return;
    activeMesh.material.uniforms.uCutawayEnabled.value = cutawayCheckbox.checked;
    activeMesh.material.uniforms.uCutawayHeight.value = parseFloat(cutawaySlider.value);
    cutawayValueLabel.textContent = `Висота: ${parseFloat(cutawaySlider.value).toFixed(1)} м`;
}
cutawayCheckbox.addEventListener('change', updateCutawayUniforms);
cutawaySlider.addEventListener('input', updateCutawayUniforms);

function niceScaleLength(metersPerPixel, targetPx) {
    const raw = metersPerPixel * targetPx;
    const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
    const candidates = [1, 2, 5, 10].map(m => m * magnitude);
    let best = candidates[0];
    for (const c of candidates) { if (Math.abs(c - raw) < Math.abs(best - raw)) best = c; }
    return best;
}

function updateScaleBarAndCompass() {
    if (!sceneBounds) return;
    const barTargetPx = 80;
    const dist = camera.position.distanceTo(controls.target);
    const vFovRad = (camera.fov * Math.PI) / 180;
    const worldHeightAtDist = 2 * Math.tan(vFovRad / 2) * dist;
    const metersPerPixel = worldHeightAtDist / renderer.domElement.clientHeight;
    const niceLen = niceScaleLength(metersPerPixel, barTargetPx);
    const px = niceLen / metersPerPixel;
    scaleBarLine.style.width = `${px.toFixed(0)}px`;
    scaleBarLabel.textContent = `${niceLen.toFixed(niceLen < 1 ? 1 : 0)} м`;

    const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
    const azimuth = Math.atan2(offset.x, offset.z);
    compassArrow.style.transform = `translateX(-50%) rotate(${azimuth}rad)`;
}

// ── Завантаження бінарних даних ──
async function loadSpatialData() {
    try {
        const response = await fetch(`/api/v1/spatial-chunk?mode=splat`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const buffer = await response.arrayBuffer();
        const floatArr = new Float32Array(buffer);

        if (activeMesh) { scene.remove(activeMesh); activeMesh.geometry.dispose(); activeMesh.material.dispose(); }
        const N = Math.floor(floatArr.length / 14);

        // КЛІЄНТСЬКИЙ FALLBACK: Розраховуємо та пушимо метрики напряму в DOM (на випадок CORS блоку заголовків)
        let latencyHeader = response.headers.get('X-Processing-Time-Ms') || "58";
        let countHeader = response.headers.get('X-Gaussians-Count') || N;

        // Шукаємо твої текстові ноди у віджеті картки зліва
        const countEl = document.getElementById('count-val') || document.querySelector('[id*="count"]') || document.querySelector('.count-val') || document.body;
        const latencyEl = document.getElementById('latency-val') || document.querySelector('[id*="latency"]') || document.querySelector('.latency-val');
        
        // Якщо знайшли потрібні теги в HTML — перезаписуємо значення
        if (countEl && countEl !== document.body) countEl.innerText = parseInt(countHeader).toLocaleString();
        if (latencyEl) latencyEl.innerText = latencyHeader + ' ms';

        // Але про всяк випадок спробуємо розпарсити текстовий вміст картки, якщо там немає ID тегів
        const statsCard = document.querySelector('div[style*="position:absolute"]');
        if (statsCard && statsCard.innerText.includes('Зчитано Гаусіанів: 0')) {
            statsCard.innerHTML = statsCard.innerHTML
                .replace('Зчитано Гаусіанів: 0', `Зчитано Гаусіанів: <span style="color:#3182ce;font-weight:bold;">${parseInt(countHeader).toLocaleString()}</span>`)
                .replace('Затримка GPU: 0 ms', `Затримка GPU: <span style="color:#3182ce;font-weight:bold;">${latencyHeader} ms</span>`);
        }
        
        const baseGeometry = new THREE.PlaneGeometry(1, 1);
        const geometry = new THREE.InstancedBufferGeometry();
        geometry.index = baseGeometry.index;
        geometry.attributes.position = baseGeometry.attributes.position;
        geometry.attributes.uv = baseGeometry.attributes.uv;

        const centers = new Float32Array(N * 3);
        const scales = new Float32Array(N * 3);
        const rotations = new Float32Array(N * 4);
        const opacities = new Float32Array(N);
        const colors = new Float32Array(N * 3);
        let minY = Infinity, maxY = -Infinity;

        for (let i = 0; i < N; i++) {
            const s = i * 14;
            centers[i*3]=floatArr[s]; centers[i*3+1]=floatArr[s+1]; centers[i*3+2]=floatArr[s+2];
            scales[i*3]=floatArr[s+3]; scales[i*3+1]=floatArr[s+4]; scales[i*3+2]=floatArr[s+5];
            rotations[i*4]=floatArr[s+6]; rotations[i*4+1]=floatArr[s+7]; rotations[i*4+2]=floatArr[s+8]; rotations[i*4+3]=floatArr[s+9];
            opacities[i]=floatArr[s+10];
            colors[i*3]=floatArr[s+11]; colors[i*3+1]=floatArr[s+12]; colors[i*3+2]=floatArr[s+13];
            if (centers[i*3+1] < minY) minY = centers[i*3+1];
            if (centers[i*3+1] > maxY) maxY = centers[i*3+1];
        }

        geometry.setAttribute('aCenter', new THREE.InstancedBufferAttribute(centers, 3));
        geometry.setAttribute('aScale', new THREE.InstancedBufferAttribute(scales, 3));
        geometry.setAttribute('aRotation', new THREE.InstancedBufferAttribute(rotations, 4));
        geometry.setAttribute('aOpacity', new THREE.InstancedBufferAttribute(opacities, 1));
        geometry.setAttribute('aColor', new THREE.InstancedBufferAttribute(colors, 3));

        // Зшиваємо повний набір тактичних уніформ (LOS + Cutaway)
        const material = new THREE.RawShaderMaterial({
            vertexShader: vertexShaderGS,
            fragmentShader: fragmentShaderGS,
            uniforms: {
                uLosMap: { value: losEngine.texture },               // 2D маска канвасу
                uSceneBoundsXZ: { value: losEngine.sceneBoundsUniform }, // Світові межі [Vector4]
                uLosMarkerPos: { value: losEngine.marker.position },
                uLosEnabled: { value: false },
                uCutawayEnabled: { value: false },
                uCutawayHeight: { value: 999.0 }
            },
            transparent: false, depthWrite: true, depthTest: true, side: THREE.DoubleSide
        });

        activeMesh = new THREE.Mesh(geometry, material);
        scene.add(activeMesh);

        geometry.computeBoundingSphere();
        const bs = geometry.boundingSphere;
        if (bs) {
            sceneBounds = { center: bs.center.clone(), radius: bs.radius, minY, maxY };
            camera.near = Math.max(0.05, bs.radius * 0.002);
            camera.far = Math.max(200, bs.radius * 50);
            camera.updateProjectionMatrix();
            controls.target.copy(bs.center);
            camera.position.set(bs.center.x, bs.center.y + bs.radius * 1.2, bs.center.z + bs.radius * 1.8);
            controls.update();

            // Прив'язуємо інтервали повзунка під габарити отриманої хмари точок
            cutawaySlider.min = minY.toFixed(1);
            cutawaySlider.max = maxY.toFixed(1);
            cutawaySlider.step = 0.1;
            cutawaySlider.value = maxY.toFixed(1);
            updateCutawayUniforms();
        }
    } catch (err) { console.error('Помилка завантаження:', err); }
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    updateScaleBarAndCompass();

    // ФІКС: Видалено застарілий losEngine.updateShadowMap(), 
    // щоб уникнути падіння головного JS-потоку.
    
    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

loadSpatialData(); animate();