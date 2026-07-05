// ═══════════════════════════════════════════════════════════════════
//  AeroSplat-GIS — Оптимізований WebGL2 Instanced 3DGS Viewer
//  v3: Апаратний Opaque-перформанс (60+ FPS) без CPU-сортування
// ═══════════════════════════════════════════════════════════════════

const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf8fafc); // Інженерний світлий фон

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000);
camera.position.set(0, 300, 600);

let isTopDown = false;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

let controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

let activeMesh = null;
let sceneBounds = null; // { center: Vector3, radius, minY, maxY }
let rawInstanceData = null;
let needsDepthSort = false; // Вимкнено на користь апаратного Z-буфера

// ── Шейдери ───────────────────────────────────────────────────────────
const vertexShaderGS = `#version 300 es
    precision highp float;
    
    in vec3 position;
    in vec2 uv;

    in vec3 aCenter;
    in vec3 aScale;
    in vec4 aRotation;
    in float aOpacity;
    in vec3 aColor;

    out vec3 vColor;
    out vec2 vUV;
    out float vOpacity;
    out vec3 vNormal;   
    out float vWorldY;  

    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;

    mat3 quatToMat(vec4 q) {
        q = normalize(q);
        float w = q.x, x = q.y, y = q.z, z = q.w;
        return mat3(
            1.0 - 2.0*(y*y+z*z),   2.0*(x*y-w*z),       2.0*(x*z+w*y),
            2.0*(x*y+w*z),          1.0 - 2.0*(x*x+z*z), 2.0*(y*z-w*x),
            2.0*(x*z-w*y),          2.0*(y*z+w*x),       1.0 - 2.0*(x*x+y*y)
        );
    }

    void main() {
        vColor = aColor;
        vUV = uv;
        vOpacity = aOpacity;

        // Обчислюємо відстань до камери для динамічної компенсації LOD
        vec4 mvPosition = modelViewMatrix * vec4(aCenter, 1.0);
        float distToCam = length(mvPosition.xyz);
        
        // Авто-розширення віддалених точок для щільності покриття сцени
        float lodScale = 1.0 + max(0.0, distToCam * 0.012);

        vec3 localVertex;
        vec3 worldPos;

        if (aScale.y < 1e-4) {
            localVertex = vec3(position.x * aScale.x * lodScale, 0.0, position.y * aScale.z * lodScale);
            worldPos = aCenter + localVertex;
            vNormal = vec3(0.0, 1.0, 0.0);
        } 
        else {
            localVertex = position * vec3(aScale.x * lodScale, aScale.y * lodScale, 1.0);
            
            mat3 R = quatToMat(aRotation);
            vec3 rotatedVertex = R * localVertex;
            worldPos = aCenter + rotatedVertex;
            vNormal = R * vec3(0.0, 0.0, 1.0);
        }

        vWorldY = worldPos.y;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(worldPos, 1.0);
    }
`;

const fragmentShaderGS = `#version 300 es
    precision highp float;
    
    in vec3 vColor;
    in vec2 vUV;
    in float vOpacity;
    in vec3 vNormal;
    in float vWorldY;
    
    out vec4 fragColor;

    const vec3 uLightDir = vec3(0.35, 0.82, 0.45);

    uniform bool uCutawayEnabled;
    uniform float uCutawayHeight;

    void main() {
        if (uCutawayEnabled && vWorldY > uCutawayHeight) discard;

        // Центруємо UV в межі [-1.0, 1.0]
        vec2 d = (vUV - vec2(0.5)) * 2.0;
        float dist_sq = dot(d, d);

        // Чітке апаратне кругове відсікання
        if (dist_sq > 1.0) discard;

        // Імітація об'єму (фаски) країв за рахунок затемнення кольору, без прозорості
        float rim = smoothstep(0.45, 1.0, dist_sq);
        vec3 shadedColor = mix(vColor, vColor * 0.68, rim * 0.35);

        float ndotl = max(dot(normalize(vNormal), normalize(uLightDir)), 0.0);
        float lightTerm = mix(0.78, 1.12, ndotl);
        vec3 litColor = shadedColor * lightTerm;

        fragColor = vec4(litColor, 1.0); 
    }
`;

// ── UI-оверлеї ───────────────────────────────────────────────────────
const uiRoot = document.createElement('div');
uiRoot.style.cssText = 'position:absolute; top:16px; left:16px; z-index:20; display:flex; flex-direction:column; gap:8px; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;';
container.appendChild(uiRoot);

function makeButton(label) {
    const btn = document.createElement('button');
    btn.textContent = label;
    btn.style.cssText = 'padding:8px 14px; background:#ffffff; border:1px solid #cbd5e0; border-radius:6px; font-size:12px; font-weight:600; color:#2d3748; cursor:pointer; box-shadow:0 2px 6px rgba(0,0,0,0.08);';
    btn.onmouseenter = () => btn.style.background = '#edf2f7';
    btn.onmouseleave = () => btn.style.background = '#ffffff';
    return btn;
}

const viewToggleBtn = makeButton('🗺️ Вид згори (Top-Down)');
uiRoot.appendChild(viewToggleBtn);

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

// ── Тактичне керування камерою ───────────────────────────────────────
function switchCamera(toTopDown) {
    isTopDown = toTopDown;

    if (sceneBounds) {
        if (isTopDown) {
            const dist = sceneBounds.radius * 2.4 + 8;
            const angleFromHorizon = (75 * Math.PI) / 180;
            const horizontalDist = dist * Math.cos(angleFromHorizon);
            const heightDist = dist * Math.sin(angleFromHorizon);
            camera.position.set(
                sceneBounds.center.x,
                sceneBounds.center.y + heightDist,
                sceneBounds.center.z + horizontalDist
            );
        } else {
            camera.position.set(
                sceneBounds.center.x,
                sceneBounds.center.y + sceneBounds.radius * 1.2,
                sceneBounds.center.z + sceneBounds.radius * 1.8
            );
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

// Заглушка для зворотної сумісності подій OrbitControls
function requestDepthSort() {}

// ── Завантаження бінарних даних ───────────────────────────────────────
async function loadSpatialData() {
    try {
        const response = await fetch(`/api/v1/spatial-chunk?mode=splat`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const latency = response.headers.get('X-Processing-Time-Ms');
        const count = response.headers.get('X-Gaussians-Count');
        if (latency) document.getElementById('latency-val').innerText = latency + ' ms';
        if (count) document.getElementById('count-val').innerText = parseInt(count).toLocaleString();

        const buffer = await response.arrayBuffer();
        const floatArr = new Float32Array(buffer);

        if (activeMesh) { scene.remove(activeMesh); activeMesh.geometry.dispose(); activeMesh.material.dispose(); }

        const N = Math.floor(floatArr.length / 14);
        
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

        rawInstanceData = { centers, scales, rotations, opacities, colors, N };

        geometry.setAttribute('aCenter', new THREE.InstancedBufferAttribute(centers, 3));
        geometry.setAttribute('aScale', new THREE.InstancedBufferAttribute(scales, 3));
        geometry.setAttribute('aRotation', new THREE.InstancedBufferAttribute(rotations, 4));
        geometry.setAttribute('aOpacity', new THREE.InstancedBufferAttribute(opacities, 1));
        geometry.setAttribute('aColor', new THREE.InstancedBufferAttribute(colors, 3));

        const material = new THREE.RawShaderMaterial({
            vertexShader: vertexShaderGS,
            fragmentShader: fragmentShaderGS,
            uniforms: {
                uCutawayEnabled: { value: false },
                uCutawayHeight: { value: 999.0 }
            },
            transparent: false, // Opaque-режим: Z-буфер працює на повну потужність
            depthWrite: true,
            depthTest: true,
            side: THREE.DoubleSide
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
            camera.lookAt(bs.center);
            controls.update();

            cutawaySlider.min = minY.toFixed(1);
            cutawaySlider.max = maxY.toFixed(1);
            cutawaySlider.step = 0.1;
            cutawaySlider.value = maxY.toFixed(1);
            updateCutawayUniforms();
        }

        needsDepthSort = false; // Примусове вимкнення CPU-сортування
    } catch (err) { console.error('Помилка завантаження 3DGS:', err); }
}

controls.addEventListener('change', updateScaleBarAndCompass);

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    updateScaleBarAndCompass();
    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    const w = window.innerWidth, h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
});

loadSpatialData();
animate();