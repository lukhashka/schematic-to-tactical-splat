// ═══════════════════════════════════════════════════════════════════
//  AeroSplat-GIS — Оптимізований WebGL2 Instanced 3DGS Viewer
// ═══════════════════════════════════════════════════════════════════

const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf8fafc); // Інженерний світлий фон

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100000);
camera.position.set(0, 300, 600);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

let activeMesh = null;

// Сучасний WebGL2 Шейдер для інстансингу еліпсів
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

        vec3 localVertex;

        // ЗАЛІЗОБЕТОННИЙ МАРКЕР ПІДЛОГИ: якщо aScale.y строго дорівнює нулю
        if (aScale.y < 1e-4) {
            // Кладемо квадрат строго в горизонт (X та Z)
            localVertex = vec3(position.x * aScale.x, 0.0, position.y * aScale.z);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(aCenter + localVertex, 1.0);
        } 
        // ЯКЩО ЦЕ СТІНА
        else {
            // Стіна використовує класичні X та Y
            localVertex = position * vec3(aScale.x, aScale.y, 1.0);
            
            mat3 R = quatToMat(aRotation);
            vec3 rotatedVertex = R * localVertex;
            
            gl_Position = projectionMatrix * modelViewMatrix * vec4(aCenter + rotatedVertex, 1.0);
        }
    }
`;

const fragmentShaderGS = `#version 300 es
    precision highp float;
    
    in vec3 vColor;
    in vec2 vUV;
    in float vOpacity;
    
    out vec4 fragColor;

    void main() {
        // Отримуємо координати від -1.0 до 1.0 відносно центру гауссіана
        vec2 d = (vUV - vec2(0.5)) * 2.0;
        float dist_sq = dot(d, d);
        
        // Обмежуємо радіус впливу
        if (dist_sq > 1.0) discard;

        // Разом замість жорстких меж рахуємо експоненційне падіння щільності (Gaussian fallback)
        // Коефіцієнт 0.5 контролює швидкість розмиття країв
        float gaussian = exp(-0.5 * dist_sq * 4.0);
        
        // Підсумкова альфа — комбінація базової прозорості та гауссової функції
        float final_alpha = vOpacity * gaussian;

        // Якщо альфа занадто мала, відсікаємо піксель для оптимізації Z-буфера
        if (final_alpha < 0.05) discard;

        fragColor = vec4(vColor, final_alpha); 
    }
`;

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
        
        // Створюємо базову геометрію одиничного квадрата (Quad)
        const baseGeometry = new THREE.PlaneGeometry(1, 1);
        const geometry = new THREE.InstancedBufferGeometry();
        geometry.index = baseGeometry.index;
        geometry.attributes.position = baseGeometry.attributes.position;
        geometry.attributes.uv = baseGeometry.attributes.uv;

        // Масиви під інстанси
        const centers = new Float32Array(N * 3);
        const scales = new Float32Array(N * 3);
        const rotations = new Float32Array(N * 4);
        const opacities = new Float32Array(N);
        const colors = new Float32Array(N * 3);

        for (let i = 0; i < N; i++) {
            const s = i * 14;
            // Позиції (Центри)
            centers[i*3]=floatArr[s]; centers[i*3+1]=floatArr[s+1]; centers[i*3+2]=floatArr[s+2];
            // Масштаби
            scales[i*3]=floatArr[s+3]; scales[i*3+1]=floatArr[s+4]; scales[i*3+2]=floatArr[s+5];
            // Кватерніони
            rotations[i*4]=floatArr[s+6]; rotations[i*4+1]=floatArr[s+7]; rotations[i*4+2]=floatArr[s+8]; rotations[i*4+3]=floatArr[s+9];
            // Прозорість та Колір
            opacities[i]=floatArr[s+10];
            colors[i*3]=floatArr[s+11]; colors[i*3+1]=floatArr[s+12]; colors[i*3+2]=floatArr[s+13];
        }

        // Зашиваємо інстанс-атрибути
        geometry.setAttribute('aCenter', new THREE.InstancedBufferAttribute(centers, 3));
        geometry.setAttribute('aScale', new THREE.InstancedBufferAttribute(scales, 3));
        geometry.setAttribute('aRotation', new THREE.InstancedBufferAttribute(rotations, 4));
        geometry.setAttribute('aOpacity', new THREE.InstancedBufferAttribute(opacities, 1));
        geometry.setAttribute('aColor', new THREE.InstancedBufferAttribute(colors, 3));

        const material = new THREE.RawShaderMaterial({
            vertexShader: vertexShaderGS,
            fragmentShader: fragmentShaderGS,
            // ВИПРАВЛЕНО: aOpacity тепер реально використовується у фрагментному шейдері,
            // тож transparent:true потрібен, інакше альфа-канал ігнорується рендерером.
            transparent: true,
            depthWrite: true,
            depthTest: true,
            side: THREE.DoubleSide // Щоб стіни було видно як зсередини, так і ззовні
        });

        activeMesh = new THREE.Mesh(geometry, material);
        scene.add(activeMesh);

        geometry.computeBoundingSphere();
        const bs = geometry.boundingSphere;
        if (bs) { controls.target.copy(bs.center); camera.position.set(bs.center.x, bs.center.y + bs.radius * 1.2, bs.center.z + bs.radius * 1.8); camera.lookAt(bs.center); controls.update(); }
    } catch (err) { console.error('Помилка завантаження 3DGS:', err); }
}

function animate() {
    requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera);
}
window.addEventListener('resize', () => { camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); });
loadSpatialData(); animate();