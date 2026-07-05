
export class TacticalLosEngine {
    constructor(scene, renderer, mainCamera, orbitControls) {
        this.scene = scene;
        this.renderer = renderer;
        this.mainCamera = mainCamera;
        this.controls = orbitControls;
        this.enabled = false;
        this.walls = [];
        
        this.bounds = { minX: -20, minZ: -20, maxX: 20, maxZ: 20 };

        // Сфера-маркер огляду
        const geo = new THREE.SphereGeometry(0.3, 16, 16);
        const mat = new THREE.MeshBasicMaterial({ color: 0x00ffcc, depthTest: false, transparent: true, opacity: 0.8 });
        this.marker = new THREE.Mesh(geo, mat);
        this.marker.position.set(0, 1.6, 0);
        this.marker.visible = false;
        this.scene.add(this.marker);

        // Offscreen-канвас для побудови векторного полігону видимості
        this.canvas = document.createElement('canvas');
        this.canvas.width = 512;
        this.canvas.height = 512;
        this.ctx = this.canvas.getContext('2d');
        
        this.texture = new THREE.CanvasTexture(this.canvas);
        this.texture.minFilter = THREE.LinearFilter;
        this.texture.flipY = false;
        
        this.sceneBoundsUniform = new THREE.Vector4(-20, -20, 20, 20);
        this.raycaster = new THREE.Raycaster();
        this.plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -1.6);

        this.loadLayoutWalls();
        this.setupEvents();
    }

    async loadLayoutWalls() {
        try {
            const res = await fetch('/api/v1/layout');
            if (!res.ok) return;
            const data = await res.json();
            if (data && Array.isArray(data.walls)) {
                // Беремо тільки конструкційні стіни, двері пропускають погляд
                this.walls = data.walls.filter(w => w.type === 'wall' || !w.type);
                this.calculateSceneBounds();
                this.renderLosMap(this.marker.position.x, this.marker.position.z);
            }
        } catch (err) {
            console.warn('Помилка зчитування стін:', err);
        }
    }

    calculateSceneBounds() {
        if (this.walls.length === 0) return;
        let minX = Infinity, minZ = Infinity, maxX = -Infinity, maxZ = -Infinity;
        
        this.walls.forEach(w => {
            minX = Math.min(minX, w.start[0], w.end[0]);
            minZ = Math.min(minZ, w.start[1], w.end[1]);
            maxX = Math.max(maxX, w.start[0], w.end[0]);
            maxZ = Math.max(maxZ, w.start[1], w.end[1]);
        });

        // Запас 5 метрів, щоб текстура не різалась на краях
        this.bounds = { minX: minX - 5, minZ: minZ - 5, maxX: maxX + 5, maxZ: maxZ + 5 };
        this.sceneBoundsUniform.set(this.bounds.minX, this.bounds.minZ, this.bounds.maxX, this.bounds.maxZ);
    }

    toggle(state) {
        this.enabled = state;
        this.marker.visible = state;
        if (state) {
            this.renderLosMap(this.marker.position.x, this.marker.position.z);
        }
    }

    setupEvents() {
        const dom = this.renderer.domElement; // Працюємо суто з DOM-елементом WebGL канвасу
        this.isDragging = false;

        // Допоміжний метод для точного розрахунку координат NDC всередині канвасу
        const getCanvasNDC = (e) => {
            const rect = dom.getBoundingClientRect();
            return new THREE.Vector2(
                ((e.clientX - rect.left) / rect.width) * 2 - 1,
                -((e.clientY - rect.top) / rect.height) * 2 + 1
            );
        };

        // 1. ПОЧАТОК ПЕРЕТЯГУВАННЯ: Перевірка кліку по маркеру
        dom.addEventListener('pointerdown', (e) => {
            if (!this.enabled) return;

            const mouse = getCanvasNDC(e);
            this.raycaster.setFromCamera(mouse, this.mainCamera);
            
            // Перевіряємо перетин променя саме з нашою сферою-маркером
            const intersects = this.raycaster.intersectObject(this.marker);

            if (intersects.length > 0) {
                this.isDragging = true;
                this.controls.enabled = false; // ПРАВИЛЬНО: Жорстко блокуємо OrbitControls
                dom.style.cursor = 'grabbing';
                e.stopPropagation(); // Запобігаємо спливанню події
            }
        });

        // 2. АКТИВНИЙ РУХ: Розрахунок площини огляду CQB оперативника
        dom.addEventListener('pointermove', (e) => {
            if (!this.enabled) return;

            const mouse = getCanvasNDC(e);
            this.raycaster.setFromCamera(mouse, this.mainCamera);

            // Логіка підсвічування токена (Hover ефект)
            if (!this.isDragging) {
                const intersects = this.raycaster.intersectObject(this.marker);
                dom.style.cursor = intersects.length > 0 ? 'pointer' : 'default';
                return;
            }

            // Якщо триває перетягування — оновлюємо позицію через проекцію на площину
            const intersection = new THREE.Vector3();
            if (this.raycaster.ray.intersectPlane(this.plane, intersection)) {
                this.marker.position.copy(intersection);
                this.renderLosMap(intersection.x, intersection.z);
            }
        });

        // 3. ЗАВЕРШЕННЯ ПЕРЕТЯГУВАННЯ: Повернення фокуса камері сцени
        const stopDragging = () => {
            if (this.isDragging) {
                this.isDragging = false;
                this.controls.enabled = true; // ПРАВИЛЬНО: Повертаємо контроль камері
                dom.style.cursor = 'default';
            }
        };

        dom.addEventListener('pointerup', stopDragging);
        dom.addEventListener('pointerleave', stopDragging); // На випадок, якщо мишу випадково смикнули за межі вікна
    }

    // Чистий 2D Raycasting по вершинах CQB геометрії
    renderLosMap(mx, mz) {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        ctx.fillStyle = '#000000'; // Все в тіні за замовчуванням
        ctx.fillRect(0, 0, w, h);

        if (this.walls.length === 0) return;

        const toPxX = (x) => ((x - this.bounds.minX) / (this.bounds.maxX - this.bounds.minX)) * w;
        const toPxZ = (z) => ((z - this.bounds.minZ) / (this.bounds.maxZ - this.bounds.minZ)) * h;

        const points = [];
        const segments = [];

        // Додаємо рамку сцени як обмежувач променів
        const edges = [
            { start: [this.bounds.minX, this.bounds.minZ], end: [this.bounds.maxX, this.bounds.minZ] },
            { start: [this.bounds.maxX, this.bounds.minZ], end: [this.bounds.maxX, this.bounds.maxZ] },
            { start: [this.bounds.maxX, this.bounds.maxZ], end: [this.bounds.minX, this.bounds.maxZ] },
            { start: [this.bounds.minX, this.bounds.maxZ], end: [this.bounds.minX, this.bounds.minZ] }
        ];

        [...this.walls, ...edges].forEach(wall => {
            const s = { x: wall.start[0], z: wall.start[1] };
            const e = { x: wall.end[0], z: wall.end[1] };
            segments.push({ a: s, b: e });
            points.push(s, e);
        });

        const angles = [];
        points.forEach(p => {
            const angle = Math.atan2(p.z - mz, p.x - mx);
            // Кастимо по 3 промені на кут (прямий + мікро-зсуви для ідеального обтікання фасок стін)
            angles.push(angle, angle - 0.0001, angle + 0.0001);
        });

        const intersections = [];

        angles.forEach(angle => {
            const dx = Math.cos(angle);
            const dz = Math.sin(angle);

            let closestIntersect = null;
            let minDst = Infinity;

            segments.forEach(seg => {
                const intersect = this.getIntersection({ x: mx, z: mz }, { x: mx + dx, z: mz + dz }, seg.a, seg.b);
                if (intersect) {
                    const dst = (intersect.x - mx) * (intersect.x - mx) + (intersect.z - mz) * (intersect.z - mz);
                    if (dst < minDst) { minDst = dst; closestIntersect = intersect; }
                }
            });

            if (closestIntersect) {
                closestIntersect.angle = angle;
                intersections.push(closestIntersect);
            }
        });

        intersections.sort((a, b) => a.angle - b.angle);

        ctx.save();
        ctx.fillStyle = '#ffffff'; // Зона видимості біла
        ctx.beginPath();
        if (intersections.length > 0) {
            ctx.moveTo(toPxX(intersections[0].x), toPxZ(intersections[0].z));
            for (let i = 1; i < intersections.length; i++) {
                ctx.lineTo(toPxX(intersections[i].x), toPxZ(intersections[i].z));
            }
        }
        ctx.closePath();
        ctx.fill();
        ctx.restore();

        this.texture.needsUpdate = true;
    }

    getIntersection(p0, p1, p2, p3) {
        const s1_x = p1.x - p0.x; const s1_z = p1.z - p0.z;
        const s2_x = p3.x - p2.x; const s2_z = p3.z - p2.z;
        const s = (-s1_z * (p0.x - p2.x) + s1_x * (p0.z - p2.z)) / (-s2_x * s1_z + s1_x * s2_z);
        const t = ( s2_x * (p0.z - p2.z) - s2_z * (p0.x - p2.x)) / (-s2_x * s1_z + s1_x * s2_z);
        
        // ФІКС: s має бути в межах [0, 1] (відрізок стіни), 
        // а t має бути просто >= 0 (промінь зору йде вперед у нескінченність)
        if (s >= 0 && s <= 1 && t >= 0) { 
            return { x: p0.x + (t * s1_x), z: p0.z + (t * s1_z) }; 
        }
        return null;
    }
}