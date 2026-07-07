// TacticalLosEngine.js — v2
//
// ЧОМУ ЦЕЙ ПІДХІД: попередня версія рендерила shadow-cubemap ПРОТИ ХМАРИ
// ТОЧОК спліату. Низька роздільність кубічної карти (512px) у поєднанні з
// джиттером і перекриттям точкових карток (навмисно додані раніше проти
// муару) означали, що межа видимості була "зубчастою" (шум на межі
// оклюзії), а деякі напрямки могли "протікати" крізь стіну, якщо промінь
// проходив між джиттерованими картками там, де насправді стіна суцільна.
//
// Ця версія рахує ТОЧНИЙ visibility polygon (класичний алгоритм
// обчислювальної геометрії — кутове промінювання по кутах вершин
// сегментів-перепон) проти РЕАЛЬНИХ сегментів стін із layout.json, а не
// проти точок хмари. Межа видимості точна аж до самої геометрії стін,
// абсолютно незалежна від point_density_per_meter.
//
// ВАЖЛИВЕ ОБМЕЖЕННЯ (задокументовано, не забуто): це 2D-рушій — він working
// на одній горизонтальній площині (висота ока маркера, за замовчуванням
// 1.6м) і НЕ підтримує наскрізну видимість між поверхами (наприклад, крізь
// діру у підлозі). Вікна (Opening з vert_start > 0) коректно враховуються:
// отвір "прорізає" стіну на висоті ока лише якщо [vert_start, vert_end]
// охоплює цю висоту — низьке вікно (наприклад head=1.2м) НЕ дасть
// видимості на висоті ока 1.6м, а нормальне вікно (0.9–2.0м) дасть.
// Коли дійдемо до мультиповерховості (Крок 2/3 архітектурного плану), цей
// модуль треба буде замінити або доповнити справжнім 3D-рейкастингом проти
// /api/v1/collision-mesh — це вже занотовано в ARCHITECTURE.md.

export class TacticalLosEngine {
    constructor(scene, renderer, mainCamera, orbitControls, options = {}) {
        this.scene = scene;
        this.renderer = renderer;
        this.mainCamera = mainCamera;
        this.controls = orbitControls;
        this.enabled = false;
        this.needsUpdate = false;

        this.eyeHeight = options.eyeHeight ?? 1.6;
        this.textureSize = options.textureSize ?? 512;

        this.segments = []; // тверді (не отвір на висоті ока) шматки стін, у світових XZ
        this.bounds = null; // { minX, minZ, maxX, maxZ }

        // 1. Маркер оперативника
        const geo = new THREE.SphereGeometry(0.3, 16, 16);
        const mat = new THREE.MeshBasicMaterial({ color: 0x00ffcc, depthTest: false, transparent: true, opacity: 0.8 });
        this.marker = new THREE.Mesh(geo, mat);
        this.marker.position.set(0, this.eyeHeight, 0);
        this.marker.visible = false;
        this.scene.add(this.marker);

        // 2. Canvas для растеризації visibility polygon -> текстура для шейдера
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.textureSize;
        this.canvas.height = this.textureSize;
        this.ctx = this.canvas.getContext('2d');
        this.texture = new THREE.CanvasTexture(this.canvas);
        this.texture.minFilter = THREE.LinearFilter;
        this.texture.magFilter = THREE.LinearFilter;
        // ВИПРАВЛЕНО: CanvasTexture за замовчуванням flipY=true (щоб звичайні
        // фото виглядали "правильно" при стандартному UV мешу). Але ми
        // семплюємо цю текстуру НЕ через mesh UV, а напряму через світові
        // XZ-координати (uSceneBoundsXZ) з ВЛАСНОЮ угодою рядків у
        // _rasterize(). Без цього вимкнення вся вісь Z виявлялась
        // ДЗЕРКАЛЬНО перевернутою відносно того, що очікує шейдер --
        // видима зона показувалась у фізично неправильній кімнаті.
        this.texture.flipY = false;

        this.raycaster = new THREE.Raycaster();
        this.plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -this.eyeHeight);

        this._loadWallData();
        this.setupEvents();
    }

    async _loadWallData() {
        try {
            const res = await fetch('/api/v1/layout');
            if (!res.ok) return;
            const layout = await res.json();
            this._buildSegmentsFromLayout(layout);
            if (this.enabled) this.needsUpdate = true;
        } catch (err) {
            console.warn('TacticalLosEngine: не вдалось завантажити layout для LOS:', err);
        }
    }

    _buildSegmentsFromLayout(layout) {
        const segments = [];
        let minX = Infinity, minZ = Infinity, maxX = -Infinity, maxZ = -Infinity;

        const walls = layout.walls || [];
        for (const w of walls) {
            const [sx, sz] = w.start;
            const [ex, ez] = w.end;
            minX = Math.min(minX, sx, ex); maxX = Math.max(maxX, sx, ex);
            minZ = Math.min(minZ, sz, ez); maxZ = Math.max(maxZ, sz, ez);

            const dx = ex - sx, dz = ez - sz;
            const length = Math.hypot(dx, dz);
            if (length < 1e-4) continue;
            const ux = dx / length, uz = dz / length;

            const openings = [];
            if ((w.type || 'wall') === 'door') {
                openings.push({ start: 0, end: length, vertStart: 0, vertEnd: 2.1 });
            }
            if (Array.isArray(w.openings)) {
                for (const op of w.openings) {
                    openings.push({
                        start: op.horiz_start, end: op.horiz_end,
                        vertStart: op.vert_start ?? 0.0, vertEnd: op.vert_end ?? 2.1
                    });
                }
            }

            // Отвори, що охоплюють ВИСОТУ ОКА -> реально прорізають стіну на цій площині
            const openAtEye = openings
                .filter(o => o.vertStart <= this.eyeHeight && o.vertEnd >= this.eyeHeight)
                .map(o => [Math.max(0, o.start), Math.min(length, o.end)])
                .sort((a, b) => a[0] - b[0]);

            let cursor = 0;
            for (const [os, oe] of openAtEye) {
                if (os > cursor + 1e-4) segments.push(this._segFromT(sx, sz, ux, uz, cursor, os));
                cursor = Math.max(cursor, oe);
            }
            if (length - cursor > 1e-4) segments.push(this._segFromT(sx, sz, ux, uz, cursor, length));
        }

        const pad = 1.0;
        this.segments = segments;
        this.bounds = (minX === Infinity)
            ? { minX: -10, minZ: -10, maxX: 10, maxZ: 10 }
            : { minX: minX - pad, minZ: minZ - pad, maxX: maxX + pad, maxZ: maxZ + pad };
    }

    _segFromT(sx, sz, ux, uz, t0, t1) {
        return { x1: sx + ux * t0, z1: sz + uz * t0, x2: sx + ux * t1, z2: sz + uz * t1 };
    }

    toggle(state) {
        this.enabled = state;
        this.marker.visible = state;
        if (state) this.needsUpdate = true;
    }

    findActiveMesh() {
        return this.scene.children.find(c => c.isMesh && c !== this.marker);
    }

    update() {
        if (!this.enabled || !this.needsUpdate || !this.bounds) return;

        const activeMesh = this.findActiveMesh();
        if (!activeMesh) return;

        const poly = this._computeVisibilityPolygon(this.marker.position.x, this.marker.position.z);
        this._rasterize(poly);
        this.texture.needsUpdate = true;

        if (activeMesh.material.uniforms) {
            if (activeMesh.material.uniforms.uLosMap) {
                activeMesh.material.uniforms.uLosMap.value = this.texture;
            }
            if (activeMesh.material.uniforms.uSceneBoundsXZ) {
                activeMesh.material.uniforms.uSceneBoundsXZ.value.set(
                    this.bounds.minX, this.bounds.minZ, this.bounds.maxX, this.bounds.maxZ
                );
            }
        }

        this.needsUpdate = false;
    }

    // ── Точний visibility polygon: кутове промінювання по кутах вершин ──
    _wrapAngle(a) {
        // ВИПРАВЛЕНО: a±EPS може вислизнути за межі (-π, π] (напр. якщо
        // справжній кут вершини лежить біля самого шва -π/π). Без
        // нормалізації такий кут чисельно сортується не там, де він
        // геометрично належить -- це й давало "метелик"/X-подібний
        // самоперетин полігону видимості на межі кута ±180°.
        while (a > Math.PI) a -= 2 * Math.PI;
        while (a <= -Math.PI) a += 2 * Math.PI;
        return a;
    }

    _computeVisibilityPolygon(originX, originZ) {
        const EPS = 1e-4;
        const angles = new Set();

        for (const seg of this.segments) {
            for (const [px, pz] of [[seg.x1, seg.z1], [seg.x2, seg.z2]]) {
                const a = Math.atan2(pz - originZ, px - originX);
                angles.add(this._wrapAngle(a - EPS));
                angles.add(a);
                angles.add(this._wrapAngle(a + EPS));
            }
        }

        if (angles.size === 0) return [];

        const sortedAngles = Array.from(angles).sort((a, b) => a - b);
        const points = [];

        for (const angle of sortedAngles) {
            const dx = Math.cos(angle), dz = Math.sin(angle);
            points.push(this._closestHit(originX, originZ, dx, dz));
        }

        return points;
    }

    _closestHit(ox, oz, dx, dz) {
        const MAX_DIST = 200;
        let closestT = MAX_DIST;

        for (const seg of this.segments) {
            const t = this._rayVsSegment(ox, oz, dx, dz, seg.x1, seg.z1, seg.x2, seg.z2);
            if (t !== null && t < closestT) closestT = t;
        }

        return [ox + dx * closestT, oz + dz * closestT];
    }

    _rayVsSegment(ox, oz, dx, dz, x1, z1, x2, z2) {
        const sx = x2 - x1, sz = z2 - z1;
        const denom = dx * sz - dz * sx;
        if (Math.abs(denom) < 1e-9) return null;

        const t = ((x1 - ox) * sz - (z1 - oz) * sx) / denom;
        const u = ((x1 - ox) * dz - (z1 - oz) * dx) / denom;

        if (t >= 0 && u >= 0 && u <= 1) return t;
        return null;
    }

    _rasterize(polyPoints) {
        const { minX, minZ, maxX, maxZ } = this.bounds;
        const w = this.canvas.width, h = this.canvas.height;
        const toCanvas = (x, z) => [
            ((x - minX) / (maxX - minX)) * w,
            ((z - minZ) / (maxZ - minZ)) * h
        ];

        if (this.segments.length === 0) {
            this.ctx.fillStyle = '#ffffff';
            this.ctx.fillRect(0, 0, w, h);
            return;
        }

        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, w, h);

        if (polyPoints.length < 3) return;

        this.ctx.fillStyle = '#ffffff';
        this.ctx.beginPath();
        const [sx0, sz0] = toCanvas(polyPoints[0][0], polyPoints[0][1]);
        this.ctx.moveTo(sx0, sz0);
        for (let i = 1; i < polyPoints.length; i++) {
            const [px, pz] = toCanvas(polyPoints[i][0], polyPoints[i][1]);
            this.ctx.lineTo(px, pz);
        }
        this.ctx.closePath();
        this.ctx.fill();
    }

    setupEvents() {
        const dom = this.renderer.domElement;
        this.isDragging = false;

        const getCanvasNDC = (e) => {
            const rect = dom.getBoundingClientRect();
            return new THREE.Vector2(
                ((e.clientX - rect.left) / rect.width) * 2 - 1,
                -((e.clientY - rect.top) / rect.height) * 2 + 1
            );
        };

        dom.addEventListener('pointerdown', (e) => {
            if (!this.enabled) return;
            const mouse = getCanvasNDC(e);
            this.raycaster.setFromCamera(mouse, this.mainCamera);
            const intersects = this.raycaster.intersectObject(this.marker);

            if (intersects.length > 0) {
                this.isDragging = true;
                this.controls.enabled = false;
                dom.style.cursor = 'grabbing';
                e.stopPropagation();
            }
        });

        dom.addEventListener('pointermove', (e) => {
            if (!this.enabled) return;
            const mouse = getCanvasNDC(e);
            this.raycaster.setFromCamera(mouse, this.mainCamera);

            if (!this.isDragging) {
                const intersects = this.raycaster.intersectObject(this.marker);
                dom.style.cursor = intersects.length > 0 ? 'pointer' : 'default';
                return;
            }

            const intersection = new THREE.Vector3();
            if (this.raycaster.ray.intersectPlane(this.plane, intersection)) {
                this.marker.position.copy(intersection);
                this.needsUpdate = true;
            }
        });

        const stopDragging = () => {
            if (this.isDragging) {
                this.isDragging = false;
                this.controls.enabled = true;
                dom.style.cursor = 'default';
            }
        };

        dom.addEventListener('pointerup', stopDragging);
        dom.addEventListener('pointerleave', stopDragging);
    }
}