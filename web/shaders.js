// shaders.js
export const vertexShaderGS = `#version 300 es
    precision highp float;
    in vec3 position;
    in vec2 uv;
    in vec3 aCenter;
    in vec3 aScale;
    in vec4 aRotation;
    in vec3 aColor;

    out vec3 vColor;
    out vec2 vUV;
    out vec3 vNormal;   
    out vec3 vWorldPos; // Передаємо позицію у фрагментник для LOS мапінгу
    out float vIsFloor;    
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
        vec3 localVertex;
        vec3 worldPos;

        if (aScale.y < 1e-4) {
            localVertex = vec3(position.x * aScale.x, 0.0, position.y * aScale.z);
            worldPos = aCenter + localVertex;
            vNormal = vec3(0.0, 1.0, 0.0);
            vIsFloor = 1.0;
        } else {
            localVertex = position * vec3(aScale.x, aScale.y, 1.0);
            mat3 R = quatToMat(aRotation);
            worldPos = aCenter + R * localVertex;
            vNormal = R * vec3(0.0, 0.0, 1.0);
            vIsFloor = 0.0;
        }

        vWorldY = worldPos.y;
        vWorldPos = worldPos;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(worldPos, 1.0);
    }
`; //

export const fragmentShaderGS = `#version 300 es
    precision highp float;
    in vec3 vColor;
    in vec2 vUV;
    in vec3 vNormal;
    in vec3 vWorldPos;
    in float vIsFloor;
    in float vWorldY;
    out vec4 fragColor;

    uniform sampler2D uLosMap;       // Текстура 2D канвасу маски видимості
    uniform vec4 uSceneBoundsXZ;     // [min_x, min_z, max_x, max_z] для інтерполяції
    uniform bool uLosEnabled;     
    uniform bool uCutawayEnabled; 
    uniform float uCutawayHeight; 

    void main() {
        if (uCutawayEnabled && vWorldY > uCutawayHeight) discard; //

        vec2 d = (vUV - vec2(0.5)) * 2.0; //[cite: 2]
        if (dot(d, d) > 1.0) discard; //[cite: 2]

        float rim = smoothstep(0.45, 1.0, dot(d, d));
        vec3 baseColor = mix(vColor, vColor * 0.68, rim * 0.35);
        vec3 litColor = baseColor;

        if (uLosEnabled && vIsFloor > 0.5) {
            // Переводимо світові координати XZ в діапазон UV текстури [0.0, 1.0]
            vec2 losUV = (vWorldPos.xz - uSceneBoundsXZ.xy) / (uSceneBoundsXZ.zw - uSceneBoundsXZ.xy);

            if (losUV.x >= 0.0 && losUV.x <= 1.0 && losUV.y >= 0.0 && losUV.y <= 1.0) {
                // Читаємо червоний канал маски (1.0 = видно/білий, 0.0 = тінь/чорний)
                float visibility = texture(uLosMap, losUV).r;
                
                if (visibility < 0.5) {
                    litColor = mix(baseColor, vec3(0.85, 0.2, 0.2), 0.65); // Мертвий кут
                } else {
                    litColor = mix(baseColor, vec3(0.2, 0.8, 0.35), 0.5);  // Пряма видимість
                }
            } else {
                litColor = mix(baseColor, vec3(0.5, 0.4, 0.4), 0.3);
            }
        }

        fragColor = vec4(litColor, 1.0);
    }
`; //[cite: 2]