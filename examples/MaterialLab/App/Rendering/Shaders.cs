namespace NumSharp.Examples.MaterialLab.App.Rendering
{
    /// <summary>
    /// GLSL 3.30 sources of the lab's render pipeline. Kept as C# constants so the app ships as a single
    /// executable with no asset files.
    /// </summary>
    /// <remarks>
    /// <para><b>Coordinate conventions.</b> Simulation units: x ∈ [0, W], y ∈ [0, 1] (y up), mapped to the
    /// whole viewport. Texture UVs follow GL (v = 0 at the bottom), which matches y-up, so no flips happen
    /// anywhere between the simulation and the screen.</para>
    /// <para><b>Pipeline.</b> (1) field splat — every smooth-material particle adds a smooth compact kernel
    /// into three HDR targets (color·w, category weights, optical/visual properties); (2) a separable blur turns
    /// the splats into continuous fields; (3) granular particles are drawn as individually lit grains;
    /// (4) the composite shades the backdrop, the walls (from the obstacle distance field), the grains and
    /// the liquid/solid surfaces (height-field normals, refraction, Beer–Lambert absorption, Fresnel, specular,
    /// foam, emission); (5) bloom and ACES tone mapping.</para>
    /// </remarks>
    internal static class Shaders
    {
        /// <summary>Full-screen triangle generated from gl_VertexID (no vertex buffer; draw 3 vertices).</summary>
        public const string FullscreenVs = @"#version 330 core
out vec2 vUv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vUv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}";

        /// <summary>
        /// Instanced particle sprite. Per-instance: (x, y, vx, vy) and (material, seed, temperature, J).
        /// <c>uMode</c> 0 = field splat (smooth materials + a grain-density contribution), 1 = grains only.
        /// Particles of a category the pass does not draw are moved off-screen (zero fragments).
        /// </summary>
        public const string ParticleVs = @"#version 330 core
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec4 aPosVel;
layout(location = 2) in vec4 aMatSeed;
uniform vec2 uDomain;
uniform float uRadiusField;
uniform float uRadiusGrain;
uniform int uMode;
uniform float uCategory[16];
uniform float uGrainScale[16];
out vec2 vLocal;
flat out int vMat;
out float vSeed;
out float vTemp;
out float vJ;
out float vSpeed;
void main() {
    int mat = int(aMatSeed.x + 0.5);
    float cat = uCategory[mat];
    bool grain = cat > 0.5 && cat < 1.5;
    // Grains overlap (scale > 1 of the particle spacing) so a pile reads as a continuous granular mass.
    float radius = uMode == 1 ? uRadiusGrain * uGrainScale[mat] * (0.85 + 0.3 * aMatSeed.y) : uRadiusField;
    if (uMode == 1 && !grain) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    vLocal = aCorner;
    vMat = mat;
    vSeed = aMatSeed.y;
    vTemp = aMatSeed.z;
    vJ = aMatSeed.w;
    vSpeed = length(aPosVel.zw);
    vec2 p = aPosVel.xy + aCorner * radius;
    gl_Position = vec4(p / uDomain * 2.0 - 1.0, 0.0, 1.0);
}";

        /// <summary>
        /// Field splat: kernel w = (1 − r²)³. Outputs (additive): color·w (smooth materials only);
        /// (fluid·w, solid·w, grain·w, heat·w); (foam·w, translucency·w, gloss·w, emissive·w).
        /// </summary>
        public const string FieldFs = @"#version 330 core
in vec2 vLocal;
flat in int vMat;
in float vSeed;
in float vTemp;
in float vJ;
in float vSpeed;
uniform vec3 uAlbedo[16];
uniform float uCategory[16];
uniform vec4 uProps[16];   // translucency, gloss, emissive, color jitter
uniform float uFoam[16];   // foaminess (water 1, oil a little, viscous liquids 0)
layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oCat;
layout(location = 2) out vec4 oProp;
void main() {
    float r2 = dot(vLocal, vLocal);
    if (r2 >= 1.0) discard;
    float k = 1.0 - r2;
    float w = k * k * k;
    float cat = uCategory[vMat];
    vec4 props = uProps[vMat];
    if (cat > 0.5 && cat < 1.5) {       // granular: only its density, for macro shading
        oColor = vec4(0.0);
        oCat = vec4(0.0, 0.0, w, 0.0);
        oProp = vec4(0.0);
        return;
    }
    vec3 albedo = uAlbedo[vMat] * (1.0 + props.w * (vSeed - 0.5) * 2.0);
    bool fluid = cat < 0.5;
    // Foam: only genuine spray — liquid torn into droplets and thin sheets. A liquid particle's J is 1/phi,
    // phi being the occupancy gathered from the grid (1 = packed liquid, about 0.6 at a calm free surface,
    // at most about 0.35 for a lone droplet), so aerated liquid is exactly where phi falls well below the
    // surface value. Whiter the faster it flies; bulk flow, however fast, stays clear.
    float occ = 1.0 / max(vJ, 1.0);
    float spray = smoothstep(0.42, 0.22, occ);
    float foam = fluid ? spray * clamp(0.35 + vSpeed * 0.06, 0.0, 1.0) * uFoam[vMat] : 0.0;
    float heat = props.z * vTemp;
    oColor = vec4(albedo * w, w);
    oCat = vec4(fluid ? w : 0.0, fluid ? 0.0 : w, 0.0, heat * w);
    oProp = vec4(foam * w, props.x * w, props.y * w, props.z * w);
}";

        /// <summary>
        /// Grain sprite: a tiny lit sphere with per-grain color variation (sand gets dark specks and bright quartz
        /// grains; snow sparkles). Output is premultiplied alpha.
        /// </summary>
        public const string GrainFs = @"#version 330 core
in vec2 vLocal;
flat in int vMat;
in float vSeed;
in float vTemp;
in float vJ;
in float vSpeed;
uniform vec3 uAlbedo[16];
uniform vec4 uProps[16];
uniform vec3 uLight;
out vec4 oColor;
float hash(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
void main() {
    float r2 = dot(vLocal, vLocal);
    if (r2 >= 1.0) discard;
    vec3 n = vec3(vLocal, sqrt(1.0 - r2));
    vec3 albedo = uAlbedo[vMat];
    float j = uProps[vMat].w;
    float s = vSeed;
    // Grain palette: most grains near the base tone, a few dark (feldspar/organic) and a few bright (quartz).
    float tone = 1.0 + j * (s - 0.5) * 2.0;
    if (s < 0.06) tone *= 0.55;
    else if (s > 0.96) tone *= 1.25;
    vec3 c = albedo * tone;
    c.r *= 1.0 + 0.10 * (hash(s * 91.0) - 0.5);
    c.b *= 1.0 + 0.12 * (hash(s * 37.0) - 0.5);
    float diff = max(dot(n, uLight), 0.0);
    vec3 col = c * (0.42 + 0.75 * diff);
    // Snow: soft fluffy flakes — bluish in shadow, low contrast, rare sparkles, feathered edges.
    bool snow = albedo.b > 0.95 && albedo.r > 0.9;
    if (snow) {
        col = c * (0.72 + 0.35 * diff);
        col = mix(col, col * vec3(0.82, 0.90, 1.06), 1.0 - diff);
        if (s > 0.985) col += vec3(0.7) * pow(max(dot(reflect(-uLight, n), vec3(0.0, 0.0, 1.0)), 0.0), 18.0);
    }
    float alpha = snow ? smoothstep(1.0, 0.25, sqrt(r2)) : smoothstep(1.0, 0.78, sqrt(r2));
    oColor = vec4(col * alpha, alpha);
}";

        /// <summary>Separable 9-tap Gaussian over the three field targets (direction in uDir, in texels).</summary>
        public const string BlurFs = @"#version 330 core
in vec2 vUv;
uniform sampler2D uT0;
uniform sampler2D uT1;
uniform sampler2D uT2;
uniform vec2 uDir;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
void main() {
    const float w0 = 0.2270270270, w1 = 0.1945945946, w2 = 0.1216216216, w3 = 0.0540540541, w4 = 0.0162162162;
    vec2 d1 = uDir, d2 = 2.0 * uDir, d3 = 3.0 * uDir, d4 = 4.0 * uDir;
    o0 = texture(uT0, vUv) * w0
       + (texture(uT0, vUv + d1) + texture(uT0, vUv - d1)) * w1
       + (texture(uT0, vUv + d2) + texture(uT0, vUv - d2)) * w2
       + (texture(uT0, vUv + d3) + texture(uT0, vUv - d3)) * w3
       + (texture(uT0, vUv + d4) + texture(uT0, vUv - d4)) * w4;
    o1 = texture(uT1, vUv) * w0
       + (texture(uT1, vUv + d1) + texture(uT1, vUv - d1)) * w1
       + (texture(uT1, vUv + d2) + texture(uT1, vUv - d2)) * w2
       + (texture(uT1, vUv + d3) + texture(uT1, vUv - d3)) * w3
       + (texture(uT1, vUv + d4) + texture(uT1, vUv - d4)) * w4;
    o2 = texture(uT2, vUv) * w0
       + (texture(uT2, vUv + d1) + texture(uT2, vUv - d1)) * w1
       + (texture(uT2, vUv + d2) + texture(uT2, vUv - d2)) * w2
       + (texture(uT2, vUv + d3) + texture(uT2, vUv - d3)) * w3
       + (texture(uT2, vUv + d4) + texture(uT2, vUv - d4)) * w4;
}";

        /// <summary>
        /// The composite: backdrop → walls → grains → liquid/solid surfaces → brush ring, in linear HDR.
        /// </summary>
        public const string CompositeFs = @"#version 330 core
in vec2 vUv;
uniform sampler2D uField0;    // color*w, w
uniform sampler2D uField1;    // fluid w, solid w, grain w, heat w
uniform sampler2D uField2;    // foam w, translucency w, gloss w, emissive w
uniform sampler2D uGrains;    // premultiplied grain layer
uniform sampler2D uPhi;       // obstacle signed distance (cells), Nx x Ny
uniform vec2 uResolution;
uniform vec2 uFieldTexel;
uniform vec2 uDomain;
uniform vec2 uGridSize;       // Nx, Ny
uniform float uCellPx;        // pixels per grid cell
uniform float uIso;           // field level of a surface
uniform float uTime;
uniform vec3 uLight;
uniform vec4 uBrush;          // x, y, radius (units), active
uniform vec3 uBrushColor;
uniform vec2 uGravityDir;
out vec4 oColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { s += a * noise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; }
    return s;
}

// Studio backdrop: a lit panel with a faint engineering grid (so refraction through liquids is visible).
vec3 backdrop(vec2 uv) {
    vec2 p = uv * uDomain;
    vec3 top = vec3(0.105, 0.125, 0.155), bottom = vec3(0.035, 0.040, 0.050);
    vec3 c = mix(bottom, top, smoothstep(0.0, 1.0, uv.y));
    float spot = exp(-dot(uv - vec2(0.32, 0.86), (uv - vec2(0.32, 0.86)) * vec2(1.4, 2.4)) * 2.2);
    c += vec3(0.070, 0.075, 0.080) * spot;
    vec2 g = abs(fract(p * 20.0) - 0.5);
    float minor = 1.0 - smoothstep(0.0, 0.035, min(g.x, g.y));
    vec2 G = abs(fract(p * 4.0) - 0.5);
    float major = 1.0 - smoothstep(0.0, 0.018, min(G.x, G.y));
    c += vec3(0.018, 0.022, 0.028) * minor + vec3(0.035, 0.042, 0.052) * major;
    c *= 0.95 + 0.05 * fbm(p * 7.0);   // faint, soft panel mottling (no visible lattice)
    return c;
}

float phiAt(vec2 posUnits) {
    vec2 t = (posUnits * uGridSize.y + 0.5) / uGridSize;
    return texture(uPhi, t).r;
}

// Walls: dark basalt with a bevelled, lit rim and stone texture; returns rgb + coverage.
vec4 walls(vec2 posUnits) {
    float phi = phiAt(posUnits);
    float cov = clamp(0.5 - phi * uCellPx, 0.0, 1.0);
    if (cov <= 0.0) return vec4(0.0);
    float e = 0.6 / uGridSize.y;
    vec2 grad = vec2(phiAt(posUnits + vec2(e, 0)) - phiAt(posUnits - vec2(e, 0)),
                     phiAt(posUnits + vec2(0, e)) - phiAt(posUnits - vec2(0, e)));
    vec2 n2 = length(grad) > 1e-5 ? normalize(grad) : vec2(0.0, 1.0);
    float rim = clamp(1.0 + phi / 1.4, 0.0, 1.0);           // 1 at the surface, 0 a cell and a half in
    vec3 n = normalize(vec3(n2 * rim * 0.9, 1.0));
    float stone = fbm(posUnits * 26.0);
    vec3 base = mix(vec3(0.085, 0.088, 0.095), vec3(0.17, 0.165, 0.16), stone);
    base *= 0.85 + 0.3 * noise(posUnits * 180.0);
    float diff = max(dot(n, uLight), 0.0);
    vec3 c = base * (0.35 + 0.9 * diff);
    c += vec3(0.25, 0.26, 0.28) * pow(max(dot(reflect(-uLight, n), vec3(0.0, 0.0, 1.0)), 0.0), 24.0) * rim;
    return vec4(c, cov);
}

// Saturated height of the smooth-material field: 0 outside, rising across the surface band, 1 inside.
float sat(float w) { return smoothstep(uIso * 0.35, uIso * 1.9, w); }

vec3 envColor(vec3 r) {
    float t = clamp(r.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 c = mix(vec3(0.05, 0.055, 0.065), vec3(0.55, 0.60, 0.68), t);
    c += vec3(1.3) * pow(max(dot(r, normalize(vec3(-0.45, 0.75, 0.5))), 0.0), 48.0);   // softbox
    return c;
}

// Blackbody-like ramp for lava temperature: dull red → orange → yellow-orange (never white — that reads as a lamp).
vec3 lavaColor(float t) {
    vec3 c1 = vec3(0.12, 0.012, 0.004), c2 = vec3(0.75, 0.07, 0.01), c3 = vec3(1.0, 0.28, 0.03), c4 = vec3(1.0, 0.55, 0.10);
    return t < 0.4 ? mix(c1, c2, t / 0.4) : (t < 0.75 ? mix(c2, c3, (t - 0.4) / 0.35) : mix(c3, c4, (t - 0.75) / 0.25));
}

vec3 behindAt(vec2 uv) {
    vec2 posUnits = uv * uDomain;
    vec3 c = backdrop(uv);
    vec4 w = walls(posUnits);
    c = mix(c, w.rgb, w.a);
    vec4 g = texture(uGrains, uv);
    c = c * (1.0 - g.a) + g.rgb;
    return c;
}

void main() {
    vec2 uv = vUv;
    vec2 posUnits = uv * uDomain;
    vec3 scene = behindAt(uv);

    // Macro shading of grain piles: light the pile as a height field, darken its interior (occlusion).
    vec4 c1 = texture(uField1, uv);
    float gw = c1.z;
    if (gw > 1e-4) {
        float gL = texture(uField1, uv - vec2(uFieldTexel.x, 0)).z, gR = texture(uField1, uv + vec2(uFieldTexel.x, 0)).z;
        float gD = texture(uField1, uv - vec2(0, uFieldTexel.y)).z, gU = texture(uField1, uv + vec2(0, uFieldTexel.y)).z;
        vec3 gn = normalize(vec3((gL - gR) * 2.2 / uIso, (gD - gU) * 2.2 / uIso, 1.0));
        float lit = 0.75 + 0.45 * dot(gn, uLight);
        float occl = 1.0 - 0.28 * smoothstep(uIso * 1.2, uIso * 3.5, gw);
        vec4 g = texture(uGrains, uv);
        scene = mix(scene, scene * lit * occl, g.a);
    }

    vec4 f0 = texture(uField0, uv);
    float wsum = f0.a;
    float cover = smoothstep(uIso * 0.55, uIso * 0.95, wsum);
    if (cover > 0.0) {
        vec4 f2 = texture(uField2, uv);
        float inv = 1.0 / max(wsum, 1e-6);
        vec3 albedo = f0.rgb * inv;
        float fluid = c1.x / max(c1.x + c1.y, 1e-6);
        float foam = clamp(f2.x * inv, 0.0, 1.0);
        float transl = f2.y * inv;
        float gloss = f2.z * inv;
        float emissive = f2.w * inv;
        float heat = f2.w > 1e-6 ? c1.w / f2.w : 0.0;

        // Height-field normal from the SATURATED density: a body of material is a flat slab behind the glass whose
        // edges round off, so the height rises only across the surface band and is flat inside — interior particle
        // noise never reaches the normals (no false glints inside a pool).
        vec2 t2 = uFieldTexel * 1.5;
        float hL = sat(texture(uField0, uv - vec2(t2.x, 0)).a), hR = sat(texture(uField0, uv + vec2(t2.x, 0)).a);
        float hD = sat(texture(uField0, uv - vec2(0, t2.y)).a), hU = sat(texture(uField0, uv + vec2(0, t2.y)).a);
        vec3 n = normalize(vec3((hL - hR) * 2.6, (hD - hU) * 2.6, 1.0));
        // Optical thickness of the slab: full inside, thinning over the rounded edge (from the saturated height,
        // so particle-density noise does not show up as cloudiness).
        float thick = 0.4 + 1.8 * sat(wsum);

        // Transmission: refract the scene behind (backdrop grid, walls, grains) and absorb along the path.
        vec2 ruv = uv - n.xy * 0.018 * transl * clamp(thick, 0.5, 2.5);
        vec3 behind = behindAt(ruv);
        vec3 absorb = (vec3(1.0) - albedo) * mix(0.9, 0.45, transl) + 0.02;
        vec3 transmitted = behind * exp(-absorb * thick * 1.25) + albedo * 0.10 * thick * transl;
        // Forward scattering in the body (subsurface look for jelly/honey; faint glow in water).
        float diff = max(dot(n, uLight), 0.0);
        vec3 opaque = albedo * (0.22 + 0.95 * diff);
        vec3 body = mix(opaque, transmitted, transl);
        body += albedo * albedo * 0.25 * transl * pow(clamp(1.0 - n.z, 0.0, 1.0), 1.5);   // rim light through thin edges

        vec3 V = vec3(0.0, 0.0, 1.0);
        vec3 H = normalize(uLight + V);
        float fres = 0.03 + 0.97 * pow(1.0 - clamp(n.z, 0.0, 1.0), 5.0);
        vec3 refl = envColor(reflect(-V, n));
        float shin = mix(18.0, 380.0, gloss);
        vec3 spec = vec3(1.0, 0.98, 0.95) * pow(max(dot(n, H), 0.0), shin) * gloss * mix(1.4, 3.2, gloss);
        vec3 col = body * (1.0 - fres * gloss) + refl * fres * gloss + spec;

        // Foam / spray on liquids.
        float fm = clamp(foam * 1.3, 0.0, 1.0) * fluid;
        col = mix(col, vec3(0.90, 0.94, 0.98) * (0.55 + 0.5 * diff), fm * 0.85);

        // Lava: incandescent core under a darkening, cracking crust as it cools.
        if (emissive > 0.02) {
            float crust = smoothstep(0.35, 0.95, fbm(posUnits * 34.0 + vec2(uTime * 0.05, -uTime * 0.11)) + (1.0 - heat) * 0.9 - 0.35);
            vec3 glow = lavaColor(clamp(heat, 0.0, 1.0)) * mix(1.9, 0.6, crust);
            vec3 rock = vec3(0.06, 0.05, 0.045) * (0.6 + 0.8 * diff);
            vec3 lava = mix(glow, rock + glow * 0.15, crust * (1.0 - heat * 0.6));
            col = mix(col, lava + spec * 0.4, clamp(emissive, 0.0, 1.0));
        }

        // A thin darker meniscus line where the surface meets air (reads as a real liquid edge).
        float edge = smoothstep(uIso * 0.55, uIso * 0.9, wsum) * (1.0 - smoothstep(uIso * 0.9, uIso * 1.4, wsum));
        col *= 1.0 - 0.18 * edge * (1.0 - fm);
        scene = mix(scene, col, cover);
    }

    // Brush ring.
    if (uBrush.w > 0.5) {
        // Distance to the ring in pixels (domain height maps to the viewport height).
        float px = abs(length(posUnits - uBrush.xy) - uBrush.z) * uResolution.y / uDomain.y;
        float ring = 1.0 - smoothstep(0.6, 1.8, px);
        scene = mix(scene, uBrushColor * 1.4 + 0.15, ring * 0.85);
    }
    oColor = vec4(scene, 1.0);
}";

        /// <summary>Bright-pass for bloom: keeps what exceeds the threshold (lava, specular glints).</summary>
        public const string BrightFs = @"#version 330 core
in vec2 vUv;
uniform sampler2D uTex;
uniform float uThreshold;
out vec4 o;
void main() {
    vec3 c = texture(uTex, vUv).rgb;
    float l = max(max(c.r, c.g), c.b);
    o = vec4(c * clamp((l - uThreshold) / max(l, 1e-4), 0.0, 1.0), 1.0);
}";

        /// <summary>Single-target separable Gaussian (bloom blur).</summary>
        public const string Blur1Fs = @"#version 330 core
in vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uDir;
out vec4 o;
void main() {
    const float w0 = 0.2270270270, w1 = 0.1945945946, w2 = 0.1216216216, w3 = 0.0540540541, w4 = 0.0162162162;
    o = texture(uTex, vUv) * w0
      + (texture(uTex, vUv + uDir) + texture(uTex, vUv - uDir)) * w1
      + (texture(uTex, vUv + 2.0 * uDir) + texture(uTex, vUv - 2.0 * uDir)) * w2
      + (texture(uTex, vUv + 3.0 * uDir) + texture(uTex, vUv - 3.0 * uDir)) * w3
      + (texture(uTex, vUv + 4.0 * uDir) + texture(uTex, vUv - 4.0 * uDir)) * w4;
}";

        /// <summary>Final: scene + bloom, ACES filmic tone map, vignette, gamma, and dither (no banding in dark gradients).</summary>
        public const string ToneFs = @"#version 330 core
in vec2 vUv;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uBloomStrength;
uniform float uExposure;
out vec4 o;
vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
void main() {
    vec3 c = texture(uScene, vUv).rgb + texture(uBloom, vUv).rgb * uBloomStrength;
    c = aces(c * uExposure);
    vec2 q = vUv - 0.5;
    c *= 1.0 - 0.28 * dot(q, q) * 2.0;
    c = pow(c, vec3(1.0 / 2.2));
    float dither = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) / 255.0;
    o = vec4(c + dither, 1.0);
}";

        /// <summary>Plain texture copy (final image → window).</summary>
        public const string CopyFs = @"#version 330 core
in vec2 vUv;
uniform sampler2D uTex;
out vec4 o;
void main() { o = texture(uTex, vUv); }";
    }
}
