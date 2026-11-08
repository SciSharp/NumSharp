using UnityEngine;
using NumSharp.Examples.GravitySandbox.Physics;

namespace NumSharp.Examples.GravitySandbox
{
    /// <summary>
    /// Small factory of runtime-created rendering resources (shaders, materials, colours, primitive
    /// meshes) shared by the view layer. It exists so the whole sample can be dropped onto ONE empty
    /// GameObject and run — nothing here needs an asset authored in the editor, a prefab, or a material
    /// file. It also papers over the biggest portability wrinkle of a code-only Unity sample: different
    /// render pipelines ship different default shaders and colour property names.
    /// </summary>
    /// <remarks>
    /// <b>Render-pipeline portability.</b> A material's shader and its colour/emission property names differ
    /// between the Built-in pipeline ("Standard", <c>_Color</c>/<c>_EmissionColor</c>) and URP
    /// ("Universal Render Pipeline/Lit", <c>_BaseColor</c>/<c>_EmissionColor</c>). Rather than assume one,
    /// this kit probes <see cref="Shader.Find"/> for a lit shader and sets whichever colour properties the
    /// chosen shader actually exposes. If no lit shader is present it falls back to an unlit colour shader
    /// so bodies are at least visible.
    /// </remarks>
    public static class SandboxRenderKit
    {
        // Cached lit/unlit shaders resolved once. Shader.Find is relatively expensive and returns null for
        // shaders not included in the build, so we resolve lazily and remember the result.
        private static Shader _litShader;
        private static Shader _unlitShader;

        /// <summary>Converts an engine-side <see cref="BodyDescriptor"/> colour (doubles 0..1) to a Unity <see cref="Color"/>.</summary>
        /// <param name="d">The descriptor carrying the RGB channels.</param>
        /// <returns>An opaque Unity colour.</returns>
        public static Color ColorOf(in BodyDescriptor d) =>
            new Color((float)d.ColorR, (float)d.ColorG, (float)d.ColorB, 1f);

        /// <summary>
        /// Resolves a lit surface shader for the active render pipeline, preferring URP's Lit, then the
        /// Built-in Standard, then an unlit fallback so a body is never invisible.
        /// </summary>
        /// <returns>The best available shader for shaded bodies.</returns>
        public static Shader LitShader()
        {
            if (_litShader != null) return _litShader;
            // Ordered by preference; the first that exists in the build wins.
            _litShader = Shader.Find("Universal Render Pipeline/Lit")
                         ?? Shader.Find("Standard")
                         ?? UnlitShader();
            return _litShader;
        }

        /// <summary>Resolves an unlit colour shader, used for trails and as the ultimate lit fallback.</summary>
        /// <returns>An unlit colour shader (guaranteed non-null in any standard Unity install).</returns>
        public static Shader UnlitShader()
        {
            if (_unlitShader != null) return _unlitShader;
            _unlitShader = Shader.Find("Universal Render Pipeline/Unlit")
                           ?? Shader.Find("Unlit/Color")
                           ?? Shader.Find("Sprites/Default");
            return _unlitShader;
        }

        /// <summary>
        /// Builds a material tinted <paramref name="color"/>, optionally self-luminous (for stars). Sets the
        /// colour under BOTH the Built-in and URP property names — whichever the resolved shader exposes —
        /// so the tint takes regardless of pipeline.
        /// </summary>
        /// <param name="color">The base colour.</param>
        /// <param name="emissive">
        /// When true, adds an emission term so the body appears to glow and (paired with a real light in
        /// <see cref="BodyView"/>) reads as a star rather than a lit sphere.
        /// </param>
        /// <returns>A new material instance owned by the caller (destroy it with the body).</returns>
        public static Material CreateBodyMaterial(Color color, bool emissive)
        {
            var mat = new Material(LitShader());
            // Set every colour property the shader might use; HasProperty guards keep it warning-free.
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", color);

            if (emissive)
            {
                // Emission needs both the keyword and the colour to show under the Built-in pipeline; URP
                // reads _EmissionColor directly. Scale below 1 so the glow doesn't blow out to pure white.
                mat.EnableKeyword("_EMISSION");
                if (mat.HasProperty("_EmissionColor")) mat.SetColor("_EmissionColor", color * 0.9f);
            }
            return mat;
        }

        /// <summary>
        /// Builds an unlit material for a trail line, tinted <paramref name="color"/>. Trails are unlit so
        /// they read as pure colour streaks independent of scene lighting.
        /// </summary>
        /// <param name="color">The trail colour.</param>
        /// <returns>A new unlit material instance owned by the caller.</returns>
        public static Material CreateTrailMaterial(Color color)
        {
            var mat = new Material(UnlitShader());
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", color);
            return mat;
        }

        /// <summary>
        /// Creates a unit-diameter sphere GameObject with its auto-added collider removed (the sample does
        /// its own collision physics; a Unity collider would only add overhead and unwanted interactions).
        /// </summary>
        /// <param name="name">The GameObject name (handy in the hierarchy while debugging).</param>
        /// <returns>A sphere GameObject positioned at the origin with unit scale.</returns>
        public static GameObject CreateSphere(string name)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = name;
            // The sample integrates its own gravity/collisions; the primitive's SphereCollider is dead
            // weight and could interact with Unity physics, so drop it.
            var col = go.GetComponent<Collider>();
            if (col != null) Object.Destroy(col);
            return go;
        }
    }
}
