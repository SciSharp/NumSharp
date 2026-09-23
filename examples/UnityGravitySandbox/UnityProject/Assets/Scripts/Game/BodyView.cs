using System;
using UnityEngine;
using NumSharp.Examples.GravitySandbox.Physics;

namespace NumSharp.Examples.GravitySandbox
{
    /// <summary>
    /// The visual representation of one simulated body: a shaded sphere, an optional glowing point light
    /// when the body is a star, and an optional rolling orbit trail. It is a plain class (not a
    /// MonoBehaviour) driven each frame by <see cref="GravitySandbox"/>, so the view is a pure function of
    /// the physics state — there is no per-body update script to schedule and no hidden coupling between
    /// views. Views are cheap to create and destroy, which is what lets the orchestrator rebuild the whole
    /// set when bodies are spawned or merged.
    /// </summary>
    public sealed class BodyView : IDisposable
    {
        private readonly GameObject _sphere;
        private readonly Material _sphereMat;
        private readonly LineRenderer _trail;   // null when trails are disabled for this run
        private readonly Material _trailMat;     // null when trails are disabled
        private readonly Vector3[] _trailBuf;    // fixed-length ring window fed to the LineRenderer
        private bool _trailPrimed;

        /// <summary>The world (engine) body index this view mirrors. Stable only until the next topology change.</summary>
        public int Index { get; }

        /// <summary>
        /// Builds the GameObjects for one body and parents them under <paramref name="parent"/>.
        /// </summary>
        /// <param name="index">The engine body index this view represents.</param>
        /// <param name="descriptor">The body's appearance and size.</param>
        /// <param name="parent">The scene transform to parent everything under (keeps the hierarchy tidy).</param>
        /// <param name="trailLength">Number of trail points to keep; 0 disables the trail entirely.</param>
        /// <param name="visualScale">Multiplier applied to the physical display radius when rendering.</param>
        /// <param name="minRenderRadius">
        /// A floor on the rendered radius so a physically tiny body (a planet seen from across a solar
        /// system) is still a visible dot. Purely cosmetic — it does not affect the physics, whose only use
        /// of size is the collision radius.
        /// </param>
        public BodyView(int index, in BodyDescriptor descriptor, Transform parent,
                        int trailLength, float visualScale, float minRenderRadius)
        {
            Index = index;
            Color color = SandboxRenderKit.ColorOf(descriptor);

            _sphere = SandboxRenderKit.CreateSphere(descriptor.Name ?? $"Body{index}");
            _sphere.transform.SetParent(parent, worldPositionStays: false);
            _sphereMat = SandboxRenderKit.CreateBodyMaterial(color, descriptor.IsStar);
            _sphere.GetComponent<MeshRenderer>().sharedMaterial = _sphereMat;

            // A primitive sphere is 1 unit in DIAMETER at unit scale, so localScale is the diameter = 2·radius.
            float renderRadius = Mathf.Max((float)descriptor.DisplayRadius * visualScale, minRenderRadius);
            _sphere.transform.localScale = Vector3.one * (2f * renderRadius);

            // A star is self-luminous: give it an actual point light so it illuminates the other bodies,
            // parented to the sphere so it tracks the body's motion for free.
            if (descriptor.IsStar)
            {
                var lightGo = new GameObject(_sphere.name + " Light");
                lightGo.transform.SetParent(_sphere.transform, worldPositionStays: false);
                var light = lightGo.AddComponent<Light>();
                light.type = LightType.Point;
                light.color = color;
                light.intensity = 1.6f;
                light.range = 200f;    // large so a distant planet is still lit
            }

            if (trailLength > 0)
            {
                var trailGo = new GameObject(_sphere.name + " Trail");
                trailGo.transform.SetParent(parent, worldPositionStays: false);
                _trail = trailGo.AddComponent<LineRenderer>();
                _trail.useWorldSpace = true;                       // trail points are absolute positions
                _trail.positionCount = trailLength;
                _trail.numCornerVertices = 0;
                _trailMat = SandboxRenderKit.CreateTrailMaterial(color);
                _trail.sharedMaterial = _trailMat;
                // Taper the trail so its head (the body) is bright/thick and its tail fades to nothing.
                _trail.widthMultiplier = 1f;
                _trail.startWidth = renderRadius * 0.9f;
                _trail.endWidth = 0.02f * renderRadius + 1e-3f;
                _trail.startColor = color;
                _trail.endColor = new Color(color.r, color.g, color.b, 0f);
                _trailBuf = new Vector3[trailLength];
            }
        }

        /// <summary>
        /// Moves the sphere to <paramref name="worldPos"/> and advances the trail. Called once per rendered
        /// frame by the orchestrator with the body's current position.
        /// </summary>
        /// <param name="worldPos">The body's current position in Unity world space.</param>
        public void UpdateVisual(Vector3 worldPos)
        {
            _sphere.transform.localPosition = worldPos;
            if (_trail == null) return;

            if (!_trailPrimed)
            {
                // Seed the whole window with the starting point so the fixed-length LineRenderer has no
                // stale (0,0,0) vertices before enough real samples have accumulated; the trail then
                // visibly grows out of a point.
                for (int i = 0; i < _trailBuf.Length; i++) _trailBuf[i] = worldPos;
                _trailPrimed = true;
            }
            else
            {
                // Slide the window one slot and append the newest point at the head. O(trailLength) per
                // frame, negligible for the few-hundred-point trails used here.
                Array.Copy(_trailBuf, 1, _trailBuf, 0, _trailBuf.Length - 1);
                _trailBuf[_trailBuf.Length - 1] = worldPos;
            }
            _trail.SetPositions(_trailBuf);
        }

        /// <summary>
        /// Destroys every GameObject and material this view owns. MUST be called when the view is retired
        /// (topology change or teardown) or the runtime-created materials and objects leak for the life of
        /// the play session.
        /// </summary>
        public void Dispose()
        {
            if (_trail != null) UnityEngine.Object.Destroy(_trail.gameObject);
            if (_trailMat != null) UnityEngine.Object.Destroy(_trailMat);
            if (_sphere != null) UnityEngine.Object.Destroy(_sphere);
            if (_sphereMat != null) UnityEngine.Object.Destroy(_sphereMat);
        }
    }
}
