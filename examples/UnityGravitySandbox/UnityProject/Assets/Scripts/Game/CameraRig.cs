using UnityEngine;

namespace NumSharp.Examples.GravitySandbox
{
    /// <summary>
    /// A self-contained orbit / pan / zoom camera controller, added at runtime to the camera GameObject by
    /// <see cref="GravitySandbox"/>. It frames a focus point (the simulation's centre of mass, which the
    /// scenarios keep at the origin) and lets the player look around a fundamentally 3-D system even though
    /// the sample scenarios happen to be planar. Kept as its own MonoBehaviour so camera control is
    /// isolated from simulation logic.
    /// </summary>
    /// <remarks>
    /// Controls: hold the RIGHT mouse button and drag to orbit, hold the MIDDLE button and drag to pan the
    /// focus, and scroll to zoom. Uses the legacy Input Manager axes ("Mouse X"/"Mouse Y"/scroll), so the
    /// project's Active Input Handling must include the old system (see the sample README).
    /// </remarks>
    public sealed class CameraRig : MonoBehaviour
    {
        /// <summary>The point the camera looks at and orbits. Panning moves it; scenarios reset it to the origin.</summary>
        public Vector3 Target = Vector3.zero;

        /// <summary>Distance from <see cref="Target"/> to the camera, in world units. Zoom changes it.</summary>
        public float Distance = 12f;

        /// <summary>Orbit angles in degrees: yaw around the vertical axis, pitch above the plane.</summary>
        public float Yaw = 35f;
        /// <summary>Pitch angle in degrees, clamped to avoid flipping over the poles.</summary>
        public float Pitch = 28f;

        /// <summary>Degrees of orbit per unit of mouse movement.</summary>
        public float OrbitSpeed = 4f;
        /// <summary>Fraction of distance panned per unit of mouse movement (so panning feels the same at any zoom).</summary>
        public float PanSpeed = 0.5f;
        /// <summary>Zoom responsiveness for the scroll wheel; larger zooms faster.</summary>
        public float ZoomSpeed = 0.15f;
        /// <summary>The closest the camera may zoom in.</summary>
        public float MinDistance = 0.5f;
        /// <summary>The farthest the camera may zoom out.</summary>
        public float MaxDistance = 500f;

        /// <summary>
        /// Re-frames the rig on a fresh scenario: recenters on the origin and sets a sensible starting
        /// distance. Called by the orchestrator whenever a scenario loads so each demo opens well-framed.
        /// </summary>
        /// <param name="distance">The starting camera distance suggested by the scenario.</param>
        public void Frame(float distance)
        {
            Target = Vector3.zero;
            Distance = Mathf.Clamp(distance, MinDistance, MaxDistance);
        }

        /// <summary>
        /// Applies input and repositions the camera after the simulation has moved for the frame. LateUpdate
        /// (not Update) is used so the camera reacts to the final positions of this frame, avoiding a
        /// one-frame lag between the bodies and the view.
        /// </summary>
        private void LateUpdate()
        {
            // Orbit while the right mouse button is held.
            if (Input.GetMouseButton(1))
            {
                Yaw += Input.GetAxis("Mouse X") * OrbitSpeed;
                Pitch -= Input.GetAxis("Mouse Y") * OrbitSpeed;
                Pitch = Mathf.Clamp(Pitch, -89f, 89f);   // never let the camera flip past vertical
            }

            // Pan the focus while the middle mouse button is held; scale by distance so the world tracks
            // the cursor at any zoom level.
            if (Input.GetMouseButton(2))
            {
                float mx = Input.GetAxis("Mouse X");
                float my = Input.GetAxis("Mouse Y");
                Vector3 right = transform.right, up = transform.up;
                Target -= (right * mx + up * my) * (PanSpeed * Distance * 0.05f);
            }

            // Zoom multiplicatively so each scroll notch changes the view by a constant PROPORTION — the
            // natural feel, and it keeps the step size sane across the huge distance range of the scenarios.
            float scroll = Input.mouseScrollDelta.y;
            if (Mathf.Abs(scroll) > 0f)
                Distance = Mathf.Clamp(Distance * Mathf.Exp(-scroll * ZoomSpeed), MinDistance, MaxDistance);

            // Compose the final transform from the orbit angles and distance.
            Quaternion rot = Quaternion.Euler(Pitch, Yaw, 0f);
            transform.rotation = rot;
            transform.position = Target - (rot * Vector3.forward) * Distance;
        }
    }
}
