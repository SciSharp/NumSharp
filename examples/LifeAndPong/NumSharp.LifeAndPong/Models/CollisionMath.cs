using System.Numerics;

namespace NumSharp.LifeAndPong.Models;

internal readonly record struct SweepHit(double Time, Vector2 Normal, float Penetration = 0);
internal readonly record struct ContactConstraint(Vector2 Normal, Vector2 SurfaceVelocity);

/// <summary>Analytic swept geometry and rigid-body impulses; no gameplay steering or speed targets.</summary>
internal static class CollisionMath
{
    internal const double TimeTolerance = 1e-8;

    internal static bool SweepRoundedBox(Vector2 position, Vector2 relativeVelocity, double maxTime,
        float ballRadius, float x, float y, float width, float height, float cornerRadius, out SweepHit hit)
    {
        var radius = Math.Clamp(cornerRadius, 0, Math.Min(width, height) / 2);
        var left = x + radius; var right = x + width - radius;
        var top = y + radius; var bottom = y + height - radius;
        var combined = ballRadius + radius;
        var closest = new Vector2(Math.Clamp(position.X, left, right), Math.Clamp(position.Y, top, bottom));
        var offset = position - closest;
        var distance = offset.Length();
        // Initial overlap recovery matters for roundoff and moving shapes, not normal collision timing.
        if (distance < combined - .0001f)
        {
            Vector2 normal; float depth;
            if (distance > .00001f) { normal = offset / distance; depth = combined - distance; }
            else
            {
                var dl = position.X - left; var dr = right - position.X;
                var dt = position.Y - top; var db = bottom - position.Y;
                var minimum = Math.Min(Math.Min(dl, dr), Math.Min(dt, db));
                normal = minimum == dl ? -Vector2.UnitX : minimum == dr ? Vector2.UnitX : minimum == dt ? -Vector2.UnitY : Vector2.UnitY;
                depth = combined + minimum;
            }
            if (Vector2.Dot(relativeVelocity, normal) < -1e-5f) { hit = new SweepHit(0, normal, depth); return true; }
        }
        var bestTime = double.PositiveInfinity; var bestNormal = Vector2.Zero;
        void Consider(double time, Vector2 normal)
        {
            if (time < -TimeTolerance || time > maxTime + TimeTolerance || time >= bestTime || Vector2.Dot(relativeVelocity, normal) >= -1e-5f) return;
            bestTime = Math.Max(0, time); bestNormal = normal;
        }
        if (relativeVelocity.X != 0)
        {
            var time = ((relativeVelocity.X > 0 ? left - combined : right + combined) - position.X) / (double)relativeVelocity.X;
            var atY = position.Y + relativeVelocity.Y * time;
            if (atY >= top - .00001 && atY <= bottom + .00001) Consider(time, relativeVelocity.X > 0 ? -Vector2.UnitX : Vector2.UnitX);
        }
        if (relativeVelocity.Y != 0)
        {
            var time = ((relativeVelocity.Y > 0 ? top - combined : bottom + combined) - position.Y) / (double)relativeVelocity.Y;
            var atX = position.X + relativeVelocity.X * time;
            if (atX >= left - .00001 && atX <= right + .00001) Consider(time, relativeVelocity.Y > 0 ? -Vector2.UnitY : Vector2.UnitY);
        }
        var a = (double)relativeVelocity.X * relativeVelocity.X + (double)relativeVelocity.Y * relativeVelocity.Y;
        if (a > 1e-12)
            for (var iy = 0; iy < 2; iy++) for (var ix = 0; ix < 2; ix++)
            {
                var cx = ix == 0 ? left : right; var cy = iy == 0 ? top : bottom;
                var mx = (double)position.X - cx; var my = (double)position.Y - cy;
                var b = mx * relativeVelocity.X + my * relativeVelocity.Y;
                var c = mx * mx + my * my - (double)combined * combined;
                var discriminant = b * b - a * c;
                if (discriminant < 0 || b >= 0) continue;
                // Stable smaller quadratic root avoids cancellation near a fast impact.
                var denominator = -b + Math.Sqrt(discriminant);
                if (denominator <= 0) continue;
                var time = c / denominator;
                var nx = mx + relativeVelocity.X * time; var ny = my + relativeVelocity.Y * time;
                if ((ix == 0 ? -nx : nx) < -.00001 || (iy == 0 ? -ny : ny) < -.00001) continue;
                var n = new Vector2((float)nx, (float)ny);
                if (n.LengthSquared() > 1e-10f) Consider(time, Vector2.Normalize(n));
            }
        hit = new SweepHit(bestTime, bestNormal);
        return double.IsFinite(bestTime);
    }

    internal static Vector2 ElasticManifold(Vector2 velocity, IReadOnlyList<ContactConstraint> contacts)
    {
        var best = velocity; var leastImpulse = double.PositiveInfinity;
        double Required(int i) => -2 * Vector2.Dot(velocity - contacts[i].SurfaceVelocity, contacts[i].Normal);
        void Consider(Vector2 candidate)
        {
            for (var k = 0; k < contacts.Count; k++)
                if (Vector2.Dot(candidate - velocity, contacts[k].Normal) < Required(k) - .003) return;
            var change = (candidate - velocity).LengthSquared();
            if (change < leastImpulse) { best = candidate; leastImpulse = change; }
        }
        // In 2D a feasible impulse has at most two independent active constraints.
        for (var i = 0; i < contacts.Count; i++)
        {
            var a = Math.Max(0, Required(i));
            Consider(velocity + contacts[i].Normal * (float)a);
            for (var j = i + 1; j < contacts.Count; j++)
            {
                var dot = (double)Vector2.Dot(contacts[i].Normal, contacts[j].Normal);
                var determinant = 1 - dot * dot;
                if (determinant < 1e-9) continue;
                var b = Math.Max(0, Required(j));
                var first = (a - dot * b) / determinant; var second = (b - dot * a) / determinant;
                if (first < -.0001 || second < -.0001) continue;
                Consider(velocity + contacts[i].Normal * (float)first + contacts[j].Normal * (float)second);
            }
        }
        if (double.IsFinite(leastImpulse)) return best;
        // Degenerate overlap recovery: project out of closing directions rather than inventing a steering angle.
        best = velocity;
        for (var pass = 0; pass < 12; pass++) foreach (var contact in contacts)
        {
            var closing = Vector2.Dot(best - contact.SurfaceVelocity, contact.Normal);
            if (closing < 0) best -= contact.Normal * closing;
        }
        return best;
    }

    internal static (Vector2 Velocity, float Spin) PaddleFriction(Vector2 incoming, Vector2 reflected,
        Vector2 paddleVelocity, Vector2 normal, float spin, float radius, float friction = .08f)
    {
        // Unit-mass solid disc: I = m*r^2/2. Coulomb-limited tangent impulse includes contact-point spin.
        var tangent = new Vector2(-normal.Y, normal.X);
        var normalImpulse = Math.Max(0, Vector2.Dot(reflected - incoming, normal));
        var slip = Vector2.Dot(reflected - paddleVelocity, tangent) - spin * radius;
        var tangentImpulse = Math.Clamp(-slip / 3, -friction * normalImpulse, friction * normalImpulse);
        return (reflected + tangent * tangentImpulse, spin - 2 * tangentImpulse / radius);
    }

    internal static Vector2 DirectionNoise(Vector2 velocity, float fraction)
    {
        if (velocity.LengthSquared() < 1e-12f) return velocity;
        fraction = Math.Clamp(fraction, -.05f, .05f);
        var perpendicular = new Vector2(-velocity.Y, velocity.X);
        return Vector2.Normalize(velocity + perpendicular * fraction) * velocity.Length();
    }
    internal static Vector2 SafeNoise(Vector2 velocity, float fraction, IReadOnlyList<ContactConstraint> contacts)
    {
        bool Valid(Vector2 candidate) => contacts.All(contact => Vector2.Dot(candidate - contact.SurfaceVelocity, contact.Normal) >= -.001f);
        var noisy = DirectionNoise(velocity, fraction);
        if (Valid(noisy)) return noisy;
        // Reduce the same random draw, never resample or redirect beyond the requested 5% bound.
        var low = 0f; var high = 1f;
        for (var i = 0; i < 20; i++)
        {
            var middle = (low + high) / 2;
            if (Valid(DirectionNoise(velocity, fraction * middle))) low = middle; else high = middle;
        }
        return DirectionNoise(velocity, fraction * low);
    }
}
