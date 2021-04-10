using System;
using System.Collections;
using System.Collections.Generic;

namespace NumSharp.Examples
{
    public class ShallowWater : IExample
    {
        public static double grid_spacing;
        public static double g;
        public static double dt;

        public void Run()
        {
            double box_size = 1;

            // Initial Conditions
            int n = 100;
            var u = np.zeros(n, n) + 0.001; // velocity in x direction
            var v = np.zeros(n, n) - 0.001; // velocity in y direction

            var eta = np.ones(n, n); // pressure deviation (like height)
            (var x, var y) = np.mgrid(np.arange(0, n, 1), np.arange(0, n, 1));

            double droplet_x = 50;
            double droplet_y = 50;

            var rr = (x - droplet_x) * (x - droplet_x) + (y - droplet_y) * (y - droplet_y);

            eta[rr < 100] = 1.1; //# add a perturbation in pressure surface

            grid_spacing = 1.0 * box_size * (1.0 / n);
            g = 1.0;

            dt = grid_spacing / 100.0;

            var trajectory = evolveEuler(eta, u, v, g, dt).GetEnumerator();

            while (trajectory.MoveNext())
            {
                (var eta_, var u_, var v_, var time) = trajectory.Current;
                Console.WriteLine(time);
            }
        }

        public static NDArray spatial_derivative(NDArray A,int axis=0)
        {
            return (A.roll(-1, axis) - A.roll(1, axis)) / (grid_spacing*2.0);
        }

        public static NDArray d_dx(NDArray A)
        {
            return spatial_derivative(A,1);
        }

        public static NDArray d_dy(NDArray A)
        {
            return spatial_derivative(A,0);
        }

        public static (NDArray,NDArray,NDArray) d_dt(NDArray eta, NDArray u,NDArray v, double g, double b )
        {
            var du_dt = -g*d_dx(eta) - b*u;
            var dv_dt = -g*d_dy(eta) - b*v;

            var H = 0; //#eta.mean() - our definition of eta includes this term
            var deta_dt = (-1) * d_dx(u * (H+eta)) - d_dy(v * (H+eta));

            return (deta_dt, du_dt, dv_dt);
        }

        public static IEnumerable<(NDArray,NDArray,NDArray,double)> evolveEuler(NDArray eta, NDArray u,NDArray v, double g, double dt)
        {
            double time = 0;

            NDArray deta_dt = null;
            NDArray du_dt = null;
            NDArray dv_dt = null;
            
            while(true)
            {
                yield return (eta,u,v,time);

                (deta_dt, du_dt, dv_dt) = d_dt(eta, u, v, g,1);           
                eta = eta + deta_dt * dt;
                u = u + du_dt * dt;
                v = v + dv_dt * dt;
                time += dt;
            }

        }
    }
}
