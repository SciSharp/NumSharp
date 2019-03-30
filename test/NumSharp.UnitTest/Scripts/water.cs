using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class ScriptCheck
    {
        public double grid_spacing;
        public double g;
        public double dt;


        [TestMethod]
        public void Water()
        {
            int box_size = 1;

            // Initial Conditions
            int n = 100;
            var u = np.zeros(n,n); // velocity in x direction
            var v = np.zeros(n,n); // velocity in y direction

            var eta = np.ones(n,n); // pressure deviation (like height)
            (var x,var y) = np.mgrid(np.arange(0,n,1),np.arange(0,n,1));

            double droplet_x = 50; 
            double droplet_y = 50;

            var rr = (x-droplet_x)*(x-droplet_x) + (y-droplet_y)*(y-droplet_y);

            eta[rr<100] = 1.1; //# add a perturbation in pressure surface

            grid_spacing =  1.0*box_size  * (1 / n);
            g = 1.0;

            dt = grid_spacing / 100.0;

            //this.Demo(eta,u,v,g,dt);
        }
        public NDArray spatial_derivative(NDArray A,int axis=0)
        {
            return (A.roll(-1, axis) - A.roll(1, axis)) / (grid_spacing*2.0);
        }
        public NDArray d_dx(NDArray A)
        {
            return this.spatial_derivative(A,1);
        }
        public NDArray d_dy(NDArray A)
        {
            return this.spatial_derivative(A,0);
        }
        public (NDArray,NDArray,NDArray) d_dt(NDArray eta, NDArray u,NDArray v, double g, double b )
        {
            var du_dt = -g*d_dx(eta) - b*u;
            var dv_dt = -g*d_dy(eta) - b*v;

            var H = 0; //#eta.mean() - our definition of eta includes this term
            var deta_dt = (-1) * d_dx(u * (H+eta)) - d_dy(v * (H+eta));

            return (deta_dt, du_dt, dv_dt);
        }
        public IEnumerable<(NDArray,NDArray,NDArray,double)> evolveEuler(NDArray eta, NDArray u,NDArray v, double g, double dt)
        {
            double time = 0;

            NDArray deta_dt = null;
            NDArray du_dt = null;
            NDArray dv_dt = null;
            
            while(true)
            {
                (deta_dt, du_dt, dv_dt) = d_dt(eta, u, v, g,0);           
                eta = eta + deta_dt * dt;
                u = u + du_dt * dt;
                v = v + dv_dt * dt;
                time += dt;
                yield return (deta_dt, du_dt, dv_dt,time);
            }

        }
        public void Demo(NDArray eta, NDArray u, NDArray v, double g, double dt, double endtime = 0.3)
        {
            var trajectory = evolveEuler(eta, u, v, g, dt).GetEnumerator();

            (var eta_, var u_, var v_, var time ) = trajectory.Current;

            while(true)
            {
                trajectory.MoveNext();

                 (eta_, u_, v_, time ) = trajectory.Current;



            }
        }
    }
}


        

            

