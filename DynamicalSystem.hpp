#include <iostream>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace std;

class DynamicalSystem {
	public:
		unsigned get_time_steps() { return time_steps; }
		unsigned get_state_size() { return state_size; }
		double get_t0() { return t0; }
		double get_t1() { return t1; }
		double* get_x0() { return x0; }
		double* get_X() { return X; }
		void print() {
			cout << "X = [" << endl;
			for (int i = 0;i < time_steps;i++) {
				cout << "[ " << X[i*state_size];
				for (int j = 1; j < state_size; j++) {
					cout << " , " << X[i*state_size + j];
				}
				cout << " ], " << endl;
			}
			cout << "]" << endl;
		}
		void f(double t, double *x, double *y) {
			for (int i = 0;i < state_size; i++) {
				y[i] = (i+1) * x[i]; //this is just a exponintal function for each i
			}
		}

		DynamicalSystem(unsigned state_size_, unsigned time_steps_, double t0_, double t1_, double *x0_) {
			state_size = state_size_;
			time_steps = time_steps_;
			t0 = t0_;
			t1 = t1_;
			x0 = x0_;
			X = new double [time_steps*state_size];
			dX = new double [time_steps*state_size];
			for (int i = 0;i < state_size;i++) {  X[i] = x0[i]; }
			solve();
		};
		~DynamicalSystem() {
			delete []X;
			delete []dX;
		}
	private:
		unsigned state_size;
		unsigned time_steps;
		double t0;
		double t1;
		double *x0;
		double *X;
		double *dX;

		void solve() {
			double *k1 = new double[state_size];
			double *k2 = new double[state_size];
			double *k3 = new double[state_size];
			double *k4 = new double[state_size];
			double *tmp = new double[state_size];
			double *x = new double[state_size];

			double h = (t1-t0)/(time_steps-1);
			double t;
			unsigned n = std::max((int) (100*h),10); // this prevents the step size from ever being larger than 0.01
			for (int i = 1;i < time_steps; i++) {
				for (int j = 0; j < state_size; j++) { x[j] = X[(i-1)*state_size+j]; }
				step(k1,k2,k3,k4,tmp,x,t0+h*(i-1),t0+h*i,n);
				for (int j = 0; j < state_size; j++) { X[i*state_size+j] = x[j]; }
			}
			delete []k1; delete []k2; delete []k3; delete []k4; delete []tmp, delete []x;
		}
		void step(double *k1, double *k2, double *k3, double *k4, double *tmp, double *x, double t0, double t1, unsigned n) {
			double tmp1;
			double h = (t1-t0) / n;
			double t;
			for (int i = 1;i<=n;i++) {
				t = t0 + i*h;
				for (int j = 0; j < state_size; j++) { tmp[j] = x[j]; }
				f(t,tmp,k1);
				for (int j = 0; j < state_size; j++) { tmp[j] = x[j] + (h/2.0)*k1[j]; }
				f(t+h/2.0, tmp, k2);
				for (int j = 0; j < state_size; j++) { tmp[j] = x[j] + (h/2.0)*k2[j]; }
				f(t+h/2.0, tmp, k3);
				for (int j = 0; j < state_size; j++) { tmp[j] = x[j] + h*k3[j]; }
				f(t+h, tmp, k4);
				for (int j = 0; j < state_size; j++) { tmp1 = x[j]; x[j] = tmp1 + (h/6.0)*(k1[j]+2*k2[j]+2*k3[j]+k4[j]); }
			}
		}
		

};

class SimplePendulum: public DynamicalSystem {
	public:
		double g;
		double l;
		double mu;
	public:
		void print() {
			DynamicalSystem::print();
		}
		void f(double t, double *x, double *y) {
			y[0] = x[1];
			y[1] = (-g / l) * std::sin(x[0]) - mu*x[1];
		}
		SimplePendulum(unsigned time_steps_, double t0_, double t1_, pybind11::array_t<double> x0_,double g_, double l_, double mu_) 
			: DynamicalSystem(2,time_steps_,t0_,t1_,(double*) x0_.request().ptr) {
				g = g_;
				l = l_;
				mu = mu_;
		};
};
