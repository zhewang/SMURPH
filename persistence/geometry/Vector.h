#ifndef VECTOR_H
#define VECTOR_H

#include <algorithm>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <math.h>

using namespace std;

class Vector  {
	public:
		Vector();
		Vector(string _file);
		Vector(int _n);
		Vector(double *_d, int _n);
		Vector(vector<double> _vector);
		Vector(const Vector & _vector);
		Vector(Vector* _vector);
		~Vector();

		int dim() const { return n; };
		double* data()  { return d; };
		void setEntry(int _n, double _entry)  { d[_n] = _entry; }
		void incrementEntry(int _n, double _entry)  { d[_n] += _entry; }
		void scaleEntry(int _n, double _scale)  { d[_n] *= _scale; }
		double const getEntry(int _n) const { return d[_n]; }

		double* dataCopy() const {
			double* copy = new double[n];
			for(int i = 0; i < n; i++)
				copy[i] = d[i];
			return copy;
		}

		Vector & operator=(const Vector & _vec)  {
			if(this == &_vec)
				return *this;
			if(n > 0)
				delete [] d;
			n = _vec.dim();
			d = new double[n];
			for(int i = 0; i < n; i++)
				d[i] = _vec.getEntry(i);
			return *this;
		}

		Vector operator+(const Vector & _vec) const {
			Vector result(n);
			if(_vec.dim() != n)  {
				cerr << "[VECTOR] unable to perform addition: mismatched dimensionality." << endl;
				return result;
			}

			for(int i = 0; i < n; i++)
				result.setEntry(i,d[i]+_vec.getEntry(i));

			return result;
		}

		Vector operator-(const Vector & _vec) const {
			Vector result(n);
			if(_vec.dim() != n)  {
				cerr << "[VECTOR] unable to perform subtraction: mismatched dimensionality." << endl;
				return result;
			}

			for(int i = 0; i < n; i++)
				result.setEntry(i,d[i]-_vec.getEntry(i));

			return result;
		}

		double operator*(const Vector & _vec) const {
			if(_vec.dim() != n)  {
				cerr << "[VECTOR] unable to perform dot product: mismatched dimensionality." << endl;
				return 0;
			}

			double result = 0;
			for(int i = 0; i < n; i++)
				result += d[i]*_vec.getEntry(i);

			return result;
		}

		Vector operator*(double _scale) const {
			Vector result(n);
			for(int i = 0; i < n; i++)
				result.setEntry(i,d[i]*_scale);
			return result;
		}

		double operator[](int _i) const  {
			return d[_i];
		}

		inline bool operator==(const Vector& _otherVec) const  {
			if(_otherVec.dim() != n)
				return false;
			for(int i = 0; i < n; i++)  {
				if(d[i] != _otherVec[i])
					return false;
			}
			return true;
		}

		double lInfNorm() const;
		double l2Norm() const;
		double l1Norm() const;
		double sqd_dist(const Vector & _vec) const;
		double length(const Vector & _vec) const;
		double l_infinity_dist(const Vector & _vec) const;
		void normalize();

		double coherence() const;

		void minAndMax(int* min_index, double* minimum, int* max_index, double* maximum);
		void min(int* index, double* minimum);
		void max(int* index, double* maximum);

		void expand_min_bound(const Vector & _vec);
		void expand_max_bound(const Vector & _vec);

		void write_to_file(string _filename, bool sort=true);

		friend std::ostream& operator <<(std::ostream &out, const Vector & _vec)  {
			for(int i = 0; i < _vec.dim(); i++)  {
				out << " " << _vec.getEntry(i);
			}
			return out;
		}

		void tokenize_line(vector<string>* tokens, const string& input, string sep);

	private:
		double* d;
		int n;
};

#endif
