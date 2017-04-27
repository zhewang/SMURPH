#ifndef FPSRIPSFILTRATION_H
#define FPSRIPSFILTRATION_H

#include "rips_filtration.h"

class FPSRipsFiltration : public RipsFiltration {

	public:
		FPSRipsFiltration(const Points & _points, int _maxD, double _eps=1.5);
		FPSRipsFiltration(int _numPoints, double** _distanceMat, int _maxD, double _eps=1.5);
		~FPSRipsFiltration();

		virtual bool build_filtration();

	private:
		double eps;
		std::vector<double> deletion_times;

		void sparse_bron_kerbosch(std::vector<Simplex>* _rips, std::vector<int> _R, const std::set<int> & _P, int _maxD);

		void fps(int* sampling, double* fps_distances);
};

#endif
